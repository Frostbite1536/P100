# p7c-cpu-ref-fp32 follow-up (2026-10-10)

Patch 0003 keeps its filename and is regenerated with `git format-patch -1 --stdout`
as **one commit**, `35d17d42cdb92fd34eff206567f3c36b79c6e4ec`, on pin
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`. It includes the original biased-V test
diff verbatim plus the CPU reference accumulator fix. The patch body records the
manager's finding: all 16 biased-V cases failed in both Maxwell M2200 builds,
with NMSE about 0.36 at 16K keys and 0.60 at 32K. The previous Maxwell outputs
are preserved. The new Maxwell comparisons are pending the manager's rerun;
nothing on a P100 has been run.

## Confirmed code path and scope

At the pin, `tests/test-backend-ops.cpp:12647-12651` obtains
`ggml_backend_cpu_set_use_ref` and calls it with `true` for the comparison backend
(the patched test file has these at lines 12695-12699). In the updated
`ggml/src/ggml-cpu/ops.cpp`, lines 9486-9490 disable KV splitting for reference
mode, line 9547 disables tiling, and line 9569 calls
`ggml_compute_forward_flash_attn_ext_f16_one_chunk` over keys `0..nek1`.
The original F16 branch at pin lines 8992-9006 scales and multiply-adds into
`VKQ16` with `ggml_vec_scale_f16` and `ggml_vec_mad_f16`.

All three F16 accumulator conditions now include `!params->use_ref`. For
reference mode with F16 V, initialization uses `VKQ32`, the existing branch
converts V with `v_to_float` into `V32` and uses `ggml_vec_scale_f32` and
`ggml_vec_mad_f32` (updated lines 9015 and 9022-9024), and the final conversion
from `VKQ16` is skipped. F16's converter is `ggml_fp16_to_fp32_row`
(`ggml/src/ggml.c:669-674`). Existing scratch buffers suffice: `V32` shares
storage with `VKQ16`, and the reference path never uses the latter.

For `params->use_ref == false`, each new condition evaluates exactly as the old
`v->type == GGML_TYPE_F16` condition; initialization, accumulation, final
conversion, and dispatch retain their prior behavior. CUDA kernels are unchanged.
This patch is test-only in purpose: it fixes reference mode and retains the
existing error metric and `5e-4` tolerance.

## Verbatim ops.cpp diff

```diff
diff --git a/ggml/src/ggml-cpu/ops.cpp b/ggml/src/ggml-cpu/ops.cpp
index c7e7c47a7..312713dab 100644
--- a/ggml/src/ggml-cpu/ops.cpp
+++ b/ggml/src/ggml-cpu/ops.cpp
@@ -8940,7 +8940,8 @@ static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(
         ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
         ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16
 
-        if (v->type == GGML_TYPE_F16) {
+        // Keep the reference output accumulator in FP32, including for F16 V.
+        if (v->type == GGML_TYPE_F16 && !params->use_ref) {
             memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));
         } else {
             memset(VKQ32, 0, DV*sizeof(float));
@@ -8989,7 +8990,7 @@ static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(
 
             const char * v_data = ((const char *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));
 
-            if (v->type == GGML_TYPE_F16) {
+            if (v->type == GGML_TYPE_F16 && !params->use_ref) {
                 if (s > M) {
                     // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                     M = s;
@@ -9030,7 +9031,7 @@ static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(
             S = S*ms + vs; // scale and increment sum with partial sum
         }
 
-        if (v->type == GGML_TYPE_F16) {
+        if (v->type == GGML_TYPE_F16 && !params->use_ref) {
             for (int64_t d = 0; d < DV; ++d) {
                 VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
             }
```

## Uniform-attention stall arithmetic

Binary16 has 11 significant bits. At a positive power of two `x`, the upward
ulp is `x/1024`. With `N` equal logits, `P = 1/N`, so the mean biased-channel
contribution is `2P = 2/N`. The CPU loop actually sums unnormalized V (weights
`vs = 1`) and divides by `S = N` at the end. Because these N values are powers
of two, the normalized values below express the same rounding thresholds.

For biased V in `[1,3]`, at raw accumulator 8192 the upward ulp is 8 and
half an ulp is 4. Every subsequent contribution is at most 3, so it rounds
back to 8192 while the FP32 denominator keeps growing.

| N | normalized stopped value `8192/N` | FP16 ulp there | half ulp | mean `2P` | largest `3P` |
| --- | --- | --- | --- | --- | --- |
| 16384 | 0.5 | 0.00048828125 | 0.000244140625 | 0.0001220703125 | 0.00018310546875 |
| 32768 | 0.25 | 0.000244140625 | 0.0001220703125 | 0.00006103515625 | 0.000091552734375 |

Thus even `3P` is below half an ulp in each row, and the ideal constant-mean
output is 2. For exactly constant `V = 2`, accumulation stops earlier:
raw sum 4096 has ulp 4; adding 2 is a midpoint and rounds back to the even
value 4096. The corresponding outputs are 0.25 at 16K and 0.125 at 32K.
These are arithmetic results for uniform attention. The manager's NMSE values
above are measurements from random Q/K/V cases.

## Rebuilds and required validation

Every shell command below starts with:

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH;
```

The sequential check starts from a clean tracked checkout at the pin, then
checks and applies each predecessor. All Git writes are inside the private
llama.cpp clone; no Git writes were made in the P100 repository.

```bash
git -C third_party/llama.cpp checkout --detach "$(cat bench/LLAMA_COMMIT)"
test -z "$(git -C third_party/llama.cpp status --porcelain --untracked-files=no)"
for p in "$PWD"/third_party/patches/0001-*.patch "$PWD"/third_party/patches/0002-*.patch "$PWD"/third_party/patches/0003-*.patch; do
    git -C third_party/llama.cpp apply --check "$p" && git -C third_party/llama.cpp apply "$p" || exit 1
done
```

```text
0001-CUDA-fewer-idle-threads-when-converting-short-non-co.patch: git apply --check exit=0
0002-CUDA-vectorized-dequantization-of-non-contiguous-q4_.patch: git apply --check exit=0
0003-tests-biased-V-flash-attention-cases.patch: git apply --check exit=0
```

The original incremental build directory is reused sequentially. Complete `bin`
snapshots keep each variant's own executable and libraries. The original
`build-upstream-snapshot` and `build-conv8-snapshot` directories are retained.

```bash
git -C third_party/llama.cpp checkout -- .
git -C third_party/llama.cpp apply "$PWD/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch"
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-upstream -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=52;60' -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF -DCMAKE_BUILD_RPATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib
cmake --build third_party/llama.cpp/build-upstream --target test-backend-ops -j 2
mkdir -p third_party/llama.cpp/build-upstream-snapshot-cpuref32
cp -a third_party/llama.cpp/build-upstream/bin third_party/llama.cpp/build-upstream-snapshot-cpuref32/
git -C third_party/llama.cpp apply "$PWD/third_party/patches/0001-CUDA-fewer-idle-threads-when-converting-short-non-co.patch"
git -C third_party/llama.cpp apply "$PWD/third_party/patches/0002-CUDA-vectorized-dequantization-of-non-contiguous-q4_.patch"
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-upstream -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=52;60' -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF -DCMAKE_BUILD_RPATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib
cmake --build third_party/llama.cpp/build-upstream --target test-backend-ops -j 2
mkdir -p third_party/llama.cpp/build-conv8-snapshot-cpuref32
cp -a third_party/llama.cpp/build-upstream/bin third_party/llama.cpp/build-conv8-snapshot-cpuref32/
git -C third_party/llama.cpp checkout -- .
git -C third_party/llama.cpp apply "$PWD/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch"
```

Both configurations exited 0. The respective final build lines were:

```text
(a) [14/14] Linking CXX executable bin/test-backend-ops
(b) [10/10] Linking CXX executable bin/test-backend-ops
```

The clone source ends at the pin plus the updated 0003. Incremental logs are
`/tmp/p7c-upstream.configure.txt`, `/tmp/p7c-upstream.build.txt`,
`/tmp/p7c-conv8.configure.txt`, and `/tmp/p7c-conv8.build.txt`.

Both builds emitted the CUDA 12.8 warning covering sm_52 and sm_60:

```text
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
```

No new warning explicitly naming sm_60 occurred.

```bash
/home/p51/Github/P100/tools/cuda-12.8/bin/cuobjdump --list-elf third_party/llama.cpp/build-upstream-snapshot-cpuref32/bin/libggml-cuda.so
/home/p51/Github/P100/tools/cuda-12.8/bin/cuobjdump --list-elf third_party/llama.cpp/build-conv8-snapshot-cpuref32/bin/libggml-cuda.so
shellcheck -x bench/phase0.sh bench/patches/run.sh bin/*.sh
bin/check-forbidden.sh
flock /tmp/p100-gpu.lock ctest --test-dir build --output-on-failure
```

```text
(a) cuobjdump exit=0: 144 sm_52 ELF images; 144 sm_60 ELF images
(b) cuobjdump exit=0: 144 sm_52 ELF images; 144 sm_60 ELF images
shellcheck exit=0 (no diagnostics)
bin/check-forbidden.sh exit=0: ok
ctest exit=0: 100% tests passed out of 28; Total Test time (real) = 74.77 sec
```

`ldd` with each new snapshot's `LD_LIBRARY_PATH` resolves both `libggml-cpu`
and `libggml-cuda` to that snapshot. The old test diff is byte-for-byte unchanged.
SHA-256 comparison confirms every pre-existing result file other than these
permitted README and command-file edits is unchanged.

## Exact manager rerun commands

These four commands are also appended to `gpu-commands.txt`. They preserve old
Maxwell outputs, select the new snapshots' own libraries, and hold the GPU lock.
The filtered timer starts after acquiring the lock and includes the CPU reference.
The new outputs have a `-cpuref32` suffix. These commands have not been run in
this follow-up; the manager will execute them.

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; ulimit -c 0; flock /tmp/p100-gpu.lock env LD_LIBRARY_PATH=/home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/llama.cpp/build-upstream-snapshot-cpuref32/bin /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/llama.cpp/build-upstream-snapshot-cpuref32/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 > /home/p51/Github/worktrees/p100-p7a-patches-biasv/bench/results/maxwell-patches-2026-10-10/upstream.FLASH_ATTN_EXT-cpuref32.txt 2>&1
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; ulimit -c 0; flock /tmp/p100-gpu.lock /usr/bin/time -f elapsed_seconds=%e -o /home/p51/Github/worktrees/p100-p7a-patches-biasv/bench/results/maxwell-patches-2026-10-10/upstream.biased-V.time-cpuref32.txt env LD_LIBRARY_PATH=/home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/llama.cpp/build-upstream-snapshot-cpuref32/bin /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/llama.cpp/build-upstream-snapshot-cpuref32/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -p v_bias > /home/p51/Github/worktrees/p100-p7a-patches-biasv/bench/results/maxwell-patches-2026-10-10/upstream.biased-V-cpuref32.txt 2>&1
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; ulimit -c 0; flock /tmp/p100-gpu.lock env LD_LIBRARY_PATH=/home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/llama.cpp/build-conv8-snapshot-cpuref32/bin /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/llama.cpp/build-conv8-snapshot-cpuref32/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 > /home/p51/Github/worktrees/p100-p7a-patches-biasv/bench/results/maxwell-patches-2026-10-10/conv8.FLASH_ATTN_EXT-cpuref32.txt 2>&1
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; ulimit -c 0; flock /tmp/p100-gpu.lock /usr/bin/time -f elapsed_seconds=%e -o /home/p51/Github/worktrees/p100-p7a-patches-biasv/bench/results/maxwell-patches-2026-10-10/conv8.biased-V.time-cpuref32.txt env LD_LIBRARY_PATH=/home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/llama.cpp/build-conv8-snapshot-cpuref32/bin /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/llama.cpp/build-conv8-snapshot-cpuref32/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -p v_bias > /home/p51/Github/worktrees/p100-p7a-patches-biasv/bench/results/maxwell-patches-2026-10-10/conv8.biased-V-cpuref32.txt 2>&1
```

Files changed in this follow-up:

- `third_party/patches/0003-tests-biased-V-flash-attention-cases.patch`
- `third_party/patches/README.md` (0003 row only)
- `bench/results/maxwell-patches-2026-10-10/gpu-commands.txt`
- `bench/results/maxwell-patches-2026-10-10/README.md`

---

The following is the original p7a sandbox report, retained as history. It describes
initial CUDA discovery failures. The manager later reran the commands and the
existing Maxwell operation outputs now contain those runs, including the 16
biased-V failures in each variant. Its original sandbox-only outcome statements
do not describe the later manager runs or the new cpuref32 snapshots.

# sm_60 patch builds and Maxwell correctness attempts (2026-10-10)

llama.cpp pin: `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`.
Compiler: CUDA 12.8, nvcc V12.8.93 (`nvcc.txt`). Targets: `52;60`, Release.
Expected local GPU: Quadro M2200, Maxwell sm_52 (ticket-provided; driver inventory unavailable in this sandbox).
`flock /tmp/p100-gpu.lock nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` exited 9 (`gpu.txt`):

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

GPU performance timings were not collected. The biased-V command is timed only to measure the total correctness run including the CPU reference. If CUDA discovery fails, its recorded attempt time is not that measurement.
No P100 correctness, performance, or FP16-accumulator behavior has been verified.
The expected upstream q4_K MUL_MAT abort cannot be classified without an actual GPU comparison.

## Preparation and builds

All shell commands start with:

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH;
```

Run from this worktree root, with no network:

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp third_party/llama.cpp
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; git -C third_party/llama.cpp checkout --detach "$(cat bench/LLAMA_COMMIT)"
```

(a) is upstream + 0003; (b) is 0001 + 0002 + 0003. The single private clone is built sequentially; the complete `bin` directory from (a) is saved before applying the conversion patches, so both binaries and their shared libraries remain available. Incremental compilation reuses the unchanged objects. Both configurations use the same compiler and architecture options.

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; git -C third_party/llama.cpp apply "$PWD/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch"
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-upstream -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=52;60' -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF -DCMAKE_BUILD_RPATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; cmake --build third_party/llama.cpp/build-upstream --target test-backend-ops -j 2
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; mkdir -p third_party/llama.cpp/build-upstream-snapshot
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; cp -a third_party/llama.cpp/build-upstream/bin third_party/llama.cpp/build-upstream-snapshot/
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; git -C third_party/llama.cpp apply "$PWD/third_party/patches/0001-CUDA-fewer-idle-threads-when-converting-short-non-co.patch"
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; git -C third_party/llama.cpp apply "$PWD/third_party/patches/0002-CUDA-vectorized-dequantization-of-non-contiguous-q4_.patch"
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-upstream -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=52;60' -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF -DCMAKE_BUILD_RPATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; cmake --build third_party/llama.cpp/build-upstream --target test-backend-ops -j 2
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; mkdir -p third_party/llama.cpp/build-conv8-snapshot
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; cp -a third_party/llama.cpp/build-upstream/bin third_party/llama.cpp/build-conv8-snapshot/
```

The exact patch filenames and the results of sequential clean-pin checks are recorded in `apply-check.txt`; build output is saved in `upstream.configure.txt`, `upstream.build.txt`, `conv8.configure.txt`, and `conv8.build.txt`.

## Biased-V cases

`biased-V-cases.txt` lists all 16 expected case names. The new grid is `hsk=hsv in {128,256}`, `kv in {16384,32768}`, `nb in {1,4,16,512}`, with `nh=1,nr23=[4,1]` (four Q heads, one KV head), F16 K/V, no mask or sinks, zero ALiBi and softcap, `v_bias=2.000000` in V channels 0..7. Other channels retain the normal initialized data, subject to the storage type's conversion/quantization. The existing normalized mean square error limit remains `5e-4`; no metric or threshold changes.

The pin supports `-p <params regex>`. Phase 0 adds `test -o FLASH_ATTN_EXT -b CUDA$device -p v_bias` immediately after the full FA correctness run, writing `fa-vbias-test-CUDA$device.txt`. All three patch runner variants include 0003 and run the same filtered check.

## Repository validation

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; shellcheck -x bench/phase0.sh bench/patches/run.sh bin/*.sh
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; bin/check-forbidden.sh
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; PYTHON=python DRYRUN=1 bash tests/fixtures/shims/dryrun "$PWD" phase0
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; PYTHON=python DRYRUN=1 bash tests/fixtures/shims/dryrun "$PWD" patches
```

The dry-run drivers invoke both scripts with `DRYRUN=1`, shim tools on PATH, isolated settings, and temporary output. They assert the filtered calls, output files, order after full FA, and 0003 application plans for every variant. Their output is explicitly FAKE and is not hardware evidence.

Repository configure and build both exited 0. CTest result: `100% tests passed out of 28`; total wall time `135.94 sec` (`repo.ctest.txt`). Shellcheck exited 0 with no diagnostics; `bin/check-forbidden.sh` exited 0 and printed `ok`.
Both dry-run drivers exited 0 and printed `FAKE plumbing checks passed; temporary results removed`.

Planned new commands captured from the fixture logs (temporary path suffixes vary):

```text

phase0.dryrun.txt (FAKE):
[19:13:42] DRYRUN: git -C /tmp/p100-phase0-lfzdkkjj/absent-llama apply --check -R /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch; apply /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch if not already applied
[19:13:43] biased-V correctness: /home/p51/Github/worktrees/p100-p7a-patches-biasv/tests/fixtures/shims/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -p v_bias
[19:13:43] biased-V correctness: /home/p51/Github/worktrees/p100-p7a-patches-biasv/tests/fixtures/shims/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA1 -p v_bias

patches.dryrun.txt (FAKE):
[19:13:44] DRYRUN: git -C /tmp/p100-patches-2k80lxoj/must-not-exist/upstream apply /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch
[19:13:44] biased-V correctness: /home/p51/Github/worktrees/p100-p7a-patches-biasv/tests/fixtures/shims/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias
[19:13:44] DRYRUN: git -C /tmp/p100-patches-2k80lxoj/must-not-exist/conv apply /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch
[19:13:45] biased-V correctness: /home/p51/Github/worktrees/p100-p7a-patches-biasv/tests/fixtures/shims/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias
[19:13:45] DRYRUN: git -C /tmp/p100-patches-2k80lxoj/must-not-exist/conv8 apply /home/p51/Github/worktrees/p100-p7a-patches-biasv/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch
[19:13:45] biased-V correctness: /home/p51/Github/worktrees/p100-p7a-patches-biasv/tests/fixtures/shims/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias
```

## Build outcomes

Both final builds exited 0. Last lines:

```text
(a) upstream + 0003: [13/13] Linking CXX executable bin/test-backend-ops
(b) 0001 + 0002 + 0003: [16/16] Linking CXX executable bin/test-backend-ops
```

The initial (a) link exited 1 because the assembled toolkit's CUDA libraries were absent from the executable's search path. Adding `-DCMAKE_BUILD_RPATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib` fixed it without changing sources or installing anything; both successful configurations use that flag. The original failure and successful retry are both retained in `upstream.build.txt`. Representative failure lines, verbatim:

```text
/usr/bin/ld: warning: libcudart.so.12, needed by bin/libggml-cuda.so.0.26.0, not found (try using -rpath or -rpath-link)
/usr/bin/ld: warning: libcublas.so.12, needed by bin/libggml-cuda.so.0.26.0, not found (try using -rpath or -rpath-link)
collect2: error: ld returned 1 exit status
```

Compiler warning in both builds (covers sm_52 and sm_60; no warning specifically names sm_60):

```text
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
```

`cuobjdump --list-elf` confirms 144 sm_52 and 144 sm_60 ELF images in each CUDA library (`upstream.architectures.txt`, `conv8.architectures.txt`). The command for each snapshot is:

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; /home/p51/Github/P100/tools/cuda-12.8/bin/cuobjdump --list-elf third_party/llama.cpp/build-upstream-snapshot/bin/libggml-cuda.so
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; /home/p51/Github/P100/tools/cuda-12.8/bin/cuobjdump --list-elf third_party/llama.cpp/build-conv8-snapshot/bin/libggml-cuda.so
```

Sequential checks were run after restoring tracked files to the clean pin and applying each predecessor:

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; git -C third_party/llama.cpp checkout -- .
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; for p in "$PWD"/third_party/patches/0001-*.patch "$PWD"/third_party/patches/0002-*.patch "$PWD"/third_party/patches/0003-*.patch; do git -C third_party/llama.cpp apply --check "$p" && git -C third_party/llama.cpp apply "$p" || exit 1; done
```

```text
clean pin: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
0001-CUDA-fewer-idle-threads-when-converting-short-non-co.patch: git apply --check exit=0
0002-CUDA-vectorized-dequantization-of-non-contiguous-q4_.patch: git apply --check exit=0
0003-tests-biased-V-flash-attention-cases.patch: git apply --check exit=0
```

## Maxwell correctness outcomes and handoff

All ten commands in `gpu-commands.txt` were executed, under `flock /tmp/p100-gpu.lock`, with each snapshot's own libraries selected via `LD_LIBRARY_PATH`. `upstream.libraries.txt` and `conv8.libraries.txt` confirm the CUDA library resolves to the respective snapshot. All commands exited 0, but CUDA0 was absent and no GPU cases executed. The output's `1/1 backends passed` and `OK` refer to a skipped CPU backend, not a successful correctness comparison.

| operation | (a) GPU case passes / failures | (b) GPU case passes / failures | result |
| --- | --- | --- | --- |
| FLASH_ATTN_EXT | 0 / 0 | 0 / 0 | not run: CUDA initialization failed |
| MUL_MAT | 0 / 0 | 0 / 0 | not run: CUDA initialization failed |
| CPY | 0 / 0 | 0 / 0 | not run: CUDA initialization failed |
| GET_ROWS | 0 / 0 | 0 / 0 | not run: CUDA initialization failed |
| biased-V (FLASH_ATTN_EXT -p v_bias) | 0 / 0 | 0 / 0 | not run: CUDA initialization failed |

Every operation output begins with this failure line, verbatim:

```text
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
```

No test-case `FAIL` lines were emitted because no GPU cases ran. The upstream q4_K MUL_MAT abort was not reached, so its comparison is unverified.
All 16 biased-V cases in `biased-V-cases.txt` are **not run** for both variants; the CPU-reference comparison runtime is **not measured** for either. The `.time.txt` files contain only the elapsed failed-initialization attempt, during which no CPU reference ran. Their values must not be used as the requested reference-cost measurement. Whether the full 16-case CPU reference cost is reasonable remains unverified.

Run the exact commands in `gpu-commands.txt` outside the sandbox to obtain the requested comparisons. Only the filtered biased-V run uses `/usr/bin/time`; its timer starts after the GPU lock is acquired and includes the CPU reference. Core dumps are disabled to keep the results text-only. No GPU performance mode is run.

After saving both binary snapshots, the private clone's tracked source is restored to the pin plus 0003, so a default Phase 0 build starts with the test patch. The snapshots preserve both completed builds. Nothing on a P100 has been run.

## Changed files

- `third_party/patches/0003-tests-biased-V-flash-attention-cases.patch`
- `third_party/patches/README.md`
- `bench/phase0.sh`
- `bench/patches/run.sh`
- `tests/fixtures/shims/dryrun`
- `tests/fixtures/shims/test-backend-ops`
- Text outputs in this results directory.

0001 and 0002 are unchanged.

## Manager runs on the Maxwell M2200 (2026-10-10)

Commands: `gpu-commands.txt` (run by the manager outside the worker sandbox). Driver 580.178.04, nvcc 12.8.93.

| run | upstream + 0003 | 0001 + 0002 + 0003 |
| --- | --- | --- |
| FLASH_ATTN_EXT (first 0003, FP16 CPU reference) | 3993/4009; the 16 failures are the biased-V cases | identical |
| biased-V only (first 0003) | 0/16 (NMSE 0.36 at 16K, 0.60 at 32K) | 0/16, same values |
| FLASH_ATTN_EXT (0003 with FP32 reference accumulation, `-cpuref32`) | 4009/4009 | 4009/4009 |
| biased-V only (`-cpuref32`) | 16/16, 39.0 s | 16/16, 39.4 s |
| MUL_MAT | 1103 OK, then the pre-existing q4_K abort | same point |
| CPY | 251/251 | 251/251 |
| GET_ROWS | 216/216 | 216/216 |

The first-0003 failures came from the CPU reference: in `use_ref` mode `ggml_compute_forward_flash_attn_ext_f16_one_chunk`
accumulates V in FP16 (`VKQ16`), which stalls at long context. 0003 now keeps that accumulator in FP32 in reference
mode only (ticket p7c). The Maxwell's CUDA FA uses FP32 accumulators, so these cases test the harness, not the
P100's FP16 path; they are meant for the P100 and V100.
