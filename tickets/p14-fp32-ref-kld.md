# Ticket p14-fp32-ref-kld: FP32-reference KL divergence step in the day-one A/B kit

Your current directory is a git worktree of the P100 repo (branch `p14-fp32-ref-kld`, based on `main` @ `65d8065`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/sm60-report.py <file.cu>` and paste its table. Spill must be 0.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/p100/attention.h`.
- Do NOT commit, push, or run git commands that write **in the P100 repo**. Leave changes uncommitted; the manager commits. In your private llama.cpp clone (Steps 1) you may apply patches, commit and `git format-patch`; that clone is gitignored scratch.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52; its timings are not P100 numbers): anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/volta-fa-p9-lessons-2026-10-11.md` §4, "Accuracy method". H1's Accept-if compares perplexity with upstream (±0.02, `docs/design/H1-pascal-decode-matvec.md`). That cannot show that `mmvh` is *more* accurate than q8_1, which `docs/research/qmv-numerics-2026-10-10.txt` predicts (7–34×).

volta-fa measured accuracy on real models with a development switch that routes quantised matmuls to dequantise + SGEMM, and llama-perplexity KL divergence of each path against that reference. The switch, from `~/Github/volta-fa/bench/results/p9-scale2-20261011-195231/p9-ref.diff`, sits at the top of `ggml_cuda_mul_mat` in `ggml-cuda.cu`:

```cpp
static const bool dev_f32_ref = getenv("GGML_CUDA_F32_REF") != nullptr; // DEV: FP32 reference (dequantize + SGEMM)
if (dev_f32_ref && ggml_is_quantized(src0->type)) { // DEV
    ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst); // DEV
    return; // DEV
} // DEV
```

`ggml_cuda_mul_mat_cublas_impl` exists at the pin (`ggml-cuda.cu` ≈ line 1409). volta-fa used it at the same pin; confirm its template parameter before relying on it.

## Design (manager decided, do not change)
1. **`third_party/patches/bench-only/0002-cuda-f32-reference-matmul-dev-switch.patch`** (against the pin, independent of the series).
   - It contains the switch above. The first time the switch takes effect, it prints one line to stderr: `GGML_CUDA_F32_REF: quantised MUL_MAT via dequantize + SGEMM`.
   - Never upstream. p13 has landed: `third_party/patches/bench-only/` and the README's "Bench-only patches" table exist, and that table is the manifest `run.sh` reads (`bench_patches_for_all` applies only rows whose applies-to cell is exactly `all variants`). Add a row for 0002 whose applies-to cell is `ref build only`, so the existing selector skips it.
   - This patch is applied **only to the `ref` build below**, not to every variant. Name that exception in the README entry and in `run.sh`'s header.
2. **New opt-in step `kld`** in `bench/patches/run.sh`. It is not in the default STEPS and requires `MODELS` and `PPL_TEXT`; otherwise it logs one line and skips.
   - **Build:** a `ref` tree = the `upstream` series (0003) + the "all variants" bench-only patches (as every tree gets) + bench-only 0002, logged like the other applies; `llama-perplexity` only. `ref` is not a variant and gets no correctness/perf/lb/ppl runs.
   - **Reference per model:**
     ```
     GGML_CUDA_F32_REF=1 llama-perplexity -m M -f PPL_TEXT -c 512 -b $KLD_UB -ub $KLD_UB -ngl 99 -fa on --chunks $KLD_CHUNKS --kl-divergence-base $OUT/<model>.ref.kld
     ```
     `KLD_UB` defaults to 8, so every matmul has n = 8 and `mmvh` is exercised (n ≥ 9 routes to cuBLAS on sm_60). `KLD_CHUNKS` defaults to 8. Check that the stderr line appeared, else fail the step.
   - **Each arm:** the same command without the env var, with `--kl-divergence-base … --kl-divergence`. mmvh arms run with and without `GGML_CUDA_DISABLE_MMVH=1`.
   - Delete each `.kld` base file after its model (they are large).
   - **Summary:** per model and arm, the `Mean KLD`, `Maximum KLD`, `RMS Δp` and `Same top p` lines, verbatim.
3. Header comment and `bench/BASELINE.md` (one line): what `kld` answers (accuracy against FP32, not against upstream) and why `KLD_UB` = 8.

## Files you may touch
- `third_party/patches/bench-only/0002-cuda-f32-reference-matmul-dev-switch.patch` (new), `third_party/patches/README.md` (bench-only section)
- `bench/patches/run.sh`, `bench/BASELINE.md` (one line)
- `tests/fixtures/shims/*` (a `llama-perplexity` shim that prints FAKE KLD lines, and the switch line when the env var is set), `tests/CMakeLists.txt` (dry-run assertions)
Nothing else.

## Steps
1. Write the patch in a private clone: `git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp build/llama-p14`, checkout `79e2e74eb` (the pin, `bench/LLAMA_COMMIT`). Do not touch `/home/p51/Github/P100/third_party/llama.cpp` itself. Export it with `git format-patch`. Show with `git apply --check` on fresh checkouts that it applies (a) on the pin alone and (b) after series 0003 + bench-only 0001 (the `ref` tree).
   - Build `test-backend-ops` in that clone for Maxwell + Pascal as in `bench/results/maxwell-patches-2026-10-10/README.md`: `-DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=52;60' -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF -DCMAKE_BUILD_RPATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib`, Release, build dir `build/llama-p14/build-ref`.
2. **Positive control on the Maxwell GPU.** Your sandbox has NO GPU access; do not try to run these. Write the exact commands (absolute paths, output files under `build/`) into your final report under a heading "Manager GPU commands"; the manager runs them and records the counts. The commands: with `GGML_CUDA_F32_REF=1`, run `test-backend-ops test -b CUDA0 -o MUL_MAT -p 'type_a=(q4_0|q8_0|q4_K|q5_K|q6_K),type_b=f32'` and report the pass count (the reference path must still match the CPU).
   - Also run it without the env var for comparison.
   - `third_party/patches/README.md` notes a pre-existing q4_K abort on Maxwell. The manager will record where it hits and filter around it; you only provide the unfiltered commands.
3. Implement the step, the shim and the dry-run assertions.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all; GPU tests may fail or skip in your sandbox because it has no GPU: name them). Add new dry-run assertions; do not remove or weaken existing ones.
- `shellcheck -x bench/patches/run.sh bin/*.sh`, `bin/check-forbidden.sh`
- `DRYRUN=1 FORCE=1 OUT=<tmp> MODELS=/FAKE/model.gguf PPL_TEXT=/FAKE/wiki.txt STEPS="build kld" bench/patches/run.sh`: paste the planned commands.

## Done means
On day one, `STEPS=kld` gives mean/max KLD per arm against an FP32 reference.

Final report:
- files changed;
- each command with its result line;
- the Manager GPU commands for the Maxwell MUL_MAT positive control (with and without the switch); the manager records the counts, labelled Maxwell;
- the dry-run log for `kld`;
- what was NOT verified (anything on a P100; no model was run).

Do not interpret.
