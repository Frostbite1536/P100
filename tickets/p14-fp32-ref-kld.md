# Ticket p14-fp32-ref-kld: FP32-reference KL divergence step in the day-one A/B kit

Your current directory is a git worktree of the P100 repo (branch `p14-fp32-ref-kld`, based on `main` @ `b13da11`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/sm60-report.py <file.cu>` and paste its table. Spill must be 0.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/p100/attention.h`.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
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
   - Never upstream. List it in `third_party/patches/README.md`'s "Bench-only patches" section, and create that section and the directory as ticket p13 describes if p13 has not landed.
   - This patch is applied **only to the `ref` build below**, not to every variant. Name that exception in the README entry and in `run.sh`'s header.
2. **New opt-in step `kld`** in `bench/patches/run.sh`. It is not in the default STEPS and requires `MODELS` and `PPL_TEXT`; otherwise it logs one line and skips.
   - **Build:** a `ref` tree = `upstream` + bench-only 0002; `llama-perplexity` only.
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
1. Write the patch in a scratch worktree of `third_party/llama.cpp` at the pin under `build/`. Build it for sm_52 with the repo's Maxwell setup (`bin/maxwell-check.sh`, or the commands in `bench/results/maxwell-patches-2026-10-10/README.md`).
2. **Positive control on the Maxwell GPU:** with `GGML_CUDA_F32_REF=1`, run `test-backend-ops test -b CUDA0 -o MUL_MAT -p 'type_a=(q4_0|q8_0|q4_K|q5_K|q6_K),type_b=f32'` and report the pass count (the reference path must still match the CPU).
   - Also run it without the env var for comparison.
   - `third_party/patches/README.md` notes a pre-existing q4_K abort on Maxwell. Filter around it if needed, and say how.
3. Implement the step, the shim and the dry-run assertions.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all)
- `shellcheck -x bench/patches/run.sh bin/*.sh`, `bin/check-forbidden.sh`
- `DRYRUN=1 FORCE=1 OUT=<tmp> MODELS=/FAKE/model.gguf PPL_TEXT=/FAKE/wiki.txt STEPS="build kld" bench/patches/run.sh`: paste the planned commands.

## Done means
On day one, `STEPS=kld` gives mean/max KLD per arm against an FP32 reference.

Final report:
- files changed;
- each command with its result line;
- the Maxwell MUL_MAT counts with and without the switch (labelled Maxwell, not P100);
- the dry-run log for `kld`;
- what was NOT verified (anything on a P100; no model was run).

Do not interpret.
