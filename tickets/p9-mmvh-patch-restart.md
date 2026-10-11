# Ticket p9-mmvh-patch: half2 decode mat-vec in llama.cpp for sm_60 (patch 0005), Maxwell-testable

Your current directory is a git worktree of the P100 repo (branch `p9-mmvh-patch`, based on `main`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;`. Never install anything.
- Do NOT commit, push, or run git commands that write **in the P100 repo**. In your private llama.cpp clone you may apply patches, commit and `git format-patch`; that clone is gitignored scratch.
- Do not loosen tolerances or edit llama.cpp's existing tests to pass; report failures.
- Never invent benchmark numbers. There is no P100 here. The local GPU is a Maxwell sm_52 (no FP16 arithmetic). Your sandbox probably cannot reach it: build everything, and leave exact commands for the manager to run on it.
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Design doc: `docs/design/H1-pascal-decode-matvec.md` (read it fully). Prototype: `bench/micro/h2_mmv.cu` (kernels, unpack tricks, pre-pass, sm_52 FP32 emulation), CPU model `src/reference/qmv_model.cpp`, numerics `docs/research/qmv-numerics-2026-10-10.txt`, SASS counts `docs/research/h2-sass-2026-10-10.csv`, Maxwell layout check `bin/maxwell-check.sh`. llama.cpp's MMVQ entry: `ggml_cuda_mul_mat_vec_q` (`mmvq.cu:1434`), selection `ggml_cuda_should_use_mmvq` (`mmvq.cu:318`), fusion gate `ggml_cuda_should_fuse_mul_mat_vec_q` (`ggml-cuda.cu:1825`), call sites `ggml-cuda.cu:1951` and the fused ones (`:3630`, `:3977`, `:4071`, `:4141`, `:4180`, `:4268`, `:4333`).

## Design (manager decided, do not change)
1. **New files** `ggml/src/ggml-cuda/mmvh.cu` / `mmvh.cuh` in llama.cpp: the h16 pre-pass and the half2 mat-vec for q4_0, q8_0 and q4_K, `ncols_dst` (batch) 1..8, `CHAIN = 16`, ported from the prototype. Changes from the prototype:
   - activations loaded with 128-bit loads (8 halves) — the prototype's 32-bit loads made loads ~40% of instructions at B ≥ 4;
   - template parameter `rows_per_block` with warps-per-row = 1, rows per CTA 1/2/4; pick the default by static instruction count only (the P100 will tune it); expose it in one constexpr table with a comment;
   - sm_52 (any arch without fast FP16) keeps the FP32-emulation fallback, used only when forced (below).
   - Non-contiguous `src1`, `ids` (MUL_MAT_ID), and any fusion are **not** handled: those cases use the existing MMVQ path.
2. **Dispatch** at the top of `ggml_cuda_mul_mat_vec_q`: use mmvh iff `ids == nullptr`, no fusion requested, src0 type ∈ {q4_0, q8_0, q4_K}, `ne11` (batch) ≤ 8, src1 contiguous F32, and either (cc ≥ 600 && cc < 610 — sm_60/sm_62, fast FP16 without dp4a — and env `GGML_CUDA_DISABLE_MMVH` unset) or env `GGML_CUDA_FORCE_MMVH` set (test hook for Maxwell). Read both env vars once (static), presence-checked like `GGML_CUDA_P2P` is.
3. **Fusion**: in `ggml_cuda_should_fuse_mul_mat_vec_q`, return false when mmvh would take the node (same predicate, factor it into one helper used by both places), so gate/up/GLU graphs run as separate mat-vecs on mmvh rather than the fused MMVQ. Write in the patch body that this trades the fusion for the faster mat-vec and must be measured.
4. **Tests**: add to `tests/test-backend-ops.cpp` nothing new unless MUL_MAT already lacks n = 1..8 cases for q4_0/q8_0/q4_K with k = 4096/14336 (check `make_test_cases_eval`; report what exists). Correctness is `test-backend-ops test -o MUL_MAT` (its own NMSE threshold).
5. **Patch** `third_party/patches/0005-CUDA-half2-mat-vec-for-sm_60-mmvh.patch` via `git format-patch`, applying after 0001–0002 at `bench/LLAMA_COMMIT` (independent of 0003/0004). Body: why (link-free summary of the design doc numbers), env vars, what is excluded.
6. **Build** in a private clone (`git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp third_party/llama.cpp`, pin, 0001+0002+0005): `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="52;60" -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF`, Release, target `test-backend-ops`. Report registers/spills of every mmvh kernel for sm_60 (`cuobjdump -res-usage` on the built object). Spill > 0 is a failing result.
7. **Static count**: no new fixtures; run `cuobjdump -sass` on the sm_60 mmvh kernels and report, per (type, B), the instruction count of the inner loop body per 32 weights (identify the loop by its branch; explain how you counted).
8. **Manager handover script** `bench/patches/mmvh-maxwell.sh` (shellcheck-clean): under `flock /tmp/p100-gpu.lock`, runs on the Maxwell `test-backend-ops test -o MUL_MAT -b CUDA0` three ways — upstream path (no env), `GGML_CUDA_FORCE_MMVH=1`, and `GGML_CUDA_FORCE_MMVH=1` with `-p` restricted to the q4_0/q8_0/q4_K n ≤ 8 cases — saving outputs under `bench/results/mmvh-maxwell-<date>/`, and prints pass/fail counts. Also `test-backend-ops perf -o MUL_MAT` with and without FORCE for those cases, saved but labelled "Maxwell, FP32 emulation of half2: not a P100 number".
9. `third_party/patches/README.md`: add the 0005 row (status: "builds for sm_60; Maxwell correctness: pending manager run; P100 not run").

## Files you may touch
- `third_party/patches/0005-*.patch` (new), `third_party/patches/README.md` (one row), `bench/patches/mmvh-maxwell.sh` (new).
Nothing else in the P100 repo. Do not write analysis into docs.

## Must pass before you finish
- `git apply --check` of 0001, 0002, 0005 in order on a clean pin; also 0005 alone on top of 0001–0003 if `third_party/patches/0003-*` exists in your worktree (it may not yet).
- the `52;60` build succeeds; zero spills on sm_60 mmvh kernels
- `shellcheck -x bench/patches/mmvh-maxwell.sh`, `bin/check-forbidden.sh`

## Done means
Final report: files changed; each command with its result line; the patch's diffstat and the dispatch/fusion hunks verbatim; the sm_60 resource table; the inner-loop instruction counts with the counting method; the exact manager commands; what was NOT verified (everything on a GPU). Do not interpret or decide whether to ship.

## Note from the manager (restart)
A previous run of this ticket in this worktree was killed by the manager by mistake at llama.cpp build step 46/288. Its partial state (an edited `third_party/patches/README.md`, `bench/patches/mmvh-maxwell.sh`, a private llama.cpp clone and build dir) is still here. Inspect it, keep what is correct, and continue; do not assume any earlier step was verified.
