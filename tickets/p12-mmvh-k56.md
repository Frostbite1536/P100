# Ticket p12-mmvh-k56: extend MMVH (patch 0005) to q5_K and q6_K as patch 0006, Maxwell-testable

Your current directory is a git worktree of the P100 repo (branch `p12-mmvh-k56`, based on `main`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;`. Never install anything. CMake for this repo needs `-DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc`.
- Do NOT commit, push, or run git commands that write **in the P100 repo**. In your private llama.cpp clone you may apply patches, commit and `git format-patch`; that clone is gitignored scratch.
- Do not loosen tolerances or edit llama.cpp's existing tests to pass; report failures.
- Never invent benchmark numbers. There is no P100 here. The local GPU is a Maxwell sm_52 (no FP16 arithmetic). Your sandbox cannot reach it: build everything, and leave exact commands for the manager to run on it.
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Patch `third_party/patches/0005-CUDA-half2-mat-vec-for-sm_60-mmvh.patch` adds `mmvh` (block-scaled FP16 activations "h16" + half2 HFMA2 dot, weights unpacked once per block for all batch columns) for q4_0, q8_0, q4_K at batch 1..8. Read the patch fully, and `docs/design/H1-pascal-decode-matvec.md`. On the Maxwell (forced, FP32 emulation of the half2 ops) decode MUL_MAT passed 255/255.

Why q5_K/q6_K now (manager's reading of `src/llama-quant.cpp` at the pin, lines ~590 and ~646-652, verify): in a Llama-style Q4_K_M file, `attn_v` and `ffn_down` are Q6_K in the `use_more_bits` layers (16 of 32 for Llama-3.1-8B) and `output.weight` is Q6_K; Q5_K appears in Q4_K_M only for fused QKV tensors, and widely in Q5_K_M / Q4_K_S / Q3_K_*. Manager's arithmetic for Llama-3.1-8B Q4_K_M: about 27% of the weight bytes read per decoded token are Q6_K. Those stay on emulated-dp4a MMVQ until this ticket.

## Design (manager decided, do not change)
1. **Patch 0006** `third_party/patches/0006-CUDA-MMVH-q5_K-and-q6_K.patch`: one `git format-patch` commit on top of 0005 in your private clone (series 0001, 0002, 0003, 0005, 0006 must apply in order at `bench/LLAMA_COMMIT`; also check 0001, 0002, 0003, 0004, 0005, 0006). Change only `mmvh.cu`/`mmvh.cuh` and `tests/test-backend-ops.cpp`. Do not touch 0005's file.
2. **q5_K unpack**: like q4_K (same 6-bit scale/min packing, `dm`, min handled through `meta.y` once per 32-weight sub-block), plus the fifth bit from `qh` (bit `sub` of `qh[l]` for element `l` of sub-block `sub`; confirm against `dequantize_row_q5_K` in `ggml/src/ggml-quants.c` and cite lines). Insert the high bit into the mantissa before the bias subtraction (for the 0x6400 low-nibble path the bit goes to mantissa bit 4; for the 0x5400 high-nibble path to bit 8), so each half is exactly bias+q, q in 0..31. Use `lop3` where it saves instructions (state the truth table in a comment as 0005 does); the sm_52 fallback stays plain C.
3. **q6_K unpack**: 6-bit q from `ql` nibble + 2 bits of `qh` (confirm the layout against `dequantize_row_q6_K`, cite lines), the -32 offset folded into the bias subtraction constant, so halves are exactly q-32 in -32..31. q6_K has **one int8 scale per 16 weights**: split each 32-weight sub-block's chain into two chains of 8 HFMA2 per column, each flushed to FP32 with `d * scales[i]`. No minimum. Integer weights here are exact in FP16; with |x̂| ≤ 1, |q| ≤ 32 and chain ≤ 16 the half2 partial sums stay ≤ 1024 (check and state the bound for both types).
4. Keep 0005's structure: `Layout<>` traits (add whatever trait q6_K's two scales need without changing the q4_0/q8_0/q4_K instruction streams — verify by comparing their sm_60 SASS inner-loop counts before/after 0006), one row of `mmvh_rows_per_block` per new type (1 on ties, as 0005), predicate and launch switch extended to `GGML_TYPE_Q5_K`, `GGML_TYPE_Q6_K`. Same env switches.
5. **Tests**: in `make_test_cases_eval`, add only the missing (type, k, n) combinations for q5_K and q6_K at k = 4096 and 14336, n = 1..8, m = 129 (same pattern as 0005; report which exist already).
6. **Build** in a private clone (`git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp third_party/llama.cpp`, checkout the pin, apply 0001, 0002, 0003, 0005, then your 0006): `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="52;60" -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF`, Release, target `test-backend-ops`, build dir `third_party/llama.cpp/build-mmvh`. Report registers/spills of every mmvh kernel for sm_60 (`cuobjdump -res-usage`). Spill > 0 on sm_60 is a failing result.
7. **Static count**: same method as p9 (`cuobjdump -sass`, sm_60, inner loop identified by its back-branch): instructions per 32 weights for q5_K and q6_K at B = 1, 2, 4, 8, plus q4_0/q8_0/q4_K B = 1 and 8 before and after 0006 (must be unchanged). Also, for comparison, llama.cpp's own `vec_dot_q5_K_q8_1` / `vec_dot_q6_K_q8_1` cost on sm_60 if `bin/vecdot-sass.sh` already reports it (read `docs/research/vecdot-sass-2026-10-10.csv`; do not re-derive).
8. `bench/patches/mmvh-maxwell.sh`: extend its filter to `q4_0|q8_0|q4_K|q5_K|q6_K` (keep behaviour otherwise). `bench/patches/run.sh`: variant `mmvh` becomes 0001, 0002, 0003, 0005, 0006 and `all` 0001–0006 (keep the one-file-per-ordinal check; update the header comment); its `mmperf` filter already includes q5_K/q6_K (verify). Update the dry-run ctest expectation in `tests/CMakeLists.txt` only if it asserts patch lists. `third_party/patches/README.md`: add the 0006 row (status "builds for sm_60; Maxwell correctness: pending manager run; P100 not run") and fix the "apply order" sentence.

## Files you may touch
`third_party/patches/0006-*.patch` (new), `third_party/patches/README.md`, `bench/patches/mmvh-maxwell.sh`, `bench/patches/run.sh`, `tests/CMakeLists.txt` (dry-run expectation only), `tests/fixtures/shims/*` (only if the dry run needs it). Nothing else in the P100 repo. Do not write analysis into docs.

## Must pass before you finish
- `git apply --check` of both orders in design item 1 on a clean pin
- the `52;60` build succeeds; zero spills on every sm_60 mmvh kernel
- `cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc && cmake --build build && ctest --test-dir build --output-on-failure` (in this worktree; a GPU-needing test may skip/fail in the sandbox — report which)
- `shellcheck -x bench/patches/*.sh bench/phase0.sh bin/*.sh`, `bin/check-forbidden.sh`
- `DRYRUN=1 FORCE=1 OUT=<tmp> bench/patches/run.sh` (paste the patch-list lines of its log)

## Done means
Final report: files changed; each command with its result line; the patch diffstat and the q5_K/q6_K unpack functions verbatim; the cited dequantize lines; the FP16 bound; the sm_60 resource table; the instruction-count table with method; the exact manager commands for the Maxwell (`bench/patches/mmvh-maxwell.sh <path to your test-backend-ops>`); what was NOT verified (everything on a GPU). Do not interpret or decide whether to ship.
