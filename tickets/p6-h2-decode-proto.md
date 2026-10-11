# Ticket p6-h2-decode-proto: prototype a half2 decode mat-vec for sm_60 (numerics model, static cost, Maxwell correctness)

Your current directory is a git worktree of the P100 repo (branch `p6-h2-decode-proto`, based on `main` @ `6554601`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything. Build into `build/` inside this worktree.
- For every kernel file run `python bin/sm60-report.py <file.cu>` and paste its table. Spill must be 0.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no P100 on this machine. The local GPU is a Maxwell sm_52 (Quadro M2200): it has no FP16 arithmetic, so it can check indexing and layout only; its timings are not P100 numbers and are not wanted in this ticket.
- GPU use: wrap every command that runs on the GPU in `flock /tmp/p100-gpu.lock <command>` (other workers share it). If CUDA cannot initialise inside your sandbox, say so, leave the exact commands in your report, and continue; the manager will run them.
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Hypothesis H1 (`docs/research/pre-hardware-findings-2026-10-10.md` §1, `docs/research/vecdot-sass-2026-10-10.csv`): llama.cpp's decode mat-vec (MMVQ, q8_1 activations, `ggml_cuda_dp4a` emulated on sm_60 at `common.cuh:740-746`) costs 3.9–7.7 ALU instructions per weight per output column on sm_60, so the model predicts 4–6-bit decode is ALU-bound at batch 1 and every quantised type is ALU-bound at batch ≥ 4, because MMVQ unpacks each weight again for every column.

GP100 runs `HFMA2` (two FP16 FMAs) at the FP32 instruction rate. Proposal: activations go to FP16 once per mat-vec, weights are unpacked once per block into `half2` with bit tricks, and the dot product is `HFMA2`, so the unpack cost is shared by all columns of a batch.

## Design (manager decided, do not change)
**Activation format `h16`** (written by a pre-pass kernel, one per mat-vec input): per block of 32 consecutive activations: `s = max|x|` (FP32; `s = 0` means all zeros, then `x̂ = 0`), `x̂_i = fp16_rn(x_i / s)` (so |x̂| ≤ 1: partial sums stay far from FP16 overflow), `sum = Σ fp16(x̂_i)` in FP32. Struct of arrays: `half x̂[K]` stored in the **permuted order** below, `float2 meta[K/32] = {s, sum}`. Permutation: within every aligned group of 4 elements, the order is (0, 2, 1, 3) — this matches the unpack order of all three weight types, so no shuffles are needed in the inner loop.

**Weight unpack** (no changes to the GGUF block formats; read `ggml/src/ggml-common.h` for the layouts):
- q4_0 (`d`, `qs[16]`; element j in the low nibble of byte j, element j+16 in the high nibble): for each 32-bit word `w` of `qs`: `lo = (w & 0x000F000F) | 0x64006400` gives half2 (1024+q) for bytes 0 and 2; `hi = (w & 0x00F000F0) | 0x54005400` gives half2 (64+q) for the high nibbles of bytes 0 and 2; repeat on `w >> 8` for bytes 1 and 3. Subtract 1032 (resp. 72) with `HADD2`/`HSUB2`, exactly. One `lop3.b32` per mask/or pair (write it as `(w & m) | c`; check the SASS shows `LOP3`).
- q8_0 (`d`, `qs[32]` int8): `u = w ^ 0x80808080` (bias to unsigned), then `prmt` with a constant 0x64646464 register to build half2 (1024+u) for bytes (0,2) and (1,3), subtract 1152.
- q4_K (super-block of 256: `d`, `dmin`, `scales[12]` 6-bit scales/mins for 8 sub-blocks of 32, `qs[128]`; byte j of each 32-byte group holds sub-block 2k (low nibble) and 2k+1 (high nibble)): same nibble trick as q4_0 without the -8 offset (subtract 1024 resp. 64); per sub-block `d·sc_j · Σ q x - dmin·m_j · Σ x`, using `meta.sum` for Σ x̂. Decode `scales` exactly as llama.cpp's `get_scale_min_k4`.
- Dot: per 32-element (sub-)block, per output column: `HFMA2` chain into a `half2` accumulator starting at 0, `CHAIN` steps per lane (`CHAIN` is a template parameter, 8 or 16), then the two halves go to FP32 and are multiplied by `d_w · s` and added to an FP32 accumulator. For q4_0/q8_0 a block has 16 half2 pairs, so `CHAIN = 16` means one flush per block.
- Batch `B` ∈ {1, 2, 4, 8} columns is a template parameter: a block is unpacked once and used for all `B` columns.

**Thread mapping**: one warp per output row, lanes stride over blocks, warp-shuffle reduction, FP32 output. Keep it simple; the point is correctness and instruction count, not tuning. Weight blocks are 2-byte aligned (q4_0 is 18 bytes): load with 16-bit loads as llama.cpp does where needed.

**sm_52 fallback**: under `__CUDA_ARCH__ < 530` emulate each `half2` operation in FP32 without FP16 rounding (this checks indexing and layout only; say so in comments).

## Steps
1. CPU numerics model `include/p100/qmv.h` + `src/reference/qmv_model.cpp` (add to the existing reference library target): block structs with `static_assert` sizes (q4_0 18, q8_0 34, q4_K 144), random block generators, and three dot products per row: (a) `exact`: FP64 dot of dequantised weights × FP32 activations; (b) `q81`: llama.cpp's path — quantise activations to q8_1 exactly as `quantize_q8_1` does (`quantize.cu`) and reproduce `vec_dot_<type>_q8_1` arithmetic from `vecdotq.cuh` (integer sums exact, then the same FP32/FP16 scale arithmetic; cite file:line); (c) `h16<CHAIN>`: the design above, emulating FP16 rounding with one rounding per `HFMA2` lane result (use the existing `round_to_fp16(double)`).
2. `tests/qmv_numerics.cpp` (ctest `qmv_numerics`): for each type × {q81, h16<8>, h16<16>}, K ∈ {4096, 14336}, 256 rows, activation distributions `normal` (N(0,1)), `outlier` (normal with 4 random channels ×50), `swiglu` (product of two independent normals × 3), `large` (normal × 1000). Seeds documented. Print median and max of |y − exact| / (‖w_row‖₂·‖x‖₂) and of |y − exact| / max(|exact|, 1e-6·‖w_row‖₂·‖x‖₂) per cell, plus the count of non-finite outputs. Exit 0 unless a non-finite value appears for any path. Write the full output to `docs/research/qmv-numerics-2026-10-10.txt`.
3. `bench/micro/h2_mmv.cu`: the pre-pass kernel and `mmv_h2_<type>_b<B>` kernels (types × B ∈ {1,2,4,8}, `CHAIN = 16`, plus q4_0 b1 with `CHAIN = 8`), and **one-block fixtures** in the same style as `bench/micro/vecdot_sass.cu` (`extern "C"` kernels that unpack exactly one weight block and dot it with B activation columns, no loop), so their SASS counts are comparable with `docs/research/vecdot-sass-2026-10-10.csv`.
4. `bin/h2-sass.sh`: like `bin/vecdot-sass.sh` (reuse its classifier by sourcing or calling; do not change `vecdot-sass.sh`), compile `-O3 -cubin -arch=sm_60` and write `docs/research/h2-sass-2026-10-10.csv` with columns `type,arch,batch,weights_per_block,sass_total,sass_alu,sass_mem,sass_ctrl,alu_per_weight_per_column,top_opcodes` for the one-block fixtures. Also count `HFMA2`, `LOP3`, `PRMT`, `HADD2`, `F2F` separately in `top_opcodes`.
5. `tests/h2_mmv_gpu.cu` (ctest `h2_mmv_gpu`, registered only when CUDA is enabled in CMake; compiled for the configured CUDA architectures): random weights and activations, run pre-pass + each kernel on the GPU, compare with the CPU `exact` path: on sm_52 (FP32 emulation) relative error ≤ 1e-4 of ‖w‖·‖x‖; on sm_60 (when one exists) compare with `h16<16>` from the CPU model at ≤ 1e-3 of ‖w‖·‖x‖. Also a ctest that compiles the fixture for sm_60 with `-Xptxas -v` via `bin/sm60-report.py` and fails on spills. Configure a second build dir `build-cuda` with `-DCMAKE_CUDA_ARCHITECTURES="52;60" -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc` (look at `CMakeLists.txt` for how CUDA is enabled) and run the GPU test there under the GPU lock.

## Files you may touch
- `include/p100/qmv.h`, `src/reference/qmv_model.cpp` (new), `src/CMakeLists.txt` or the root `CMakeLists.txt` only to add that source to the reference library.
- `tests/qmv_numerics.cpp`, `tests/h2_mmv_gpu.cu` (new), `tests/CMakeLists.txt` (register them).
- `bench/micro/h2_mmv.cu`, `bin/h2-sass.sh` (new).
- `docs/research/qmv-numerics-2026-10-10.txt`, `docs/research/h2-sass-2026-10-10.csv` (new, generated).
Nothing else. Do not edit llama.cpp, `bin/decode-model.py` or `bin/vecdot-sass.sh`. Do not write analysis or conclusions into docs; the manager does that.

## Must pass before you finish
- `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure` (all pass)
- the `build-cuda` configure/build and `ctest --test-dir build-cuda -R h2_mmv --output-on-failure` (or the sandbox reason it could not run)
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`
- `python bin/sm60-report.py bench/micro/h2_mmv.cu` (paste the table)

## Done means
CPU model, kernels, fixtures and counts exist and the GPU check ran (or is handed over with exact commands). Final report: files changed; each command with its result line; the sm60-report table; the h2 SASS CSV verbatim; the numerics summary (median/max per type × path × distribution, verbatim); SASS lines showing the LOP3/PRMT unpack and HFMA2 dot for q4_0 b1; what was NOT verified (FP16 arithmetic on a GPU, any timing). Do not interpret or decide whether to ship.
