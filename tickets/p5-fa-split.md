# Ticket p5-fa-split: trace llama.cpp's FA key split on sm_60 and model it

Your current directory is a git worktree of the P100 repo (branch `p5-fa-split`, based on `main` @ `6554601`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything. Build into `build/` inside this worktree.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead. `include/p100/tolerance.h`, `tests/error_budget.cpp`, `tests/error_budget_sm60.cpp`, and the existing cases in `tests/test_tiled_model.cpp` and `tests/test_tiled_model_sm60.cpp` must stay byte-identical.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52): anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/pre-hardware-findings-2026-10-10.md` §2: the CPU model of llama.cpp's sm_60 TILE kernel shows that the FP16 output accumulator (`half2 VKQ`, `fattn-tile.cuh:888`) loses small P·V additions at long context (worst case 61% max relative error at 32K keys with a biased V channel). That figure assumes ONE accumulator per query row over ALL keys. The real kernel divides the keys:

- **Parallel blocks** (`fattn-common.cuh:1136-1223`, `launch_fattn`): `parallel_blocks` starts at the occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`), is capped by `ntiles_KV`, then raised while it improves wave efficiency over `nsm`. Each block handles KV tiles `blockIdx.y, blockIdx.y + gridDim.y, ...` of `nbatch_fa` keys (`fattn-tile.cuh:954-977`), with its own running max and sum; the parts are combined in FP32 (`flash_attn_combine_results`, `fattn-common.cuh:~928`).
- **Warps per column** (`np`, `fattn-tile.cuh:499,584`): within a block, key `k1 + threadIdx.y % np` of each V sub-batch goes to warp `threadIdx.y % np` (`:727-753`); each warp has its own `half2 VKQ`, all share the block's max (`:649-659`); at the end the `np` partial `VKQ` are added **in half2** (`:986-1048`).
- Config per (DKQ, DV, ncols): `nthreads, occupancy, nbatch_fa, nbatch_K` from `ggml_cuda_fattn_tile_get_config_nvidia_fp16` (`fattn-tile.cuh:21+`); `ncols = ncols1*ncols2` chosen by the launch code from the number of Q columns and the GQA ratio. Kernel choice VEC/TILE: `ggml_cuda_get_best_fattn_kernel` (`fattn.cu:541-719`).
- The VEC kernel (`fattn-vec.cuh`) is used for decode with a quantised KV cache. Find out whether its output accumulator is FP16 under `FAST_FP16_AVAILABLE` and how it splits keys; model it the same way if so.

llama.cpp at the pinned commit is at `/home/p51/Github/P100/third_party/llama.cpp` (read-only; do not modify). nvcc 12.8 and cuobjdump: `/home/p51/Github/P100/tools/cuda-12.8/bin/`.

## Design (manager decided, do not change)
1. `bin/fa-split.py` (standard library only). Given `--cc 600|700`, `--nsm` (default 56 for 600, 80 for 700), `--d`, `--heads-q`, `--heads-kv`, `--n-q` (Q columns per sequence), `--seq-kv`, `--kv-type f16|q8_0|q4_0`, it reproduces llama.cpp's decisions and prints: kernel (VEC/TILE/other), ncols1, ncols2, nwarps, np, nbatch_fa, occupancy used, ntiles_dst, ntiles_KV, parallel_blocks, and **keys per FP16 accumulator** (max over accumulators of the number of keys it receives). Every reproduced rule carries a comment with its `file:line`. `--table` prints the grid in step 3 as CSV.
   - Occupancy: llama.cpp asks the runtime. Compute it from (a) the config's launch-bounds `occupancy`, (b) shared memory per block vs per-SM capacity (sm_60: 64 KB/SM, 48 KB/block; sm_70: 96 KB/SM), (c) 2048 threads and 32 blocks per SM, and (d) registers per thread from the compiled kernel when available: compile the needed `template-instances/fattn-tile-instance-*.cu` (and VEC instances if relevant) with `nvcc -arch=sm_60 -cubin` (and `sm_70`) into `build/` and read `cuobjdump -res-usage`. Commit the measured register/smem counts as a small CSV `docs/research/fa-tile-resources-sm60-sm70-2026-10-10.csv` that the script reads. If an instance cannot be compiled standalone, say which and what it needed; fall back to the launch-bounds occupancy and say so in the output.
2. Model: append to `TiledModelConfig` (after the existing members):
   - `int kv_parallel_blocks = 1;` KV tile `t` (tiles of `bc` keys) goes to part `t % kv_parallel_blocks`; each part has its own running max, sum and O accumulator; parts are combined in FP32 by their (max, sum) as `flash_attn_combine_results` does (read it and mirror its arithmetic).
   - `int o_accum_np = 1;` within a part, key `j` of a tile goes to accumulator `j % o_accum_np` (all share the part's max and rescale); at the end of the part the accumulators are added in order 0+1+2+… in the accumulator's precision (FP16 when `o_accum_fp16`, with one rounding per addition).
   - Defaults reproduce today's behaviour exactly (the existing tests prove it). Throw `std::invalid_argument` for values < 1.
3. Grid for `--table` (both `--cc 600` and `--cc 700`): d=128 with heads 32/8 (Llama-3.1-8B), d=256 with heads 24/4 and d=128 with heads 40/8; `n-q` ∈ {1, 2, 4, 8, 16, 512}; `seq-kv` ∈ {4096, 16384, 32768}; kv-type f16 and q8_0. Write it to `docs/research/fa-split-sm60-sm70-2026-10-10.csv`.
4. New executable `tests/error_budget_split.cpp` (ctest `error_budget_split`): reads that CSV, and for each distinct (cc, d, kernel, bc=nbatch_fa, np, parallel_blocks, seq_kv) runs the `L60` configuration of `error_budget_sm60.cpp` (copy its definition; do not include or modify that file) with `kv_parallel_blocks`/`o_accum_np` set from the row, and the same with the split forced to 1/1 as the worst-case reference. Inputs: `normal` and `vbias` exactly as in `error_budget_sm60.cpp` (same seeds), seq_q = 16, non-causal, batch = heads = 1. Print per row: the CSV key columns, keys per accumulator, max_rel and lse_abs for split and unsplit, and the number of elements outside `out_within_tolerance`. Exit 0 unless a run produces a non-finite value or the split reference (1/1) differs from `L60` in `error_budget_sm60` for an identical shape (consistency check on one shape). Runtime: if > 120 s, ctest runs the seq_kv ≤ 16384 rows and `--long` runs all.
5. Unit tests in a new `tests/test_tiled_model_split.cpp` (ctest `test_tiled_model_split`): `kv_parallel_blocks = 4` and `o_accum_np = 4` with FP32 everything match the default config within tolerance; with `o_accum_fp16` and the swamping case of `test_tiled_model_sm60` (equal logits, V = 1, 8192 keys, d = 64), print the result for splits 1/1, 1/4, 8/1, 8/4 and assert that each split result is at least as close to 1.0 as the 1/1 result (more accumulators cannot be worse on this input); invalid values throw.
6. Write the full `--long` output to `docs/research/error-budget-split-2026-10-10.txt` with the command line and build type at the top.

## Files you may touch
- `bin/fa-split.py` (new); `docs/research/fa-tile-resources-sm60-sm70-2026-10-10.csv`, `docs/research/fa-split-sm60-sm70-2026-10-10.csv`, `docs/research/error-budget-split-2026-10-10.txt` (new, generated).
- `include/p100/attention.h`: only the two `TiledModelConfig` additions. `src/reference/attention_tiled_model.cpp`: the new modes.
- `tests/error_budget_split.cpp`, `tests/test_tiled_model_split.cpp` (new), `tests/CMakeLists.txt` (register them; also a ctest `fa_split_table` that runs `bin/fa-split.py --table` and checks it equals the committed CSV).
Nothing else. Do not write analysis or conclusions into docs; the manager does that.

## Must pass before you finish
- `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure` (all pass)
- `git diff --stat -- include/p100/tolerance.h tests/error_budget.cpp tests/error_budget_sm60.cpp tests/test_tiled_model.cpp tests/test_tiled_model_sm60.cpp` prints nothing
- `bin/check-forbidden.sh`
- `python bin/fa-split.py --cc 600 --d 128 --heads-q 32 --heads-kv 8 --n-q 1 --seq-kv 32768 --kv-type f16` and the same with `--n-q 512` (paste both)

## Done means
The script reproduces llama.cpp's split with file:line citations, the model expresses it, and the split budget file exists. Final report: files changed; each command with its result line; the resource CSV verbatim; the VEC kernel accumulator finding with file:line; the split-budget summary rows (verbatim, not interpreted); which rules you could not reproduce exactly and what you assumed; what was NOT verified (anything on a GPU).
