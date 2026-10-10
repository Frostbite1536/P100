# Ticket p3-sm60-numerics: model llama.cpp's sm_60 FlashAttention arithmetic and FP16 alternatives on the CPU

Your current directory is a git worktree of the P100 repo (branch `p3-sm60-numerics`, based on `main`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything. Build into `build/` inside this worktree.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead. `include/p100/tolerance.h`, `tests/error_budget.cpp` and the existing cases in `tests/test_tiled_model.cpp` must stay byte-identical.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52; its timings are not P100 numbers): anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
INV-NUM-005 (`docs/INVARIANTS.md`) allows FP16 products or short `half2` chains in a FlashAttention kernel only with an error-budget row. The CPU model `p100::attention_tiled_model` (`src/reference/attention_tiled_model.cpp`, config in `include/p100/attention.h`) currently models FP32 products, FP16-rounded P, FP16 output, and a 16-element FP16 QK accumulation (`qk_accum_fp16`). It cannot express what llama.cpp's TILE kernel does on sm_60 (`FAST_FP16_AVAILABLE`), read at commit 79e2e74 in `/home/p51/Github/P100/third_party/llama.cpp/ggml/src/ggml-cuda/` (read-only):
- **QKᵀ**: Q is multiplied by the softmax scale in FP32, rounded to FP16, then multiplied by 1/4 in `half2` (`fattn-tile.cuh:918-937`); each `half2` product `v*u` is rounded to FP16, converted to float and added to an FP32 accumulator; the FP32 result is multiplied by 4 afterwards (`common.cuh:764-776` `ggml_cuda_mad(float&, half2, half2)`; `fattn-tile.cuh:624-632`).
- **Softmax**: the running max is `max(prev, x + FATTN_KQ_MAX_OFFSET)` with `FATTN_KQ_MAX_OFFSET = 3*0.6931` (`fattn-common.cuh:19`, `fattn-tile.cuh:642`), so P = exp(x − m) ≤ 1/8. P is computed in FP32, its FP32 value is added to the FP32 row sum, and P is stored to shared memory as FP16 (`fattn-tile.cuh:662-700`).
- **P·V**: the output accumulator is **`half2 VKQ`** (`fattn-tile.cuh:888`), rescaled by an FP16 `KQ_max_scale_h2` once per KV tile (`:688-692`) and updated with `VKQ += V_k*KQ_k` in `half2` (`:753`) over the whole KV range handled by that block.

## Design (manager decided, do not change)
1. Append to `TiledModelConfig` (after the existing members, so existing aggregate initialisers keep their meaning):
   - `bool qk_scale_in_q = false;` the softmax scale is multiplied into Q in FP32 and Q is rounded to FP16 before the products (instead of scaling the dot product afterwards).
   - `float qk_q_prescale = 1.0f;` the (FP16) Q is multiplied by prescale and rounded to FP16 again before the products; the FP32 dot product is divided by prescale afterwards.
   - `bool qk_products_fp16 = false;` each Q·K product is rounded to FP16, then added to the FP32 accumulator.
   - `int qk_half2_chain = 0;` if > 0: two lanes (even and odd `d`) each accumulate in FP16 with fused multiply-add (one rounding per step) for N steps, then both lane partials are added to the FP32 accumulator and the lanes restart at 0 (N steps per lane = 2N elements).
   - `float kq_max_offset = 0.0f;` running max is `max(m_prev, tile_max + kq_max_offset)`; LSE stays `m + log(l)`.
   - `bool l_from_unrounded_p = false;` the row sum adds P before the FP16 rounding that P·V uses.
   - `bool o_accum_fp16 = false;` the O accumulator is FP16: per tile, rescale `acc = fp16(acc * fp16(alpha))`, then for each key `acc = fp16(acc + p*v)` with a single rounding (fused multiply-add). This replaces the separate FP32 `tile_acc` in this mode.
   - Rounding helper: compute fused results in `double` and round to FP16 once. A `double` overload of the FP16 rounding (in the model's `.cpp`, or a new declaration in `attention.h`) is allowed; existing `round_to_fp16(float)` behaviour must not change.
   - `attention_tiled_model` throws `std::invalid_argument` if more than one of `qk_accum_fp16`, `qk_products_fp16`, `qk_half2_chain > 0` is set, or if `qk_q_prescale <= 0`.
2. New executable `tests/error_budget_sm60.cpp` (registered in ctest as `error_budget_sm60`; keep the existing `error_budget` untouched). Configurations:
   - `PO`: the existing correct-kernel config `{64, 64, false, true, true}` (reference point; must stay within `tolerance.h`, as in `error_budget`).
   - `L60`: llama.cpp sm_60 TILE: `qk_scale_in_q`, `qk_products_fp16`, `qk_q_prescale = 0.25`, `p_fp16`, `kq_max_offset = 3*0.6931f`, `l_from_unrounded_p`, `o_accum_fp16`, `o_fp16_out`.
   - `L60o32`: `L60` with `o_accum_fp16 = false` (isolates the FP16 output accumulator).
   - `C2`, `C4`, `C8`: `qk_half2_chain = 2/4/8`, `qk_scale_in_q`, `qk_q_prescale = 0.25`, `p_fp16`, `o_fp16_out`, FP32 O accumulation (candidate faster inner loops for hypothesis H2).
   - `br = 32, bc = 64` for all (sm_60 tiles are smaller than Volta's; the numerics depend little on them).
   - Distributions: the existing `normal`, `outlier` (Q,K channels 0..3 ×20), `peaked` (scale 64/√d), plus `vbias`: V ~ N(0,1) + 2.0 on channels 0..7 (a positive-mean V channel: the case where FP16 accumulation can swamp small additions).
   - Shapes: d ∈ {64, 128, 256}; seq_kv ∈ {512, 4096, 32768}; seq_q = 16; causal off and on; batch = heads = 1. Inputs rounded to FP16 as in `error_budget`. Seeds documented like `error_budget`'s.
   - Output: the same per-row columns as `error_budget`, plus a final summary table per config: worst max_rel, worst lse_abs, number of elements outside `out_within_tolerance`/`lse_within_tolerance`, non-finite count.
   - Exit code: 0 unless `PO` has any element out of tolerance or any non-finite value. Out-of-tolerance or non-finite results for `L60`, `L60o32`, `C*` are **findings**, printed and not failed.
   - Runtime: measure it. If the full run exceeds 120 s in a Release build, ctest runs it without the 32768 shapes and a `--long` flag runs everything; say which you chose.
3. New unit tests `tests/test_tiled_model_sm60.cpp` (ctest `test_tiled_model_sm60`) for the new switches, each with a hand-checkable expectation:
   - `qk_q_prescale = 0.25` with FP32 products matches the default config to ≤ 1e-6 relative on random data (powers of two are exact).
   - `o_accum_fp16` swamping: equal logits, V = 1.0 everywhere, seq_kv = 8192, d = 64. The exact output is 1.0; assert the FP32 path gives 1.0 within tolerance, and print (do not assert a guess) what the FP16-accumulator path gives.
   - `qk_half2_chain = N` with N ≥ d/2 equals a pure FP16 FMA accumulation per lane, checked against a direct scalar loop on a small vector.
   - `kq_max_offset` leaves O and LSE unchanged within tolerance (FP32 everything else).
   - `qk_scale_in_q` with FP32 products matches the default config within tolerance.
   - Invalid combinations throw.
4. Check the FMA assumption: compile with `/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -arch=sm_60 -cubin` a minimal kernel doing `half2 acc; acc += a*b;` in a loop (as `fattn-tile.cuh:753` does) and look at `cuobjdump -sass` for `HFMA2` vs `HMUL2`+`HADD2`. Report what you see. If the SASS shows separate multiply and add, STOP and report before finishing step 1's `o_accum_fp16` rounding (the manager decides the model).
5. Write the full output of the long run to `docs/research/error-budget-sm60-2026-10-10.txt` (with the command line and build type at the top).

## Files you may touch
- `include/p100/attention.h`: only the `TiledModelConfig` additions and an optional `double` rounding declaration.
- `src/reference/attention_tiled_model.cpp`: the new modes.
- `tests/error_budget_sm60.cpp`, `tests/test_tiled_model_sm60.cpp` (new), `tests/CMakeLists.txt` (register them).
- `docs/research/error-budget-sm60-2026-10-10.txt` (new).
Nothing else. Do not write analysis or conclusions into docs; the manager does that.

## Must pass before you finish
- `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure` (all pass, including the two new tests and the unchanged `error_budget`)
- `git diff --stat -- include/p100/tolerance.h tests/error_budget.cpp tests/test_tiled_model.cpp` prints nothing
- `bin/check-forbidden.sh`

## Done means
The model expresses the seven switches as designed, the unit tests pin their behaviour, and the budget file exists. Final report: files changed; each command with its result line; the SASS finding from step 4 (quote the relevant lines); the per-config summary table from the budget run, copied verbatim; runtime; what was NOT verified (anything on a GPU). Do not interpret the findings or decide whether to ship.
