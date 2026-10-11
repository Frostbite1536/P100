# Ticket p8-fa-vkq-flush: flush llama.cpp's FP16 FA output accumulator to FP32 once per KV tile (patch 0004 + model)

Your current directory is a git worktree of the P100 repo (branch `p8-fa-vkq-flush`, based on `main`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything. Build into `build/` inside this worktree.
- Do NOT commit, push, or run git commands that write **in the P100 repo**. In your private llama.cpp clone you may `git apply`, `git checkout -- .`, `git diff`, `git format-patch`/`git commit` (to produce the patch); that clone is gitignored scratch.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead. `include/p100/tolerance.h` and all existing test files except `tests/error_budget_split.cpp` must stay byte-identical.
- Never invent benchmark numbers. There is no P100 here. The local GPU is a Maxwell sm_52 without `FAST_FP16_AVAILABLE`, so this patch does not change what runs there; your sandbox probably cannot reach the GPU anyway. Static checks only.
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/error-budget-split-2026-10-10.txt` (p5): with the real key split of llama.cpp's sm_60 TILE kernel, the FP16 output accumulator (`half2 VKQ`, `fattn-tile.cuh:888`, updated at `:753`, rescaled per tile at `:688-692` and `:1060-1080`, warp partials combined in half2 at `:986-1048`) gives max relative error 0.1–2.6% for decode (21 parallel blocks) but 2.8% / 18% / 33% for a 512-token prefill at 4K/16K/32K keys with a biased V channel (3 parallel blocks, ~11K keys per accumulator). With 256 keys per accumulator the error was ≤ 0.24%. The same kernel runs on Volta (sm_70) for TILE-routed shapes.

## Design (manager decided, do not change)
1. **Patch `third_party/patches/0004-CUDA-FA-tile-flush-FP16-VKQ-to-FP32-per-KV-tile.patch`** against `bench/LLAMA_COMMIT`, applying cleanly after 0001–0002 (it must not depend on 0003, which only touches tests). Only `ggml/src/ggml-cuda/fattn-tile.cuh` changes, only inside `#ifdef FAST_FP16_AVAILABLE` paths:
   - keep `half2 VKQ` as the per-tile accumulator; add `float2 VKQ_f32[same count]` = 0;
   - where the per-tile rescale happens (`:688-692` and the corresponding place in the second loop around `:1060-1080`), rescale `VKQ_f32` in FP32 by `KQ_max_scale` instead of rescaling `VKQ` in half2;
   - after each KV tile's P·V loop, `VKQ_f32 += __half22float2(VKQ)` and reset `VKQ` to zero;
   - the `np > 1` warp combine (`:986-1048`) and the output write use `VKQ_f32` (combine in FP32; the shared buffer `VKQ_combine` becomes float, check it still fits the shared-memory buffer it aliases and say how);
   - no other behaviour change; the non-FAST_FP16 path is untouched (diff must show that).
   Generate the patch with `git format-patch` from your clone, with a subject line and a short body saying why (FP16 accumulation over thousands of keys loses small P·V terms; numbers from p5).
2. **Resources**: compile the TILE instances for sm_60 and sm_70 (same method as p5's `docs/research/fa-tile-resources-sm60-sm70-2026-10-10.csv`; read `bin/fa-split.py` for how) with and without 0004, and write `docs/research/fa-tile-resources-0004-2026-10-10.csv` with columns `cc,d,ncols1,ncols2,regs_before,regs_after,spill_st_after,spill_ld_after,stack_after,smem_before,smem_after`. **Any spill or stack growth is a finding: report it prominently; do not try to fix it by changing tile configs.** Also re-run `bin/fa-split.py` logic for occupancy with the new register counts and list rows whose `parallel_blocks` would change.
3. **Model**: append `bool o_accum_fp16_tile_flush = false;` to `TiledModelConfig`: with `o_accum_fp16`, each tile's P·V contributions accumulate in FP16 (one rounding per fused update, starting at 0), then are converted and added to an FP32 accumulator that carries the rescale; the `o_accum_np` partials are flushed the same way and combined in FP32. Unit test in a new `tests/test_tiled_model_flush.cpp` (ctest `test_tiled_model_flush`): on the swamping input of `test_tiled_model_sm60` (equal logits, V = 1, 8192 keys, d = 64, bc = 64) the flushed result is within `tolerance.h` of 1.0; flag off reproduces today's output bit-exactly on a random case.
4. `tests/error_budget_split.cpp`: add `flush` columns (L60 + `o_accum_fp16_tile_flush` at the row's split) next to `split` and `unsplit`, and a `flush` row in the summary. Regenerate `docs/research/error-budget-split-2026-10-10.txt` with `--long`. Existing `split`/`unsplit` numbers must not change (diff them and say so).
5. Build check: in a private clone (`git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp third_party/llama.cpp`, checkout the pin, apply 0001, 0002, 0004), build `test-backend-ops` with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;70" -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF` (Release). Paste the last build line. Optional if time allows: `52` too, and leave a command for the manager to run `test-backend-ops test -o FLASH_ATTN_EXT` on the Maxwell (it exercises only the unchanged FP32 path there; say so).
6. `third_party/patches/README.md`: add the 0004 row (provenance: this repo; status: "builds for sm_60/sm_70; P100 not run"). Facts only.

## Files you may touch
- `third_party/patches/0004-*.patch` (new), `third_party/patches/README.md` (one row).
- `include/p100/attention.h` (the one config addition), `src/reference/attention_tiled_model.cpp`.
- `tests/test_tiled_model_flush.cpp` (new), `tests/error_budget_split.cpp`, `tests/CMakeLists.txt`.
- `docs/research/fa-tile-resources-0004-2026-10-10.csv` (new), `docs/research/error-budget-split-2026-10-10.txt` (regenerated).
Nothing else. Do not write analysis or conclusions into docs.

## Must pass before you finish
- `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure` (all pass)
- `git diff --stat -- include/p100/tolerance.h tests/error_budget.cpp tests/error_budget_sm60.cpp tests/test_tiled_model.cpp tests/test_tiled_model_sm60.cpp tests/test_tiled_model_split.cpp` prints nothing
- patch applies after 0001–0002 (`git apply --check`), and the sm_60;70 build succeeds
- `bin/check-forbidden.sh`

## Done means
Final report: files changed; each command with its result line; the patch diff (verbatim); the resource CSV verbatim with spills highlighted; the split-budget summary rows for split/unsplit/flush (verbatim); what was NOT verified (anything on a P100 or V100: correctness and speed). Do not interpret or decide whether to ship.
