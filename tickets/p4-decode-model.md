# Ticket p4-decode-model: per-type SASS counts of llama.cpp's decode dot products and a bandwidth-vs-ALU decode model for sm_60

Your current directory is a git worktree of the P100 repo (branch `p4-decode-model`, based on `main`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain. (Exception for this ticket: the SASS-count fixture is ALSO compiled for sm_61 as a comparison, where llama.cpp's own code uses `__dp4a`; do not write `__dp4a` yourself.)
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything. Build into `build/` inside this worktree.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52; its timings are not P100 numbers): anything needing one is "not run". The model's outputs are predictions from datasheet peaks and must be labelled so.
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Hypothesis H1 (`docs/research/volta-fa-lessons.md` §2, §5): on sm_60 llama.cpp's quantised decode mat-vec (MMVQ) emulates `__dp4a` (`common.cuh` `ggml_cuda_dp4a`, `__CUDA_ARCH__ < 610`), so it may be ALU-bound rather than HBM-bound. A hand-written replica of one q8_0 block measured 120 SASS instructions on sm_60 vs 36 on sm_61. That replica is not llama.cpp's code, and other weight types (4-bit unpacking, k-quant scales) differ. This ticket measures the real per-type cost statically and turns it into a model that Phase 0 will calibrate.

llama.cpp at the pinned commit is checked out read-only at `/home/p51/Github/P100/third_party/llama.cpp` (do not modify it). The decode dot products are `vec_dot_<type>_q8_1(vbq, bq8_1, kbx, iqs)` in `ggml/src/ggml-cuda/vecdotq.cuh`, with `VDR_<TYPE>_Q8_1_MMVQ` ints per call and `QI<type>` ints per block; `mmvq.cu` shows how a block is covered (`iqs` stepping by VDR). Full nvcc 12.8 is at `/home/p51/Github/P100/tools/cuda-12.8/bin/` (`nvcc`, `cuobjdump`).

## Design (manager decided, do not change)
1. `bench/micro/vecdot_sass.cu`: includes llama.cpp's `vecdotq.cuh` (include dirs `ggml/include`, `ggml/src`, `ggml/src/ggml-cuda`). For each type in {q4_0, q4_1, q5_0, q5_1, q8_0, q4_K, q5_K, q6_K, iq4_xs}: one `extern "C" __global__` kernel that covers exactly one weight block for one output column the way MMVQ does (loop `iqs` over the block in steps of VDR, summing the `vec_dot_*` results), reading the block and the matching `block_q8_1` data from global pointers and writing one float. Add `static_assert`s on `sizeof(block_<type>)` matching the bytes-per-block table in step 3, so a wrong table fails to compile. If `vecdotq.cuh` cannot be compiled standalone without editing llama.cpp, STOP and explain what it needs.
2. `bin/vecdot-sass.sh`: compiles that file with `-O3 -cubin` for `sm_60` and `sm_61` (nvcc from `$P100_NVCC`, default `tools/cuda-12.8/bin/nvcc` of the main checkout), runs `cuobjdump -sass`, and writes a CSV `type,arch,weights_per_block,sass_total,sass_alu,sass_mem,sass_ctrl,top_opcodes` where `sass_mem` = `LD*`/`ST*`/`LDG`/`STG`/`LDS`/`STS`/`LDC`, `sass_ctrl` = `S2R`/`EXIT`/`BRA`/`RET`/`NOP`/`BAR`/`SYNC`/`SSY`/`PBK`/`BRK`/`CAL`, and `sass_alu` = everything else. Write the result to `docs/research/vecdot-sass-2026-10-10.csv`.
3. `bin/decode-model.py` (standard library only):
   - Bytes per weight from the block sizes (q4_0 18/32, q4_1 20/32, q5_0 22/32, q5_1 24/32, q8_0 34/32, q4_K 144/256, q5_K 176/256, q6_K 210/256, iq4_xs 136/256, f16 2), cross-checked by the static_asserts above.
   - ALU instructions per weight per output column = `sass_alu / weights_per_block` from the CSV, per arch.
   - Presets: `llama3.1-8b` (32 layers, hidden 4096, FFN 14336, 32 heads, 8 KV heads, head dim 128, vocab 128256: q/k/v/o, gate/up/down and the output head as mat-vec weights; embeddings excluded) and `custom --weights <billions>`. All mat-vec weights are assumed to be the one `--type` (say so in the output).
   - GPU from `--variant pcie16|pcie12|sxm2` (default `pcie16`, the owner's cards): 56 SMs × 64 lanes × boost clock (1303/1303/1480 MHz) for the issue rate, HBM 732/549/732 GB/s.
   - For `--batch` 1..8: `t_bw = bytes/(HBM × --bw-eff)`, `t_alu = weights × alu_per_weight × batch/(issue_rate × --issue-eff)` for sm_60 and, as a reference line, with the sm_61 counts ("if dp4a existed"). Print bound (`bw`/`alu`), `max(t_bw, t_alu)` and the implied tokens/s ceiling. Defaults `--bw-eff 1.0 --issue-eff 1.0`; header says "datasheet peaks, uncalibrated; ignores activations, KV cache, attention and launch overhead".
   - `--csv` path option (default the committed CSV); `--all-types` prints one row per type at the given batch.
4. ctest: `decode_model` runs the script on `llama3.1-8b` for q8_0 and q4_0 at batch 1 and 4 and checks: q8_0 bytes per token = weights × 34/32 (±0.1%); `t_alu` at batch 4 = 4 × batch 1 (±0.1%); sm_61 ALU < sm_60 ALU for every type in the CSV. `vecdot_sass` (runs `bin/vecdot-sass.sh` into a temp file) is registered only when the nvcc path exists, and checks one row per type and arch.

## Files you may touch
- `bench/micro/vecdot_sass.cu`, `bin/vecdot-sass.sh`, `bin/decode-model.py` (new).
- `docs/research/vecdot-sass-2026-10-10.csv` (new, generated).
- `tests/CMakeLists.txt` and a new `tests/decode_model_check.py` (or equivalent) for the ctest entries.
Nothing else. Do not edit llama.cpp. Do not write analysis or conclusions into docs; the manager does that.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all pass, including `decode_model` and `vecdot_sass`)
- `bin/check-forbidden.sh` (the fixture lives under `bench/`; if the scan flags llama.cpp-originated `__dp4a` inside the included header it will not, since `third_party/` is not scanned. If it flags your file, fix your file.)
- `shellcheck -x bin/*.sh`
- `python bin/decode-model.py --model llama3.1-8b --all-types --batch 1` and `--batch 4` (paste both outputs)

## Done means
The CSV holds real per-type SASS counts for sm_60 and sm_61 from llama.cpp's own `vec_dot_*` code, and the model turns them into labelled predictions. Final report: files changed; each command with its result line; the CSV verbatim; both model outputs; anything about the counting that may mislead (e.g. per-block isolation omits MMVQ's loop and warp reduction); what was NOT verified (everything on a GPU). Do not interpret or decide whether to ship.
