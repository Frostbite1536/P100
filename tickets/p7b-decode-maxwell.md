# Ticket p7b-decode-maxwell: test the decode model's structure on the Maxwell GPU (same no-dp4a path) and add a calibration tool

Your current directory is a git worktree of the P100 repo (branch `p7b-decode-maxwell`, based on `main` @ `6554601`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores and NO __dp4a/__dp2a (sm_61+). Do NOT write `__dp4a` yourself. If you think you need a forbidden feature, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything. Build into `build/` inside this worktree.
- Do NOT commit, push, or run git commands that write **in the P100 repo**. Your private llama.cpp clone (step 1) is gitignored scratch.
- Never invent numbers. There is no P100 on this machine. The local GPU is a Maxwell sm_52 (Quadro M2200). Its numbers are labelled "Maxwell M2200" everywhere and are never presented as P100 numbers.
- GPU use: wrap every command that runs on the GPU in `flock /tmp/p100-gpu.lock <command>` (other workers share it; hold the lock for a whole measurement round, not per command, so another worker cannot interleave). If CUDA cannot initialise inside your sandbox, say so, leave the exact commands in your report, and continue; the manager will run them.
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`bin/decode-model.py` predicts, from llama.cpp's own per-type SASS counts (`docs/research/vecdot-sass-2026-10-10.csv`, `bin/vecdot-sass.sh`), whether decode mat-vec is bandwidth- or ALU-bound on a P100. It is uncalibrated. The local Maxwell GPU also lacks `__dp4a` (`GGML_CUDA_CC_DP4A` = 610), so llama.cpp runs the **same** emulated MMVQ path there. Its ALU-to-bandwidth ratio differs from the P100's, so the model predicts a different crossover (bandwidth-bound → ALU-bound as the batch grows). If the model's predicted crossover by type and batch matches Maxwell measurements, that supports its structure before the P100s arrive; if not, we learn where it is wrong.

## Design (manager decided, do not change)
1. Private llama.cpp clone: `git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp third_party/llama.cpp` (gitignored), checkout `bench/LLAMA_COMMIT`, **no patches**. Build Release with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52 -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF`, target `test-backend-ops`.
2. `bin/vecdot-sass.sh`: add `sm_52` as a third arch (same classifier); regenerate `docs/research/vecdot-sass-2026-10-10.csv` (rows for sm_60/sm_61 must be unchanged — check with diff and say so).
3. `bin/decode-model.py`:
   - `--variant m2200` with the Maxwell's parameters **measured on this machine**, not from a datasheet: SM count, max SM clock and memory bus width/clock from `cudaGetDeviceProperties` and `nvidia-smi -q -d CLOCK` (write a tiny `bench/micro/devprops.cu` or use `nvidia-smi`), lanes per SM = 128 for sm_52; record them and where each came from in a comment. HBM/GDDR bandwidth: theoretical from bus width × memory clock; also a `--bw-eff` measured in step 4.
   - Variant implies arch for the ALU counts (`pcie16|pcie12|sxm2` → sm_60, `m2200` → sm_52; keep the sm_61 reference line).
   - `--shape m,k` option: predict one mat-vec of an m×k weight (instead of a whole model), so it can be compared with `test-backend-ops` MUL_MAT cases directly.
   - Existing outputs for existing options must not change (the existing ctest proves it; do not edit it except to add new checks).
4. `bin/calibrate-decode.py` (stdlib only): reads `test-backend-ops perf -o MUL_MAT --output sql` output (format: see `tests/fixtures/tbo-perf-cpu.sql` and `bench/phase0.sh`), selects rows with quantised `type_a`, `type_b=f32`, `n` 1..8, contiguous inputs; for each (type, m, k, n) prints measured time per run and achieved weight GB/s next to the model's prediction for that shape and variant (bound, t_bw, t_alu). Fits `bw_eff` from rows the model calls bandwidth-bound and `issue_eff` from rows it calls ALU-bound (median of measured/predicted), prints both, then re-prints predictions with the fitted values and the ratio measured/predicted per row. Prints which rows changed bound after fitting. ctest `calibrate_decode` on a small SQL fixture you write by hand **marked as synthetic** in its first line (check parsing and arithmetic only).
5. Measurement on Maxwell (under the GPU lock for the whole round): `test-backend-ops perf -o MUL_MAT -b CUDA0 --output sql`, filtered to the decode shapes if the tool allows (`-p`), otherwise run all and filter in the script. Three rounds; between rounds re-read `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv` and record it (the laptop GPU throttles). Save raw SQL + clocks under `bench/results/maxwell-decode-<date>/` with a README (commands, llama.cpp commit, nvcc version, GPU, round-to-round spread per row). Run `bin/calibrate-decode.py --variant m2200` on the median of the rounds and save its output there.
6. Repeat the `--variant m2200` prediction without calibration and save it next to the calibrated one, so the uncalibrated model's crossover can be compared with the measurement.

## Files you may touch
- `bin/vecdot-sass.sh`, `docs/research/vecdot-sass-2026-10-10.csv` (sm_52 rows added), `bin/decode-model.py`, `tests/decode_model_check.py` (new checks only), `bin/calibrate-decode.py` (new), `bench/micro/devprops.cu` (new, optional), `tests/fixtures/tbo-perf-mulmat-synthetic.sql` (new), `tests/CMakeLists.txt` (register new ctests).
- `bench/results/maxwell-decode-<date>/` (new).
Nothing else. Do not edit llama.cpp. Do not write analysis or conclusions into docs; the manager does that.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all pass)
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`
- `python bin/decode-model.py --model llama3.1-8b --all-types --batch 1` output identical to the committed numbers in `docs/research/pre-hardware-findings-2026-10-10.md` §1 (paste)
- `python bin/decode-model.py --variant m2200 --shape 4096,14336 --all-types --batch 1` and `--batch 4` (paste)

## Done means
sm_52 counts exist, the model has a measured Maxwell variant and a shape mode, the calibration tool works, and Maxwell measurements (or exact handover commands) are saved. Final report: files changed; each command with its result line; the measured device parameters and their sources; the calibrate output (uncalibrated and fitted, verbatim); round-to-round spread; what may mislead (laptop throttling, Maxwell vs Pascal ISA differences, MMVQ launch parameters differing by arch — check `mmvq.cu` `get_device_table_id` and say which table Maxwell and P100 use); what was NOT verified (anything on a P100). Do not interpret or decide whether to ship.
