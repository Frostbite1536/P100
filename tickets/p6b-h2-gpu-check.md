# Ticket p6b-h2-gpu-check: fix the sm_52 reference of h2_mmv_gpu (follow-up to p6)

Your current directory is the p6 worktree (branch `p6-h2-decode-proto`) with p6's uncommitted changes. Work only here. All p6 rules apply unchanged (sm_60 only, venv PATH prefix, no git writes, no tolerance loosening, no invented numbers, GPU commands under `flock /tmp/p100-gpu.lock`, stay in file scope).

## What the manager found
The manager ran your hand-over commands on the Maxwell M2200:
```
q4_0 B=8 K=32 row=10 col=4 chain=16 error=0.000104088177
sm_52: Quadro M2200; seed=600602; FP32 emulation checks indexing/layout only
exit 1
```
This is a flaw in the ticket, not in the kernel: on sm_52 the test compares with `exact` (FP32 activations), but the design stores activations as FP16 `x̂`, so a K=32 row can legitimately differ by ~1e-4 of ‖w‖‖x‖. The sm_52 check must isolate indexing/layout.

## Design (manager decided)
1. CPU model: add a path `h16_exact`: the same `h16` activations (`s`, FP16 `x̂`, `sum`) as the kernel's pre-pass, dot product in FP64 with no further rounding. Use it as the sm_52 reference with threshold 1e-5 of ‖w_row‖·‖x‖ (FP32 accumulation only). Keep the sm_60 reference (`h16<CHAIN>` at 1e-3) unchanged.
2. The test reports every failing case (type, B, K, row, col, chain, error) up to 20 lines, then a summary `checked N, failed M, worst error E (case)`, and exits non-zero if M > 0. It must not stop at the first failure.
3. `bin/maxwell-check.sh`: builds `tests/h2_mmv_gpu.cu` for `sm_52` and `sm_60` with the CUDA 12.8 nvcc and the built reference library (the hand-over command), and runs it under the GPU lock. The CMake guard (INV-ARCH-001) stays as it is; this script is the documented way to run the Maxwell check. shellcheck-clean.
4. Add `h16_exact` to `tests/qmv_numerics.cpp` output only if cheap; otherwise leave the numerics file unchanged. If you change the output, regenerate `docs/research/qmv-numerics-2026-10-10.txt`.

## Files you may touch
`include/p100/qmv.h`, `src/reference/qmv_model.cpp`, `tests/h2_mmv_gpu.cu`, `tests/qmv_numerics.cpp`, `docs/research/qmv-numerics-2026-10-10.txt`, `bin/maxwell-check.sh` (new), `tests/CMakeLists.txt`. Nothing else.

## Must pass before you finish
- `cmake --build build && ctest --test-dir build --output-on-failure` (all pass)
- `shellcheck -x bin/*.sh`, `bin/check-forbidden.sh`
- `bin/maxwell-check.sh` builds (running it will fail in your sandbox; say so)

## Done means
Final report: files changed; each command with its result line; the exact command for the manager to run. Do not interpret.
