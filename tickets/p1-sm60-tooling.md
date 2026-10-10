# Ticket p1-sm60-tooling: static sm_60 tooling (report, forbidden-feature check, tile planner)

Your current directory is a git worktree of the P100 repo (branch `p1-sm60-tooling`, based on `main`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything. Build into `build/` inside this worktree.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52; its timings are not P100 numbers): anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
The repo was scaffolded from `~/Github/volta-fa` (V100, sm_70). Three Volta tools were copied into this worktree, unmodified and untracked, as starting points: `bin/sm60-report.py` (was `sm70-report.py`), `bin/check-forbidden.sh`, `bin/plan-tiles.py`. Each still encodes Volta. Port them to sm_60 (INV-ARCH-001, INV-ARCH-002 in `docs/INVARIANTS.md`; hardware table in `docs/research/volta-fa-lessons.md` §1). `CLAUDE.md`, `docs/WORKFLOW.md`, `.github/workflows/ci.yml` and `bin/setup-dev.sh` already call `bin/sm60-report.py` and `bin/check-forbidden.sh`.

## Design (manager decided, do not change)
1. `bin/sm60-report.py`: NVRTC at `--gpu-architecture=compute_60`, ptxas `-arch=sm_60 -v`. Same output table and CLI as now. Exit 1 if any kernel spills (stores or loads > 0) or static smem > 48 KB (`SMEM_LIMIT = 48 * 1024`). Docstring and messages say sm_60.
2. `bin/check-forbidden.sh`: keep the current pattern (BF16, FP8, cp.async, ldmatrix, m16n8k*, wgmma, clusters, TMA ...) and ADD the Volta+/sm_61+ features that are now forbidden: `nvcuda::wmma`, `#include <mma.h>`, every inline `mma.sync`, `__dp4a`, `__dp2a_lo`, `__dp2a_hi`, PTX `dp4a`/`dp2a`, `__reduce_(add|min|max|and|or|xor)_sync`. The error message says "forbidden for sm_60 (INV-ARCH-001)". Exclude `third_party/` (it is not in the scanned dirs today; keep it that way).
3. `bin/plan-tiles.py`: GP100 limits: 56 SMs, 64 KB smem per SM, 48 KB max per block (no carveout options; drop the carveout list), 65536 registers per SM, 255 max per thread, 64 max warps per SM, 32 max blocks per SM. Peaks as CLI-selectable variants `--variant pcie16|pcie12|sxm2` (default `pcie16`): FP32 9.3/9.3/10.6 TFLOPS, FP16 (half2) 18.7/18.7/21.2, HBM 732/549/732 GB/s. Remove the WMMA/m8n8k4 smem layout choice: S and P live in registers or smem by a `--s-in-smem` flag (default off). Roofline floor = max(FP16 half2 compute floor, DRAM floor), labelled as datasheet peaks. Help text says sm_60 and points at `bin/sm60-report.py` for real register counts.

## Files you may touch
- `bin/sm60-report.py`, `bin/check-forbidden.sh`, `bin/plan-tiles.py`: the port.
- `tests/fixtures/sm60/` (new): small `.cu` fixtures for the negative controls below.
- `tests/CMakeLists.txt`: add ctest entries for the controls (only if `.venv` NVRTC is found; skip cleanly otherwise).
Nothing else.

## Steps
1. Port the three tools per the design.
2. Fixtures: `ok.cu` (a small `extern "C" __global__` half2 kernel, no spills), `spill.cu` (forces spills, e.g. a large indexed local array), `smem.cu` (static `__shared__` of 49 KB), `dp4a.cu` (uses `__dp4a`), `wmma.cu` (includes `<mma.h>` and uses `nvcuda::wmma`).
3. ctest entries: `sm60_report_ok` passes on `ok.cu`; `sm60_report_spill` and `sm60_report_smem` expect failure (use `WILL_FAIL`); `forbidden_dp4a`/`forbidden_wmma` run the grep part on a fixture copy and expect failure. Keep fixtures out of the default `check-forbidden.sh` scan (they live under `tests/fixtures/`, so either scan excludes them or the script takes optional paths: choose the smallest change and say which).

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all pass, including the new controls)
- `bin/check-forbidden.sh` (ok on the repo)
- `shellcheck -x bin/*.sh`
- `python bin/sm60-report.py tests/fixtures/sm60/ok.cu` (table printed, exit 0)
- `python bin/plan-tiles.py --d 128 --br 64 --bc 64 --warps 4` and `--variant pcie12` (both print; paste output)

## Done means
The three tools target sm_60 as designed, the negative controls fail as expected, and everything above passes. Final report: files changed; each command with its result line; the sm60-report table for `ok.cu`; which negative controls failed as expected; what was NOT verified. Do not decide whether to ship.
