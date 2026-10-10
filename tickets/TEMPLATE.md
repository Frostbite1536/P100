# Ticket @NAME@: @ONE-LINE GOAL@

Your current directory is a git worktree of the P100 repo (branch `@BRANCH@`, based on `main` @ `@SHA@`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=@REPO@/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/sm60-report.py <file.cu>` and paste its table. Spill must be 0.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/p100/attention.h`.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52; its timings are not P100 numbers): anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
@WHY. Link the invariant IDs. Give file:line anchors. For optimisation tickets paste the profile summary (kernel, launch config, regs, spill, smem, occupancy, FP32/FP16 pipe %, DRAM %, top bank-conflict lines).@

## Design (manager decided, do not change)
@tile sizes, smem layout/padding, warp->tile mapping, register/smem budget, target occupancy@

## Files you may touch
- @path@: @what may change@
Nothing else.

## Steps
1. @...@

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build`
- `ctest --test-dir build -R @focused tests@ --output-on-failure`
- `bin/check-forbidden.sh`
- `python bin/sm60-report.py @kernel files@` (kernel tickets only)

## Done means
@exact state@. Final report: files changed; each command with its result line; sm60-report table; which negative controls failed as expected; what was NOT verified. Do not decide whether to ship.
