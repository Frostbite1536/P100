# Ticket p2-phase0-kit: Phase 0 baseline kit and patch A/B harness for 2× P100

Your current directory is a git worktree of the P100 repo (branch `p2-phase0-kit`, based on `main`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything. Build into `build/` inside this worktree.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52; its timings are not P100 numbers): anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Phase 0 (`docs/ROADMAP.md`) must answer hypotheses H1–H5 of `docs/research/volta-fa-lessons.md` §5 on the first day with the cards. volta-fa's kit was copied into this worktree, unmodified and untracked, as the starting point: `bench/phase0.sh` and `bench/patches/run.sh` (V100 versions). Both still check for a V100, build for sm_70 and run Volta-specific cases and variants. Lessons to keep: interleaved rounds, clocks/temperature per round, the f16-KV reference in the same session (INV-PERF-004), `set +e` around optional tools, results under `bench/results/<kind>-<stamp>-<host>/`.

## Design (manager decided, do not change)
`bench/phase0.sh`:
1. GPU check: name contains `P100` (FORCE=1 overrides); CUDA 12.8 check kept (message: 13 dropped sm_60). `ngpu` counts P100s. Build dir `$LLAMA/build-sm60`, `-DCMAKE_CUDA_ARCHITECTURES=60`.
2. `env` step also records: `nvidia-smi -q` (to `nvidia-smi-q.txt`), `nvidia-smi topo -m` and `topo -p2p r`, `ncu --list-chips` filtered for GP100 (or "ncu: no GP100" / "ncu: not installed"), `lscpu | head -20`.
3. Steps `support`, `test`, `perf` run per GPU index (`-b CUDA0`, `-b CUDA1` when present) for `FLASH_ATTN_EXT`, and `test`/`perf` also for `MUL_MAT` (H1). File names carry the op and device: `fa-test-CUDA0.txt`, `mulmat-perf-CUDA0.sql`, etc.
4. `bench` step: keep the depth (FA on/off), kv-quant, small-batch and ubatch sweeps on `CUDA0`; add a decode sweep `-p 0 -n 128 -d 0,4096,16384` per model (H1 end to end).
5. New `p2p` step: run `build/bench/p2p_bandwidth --csv` (this repo's binary; skip with a log line if it is not built) into `p2p.csv`.
6. `multi` step: unchanged logic (layer split; tensor split nccl/internal × GGML_CUDA_P2P on/off), on the first model that is listed in `bench/models.txt`.
7. Default `STEPS="env build support test perf bench p2p multi"`.

`bench/patches/run.sh`:
1. Variants (cumulative over `third_party/patches/`): `upstream` (none), `conv` (0001), `conv8` (0001–0002). Same worktree/ccache mechanics. GPU check P100, sm_60.
2. Correctness: full `FLASH_ATTN_EXT` and `MUL_MAT` per variant on CUDA0.
3. Kernel perf cases (local-only, injected like now): quantised K/V (`q8_0`, `q4_0`) at 3, 4, 8, 16 tokens (the TILE + conversion range on Pascal), D 128/256, GQA 4/6/8, KV 4096/32768; plus the same shapes with f16 K/V as the reference. Remove the Volta P-6/P-8 case lists and any reference to Volta patches.
4. Microbench step: build `bench/micro/dequant_nc.cu` with `-arch=sm_60`.
5. llama-bench: per model, `-ctk/-ctv f16,q8_0,q4_0 -p 8 -n 64 -d 16384` for every variant, interleaved rounds.
6. Summary as now (`bench/analyze.py tbo` / `lb`). Drop `bench/patches/check.py`-style acceptance logic; there is no design doc yet.

Dry-run test: `tests/fixtures/shims/` with stand-in `nvidia-smi`, `nvcc`, `test-backend-ops`, `llama-bench`, `ncu` scripts (print plausible, clearly fake output; `nvidia-smi` reports two `Tesla P100-PCIE-16GB`), and a ctest `phase0_dryrun` that runs `bench/phase0.sh` with `PATH` shimmed, `STEPS="env support test perf"` and a temp results dir (add an `OUT` override env var if needed), then checks the expected files exist. Same for `bench/patches/run.sh` if it can be done without building llama.cpp (a `STEPS` or `DRYRUN` switch that skips `git worktree`/build is acceptable); if not, STOP and explain before inventing a mechanism.

## Files you may touch
- `bench/phase0.sh`, `bench/patches/run.sh`: the port.
- `bench/models.example.txt`: models for 2× P100 (Llama-3.1-8B Q8_0 fits one card; Qwen3.8-27B-UD-Q4_K_XL 17.6 GB needs both on 16 GB cards; a D=256 or GQA-6 model).
- `tests/fixtures/shims/` (new), `tests/CMakeLists.txt`: the dry-run tests.
Nothing else.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all pass, including `phase0_dryrun`)
- `shellcheck -x bench/phase0.sh bench/patches/run.sh tests/fixtures/shims/*`
- `git status --short` shows no files written outside the worktree's allowed paths and no results directories left behind by the tests.

## Done means
Both scripts target 2× P100 / sm_60 as designed, and the dry run proves the step plumbing without hardware. Final report: files changed; each command with its result line; what the dry run does and does not prove; what was NOT verified (everything on real hardware). Do not decide whether to ship.
