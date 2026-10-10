# P100 agent instructions

Inference speed-ups for two NVIDIA Tesla P100s (Pascal GP100, sm_60): decode mat-vec, FlashAttention and two-GPU P2P. CUDA C++17, CMake. Baseline and integration target: llama.cpp (`third_party/llama.cpp`, gitignored clone; our changes live as patches in `third_party/patches/`). Sibling of `~/Github/volta-fa` (V100) and `~/Github/A770` (Arc): same process, different hardware. What carried over and why: `docs/research/volta-fa-lessons.md`.

## Hard rules (every agent, every ticket)

- **Pascal sm_60 ONLY.** There are **no tensor cores**: no `nvcuda::wmma`, no `mma.sync` of any shape. There is **no `__dp4a`/`__dp2a`** (those are sm_61, GP102/104); llama.cpp emulates them on sm_60 at ~3× the instructions. FP16 math goes through `half2` (`__hfma2`, `__hmul2`; 2× FP32 rate on GP100). Also forbidden: BF16, FP8, `cp.async`, `ldmatrix`, TMA, clusters, and anything copied from FlashAttention-2/3. If you think you need one, STOP and say so.
- **Shared memory ≤ 48 KB per block** (Pascal has no opt-in above 48 KB; 64 KB per SM).
- **CUDA 12.x only** (12.8 pinned, same as volta-fa). CUDA 13 dropped sm_60. Driver R580 is the last branch for Pascal.
- Every kernel change reports registers, spill stores/loads, stack and smem (`bin/sm60-report.py`, or `--ptxas-options=-v` on the P100 box). Spill > 0 is a failing result.
- Numerics: QKᵀ accumulates in FP32. Rounding products (or short `half2` chains) to FP16 is allowed only with an `error_budget` entry showing the tolerance still holds (INV-NUM-005). Never write the N×N score matrix to global memory.
- Correctness before speed: compare against the reference in `tests/`. Do not loosen tolerances or edit existing tests to pass; report the real error.
- Do not invent benchmark numbers. Anything not run on a P100 is "not run". V100, P40 (sm_61) and laptop (Maxwell sm_52) numbers are not P100 numbers.
- Read `docs/INVARIANTS.md` before changing kernel behaviour; changing an invariant needs a manager decision.
- Never read or commit `.env`, tokens, model weights, or raw `*.ncu-rep`/`*.nsys-rep`/`*.nvprof`.
- Workers: no git commands that write; leave changes uncommitted. The manager commits.
- Stay in the ticket's file scope. Out-of-scope need → STOP and explain.

## Commands

```bash
# dev machine (Quadro M2200 = Maxwell sm_52; no P100). Setup: docs/TOOLCHAIN.md
bin/setup-dev.sh                   # .venv (cmake, ninja, NVRTC/ptxas wheels); WITH_NVCC=1 adds full nvcc 12.8 in tools/
export PATH=$PWD/.venv/bin:$PATH   # once per shell
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
python bin/sm60-report.py src/<dir>/<file>.cu   # static regs/spill/smem for sm_60
python bin/plan-tiles.py --d 128 --br 32 --bc 64 --warps 4 [--regs N]   # smem/occupancy/roofline (default variant pcie16 = the owner's cards)
bin/check-forbidden.sh && shellcheck -x bin/*.sh

# P100 box (2 GPUs)
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=60 && cmake --build build && ctest --test-dir build --output-on-failure
bench/phase0.sh                    # Phase 0 baseline kit: llama.cpp FA/MUL_MAT test+perf per GPU, llama-bench grid, P2P, 2-GPU split modes
build/bench/p2p_bandwidth --csv    # staged vs peer copy bandwidth between the two P100s
```

## Conventions

- Files < ~1,500 lines; one kernel family per file.
- Tile sizes are template parameters (`BR`, `BC`, `D`, warps); no magic numbers inline.
- Tensor layout, mask and causal semantics are defined in `include/p100/attention.h`; do not reinterpret them per kernel.
- Every CUDA API call and launch is checked (`P100_CUDA_CHECK`).
- Comments explain *why* (layout, padding, occupancy trade-off), not what.
- Commits say why each test exists. Worker-written code: "Implemented by <worker> from ticket <name>; verified by <manager>".

## Docs map

[ARCHITECTURE](docs/ARCHITECTURE.md) · [INVARIANTS](docs/INVARIANTS.md) · [ROADMAP](docs/ROADMAP.md) · [WORKFLOW](docs/WORKFLOW.md) · [PROFILING](docs/PROFILING.md) · [TOOLCHAIN](docs/TOOLCHAIN.md) · [research/](docs/research/) · [REVIEW.md](REVIEW.md) · [tickets/](tickets/)
