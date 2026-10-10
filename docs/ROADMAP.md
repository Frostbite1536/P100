# Roadmap

Goal: faster llama.cpp inference on two Tesla P100s (sm_60), measured against upstream at a pinned commit, with
every change small enough to upstream or carry as a patch. Background and hypotheses:
`docs/research/volta-fa-lessons.md`.

## Pre-hardware (now, dev laptop)

- P-0 Scaffold: volta-fa's process, numerics contract (reference, tiled model, tolerances, error budget), P2P
  layer and analysis tools, renamed for P100. ✅ CPU build + 17 tests pass.
- P-1 Static tooling for sm_60 (ticket `tickets/p1-sm60-tooling.md`): `bin/sm60-report.py`,
  `bin/check-forbidden.sh` for sm_60, `bin/plan-tiles.py` with GP100 limits.
- P-2 Phase 0 kit (ticket `tickets/p2-phase0-kit.md`): `bench/phase0.sh` for one and two P100s, `bench/patches/run.sh`,
  box setup script; dry-run with stand-in commands.
- P-3 Numerics for FP16 products (H2): extend the tiled CPU model with "products rounded to FP16, Q pre-scaled
  by s" and "half2 chain of N terms", and add `error_budget` rows for llama.cpp's current TILE scheme and for
  N = 2, 4, 8. Answers whether a faster FA TILE inner loop is allowed before anyone writes it.
- P-4 Decode cost model (H1): `bin/decode-model.py`, bytes vs ALU instructions per weight type for MMVQ on
  sm_60 (from SASS counts like `docs/research/volta-fa-lessons.md` §2), to be calibrated on day one.
- Patches `third_party/patches/0001..0002` (volta-fa 0005/0006): check they apply at the pinned commit and
  build for sm_60 (`-DCMAKE_CUDA_ARCHITECTURES=60` with the local nvcc); MUL_MAT + FA correctness on the
  laptop's Maxwell GPU.

**Excludes**: any performance claim.

## Phase 0: Baseline on the P100s (first day on hardware)

Record the variant (12/16 GB, PCIe/SXM2), `nvidia-smi -q`, `nvidia-smi topo -m`, host CPU and PCIe layout in
`bench/BASELINE.md`. Then, in this order:

1. Full `test-backend-ops test -o FLASH_ATTN_EXT` and `-o MUL_MAT` on each GPU (crash classes, 3.).
2. **H1**: `test-backend-ops perf -o MUL_MAT` at batch 1–8 per weight type: achieved GB/s vs HBM roofline.
3. **H2/H3**: FA perf by head dim; llama-bench FA on/off × depth 0–32K.
4. Quantised KV: llama-bench tg/pp8 at 16K, f16/q8_0/q4_0, upstream vs patches 0001–0002 (**H5**).
5. **H4**: `p2p_bandwidth` (peer and staged), then llama-bench `-sm layer|row|tensor` on a model that needs
   both GPUs (Qwen3.8-27B Q4_K_XL, 17.6 GB).
6. Profiler check: `ncu --list-chips` for GP100; otherwise nsys timelines + test-backend-ops timings.

**Exit**: `docs/research/phase0-<date>.md` with a go/no-go per hypothesis and the measured bottleneck.

## Phase 1+: chosen from Phase 0

Candidates, in expected order of value until data says otherwise: decode mat-vec without dp4a (H1), conversion
patches upstream (H5), FA TILE inner loop within the FP16 budget (H2/P-3), padded GQA for TILE with per-shape
routing (lesson 3.2), two-GPU split and P2P overlap (H4). Each gets a design doc in `docs/design/` with an
"Accept if" section before code.
