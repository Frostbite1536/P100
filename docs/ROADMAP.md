# Roadmap

Goal: faster llama.cpp inference on two Tesla P100s (sm_60), measured against upstream at a pinned commit, with
every change small enough to upstream or carry as a patch. Background and hypotheses:
`docs/research/volta-fa-lessons.md`.

## Pre-hardware (now, dev laptop)

- P-0 Scaffold: volta-fa's process, numerics contract (reference, tiled model, tolerances, error budget), P2P
  layer and analysis tools, renamed for P100. ✅
- P-1 Static tooling for sm_60: `bin/sm60-report.py`, `bin/check-forbidden.sh`, `bin/plan-tiles.py` with GP100
  limits. ✅ (ticket p1; note: volta-fa's 64×64 D=128 tile needs 51 KB smem, over sm_60's 48 KB, so P100 FA tiles are smaller)
- P-2 Phase 0 kit: `bench/phase0.sh` for one and two P100s, `bench/patches/run.sh` (upstream/conv/conv8);
  dry-run tests with stand-in commands. ✅ (ticket p2)
- P-3 Numerics model for llama.cpp's sm_60 FA scheme and `half2` chains (H2). ✅ (ticket p3) Findings:
  `docs/research/pre-hardware-findings-2026-10-10.md` §2: the FP16 output accumulator is a long-context risk
  (model worst case 61% at 32K keys with biased V; real key split not yet modelled); `half2` chains: 2 terms at most.
- P-4 Decode cost model (H1). ✅ (ticket p4) `bin/decode-model.py` + llama.cpp's real per-type SASS counts:
  predicted ALU-bound at batch 1 for q4_0/q4_K/q5_K/q6_K, every quantised type at batch ≥ 4 (uncalibrated; §1).
- P-5 FA key split traced (`bin/fa-split.py`) and modelled. ✅ (p5) Decode: 21 parallel blocks, ≤ 2.6% (biased V,
  32K); 512-token prefill: 3 blocks, 33%. VEC is FP32 on NVIDIA. `pre-hardware-findings-2-2026-10-10.md` §1.
- P-6 half2 decode mat-vec prototype (H1): CPU numerics, static SASS, Maxwell layout check. ✅ (p6/p6b)
  Design: `docs/design/H1-pascal-decode-matvec.md`.
- P-7 Patches 0001–0002 built for sm_60 and checked on Maxwell (no new failures); patch 0003 biased-V FA cases +
  FP32 CPU reference (llama.cpp's reference accumulated in FP16); decode model checked on Maxwell (issue
  efficiency 0.89–0.93) with `bin/calibrate-decode.py`. ✅ (p7a/p7b/p7c)
- P-8 Patch 0004: flush the TILE FP16 output accumulator to FP32 per KV tile (model worst 0.17%). ✅ (p8)
- P-9 Patch 0005 `mmvh`: half2 decode mat-vec in llama.cpp (q4_0/q8_0/q4_K, batch 1–8). ✅ (p9)
- P-10 `test_p2p` setup race fixed. ✅ (p10)
- P-11 Day-one A/B kit: variants `upstream conv8 flush mmvh`, MUL_MAT perf n = 1..8, tg128, perplexity. ✅ (p11)

Still open before hardware: mmvh for q5_K/q6_K (the other half of Q4_K_M files); upstream-quality write-ups are
the owner's (llama.cpp forbids AI-written PR text).

**Excludes**: any performance claim.

## Phase 0: Baseline on the P100s (first day on hardware)

Variant: PCIe 16 GB (ordered). Record `nvidia-smi -q`, `nvidia-smi topo -m`, host CPU and PCIe layout in
`bench/BASELINE.md`. Then, in this order:

0. `bench/patches/run.sh` (all variants, one session): answers 0004 (biased-V fixed? cost?) and 0005 (H1 Accept-if).
1. Full `test-backend-ops test -o FLASH_ATTN_EXT` (incl. biased-V cases) and `-o MUL_MAT` on each GPU.
2. **H1**: `test-backend-ops perf -o MUL_MAT` at batch 1–8 per weight type: achieved GB/s vs HBM roofline;
   calibrate `bin/decode-model.py` (`--bw-eff`, `--issue-eff`) and record where it was wrong.
3. **H2/H3**: FA perf by head dim; llama-bench FA on/off × depth 0–32K.
4. Quantised KV: llama-bench tg/pp8 at 16K, f16/q8_0/q4_0, upstream vs patches 0001–0002 (**H5**).
5. **H4**: `p2p_bandwidth` (peer and staged), then llama-bench `-sm layer|tensor` (row split no longer exists for CUDA, llama.cpp #24216) on a model that needs
   both GPUs (Qwen3.8-27B Q4_K_XL, 17.6 GB).
6. Profiler check: `ncu --list-chips` for GP100; otherwise nsys timelines + test-backend-ops timings.

**Exit**: `docs/research/phase0-<date>.md` with a go/no-go per hypothesis and the measured bottleneck.

## Phase 1+: chosen from Phase 0

Candidates, in expected order of value until data says otherwise: decode mat-vec without dp4a (H1), conversion
patches upstream (H5), FA TILE inner loop within the FP16 budget (H2/P-3), padded GQA for TILE with per-shape
routing (lesson 3.2), two-GPU split and P2P overlap (H4). Each gets a design doc in `docs/design/` with an
"Accept if" section before code.
