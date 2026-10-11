# Pre-hardware findings, round 2 (2026-10-10): FA key split, half2 decode, Maxwell checks

Tickets p5–p11. CPU models, static SASS, and runs on the laptop's Maxwell M2200 (sm_52: no dp4a, no fast FP16).
**Nothing here ran on a P100.** Round 1: `pre-hardware-findings-2026-10-10.md`.

## 1. FlashAttention FP16 output accumulator (H2 risk)

**Real key split** (`bin/fa-split.py`, reproduces llama.cpp 79e2e74's kernel choice, TILE config, occupancy from
nvcc 12.8 register/smem counts, and `parallel_blocks`; `docs/research/fa-split-sm60-sm70-2026-10-10.csv`):
for Llama-3.1-8B (D 128, GQA 4) at 32K keys on a 56-SM P100, decode uses TILE with 21 parallel blocks
(1,600 keys per FP16 accumulator); a 512-token prefill uses 3 parallel blocks (10,944 keys per accumulator).
The VEC kernel (decode with a quantised KV cache) accumulates in **FP32** on NVIDIA (`fattn-vec.cuh:124-130`),
so it is not affected.

**Error with that split** (CPU model of the TILE arithmetic, biased V = +2 on 8 channels, seq_q 16,
`docs/research/error-budget-split-2026-10-10.txt`), worst max relative error, D 128:

| shape (sm_60) | 4K keys | 16K | 32K |
| --- | --- | --- | --- |
| decode, 21 blocks | 0.14% | 0.7% | 2.6% |
| 512-token prefill, 3 blocks | 2.8% | 18% | 33% |
| one accumulator per row (unsplit) | 13% | 43% | 58% |
| with patch 0004 (flush per KV tile), worst over all 75 traced shapes | | | 0.17% |

So the risk concentrates in **long-context prefill** on Pascal (no tensor-core FA path there). Zero-mean V
(`normal`) inputs: ≤ 2% with the real split at 32K (0.8% decode, 2.0% prefill), 3.7% unsplit. Whether real models have V channels with a large mean relative to their
spread is unmeasured; the biased-V case is a stress test, not a typical input.

**Patch 0004** keeps half2 for one KV tile (≤ 64–128 keys) and carries rescale and sum in FP32. sm_60: no new
spills (D 256 ncols 2×1 already spilled 16 B before). sm_70: D 256 configs 4×8, 8×4, 32×1 gain 4–32 B spills
(`docs/research/fa-tile-resources-0004-2026-10-10.csv`). The same TILE path runs on Volta for TILE-routed
shapes, so 0004 concerns volta-fa too.

## 2. llama.cpp's CPU reference had the same flaw

`test-backend-ops` puts the CPU backend into reference mode; its vec FA path then accumulates an F16 V in
**FP16** (`ggml_compute_forward_flash_attn_ext_f16_one_chunk`, `VKQ16`, ops.cpp ~8940-9035) over all keys. On
the Maxwell, whose CUDA FA is FP32, all 16 biased-V cases failed against that reference (NMSE 0.36 at 16K, 0.60
at 32K: the FP16 sum stalls at 8,192 once each term is below half an ulp). With patch 0003 keeping that
accumulator in FP32 in reference mode only, they pass 16/16, and full FLASH_ATTN_EXT passes 4009/4009 for
upstream and for 0001+0002.

**Measured (CPU only, `bench/results/cpu-fa-fp16-2026-10-10/`):** the same FP16 accumulation runs in **normal
CPU inference** for F16 V when the tiled path is not taken (fewer than 64 query rows), including the decode
split-KV path (per-thread chunks of `nek1/nth` keys). `test-backend-ops -b CPU` on the 16 biased-V cases: 4/16 pass
(only nb 512, the tiled path); NMSE at 32K keys 0.43 (nb 1) and 11.6–12.1 (nb 4/16). Disabling only the FP16
branch: 16/16. Why nb 4/16 exceed NMSE 1 is not explained. Stress inputs; real-model impact not measured;
reporting upstream is the owner's call (llama.cpp's contribution rules).

## 3. Decode mat-vec without dp4a (H1)

Design: `docs/design/H1-pascal-decode-matvec.md`. Prototype (p6) → llama.cpp patch 0005 `mmvh` (p9).

- **Static cost on sm_60**, all instructions in the compiled inner loop of `mmvh`, per 32 weights:
  batch 1: q4_0 98, q8_0 115, q4_K 103; batch 4: 200 / 212 / 213; batch 8: 331 / 343 / 362. llama.cpp's MMVQ
  per column costs 3.9–5.5 **ALU** instructions per weight (125–176 per 32) on sm_60, and that repeats per
  column.
- **Numerics** (`qmv-numerics-2026-10-10.txt`): block-scaled FP16 activations + half2 chains are 7–46× more
  accurate (median, normalised) than llama.cpp's q8_1 path, because FP16 keeps 11 bits per activation.
- **Correctness**: indexing/layout verified on the Maxwell with FP32 emulation of the half2 ops (prototype
  29,543 cases; `mmvh` forced: decode MUL_MAT 255/255, full MUL_MAT 1,139 pass / 0 fail up to the pre-existing
  q4_K abort). FP16 arithmetic on a GPU: not run.
- **Model structure check** on the Maxwell (same emulated MMVQ, `MMVQ_PARAMETERS_GENERIC` as sm_60), 74 decode
  shapes × 3 rounds, two runs, host not idle: on rows the model calls ALU-bound, measured/predicted fits
  issue efficiency 0.93 / 0.89; at n = 2–4 every type is within about ±30%. At n = 1 every type is 1.3–1.8×
  slower than the bandwidth bound (fitted bandwidth efficiency 0.62 / 0.63). Rows with large round-to-round
  spread (mostly n = 5, 8) disagree between runs and are not used.

Prediction for the P100 (uncalibrated; Llama-3.1-8B; tokens/s per batch step): q4_K batch 1 113 → 173
(bandwidth-bound), batch 4 113 → 443. If the P100's bandwidth efficiency is as low as the Maxwell's (0.6), the
batch-1 gain shrinks accordingly; Phase 0 measures it.

## 4. Patches 0001–0002 (conversion speed-ups from volta-fa)

Apply and build for sm_60 at the pin. Maxwell: FA 4009/4009, CPY 251/251, GET_ROWS 216/216, MUL_MAT identical
to upstream up to the pre-existing q4_K abort (also in volta-fa's notes). Speed on a P100: not run.

## 5. P2P test race (fixed)

`p2p_buffer_reuse` failed intermittently (53/400 under GPU load). Cause: the tests filled device buffers with a
pageable `cudaMemcpy`, which may return before its DMA completes, then read them on `cudaStreamNonBlocking`
streams. Fixed in the test (sync after the fill): 0/400 under load vs 4/200 for the old binary in the same
session. The P2P library was not at fault. volta-fa's `tests/test_p2p.cpp` has the same three lines.

## 6. What Phase 0 now runs

`bench/patches/run.sh` default variants `upstream conv8 flush mmvh` (0003 everywhere): full FA, biased-V cases,
MUL_MAT (mmvh also with `GGML_CUDA_DISABLE_MMVH=1`), FA perf, MUL_MAT perf at n = 1..8, llama-bench kv and tg128,
and perplexity when `PPL_TEXT` is given. `bin/calibrate-decode.py --variant pcie16` turns the MUL_MAT perf SQL
into calibrated decode predictions.

Open: biased-V on a real P100/V100 (does upstream fail as the model predicts, does 0004 fix it, at what cost);
mmvh FP16 correctness and speed; whether real models have biased V channels.
