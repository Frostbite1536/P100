# Pre-hardware findings: decode cost (P-4) and FlashAttention numerics (P-3), 2026-10-10

Static and CPU-model results only. Nothing here ran on a P100; every time below is a prediction from datasheet
peaks (PCIe 16 GB: 732 GB/s, 56 SMs × 64 lanes × 1303 MHz) and must be calibrated in Phase 0.

## 1. Decode mat-vec (H1): llama.cpp's own dot products, counted

`docs/research/vecdot-sass-2026-10-10.csv` (`bin/vecdot-sass.sh`): SASS of llama.cpp's `vec_dot_<type>_q8_1`
for one block and one output column, sm_60 vs sm_61. ALU instructions per weight:

| type | sm_60 | sm_61 (dp4a) | ratio |
| --- | --- | --- | --- |
| q4_0 | 4.2 | 1.3 | 3.2× |
| q8_0 | 3.9 | 1.1 | 3.5× |
| q5_0 | 7.2 | 3.4 | 2.1× |
| q4_K | 5.5 | 2.7 | 2.1× |
| q6_K | 7.7 | 3.8 | 2.0× |
| iq4_xs | 5.6 | 2.7 | 2.1× |

The q8_0 count (125 ALU per 32 weights) agrees with an independent hand-written replica (120 total). The
classifier counts `MOV`/`DEPBAR` as ALU, which overstates ALU time slightly.

`bin/decode-model.py --model llama3.1-8b --all-types` (uncalibrated):

| type | batch 1 bound, predicted tok/s | bandwidth ceiling | batch 4 |
| --- | --- | --- | --- |
| q8_0 | bandwidth, 92 | 92 | ALU |
| q4_1 | bandwidth, 156 | 156 | ALU |
| q4_0 | **ALU**, 148 | 173 | ALU |
| q4_K | **ALU**, 113 | 173 | ALU |
| q5_K | **ALU**, 101 | 142 | ALU |
| q6_K | **ALU**, 81 | 119 | ALU |

Reading: if the model holds, 4- to 6-bit decode on the P100 leaves 15–35% of the bandwidth ceiling unused at
batch 1, and at batch 2–8 (speculative-decoding verify) MMVQ's per-column unpacking makes every quantised type
ALU-bound, so extra verify tokens cost almost linear time. That ranks H1 first: a Pascal decode kernel that
unpacks each weight once per batch, or dots in `half2`, has a predicted upside. Phase 0 step 2 (MUL_MAT perf GB/s
by type and batch) tests it directly.

## 2. FlashAttention numerics on sm_60 (H2), CPU model

`docs/research/error-budget-sm60-2026-10-10.txt` (`tests/error_budget_sm60 --long`). Configurations: `PO` (FP32
products and accumulators; meets `tolerance.h` everywhere), `L60` (llama.cpp's sm_60 TILE scheme as read at
79e2e74), `L60o32` (L60 with an FP32 output accumulator), `C2/C4/C8` (`half2` FMA chains of N for QKᵀ, FP32
output accumulator). Worst relative error by input:

| config | normal 512 → 32K keys | outlier channels | peaked | biased V 512 → 4K → 32K |
| --- | --- | --- | --- | --- |
| L60 | 0.8% → 4.6% | 1.0–1.8% | 0.5–1.2% | 0.9% → 13% → **61%** |
| L60o32 | ≤ 0.06% | 1.0–1.8% | 0.5–1.2% | ≤ 0.05% |
| C2 | ≤ 0.07% | 1.1–2.6% | 0.4–1.0% | ≤ 0.05% |
| C4 | ≤ 0.06% | 1.6–5.1% | 0.7–1.7% | ≤ 0.05% |
| C8 | ≤ 0.08% | 2.5–11% | 0.8–2.8% | ≤ 0.05% |

Two separate effects:

1. **FP16 output accumulator (`half2 VKQ`, fattn-tile.cuh:888).** Error grows with the number of keys and with a
   non-zero-mean V channel: small P·V additions are lost against a large FP16 running sum (unit test: exact 1.0,
   FP16 path 0.25 at 8192 keys). **This is the model's worst case**: it gives each row one accumulator over all
   keys, while the real kernel splits keys across warps (`np`) and parallel blocks, combined in FP32, which divides
   the keys per FP16 accumulator. The magnitude on hardware is therefore unknown. The same code path runs on Volta
   for TILE-routed small batches (sm_70 also has fast FP16), so if confirmed it concerns volta-fa too.
2. **FP16 products / chains for QKᵀ.** Length-independent, driven by outlier channels and peaked logits.
   llama.cpp's current product rounding already costs 1–2% there; `half2` chains of 2 cost about the same, 4–8 cost
   more. H2's room is therefore a 2-term chain at most, and only with an FP32 output accumulator.

llama.cpp's own FLASH_ATTN_EXT tests use zero-mean random V and an NMSE threshold, so they would not exercise
effect 1 (not verified case by case); "outside tolerance" here means this repo's stricter `tolerance.h`.

## 3. Next steps this suggests

- Refine the model: keys per FP16 accumulator as a parameter, set from a trace of llama.cpp's `np` and
  parallel-block split for decode and small-batch shapes; re-run the biased-V rows.
- Phase 0 addition: a `test-backend-ops` FLASH_ATTN_EXT case with a biased V at 16K–32K keys, compared against
  the CPU backend, on the P100 (and on the next V100 session).
- H1 design doc for a Pascal decode mat-vec once Phase 0 confirms the ALU bound.
