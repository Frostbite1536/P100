# H1: decode mat-vec without dp4a (half2 path for sm_60)

Status: design, pre-hardware. Prototype measured statically and on the CPU (`p6`); llama.cpp patch is ticket `p9`.
Nothing here ran on a P100.

## Problem

llama.cpp's quantised decode mat-vec (MMVQ) quantises activations to q8_1 and dots them with `ggml_cuda_dp4a`,
which sm_60 emulates (`common.cuh:740-746`). Evidence that this makes P100 decode ALU-bound:

- llama.cpp's own `vec_dot_*_q8_1`, compiled for sm_60: 3.9–7.7 ALU instructions per weight per output column,
  2.0–3.5× sm_61 (`docs/research/vecdot-sass-2026-10-10.csv`, p4).
- `bin/decode-model.py` (datasheet peaks): q4_0/q4_K/q5_K/q6_K ALU-bound at batch 1 on a P100 PCIe 16 GB; every
  quantised type ALU-bound at batch ≥ 4, because MMVQ unpacks each weight again for every column.
- Structure check on the Maxwell M2200 (also no dp4a, same `MMVQ_PARAMETERS_GENERIC` table), `test-backend-ops
  perf -o MUL_MAT`, 74 decode shapes × 3 rounds (p7b): for rows the model calls ALU-bound, measured/predicted
  median 1.08 (fitted issue efficiency 0.93); for bandwidth-bound rows the fitted bandwidth efficiency is 0.62.
  First run was taken under heavy CPU load (round spread up to 190%); a clean re-run is pending. This tests the
  model's structure on a sibling architecture, not P100 numbers.

## Proposal

1. **Activations in FP16, block-scaled ("h16")**: a pre-pass replaces `quantize_q8_1` for the affected mat-vecs.
   Per 32 activations: `s = max|x|` (FP32), `x̂ = fp16(x/s)` (|x̂| ≤ 1, so FP16 partial sums cannot overflow),
   `sum = Σx̂` (FP32, for k-quant minimums). Stored permuted (0,2,1,3 within each group of 4) to match the
   unpack order, so the inner loop needs no shuffles.
2. **Weights unpacked once per block to half2 by bit tricks**: q4 nibbles via `lop3 (w & mask) | 0x6400/0x5400`
   (the halves are exactly 1024+q or 64+q), one `HADD2` to remove the bias; q8_0 via `xor 0x80` + `prmt` into
   `0x64xx`. Unchanged GGUF formats.
3. **Dot with HFMA2**, chain of 16 per lane per 32-weight block, flushed to an FP32 accumulator with the scales.
   The unpack is shared by all columns of a batch.

## Evidence so far

Static SASS for sm_60, one-block fixtures comparable with p4 (`docs/research/h2-sass-2026-10-10.csv`), ALU
instructions per weight per column (`MOV`/`DEPBAR` counted as ALU in both):

| type | llama.cpp B=1..8 | h2 B=1 | B=2 | B=4 | B=8 |
| --- | --- | --- | --- | --- | --- |
| q4_0 | 4.2 | 2.63 | 1.75 | 1.27 | 1.07 |
| q8_0 | 3.9 | 2.78 | 1.89 | 1.35 | 1.11 |
| q4_K | 5.5 | 2.78 | 1.83 | 1.40 | 1.19 |

Model predictions for Llama-3.1-8B on a P100 PCIe 16 GB (datasheet peaks, uncalibrated; tokens/s per batch step):

| type | batch | llama.cpp | h2 | h2 if every instruction (loads too) cost an ALU slot |
| --- | --- | --- | --- | --- |
| q4_0 | 1 | 148 (ALU) | 173 (bandwidth) | 173 |
| q4_K | 1 | 113 (ALU) | 173 (bandwidth) | 170 |
| q8_0 | 1 | 92 (bandwidth) | 92 | 92 |
| q4_K | 4 | 113 | 443 | 308 |
| q8_0 | 4 | 160 | 367 | 302 |

Numerics (`docs/research/qmv-numerics-2026-10-10.txt`, 256 rows, K 4096/14336, four activation
distributions): median error normalised by ‖w‖‖x‖ is 2.2e-6 to 7.7e-6 for h16 vs 2.6e-5 to 2.3e-4 for
llama.cpp's q8_1 path; per cell, q8_1's median error is 7–34× h16 chain 16's (11–46× chain 8's), i.e.
**the faster path is also more accurate**, because FP16 keeps 11 bits per activation where q8_1 keeps 8
relative to the block maximum. Chain 8 is slightly more accurate than 16; both are far inside the q8_1 error.

Correctness of layout and indexing: `bin/maxwell-check.sh` on the M2200 (FP32 emulation of the half2 ops),
29,543 cases, 0 failures, worst 4e-8 of ‖w‖‖x‖. FP16 arithmetic on a GPU: not run (needs sm_53+).

## Risks and open points

- Load instructions: the prototype loads activations as 32-bit `half2` (16 `LDG` per block per column). At
  B ≥ 4 loads are ~40% of instructions; the patch must use 128-bit loads or stage activations in shared memory.
- k-quants other than q4_K (q5_K, q6_K: the other half of Q4_K_M files) need their own unpack; q6_K's 2-bit high
  parts need a second `lop3`. Not yet counted.
- llama.cpp fuses gate/up mat-vecs with MMVQ (`ggml_cuda_should_fuse_mul_mat_vec_q`); the new path must either
  support the fusion or be excluded from it without slowing those nodes.
- The pre-pass is per input vector; llama.cpp shares one q8_1 quantisation between consumers of the same
  input. Keep that sharing for h16.
- MMVQ's thread mapping (`MMVQ_PARAMETERS_GENERIC`: warps per row, rows per block) is tuned for dp4a costs; the
  half2 path may want more rows per warp to amortise activation loads. Tune on the P100, not before.

## Accept if (on the P100, same-session interleaved A/B, INV-PERF-004)

- `test-backend-ops test -o MUL_MAT` (and `MUL_MAT_ID` if touched): 0 new failures.
- `test-backend-ops perf -o MUL_MAT`, n = 1: q4_0 and q4_K ≥ 1.10×, no type < 0.98×; n = 4: q4_K ≥ 2×.
- llama-bench tg128 Llama-3.1-8B Q4_K_M ≥ 1.10× upstream, perplexity (wikitext-2, 32 chunks) within ±0.02.

**Kill if** Phase 0 shows upstream MMVQ at n = 1 already reaching ≥ 85% of measured copy bandwidth for q4_K
(then decode is not ALU-bound at batch 1 and the remaining case is speculative decoding only).
