# llama.cpp patch series

Applied in order with `git apply` on the commit in `bench/LLAMA_COMMIT`.

| patch | provenance | why it applies to sm_60 | status on P100 |
| --- | --- | --- | --- |
| 0001 fewer idle threads when converting short non-contiguous rows | volta-fa `0005` (byte-identical) | architecture-independent K/V-to-F16 conversion, used by the TILE FA kernel with a quantised cache (≥ 3 tokens on Pascal) | builds for sm_60; Maxwell (with 0002, 0003): FA 4009/4009, CPY 251/251, GET_ROWS 216/216, MUL_MAT identical to upstream up to the pre-existing q4_K abort; P100 not run |
| 0002 vectorised q4_0/q8_0 non-contiguous dequantisation | volta-fa `0006` (byte-identical) | same | same as 0001 |
| 0003 biased-V FLASH_ATTN_EXT cases and FP32 CPU reference | this repo (test-only; one format-patch commit) | tests a non-zero V mean in F16 K/V attention at 16K/32K keys; F16 V output accumulates in FP32 only in CPU reference mode (`params->use_ref`); non-reference CPU and CUDA kernels unchanged | builds for sm_60; Maxwell (FP32 CUDA FA): biased-V 16/16 (with the FP16 reference before the fix: 0/16, NMSE 0.36/0.60); P100 not run |
| 0004 flush FP16 TILE VKQ to FP32 per KV tile | this repo | sm_60 FAST_FP16_AVAILABLE TILE path; FP32 carry, rescale and warp combine | builds for sm_60/sm_70; sm_60 no new spills; P100 not run |
| 0005 half2 decode mat-vec (MMVH) | H1 design / h2 prototype, private llama.cpp port | block-scaled h16 and half2 for q4_0/q8_0/q4_K, batch 1–8; forced FP32 emulation on Maxwell | builds for sm_60; Maxwell (forced, FP32 emulation): decode MUL_MAT 255/255, full MUL_MAT 1139/0 fail before the pre-existing q4_K abort; P100 not run |
| 0006 MMVH q5_K and q6_K | this repo (one format-patch commit on 0005) | q5_K fifth-bit unpack; q6_K signed six-bit unpack with two eight-HFMA2 chains per 32 weights, batch 1–8; forced FP32 emulation on Maxwell | builds for sm_60, 0 spills (max 198 registers, q6_K B=7); Maxwell (forced, FP32 emulation): q4_0–q6_K n ≤ 8 MUL_MAT 375/375, full MUL_MAT 1161/0 fail before the pre-existing q4_K abort; P100 not run |

0001–0003 apply in order; 0004 and 0005 each apply after 0001–0002 and together with 0003; 0006 requires 0005 (checked: 0001–0003+0005+0006 and 0001–0006). Maxwell commands and
outputs: [bench/results/maxwell-patches-2026-10-10/README.md](../../bench/results/maxwell-patches-2026-10-10/README.md).

volta-fa measured both on a V100 (bit-identical, conversion 1.7–6.9× faster) and a Maxwell laptop GPU
(2.6–5.2×): `~/Github/volta-fa/docs/research/patches-cloud-2026-10-10.md` §2. volta-fa `0001`–`0004` are
Volta tensor-core or Volta-routing changes and do not apply here.

## Bench-only patches

These patches live in `bench-only/`, apply after each variant's series patches,
and are never part of the upstream series. The runner selects rows marked
`all variants` by filename; patches scoped to one build must use a different
entry in the applies-to column.

| patch | applies to | provenance | purpose / upstream policy |
| --- | --- | --- | --- |
| `bench-only/0001-tests-mul_mat-perf-cases-1-16.patch` | all variants | this repo, one `git format-patch` commit against `79e2e74eb` | 280 additional MUL_MAT perf cases: five quantised types, four Llama-3.1-8B shapes, n=1–16, skipping existing cases; eval list untouched; never upstream |
