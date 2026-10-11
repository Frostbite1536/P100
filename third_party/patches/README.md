# llama.cpp patch series

Applied in order with `git apply` on the commit in `bench/LLAMA_COMMIT`.

| patch | provenance | why it applies to sm_60 | status on P100 |
| --- | --- | --- | --- |
| 0001 fewer idle threads when converting short non-contiguous rows | volta-fa `0005` (byte-identical) | architecture-independent K/V-to-F16 conversion, used by the TILE FA kernel with a quantised cache (≥ 3 tokens on Pascal) | not run |
| 0002 vectorised q4_0/q8_0 non-contiguous dequantisation | volta-fa `0006` (byte-identical) | same | not run |
| 0004 flush FP16 TILE VKQ to FP32 per KV tile | this repo | sm_60 FAST_FP16_AVAILABLE TILE path; FP32 carry, rescale and warp combine | builds for sm_60/sm_70; P100 not run |

volta-fa measured both on a V100 (bit-identical, conversion 1.7–6.9× faster) and a Maxwell laptop GPU
(2.6–5.2×): `~/Github/volta-fa/docs/research/patches-cloud-2026-10-10.md` §2. volta-fa `0001`–`0004` are
Volta tensor-core or Volta-routing changes and do not apply here.
