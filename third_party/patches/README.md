# llama.cpp patch series

Applied in order with `git apply` on the commit in `bench/LLAMA_COMMIT`.

| patch | provenance | why it applies to sm_60 | status on P100 |
| --- | --- | --- | --- |
| 0001 fewer idle threads when converting short non-contiguous rows | volta-fa `0005` (byte-identical) | architecture-independent K/V-to-F16 conversion, used by the TILE FA kernel with a quantised cache (≥ 3 tokens on Pascal) | builds for sm_60; Maxwell correctness not run (CUDA unavailable in sandbox); P100 not run |
| 0002 vectorised q4_0/q8_0 non-contiguous dequantisation | volta-fa `0006` (byte-identical) | same | builds for sm_60; Maxwell correctness not run (CUDA unavailable in sandbox); P100 not run |
| 0003 biased-V FLASH_ATTN_EXT cases and FP32 CPU reference | this repo (test-only; one format-patch commit) | tests a non-zero V mean in F16 K/V attention at 16K/32K keys; F16 V output accumulates in FP32 only in CPU reference mode (`params->use_ref`); non-reference CPU and CUDA kernels unchanged | builds for sm_60; previous Maxwell biased-V: all 16 fail in both builds (NMSE about 0.36 at 16K, 0.60 at 32K) with FP16 CPU reference; FP32-reference rerun pending; P100 not run |

Build checks at the pin: upstream + 0003 and 0001 + 0002 + 0003. CUDA initialization failed before any Maxwell cases ran.
[Commands and outputs](../../bench/results/maxwell-patches-2026-10-10/README.md).

volta-fa measured both on a V100 (bit-identical, conversion 1.7–6.9× faster) and a Maxwell laptop GPU
(2.6–5.2×): `~/Github/volta-fa/docs/research/patches-cloud-2026-10-10.md` §2. volta-fa `0001`–`0004` are
Volta tensor-core or Volta-routing changes and do not apply here.
