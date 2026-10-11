# What volta-fa's small-batch matmul work (P-9) teaches the P100 (2026-10-11)

Follow-up to `volta-fa-lessons.md` (2026-10-10), which covers the attention and conversion work. This note covers
volta-fa's P-9: a Volta tensor-core matmul for quantised weights at 5–16 tokens, llama.cpp patch 0008. It was
measured on a rented V100 on 2026-10-11 (`~/Github/volta-fa/docs/design/P9-small-batch-qmatmul.md`). Nothing below
has run on a P100. Items marked **hypothesis** are reasoning from V100 data and llama.cpp code at the shared pin
`79e2e74`, not measurements.

## 1. On the V100, the gap was matmul at 5–16 tokens, not attention

- **Where the time went.** The llama-superfast-v100 fork was 85–89% faster than upstream on Qwen3.8-27B at 8–16
  tokens. About 95% of that came from its quantised matmul, not attention: 68.5 → 32.8 ms of a 78.7 ms upstream
  step (`volta-fa/docs/research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md`). volta-fa had assumed
  attention.
- **What upstream did per batch size.** It streamed weights at 570–775 GB/s at 1 token and 205–365 GB/s at 9–16:
  MMVQ redoes the dequantisation for every column, and Volta's MMQ path uses dp4a.
- **Result.** Patch 0008 reads each weight once in GGUF layout, decodes it to FP16 and feeds tensor cores. Qwen
  +34–39% at 8–16 tokens, Llama-3.1-8B +27% at 12–16, unchanged at 1–4.

**Lesson:** before choosing kernel work, profile one whole 8- and 16-token step per kernel (nsys, per kernel grid
→ shape) and measure MUL_MAT GB/s against the HBM roofline at every batch size from 1 to 16.

## 2. P100 at 9–16 tokens is probably the weakest range (hypothesis)

At the pin, dense quantised MUL_MAT on sm_60 routes as follows (`ggml-cuda.cu` `ggml_cuda_mul_mat`, `mmvq.cu`,
`mmq.cu`):
- **1–8 tokens:** MMVQ (`MMVQ_MAX_BATCH_SIZE` 8; sm_60 has no per-type table). Patch 0005/0006 `mmvh` replaces
  this range.
- **9 tokens and up:** `ggml_cuda_should_use_mmq` returns false for dense weights because
  `ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A` (MMQ is only kept for MoE on Pascal). So these go to
  `ggml_cuda_mul_mat_cublas`: the whole weight matrix is dequantised to an FP16 temporary, then multiplied with
  HGEMM.

For Q4_K (≈ 0.56 bytes per weight) that is about 0.56 read + 2 written + 2 read ≈ 4.6 bytes per weight, roughly 8×
the bytes of reading the weights once.

So the P100 likely has a cliff exactly where the V100 had its biggest gap, and the 9–16 range matters for
speculative decoding and parallel slots.

**Suggested:**
- Extend the day-one kit's `mmperf` from n = 1..8 to n = 1..16 so Phase 0 shows the cliff.
- If it shows, extend `mmvh` to 16 columns. The V100 data says staging the weights through shared memory, in their
  native layout, is enough (§3).

## 3. Native layout plus shared-memory staging was enough on the V100

- **Microbenchmark** (`volta-fa/bench/micro/qload.cu`): the GGUF layout staged through shared memory, with 128-byte
  row reads, reached 0.76–0.90× of a fully repacked fragment-order layout once decode and math were included.
  Naive native reads reached 0.29–0.64×.
- **What closing the rest would take.** The fork repacks weights in place at load time. After that every path has to
  read the new layout, and its activation prep is fused into model-specific producers. That is far too invasive to
  upstream.

**Lesson:** try staging before any repacking, and treat repacking as an owner decision.

## 4. Numerics: what transfers to `mmvh`

- **Activation range.** `mmvh`'s prepass (`x̂ = fp16(x/s)`, `s` = max |x| per 32, sum of x̂ in FP32) cannot
  overflow. Upstream's q8_1 can: `block_q8_1` stores d and the block sum of the raw values in FP16, so sums above
  65504 become inf at activations around 1e5, and d underflows around 1e-5.
  - volta-fa's range tests (`volta-fa/third_party/patches-local/0009-*.patch`, `test_mul_mat_range`: src1 × 1e5
    and × 1e-5, three magnitudes per token): the V100 current path passed 18/35, patch 0008 35/35.
  - **Suggested:** port them as a local-only P100 patch. They fail on unmodified upstream, so they must not go into
    an upstream test list.
  - Note: an infinite activation gives `s = inf` and `inf/inf = NaN` in `mmvh` (FP32 would give inf). Both are
    non-finite; just know which one the tests expect.
- **Power-of-two scale (optional).** volta-fa 0008 scales each slice by 2^k, with k computed from the exponent bits
  of the maximum (maximum in [2^14, 2^15), |k| ≤ 112, k = 0 for zero or non-finite slices). Multiplying by 2^k is
  exact and needs no division (`__fdiv_rn` in the prepass). This is minor for `mmvh`: the prepass is cheap and the
  FP16 rounding happens anyway.
- **Decode.** `mmvh` already uses the 0x6400 OR trick, including for Q6_K (subtract 1056). volta-fa's single-HFMA2
  Q6_K variant (codes at bits 4–9, then `x*(sc/16) + (-96*sc)`) only saves an instruction when the scale is applied
  in FP16. `mmvh` applies it at the FP32 flush, so there is nothing to change.
- **Accuracy method.** volta-fa added a development switch that routes quantised matmuls to dequantise + SGEMM (an
  FP32 reference). It then compared llama-perplexity KL divergence of each path against that reference: mean KLD
  0.0017 (upstream q8_1) vs 0.0001 (FP16 activations) on Qwen.
  - The P100 H1 Accept-if uses perplexity vs upstream (±0.02). That cannot show that `mmvh` is *more* accurate,
    which `qmv-numerics-2026-10-10.txt` predicts (7–34×).
  - **Suggested:** add the FP32-reference KLD to the day-one kit.

## 5. Measurement lessons from P-9

- **The benchmark keeps small matrices in cache.** test-backend-ops perf repeats one matrix, so a small one stays in
  L2. On the V100 (6 MB L2), N = 48 measured 16 µs there and a median of 23 µs inside the model. The P100 has 4 MB
  of L2: a 4096×1024 Q4_K matrix (≈ 2.4 MB) fits. Confirm small-matrix gains with nsys in the model.
- **Small ops jump 20–70 µs between runs** on both paths. Use the minimum of 3 or more interleaved rounds for them.
- **Thresholds differ per type and per N.** On the V100 the crossover against MMVQ was 5 tokens for Q4_K, 6 for
  Q5_K, 7 for Q6_K and 8 for IQ4_XS/Q8_0, and small matrices (N < 4096) lost. Measure each type separately before
  routing.
- **Query occupancy instead of assuming it.** Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` when choosing a K
  split. Short blocks in 2–3 waves cost the V100 kernel 10–20%.
- **Split-K reductions:** load all partials of a step together. One chain of dependent L2 loads per element cost
  10–15 µs per launch on the V100.
- **Find out which side you are waiting on.** Use development switches: weights from L2 only (clamp the row index),
  or loads only (no decode or math, with shared memory padded so the same number of blocks fit). On the V100 each
  took about 80% of the total time. The P100 cost model (`bin/decode-model.py`) predicts ALU-bound decode, and this
  experiment tests that directly.
- **Process:**
  - When the built kernel departs from its design doc, write it down at the time. volta-fa's activation scaling
    drifted silently and had to be restored.
  - Before releasing a rented machine, diff its results folders against the local copies. Four uncopied folders
    were found that way, one of them cited in a doc.
