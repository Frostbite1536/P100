# What volta-fa teaches about the P100 (2026-10-10)

Sources: volta-fa at `19c8b05` (two cloud V100 sessions, patch series 0001–0006), llama.cpp `79e2e74` read for
the sm_60 code paths, and one static compile below. Nothing here has run on a P100: every speed-up is a hypothesis
for Phase 0.

## 1. The hardware difference that matters

| | V100 PCIe 32 GB (volta-fa) | P100 PCIe 16 GB | P100 PCIe 12 GB | P100 SXM2 16 GB |
| --- | --- | --- | --- | --- |
| arch / cc | Volta / 7.0 | Pascal GP100 / 6.0 | same | same |
| SMs, boost | 80, 1380 MHz | 56, 1303 MHz | 56, 1303 MHz | 56, 1480 MHz |
| FP32 | 14 TFLOPS | 9.3 | 9.3 | 10.6 |
| FP16 | 112 TFLOPS (tensor cores) | 18.7 (`half2`, no tensor cores) | 18.7 | 21.2 |
| int8 dot (`dp4a`) | native | **none** (emulated) | none | none |
| HBM2 | 900 GB/s | 732 GB/s | 549 GB/s | 732 GB/s |
| smem per SM / per block | 96 / 96 KB | 64 / 48 KB | same | same |
| GPU link | PCIe 3.0 | PCIe 3.0 | PCIe 3.0 | NVLink 1.0 |

Datasheet values, not measured. **The owner's cards are the PCIe 16 GB variant** (732 GB/s, PCIe 3.0 between the cards, no NVLink); the other columns are for comparison only.

What it means for llama.cpp at `79e2e74` (traced in the source):

| work | V100 path | P100 path |
| --- | --- | --- |
| Quantised decode mat-vec (batch ≤ 8) | MMVQ, native `__dp4a` | MMVQ, **`__dp4a` emulated** (`common.cuh` `ggml_cuda_dp4a`, `__CUDA_ARCH__ < 610`) |
| Quantised prefill mat-mul, dense | MMQ (dp4a) for small batches, else cuBLAS | **MMQ disabled** below cc 6.1 except MoE (`mmq.cu`): dequantise to F16 + cuBLAS |
| FlashAttention, prefill | MMA m8n8k4 tensor cores (≥ 32 columns) | **TILE kernel** (no tensor cores anywhere) |
| FA decode, f16 K/V | VEC or TILE by GQA | VEC only when the GQA optimisation does not apply; with GQA and a mask → TILE |
| FA decode, quantised K/V | TILE + whole-cache conversion to F16 (volta-fa P-8's problem) | **VEC for ≤ 2 tokens** (no conversion); TILE + conversion from 3 tokens |
| FA arithmetic | tensor cores, FP32 accumulate | `half2` products rounded to FP16, Q pre-scaled by ¼, FP32 accumulate (`ggml_cuda_mad(float&, half2, half2)`) |

## 2. Static evidence: the dp4a emulation

llama.cpp's `ggml_cuda_dp4a` compiled for one q8_0 block (8 words = 32 int8 products), nvcc 12.8, `-O3`:

| target | SASS instructions | of which |
| --- | --- | --- |
| sm_61 (has dp4a) | 36 | 8 `IDP` |
| **sm_60 (P100)** | **120** | 48 `BFE`, 35 `XMAD`, 18 `SHR`, 6 `IADD` |

About 3.75 instructions per int8 product on the P100 against 0.25 with dp4a. Back of the envelope for
Llama-3.1-8B (8.0 G weights per token): dot-product ALU work ≈ 30 G thread-instructions ≈ 6.4 ms even at
full issue rate (56 SMs × 64 lanes × 1.303 GHz), against 11.6 ms (Q8_0, 8.5 GB) or 6.3 ms (Q4_0) to stream the
weights at 732 GB/s. Unpacking, scales and address math come on top, and `BFE`/`XMAD` may not issue at full
rate. So **4-bit decode on a P100 is plausibly ALU-bound**, and Q8_0 may be close; this applies to the owner's 16 GB
cards at 732 GB/s (the 12 GB card's 549 GB/s would shift the balance back). That is Phase 0 question 1:
`test-backend-ops perf -o MUL_MAT` at batch 1 gives achieved GB/s per type against the HBM roofline.

## 3. What transfers from volta-fa

1. **Patches volta-fa 0005/0006 (K/V-to-F16 conversion), as P100 `0001`/`0002`.** Architecture-independent,
   bit-exact, 1.7–6.9× faster conversion on the V100 and 2.6–5.2× on a Maxwell laptop GPU. On the P100 they
   matter wherever the TILE kernel converts a quantised cache: ≥ 3-token batches (speculative verify, chunked
   prefill) and prefill with a quantised KV cache. Not single-token decode, which already takes VEC on Pascal.
   Lowest-risk first patch; the microbenchmark `bench/micro/dequant_nc.cu` builds for sm_60 as is.
2. **Padded GQA heads per tile (volta-fa P-6, A770 p6b).** The TILE kernel's `ncols2` uses only power-of-two
   divisors of the GQA ratio (GQA 6 → 2), so non-power-of-two GQA models pay K/V re-reads at small batches.
   The same padding idea applies to CUDA TILE on the P100, but the V100 data says padding *lost* (0.69–0.91×)
   wherever upstream's tile was already good (D=128, GQA 3/6, 8–16 tokens). Measure per (D, GQA, batch) and
   route only the winners.
3. **The crash class from volta-fa 0001**: kernel selection and dispatch computing the GQA tile differently,
   so a never-compiled instance is launched. Run the full `test-backend-ops -o FLASH_ATTN_EXT` on day one
   (~3 min on a V100); any abort is a candidate upstream report.
4. **Measurement discipline**: interleaved rounds, medians with ranges, clocks and temperature per round,
   the f16-KV reference in the same session (INV-PERF-004), FA on and off.
5. **Model accounting**: Qwen3.8-27B runs FlashAttention in ~16 of its 65 layers, so attention gains shrink
   ×4 end to end on it. Predict end-to-end from kernel savings × FA layers before claiming anything.
6. **The cost-model + static-report loop.** `sm70-report` becomes `sm60-report`; `plan-tiles` gets GP100
   limits. Expect models to be wrong on day one and say where (volta-fa: q4_0 VEC was not bandwidth-bound;
   padding cost at D=128 was mispredicted).
7. **The P2P layer.** volta-fa wrote the peer/staged copy code, bandwidth bench and tests, but the peer path
   never ran (one-GPU instances). The P100 pair is its first real test; results feed back to volta-fa.
8. **Process**: the invariants, tickets, worker loop, "not run", and llama.cpp's contribution rules (the
   owner writes upstream text, AI use disclosed).
9. **Toolchain**: CUDA 12.8 + R580 are already volta-fa's pins and still support sm_60. The dev laptop's
   Maxwell GPU (sm_52) runs llama.cpp's same no-dp4a, no-tensor-core code paths, so it can check correctness
   of MMVQ/TILE changes (not P100 timing).

## 4. What does not transfer

Everything tensor-core: m8n8k4/WMMA kernels, volta-fa 0002–0004 (MMA padding, D=256 MMA configs, the Volta VEC
routing), PR #27997 configs, the superfast-v100 fork, and their numbers. Volta's ITS-based warp behaviour is also
absent: Pascal warps run in lockstep.

## 5. New, P100-specific hypotheses (ranked only after Phase 0)

| # | hypothesis | why it might pay | first measurement |
| --- | --- | --- | --- |
| H1 | Quantised decode mat-vec is ALU-bound by the dp4a emulation (§2) | decode is most users' bottleneck; a cheaper int8 or `half2` dot could approach the HBM roofline | MUL_MAT batch 1–8 GB/s per type vs 732/549 GB/s |
| H2 | FA TILE spends 5 instructions per 2 multiply-adds (HMUL2, 2 conversions, 2 FADD) | short `half2` FMA chains would cut that ~2×, if the error budget allows (INV-NUM-005) | FA perf by head dim; `error_budget` with product rounding + chain length |
| H3 | Dense prefill = dequantise + cuBLAS HGEMM; FA TILE may lose to FA off at some shapes | picks the right default before writing kernels | llama-bench FA on/off × depth |
| H4 | Two P100s: `-sm layer` vs `row` vs `tensor`, P2P vs staged | 27B-class models need both cards | `p2p_bandwidth`, `nvidia-smi topo -m`, llama-bench split modes |
| H5 | Conversion patches (§3.1) | known-good, small | `dequant_nc` + quantised-KV pp8 |

## 6. Not known yet

The host's PCIe topology (same root complex? P2P allowed?), whether
Nsight Compute 2025.1 still profiles GP100 (`ncu --list-chips`) or only nsys/nvprof are left, and every number
in §5.
