# System Invariants

Non-negotiable rules. A change that violates one is a bug or an explicit product decision, never an implementation detail. Changes to this file require a justification in the change log at the bottom. Inherited from volta-fa where the rule is hardware-independent; the ID is kept so the two repos can be compared.

## Architecture invariants

### INV-ARCH-001: sm_60 only
**Rule**: Device code uses only features available on compute capability 6.0 (GP100). FP16 arithmetic uses `half2` intrinsics (`__hfma2`, `__hmul2`, `__hadd2`, `__h2div` etc.), which run at 2× the FP32 rate on GP100. Integer dot products are written out (byte/half-word extracts, `__byte_perm`, `__vabsdiffu4`-class SIMD-in-word intrinsics that exist on sm_60), never `__dp4a`/`__dp2a`.
**Rationale**: The project exists for P100. sm_61+ intrinsics fail to compile for sm_60 or are emulated; Volta+ features do not exist.
**Valid**: `half2` math; `__shfl_*_sync`; `__ldg`; `atomicAdd` on `half2` (sm_60+); unified-memory hints.
**Invalid**: `__dp4a`/`__dp2a` (sm_61), `nvcuda::wmma` and every `mma.sync` (sm_70+), independent-thread-scheduling assumptions (sm_70+), BF16, FP8, `cp.async`, `ldmatrix`/`stmatrix`, `__reduce_*_sync` (sm_80), clusters, TMA, `wgmma`.
**Enforcement**: `bin/sm60-report.py` compiles at `compute_60` (rejects sm_61+ intrinsics); `bin/check-forbidden.sh` greps source and inline PTX; CI runs both.

### INV-ARCH-002: Per-block shared memory budget
**Rule**: Static + dynamic shared memory per block ≤ 48 KB, and the occupancy target per kernel is documented next to its launch config.
**Rationale**: GP100 has 64 KB shared memory per SM and no opt-in above 48 KB per block (llama.cpp's MMQ refuses devices under 48 KB per block for the same reason). Two 32 KB blocks per SM is the most a 64 KB SM holds.
**Enforcement**: `bin/sm60-report.py` fails above 48 KB; `--ptxas-options=-v` on the box.

### INV-ARCH-003: CUDA 12.x toolchain
**Rule**: Build with CUDA 12.8 and an R580-or-older driver branch. Never CUDA 13+.
**Rationale**: CUDA 13.0 removed offline compilation for sm_60 (12.8 already warns that pre-sm_75 targets are deprecated). R580 is the last driver branch for Pascal. 12.8 is also volta-fa's pin (it avoids llama.cpp #28416 there), so llama.cpp builds and ccache can be shared between the repos.
**Enforcement**: `bin/sm60-report.py` uses the pinned NVRTC 12.8 wheel; CMake refuses CUDA ≥ 13.

## Correctness invariants

### INV-NUM-001: No materialised score matrix
**Rule**: The fused path never writes the N×N scores/probabilities to global memory. Only O (and the log-sum-exp statistic if needed for split-KV) leave the SM.

### INV-NUM-002: Online softmax is numerically safe
**Rule**: Running max and sum are kept in FP32; fully masked rows produce O = 0 and LSE = -inf, never NaN; no intermediate overflows for logits up to the FP16 range.
**Enforcement**: `tests/` includes fully-masked-row, very-large-logit and all-equal-logit cases.

### INV-NUM-003: Reference parity
**Rule**: Output matches an FP32 reference within the documented tolerance for every supported shape (head dim, GQA ratio, causal/non-causal, seq lens including non-multiples of the tile).
**Enforcement**: `ctest`. Tolerances live in `include/p100/tolerance.h` (element-wise `|o-ref| <= 1e-3 + 2e-3*|ref|`, LSE `1e-3`), copied unchanged from volta-fa, and change only with a manager decision.

### INV-NUM-004: Causal alignment is bottom-right
**Rule**: With `causal`, query row `i` (0-based) may attend key `j` iff `j <= i + (seq_kv - seq_q)`.

### INV-NUM-005: QKᵀ accumulates in FP32; FP16 products need a budget
**Rule**: The S = QKᵀ sum is accumulated in FP32. Without tensor cores the products themselves are computed by the kernel: a kernel may form them in FP16 (`__hmul2`) or accumulate a short `half2` chain of at most N terms before adding to FP32 only if `tests/error_budget` models exactly that scheme (product rounding, chain length N, any pre-scaling of Q) with the outlier distribution and stays within INV-NUM-003's tolerance. N and the scaling are recorded in the kernel's design doc.
**Rationale**: volta-fa's CPU model showed FP16 *accumulation* reaching 0.46 absolute error with LLM-style outlier channels, against ~1e-3 with FP32. llama.cpp's TILE kernel on sm_60 already rounds each product to FP16 (`ggml_cuda_mad(float&, half2, half2)`: `__half22float2(v*u)`, with Q pre-scaled by 1/4 against overflow), a regime volta-fa never modelled. FP16 products are where the P100's 2× `half2` rate lives, so the trade-off must be measured, not assumed.
**Enforcement**: design review; `error_budget` entry per scheme; parity tests with the outlier distribution.

## Performance honesty invariants

### INV-PERF-001: No claim without an artifact
**Rule**: Every speedup/percentage in README, PRs or posts links to a committed file in `profiles/` or `bench/results/` with hardware (P100 PCIe 12/16 GB or SXM2), driver, CUDA version, commit and baseline commit recorded.

### INV-PERF-002: Compare against the real baseline
**Rule**: The baseline is llama.cpp's current CUDA path on cc 6.0 at a pinned commit (MMVQ with the emulated dp4a for quantised decode, dequantise + cuBLAS for dense prefill, the TILE/VEC FlashAttention kernels), with FA both on and off reported, not a naive implementation.
**Enforcement**: baseline commit pinned in `bench/BASELINE.md`.

### INV-PERF-003: No spills by default
**Rule**: Shipped kernels have 0 bytes spill stores/loads unless a documented exception exists.

### INV-PERF-004: Same-session baselines, interleaved rounds
**Rule**: A/B timing runs variants interleaved within one session (≥ 3 kernel rounds, ≥ 2 llama-bench rounds), reports medians with min–max, and includes the unmodified baseline and the f16-KV reference in the same session. Differences inside the measured spread are not claimed.
**Rationale**: volta-fa session 2 found that session 1's decode numbers were host-affected (28 vs 49 t/s for the same model and build class), and a first cool-GPU round reads high.

## P2P / multi-GPU invariants

### INV-P2P-001: Capability is checked, with a fallback
**Rule**: Use `cudaDeviceCanAccessPeer` before enabling P2P; fall back to staged host copies and log which path is active. Never assume the topology (PCIe switch vs root complex vs NVLink on SXM2 changes everything).
**Enforcement**: `tests/` covers the fallback path via a forced-disable env var; the peer test runs when two GPUs are present.

### INV-P2P-002: Streams and events, not device syncs
**Rule**: Communication overlaps compute through streams/events; no `cudaDeviceSynchronize()` in the hot path.

### INV-P2P-003: Buffer lifetime
**Rule**: Peer buffers are freed only after all consuming streams have passed an event recorded after the last use.

## Agent-process invariants

### INV-PROC-001: Workers do not decide
**Rule**: Workers implement and report; they do not decide ship/no-ship or loosen invariants/tolerances. Stop conditions are honoured.

### INV-PROC-002: "Not run" is a valid result
**Rule**: Anything not executed on real P100 hardware is labelled "not run", never inferred. The dev laptop's Maxwell GPU (sm_52) shares llama.cpp's no-dp4a and no-tensor-core code paths and is good for correctness only.

## Change log

| Date | Invariant | Change | Justification |
| --- | --- | --- | --- |
| 2026-10-10 | all | Initial version, adapted from volta-fa | Repo created for 2× P100 |
| 2026-10-10 | ARCH-001, ARCH-002 | sm_60 rules: no tensor cores, no dp4a, 48 KB smem per block | GP100 limits; llama.cpp source at 79e2e74 |
| 2026-10-10 | NUM-005 | FP32 accumulate kept; FP16 products allowed only with an error-budget entry | llama.cpp TILE on sm_60 rounds products to FP16 (fattn-tile.cuh, common.cuh `ggml_cuda_mad`) |
| 2026-10-10 | PERF-004 | New: same-session baselines and interleaved rounds | volta-fa session 2 host-variance finding |
