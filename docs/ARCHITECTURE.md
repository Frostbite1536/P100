# Architecture

## Components

```
            llama.cpp (third_party/, pinned commit) ── changes kept as third_party/patches/*.patch
                  │  ggml CUDA backend
                  ▼
        ┌─────────────────────┐        ┌──────────────────────┐
        │ include/p100/       │        │ src/p2p/             │
        │ shared contracts    │        │ peer access, copies, │
        └─────────┬───────────┘        │ staged fallback      │
                  ▼                    └──────────┬───────────┘
        ┌─────────────────────┐                   │
        │ src/<kernel family>/│◄── split / TP ────┘
        │ sm_60 kernels       │
        └─────────────────────┘
```

| Component | Responsibility | Interface |
| --- | --- | --- |
| `include/p100/attention.h` | Tensor layout, mask/causal semantics, reference API, tiled-model config | header |
| `include/p100/tolerance.h` | Parity tolerances (INV-NUM-003) | header |
| `src/reference/` | FP32 attention reference and the tiled numerics model (CPU) | `p100::attention_reference`, `p100::attention_tiled_model` |
| `src/p2p/` | Peer-access probe, async peer copies, host-staged fallback, forced-disable switch | `p100::p2p::probe`, `describe`, `path_name` (`include/p100/p2p.h`) |
| `tests/` | Reference/tiled-model tests, error budget, FP16 helpers, P2P tests, analysis-tool fixtures | ctest |
| `bench/` | `p2p_bandwidth`, conversion microbenchmark, `analyze.py` (test-backend-ops SQL and llama-bench JSONL tables), Phase 0 kit | scripts |
| `third_party/patches/` | llama.cpp patch series, numbered for this repo (provenance in each patch header) | `git apply` |

Kernel families (`src/matvec/`, `src/attention/`) are created by Phase 1 design docs, not before.

## Key design decisions

| Decision | Status | Evidence |
| --- | --- | --- |
| Work in llama.cpp patches vs standalone kernels | patches first (as volta-fa); standalone only for kernels llama.cpp cannot host | volta-fa P-6/P-8 |
| QKᵀ accumulation | **FP32** (INV-NUM-005); FP16 products only with an error-budget row | volta-fa error budget; P-3 |
| Decode int8 dot without dp4a | open (H1): emulated int8 vs `half2` dequant-dot vs SIMD-in-word tricks | Phase 0 MUL_MAT GB/s, P-4 model |
| Two-GPU split mode | open (H4) | Phase 0 step 5 |

## Failure modes and limits

- No P100 → CPU tests, static sm_60 reports and Maxwell-laptop correctness only; hardware results are "not run".
- PCIe topology decides P2P (same root complex vs across sockets; ACS/IOMMU). Measure before designing overlap.
- 48 KB smem per block and 64 KB per SM bound tile sizes harder than on Volta; register pressure vs occupancy
  is the main cliff (docs/PROFILING.md).
