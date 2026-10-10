# Review rules

Severity: **Critical** (blocks merge), **Nit** (fix if easy), **Pre-existing** (file separately). Review changed files in full, not just the diff.

## Checklist

- [ ] No device feature outside sm_60 (INV-ARCH-001): no tensor cores (`wmma`, any `mma.sync`), no `__dp4a`/`__dp2a`, no BF16/FP8/`cp.async`/`ldmatrix`.
- [ ] `bin/sm60-report.py` output attached: regs, spill = 0, smem ≤ 48 KB incl. dynamic (INV-ARCH-002, INV-PERF-003).
- [ ] Any FP16 product or `half2` chain in a QKᵀ/PV path has an `error_budget` row (INV-NUM-005).
- [ ] Parity tests cover tail tiles (N not a multiple of tile), fully masked rows, large logits, GQA ratios, `seq_q < seq_kv` causal (INV-NUM-002/003/004).
- [ ] Tolerances unchanged, or the change is a recorded manager decision.
- [ ] No N×N intermediate in global memory (INV-NUM-001).
- [ ] Out-of-bounds: every shared/global access guarded for partial tiles; `__syncthreads()` not inside divergent branches; warp shuffles use correct masks.
- [ ] Every launch and API call is error-checked.
- [ ] P2P: capability check + fallback, event-based sync, buffer lifetime (INV-P2P-*).
- [ ] Any performance claim links to an artifact with hardware/driver/CUDA/commit/baseline, from interleaved same-session rounds (INV-PERF-001/002/004).
- [ ] Edits to existing tests are justified.
- [ ] Docs updated when behaviour changes.
