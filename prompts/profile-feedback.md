# Profile feedback prompt

Paste into the manager (or into a ticket Background section).

```
Kernel: <name>, launch <grid>x<block>, dyn smem <bytes>
sm60-report: regs=<n> spill_stores=<n> spill_loads=<n> stack=<n> smem=<n>
ncu: achieved occupancy <x>% (theoretical <y>%, limiter: <regs|smem|warps>), fp32/fp16 pipe <x>%, issue-slot utilisation <x>%, DRAM <x>% of peak,
     bank conflicts <n> at <file:line ...>
Baseline: <commit> <metric>.

Constraints: Pascal sm_60; no tensor cores, no dp4a, smem <= 48 KB/block. Invariants: docs/INVARIANTS.md.
Task: propose ONE change targeting the largest measured limiter (e.g. smaller BR to cut registers, padding K tile rows by 8 halves to remove conflicts). State the expected effect on each metric above so it can be falsified by the next run.
```

One change per iteration, so the next profile attributes the effect.
