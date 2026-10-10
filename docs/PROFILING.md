# Profiling loop

Run on the P100 box. Commit **text summaries** to `profiles/`; raw reports stay local (`profiles/raw/`, gitignored).

## First: which profiler works on GP100

Nsight Compute support for Pascal in the CUDA 12.8 release (2025.1) is unverified. On day one:

```bash
ncu --list-chips | tr ',' '\n' | grep -i gp100 || echo "ncu: no GP100"
nsys --version
```

Record the answer in `bench/BASELINE.md`. Without ncu, kernel questions are answered with test-backend-ops timings,
SASS instruction counts (`cuobjdump -sass`) and `bin/sm60-report.py`, and the system view with nsys.

## System view (nsys)

```bash
nsys profile --stats=true -o profiles/raw/llama-decode \
  ./third_party/llama.cpp/build/bin/llama-cli -m model.gguf -p "..." -n 64 -ngl 99 -fa on
nvidia-smi topo -m
```

Look for: share of decode time in `mul_mat_vec_q` vs attention vs everything else; gaps between kernels; peer or
host copies in two-GPU runs.

## Kernel view (ncu, if it supports GP100)

| Question | Metric (check names with `ncu --query-metrics`) |
| --- | --- |
| Bandwidth-bound? | `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes.sum` |
| ALU-bound? (H1: emulated dp4a) | `sm__inst_executed.avg.per_cycle_active`, `smsp__issue_active.avg.pct_of_peak_sustained_active`, pipe utilisation |
| Occupancy limiter | `sm__warps_active.avg.pct_of_peak_sustained_active`, launch stats (regs/thread, smem/block) |
| Bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` with `-lineinfo` |
| Spills | `bin/sm60-report.py` or ptxas `-v`; at runtime local-memory sectors |

The quick ALU-vs-bandwidth test needs no profiler: achieved GB/s from `test-backend-ops perf -o MUL_MAT`
against the HBM2 roofline of the installed variant (732 or 549 GB/s).

## Feeding results back to a model

Paste only: kernel name, launch config, regs/thread, spill bytes, smem/block, occupancy, issue/pipe %, DRAM %,
top flagged source lines. Use `prompts/profile-feedback.md`.

## Rules

- Same input, same clocks (`nvidia-smi -lgc` if permitted), warm-up discarded, clocks and temperature logged per round.
- 3+ interleaved rounds, medians and spread (INV-PERF-004).
- Record GPU variant, driver, CUDA version, commit and baseline commit with every summary.
