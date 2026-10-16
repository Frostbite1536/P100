# Baseline record

Fill in on the first day with the P100s. Every result file links back here (INV-PERF-001).

| Field | Value |
| --- | --- |
| Where | (home box / cloud provider + instance type) |
| GPUs | 2× Tesla P100 **PCIe 16 GB** (ordered 2026-10-10; datasheet: 56 SMs, 1303 MHz boost, 732 GB/s HBM2, 250 W, no NVLink). Record VBIOS, power limit and measured clocks on arrival |
| Topology | `nvidia-smi topo -m` (PIX / PXB / PHB / SYS / NV#), PCIe gen and width per card |
| Host | CPU, RAM, board, BIOS settings (Above 4G, ACS, IOMMU) |
| Driver | (R580.x) |
| CUDA | (12.8.x, `nvcc --version`) |
| Profilers | `ncu --list-chips` has GP100? nsys version |
| llama.cpp | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (`bench/LLAMA_COMMIT`, shared with volta-fa) |
| Models | (`bench/models.txt`, never committed; record file names and sizes here) |

**Day-one A/B.** Run `MODELS="/models/Llama-3.1-8B-Q4_K_M.gguf" PPL_TEXT=/data/wikitext-2-test.txt bench/patches/run.sh` (defaults: upstream, conv8, flush, mmvh; 3 kernel rounds, 2 llama-bench rounds × 3 reps). Planning estimates from case counts, **not P100 measurements**: micro seconds (one 50-iteration run); build minutes to tens of minutes (four trees); correctness minutes to hours (four full FA/biased-V/MUL_MAT suites plus one disabled-MMVH MUL_MAT suite); FA perf minutes to tens of minutes (1,728 selected cases); mmperf tens of minutes or longer (320 type/shape/width cases × five arms × three rounds); llama-bench minutes to tens of minutes per model (30 invocations: nine kv configurations or one of two tg128 depths); PPL tens of minutes to hours per model (five arms × 32 chunks). Actual duration depends on the host, shapes and model. `flush.fa-vbias-test-CUDA0.txt` answers biased-V correctness; `*.perf.r*.sql` and `*.kv.r*.jsonl`/`*.tg128-d*.r*.jsonl` record FA and pp/tg cost. `*.mulmat-test-CUDA0.txt`, `*.mmperf.r*.sql`, `*.tg128-d*.r*.jsonl` and `*.ppl.txt` cover H1 correctness, n=1..16 timing, tg128 and final PPL ± error; `mmvh-nommvh.*` is the same-build switch baseline. `summary.md` labels the comparisons and prints each final PPL line.

`mmperf` covers Q4_0/Q8_0/Q4_K/Q5_K/Q6_K at n=1–16 for (m,k)=(4096,4096), (1024,4096), (14336,4096), (4096,14336); 9–16 uses upstream’s dequantise-to-FP16 + cuBLAS path on sm_60, and `summary.md` reports effective GB/s against 732 GB/s ([research §§2,5](../docs/research/volta-fa-p9-lessons-2026-10-11.md)).
`STEPS="build kld"` (then `STEPS=kld` to reuse builds), with explicit `MODELS` and `PPL_TEXT`, compares each arm against an FP32 reference rather than upstream; `KLD_UB=8` keeps n ≤ 8 in MMVH (n ≥ 9 routes to cuBLAS on sm_60), `KLD_CHUNKS=8` by default, and `summary.md` preserves Mean/Maximum KLD, RMS Δp and Same top p lines.
`*.range.txt` answers whether each arm preserves activation range at 1e5 and 1e-5 for n=1/2/4/8/16 (Q4_0/Q8_0 skip 1e-5 because their CPU reference underflows); upstream q8_1 failures are expected by design, not a regression signal.
