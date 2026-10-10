# Baseline record

Fill in on the first day with the P100s. Every result file links back here (INV-PERF-001).

| Field | Value |
| --- | --- |
| Where | (home box / cloud provider + instance type) |
| GPUs | 2× Tesla P100 — variant: (PCIe 12 GB / PCIe 16 GB / SXM2 16 GB), VBIOS, power limit |
| Topology | `nvidia-smi topo -m` (PIX / PXB / PHB / SYS / NV#), PCIe gen and width per card |
| Host | CPU, RAM, board, BIOS settings (Above 4G, ACS, IOMMU) |
| Driver | (R580.x) |
| CUDA | (12.8.x, `nvcc --version`) |
| Profilers | `ncu --list-chips` has GP100? nsys version |
| llama.cpp | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (`bench/LLAMA_COMMIT`, shared with volta-fa) |
| Models | (`bench/models.txt`, never committed; record file names and sizes here) |
