# Toolchain

## Version pins (INV-ARCH-003)

| Piece | Pin | Why |
| --- | --- | --- |
| CUDA toolkit | 12.8.x | CUDA 13.0 dropped sm_60; 12.8 warns that pre-sm_75 targets are deprecated but builds them. Same pin as volta-fa, so llama.cpp builds and ccache can be shared. |
| Driver | R580 branch | Last branch supporting Pascal. |
| Profilers | ships with 12.8 | Nsight Compute support for GP100 is **unverified**: check `ncu --list-chips` on the box. nsys timelines work. |
| llama.cpp | `bench/LLAMA_COMMIT` (`79e2e74`, volta-fa's pin) | INV-PERF-002; one base for both repos' patch series |

## Dev machine (no P100)

No sudo, no system CUDA. Everything lives in the gitignored `.venv/` and `tools/`, installed by one script (pinned, checksum-verified downloads):

```bash
bin/setup-dev.sh                 # .venv: cmake, ninja, shellcheck, NVRTC/ptxas/cudart wheels (CUDA 12.8)
WITH_NVCC=1 bin/setup-dev.sh     # + tools/cuda-12.8: full nvcc 12.8, cudart, cuBLAS headers from conda-forge (~2.4 GB)
```

volta-fa's `tools/cuda-12.8` is the same install; a symlink (`ln -s ~/Github/volta-fa/tools/cuda-12.8 tools/cuda-12.8`) avoids a second download.

What works here:

- `export PATH=$PWD/.venv/bin:$PATH`, then the CPU build + tests: `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build`.
- **Static sm_60 reports**: `python bin/sm60-report.py <file.cu>` (NVRTC + ptxas: registers, spills, stack, smem).
- **SASS inspection**: `tools/cuda-12.8/bin/nvcc -arch=sm_60 -cubin` + `cuobjdump -sass` (how `docs/research/volta-fa-lessons.md` §2 was measured).
- **Host-only CUDA code (P2P)** builds against the pip `libcudart` and runs its staged path on the laptop GPU; the peer test skips with one GPU.
- **llama.cpp correctness on the laptop's Quadro M2200 (sm_52)**: same no-dp4a MMVQ fallback and no-tensor-core FA TILE/VEC paths as the P100 (but no fast FP16: Maxwell takes the float branch). Correctness only.

What does not work here: running sm_60 code, timing, profiler metrics, the peer (2-GPU) P2P path.

## P100 box (when it arrives)

1. Driver R580, CUDA 12.8 toolkit (full, with `nvcc` and nsys).
2. `cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=60` builds the CUDA targets too.
3. Record `nvidia-smi -q`, `nvidia-smi topo -m`, `nvcc --version`, `ncu --list-chips | grep -i gp100` in `bench/BASELINE.md`.
4. P2P: BIOS Above 4G decoding; check ACS/IOMMU (`iommu=pt`); both cards on the same root complex if the board allows. Then `build/bench/p2p_bandwidth --csv` and cuda-samples `p2pBandwidthLatencyTest`.
5. Profiling without root if needed: `NVreg_RestrictProfilingToAdminUsers=0`.
