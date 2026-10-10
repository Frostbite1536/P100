# P100

Making llama.cpp faster on two NVIDIA Tesla P100s (Pascal GP100, sm_60): quantised decode without `dp4a`,
FlashAttention without tensor cores, and two-GPU splits over PCIe P2P. Changes are carried as a llama.cpp
patch series (`third_party/patches/`) and measured against upstream at a pinned commit.

Sibling of [volta-fa](../volta-fa) (V100) and [A770](../A770) (Intel Arc), with the same process: invariants,
narrow tickets, static reports before hardware, and "not run" for anything not measured on a P100.

Status: pre-hardware. The CPU numerics contract, the P2P layer and the analysis tools are in place; nothing has
run on a P100 yet, so there are no performance claims here.

- Why these targets: [docs/research/volta-fa-lessons.md](docs/research/volta-fa-lessons.md)
- Plan: [docs/ROADMAP.md](docs/ROADMAP.md) · Rules: [docs/INVARIANTS.md](docs/INVARIANTS.md) · Setup: [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md)

```bash
bin/setup-dev.sh && export PATH=$PWD/.venv/bin:$PATH
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
```
