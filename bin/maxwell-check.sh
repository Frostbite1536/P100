#!/bin/bash
# Documented Maxwell indexing/layout check; CMake remains sm_60-only.
# From this worktree: cmake --build build && bin/maxwell-check.sh
set -euo pipefail
export PATH=/home/p51/Github/P100/.venv/bin:$PATH
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc
reference="$root/build/src/reference/libp100_ref.a"
output="$root/build/tests/h2_mmv_gpu_maxwell"
if [ "$#" -ne 0 ]; then
    echo "usage: $0" >&2
    exit 2
fi
if [ ! -f "$reference" ]; then
    echo "missing $reference; run cmake --build build first" >&2
    exit 1
fi
"$nvcc" -O3 -std=c++17 \
    -gencode arch=compute_52,code=sm_52 \
    -gencode arch=compute_60,code=sm_60 \
    -I "$root/include" "$root/tests/h2_mmv_gpu.cu" "$reference" -o "$output"
echo "built $output (sm_52;sm_60)"
flock /tmp/p100-gpu.lock "$output"
