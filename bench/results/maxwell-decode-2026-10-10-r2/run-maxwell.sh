#!/bin/bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH
export LD_LIBRARY_PATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
set -euo pipefail
cd /home/p51/Github/worktrees/p100-p7b-decode-maxwell
result_dir=bench/results/maxwell-decode-2026-10-10-r2
nvcc=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc

# Local source only, no patches, network, models or installs.
test "$(git -C third_party/llama.cpp rev-parse HEAD)" = "$(cat bench/LLAMA_COMMIT)"
test -z "$(git -C third_party/llama.cpp status --porcelain --untracked-files=no)"
cmake -S third_party/llama.cpp -B build/llama-sm52 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52 \
    -DCMAKE_CUDA_COMPILER="$nvcc" -DLLAMA_CURL=OFF > "$result_dir/llama-configure.log" 2>&1
cmake --build build/llama-sm52 --target test-backend-ops -j 4 > "$result_dir/llama-build.log" 2>&1
"$nvcc" --version > "$result_dir/nvcc-version.txt"
"$nvcc" -O2 -std=c++17 -arch=sm_52 bench/micro/devprops.cu -o build/devprops

# Hold ONE lock through the entire measurement session (all three rounds and
# their clock readings), so no worker can interleave a GPU workload.
flock /tmp/p100-gpu.lock bash -s -- "$result_dir" <<'ROUND'
export PATH=/home/p51/Github/P100/.venv/bin:$PATH
export LD_LIBRARY_PATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
set -euo pipefail
result_dir=$1
build/devprops > "$result_dir/devprops.json" 2> "$result_dir/devprops.stderr.txt"
nvidia-smi -q -d CLOCK > "$result_dir/nvidia-clocks.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version --format=csv > "$result_dir/gpu.csv"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > "$result_dir/clocks-before.csv"
# test-backend-ops -p is a regex search against vars(); these dimensions are
# present at bench/LLAMA_COMMIT. Batches 6/7 are absent from that upstream grid.
params='^type_a=(q4_0|q4_1|q5_0|q5_1|q8_0|q4_K|q5_K|q6_K|iq4_xs),type_b=f32,m=(4096|6144|6272|14336),n=[1-8],k=(4096|14336),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,src_overlap=0,m_v=0,pad=0$'
for round in 1 2 3; do
    if build/llama-sm52/bin/test-backend-ops perf -o MUL_MAT -b CUDA0 -p "$params" --output sql \
        > "$result_dir/round-$round.sql" 2> "$result_dir/round-$round.stderr.txt"; then
        status=0
    else
        status=$?
    fi
    printf '%s\n' "$status" > "$result_dir/round-$round.status"
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv \
        > "$result_dir/clocks-after-$round.csv" 2>&1
done
ROUND

rounds=("$result_dir/round-1.sql" "$result_dir/round-2.sql" "$result_dir/round-3.sql")
python bin/calibrate-decode.py --variant m2200 --device-props "$result_dir/devprops.json" \
    --median-sql "$result_dir/median.sql" --no-fit "${rounds[@]}" > "$result_dir/uncalibrated.txt"
python bin/calibrate-decode.py --variant m2200 --device-props "$result_dir/devprops.json" \
    "$result_dir/median.sql" > "$result_dir/calibrated-median.txt"
# The same median timing fit, retaining round spread on every row.
python bin/calibrate-decode.py --variant m2200 --device-props "$result_dir/devprops.json" \
    "${rounds[@]}" > "$result_dir/calibrated.txt"
for batch in 1 2 3 4 5 6 7 8; do
    python bin/decode-model.py --variant m2200 --device-props "$result_dir/devprops.json" \
        --shape 4096,14336 --all-types --batch "$batch" > "$result_dir/maxwell-prediction-batch$batch.txt"
done
