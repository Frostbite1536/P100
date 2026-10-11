#!/bin/bash
# Run the patched 52;60 binary on Maxwell. No model files are needed.
set -euo pipefail
export PATH=/home/p51/Github/P100/.venv/bin:$PATH
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if (( $# > 1 )); then
    echo "usage: $0 [path/to/test-backend-ops]" >&2
    exit 2
fi
tbo=${1:-"$root/third_party/llama.cpp/build-mmvh/bin/test-backend-ops"}
if [[ ! -x "$tbo" ]]; then
    echo "missing executable: $tbo" >&2
    exit 2
fi
# All q4_0/q8_0/q4_K MUL_MAT cases with F32 input and n=1..8, including strided controls.
params='type_a=(q4_0|q8_0|q4_K),type_b=f32,.*n=[1-8],k='
exec 9>/tmp/p100-gpu.lock
flock 9
gpu_index=${MMVH_MAXWELL_GPU:-0}
cc=$(nvidia-smi -i "$gpu_index" --query-gpu=compute_cap --format=csv,noheader)
if [[ "$cc" != 5.2 ]]; then
    echo "expected Maxwell sm_52 at GPU $gpu_index, got $cc" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_index"
out_base="$root/bench/results/mmvh-maxwell-$(date +%F)"
mkdir -p "$out_base"
out=$(mktemp -d "$out_base/run-XXXXXX")
label='Maxwell, FP32 emulation of half2: not a P100 number'
printf '%s\nBinary: %s\nFilter: %s\n' "$label" "$tbo" "$params" | tee "$out/run.txt"
nvidia-smi -i "$gpu_index" --query-gpu=name,compute_cap,driver_version --format=csv,noheader >> "$out/run.txt"
failed=0
run_case() {
    local name=$1 mode=$2 forced=$3 restricted=$4 rc=0 counts
    local -a command=(env -u GGML_CUDA_DISABLE_MMVH -u GGML_CUDA_FORCE_MMVH)
    if [[ "$forced" == 1 ]]; then
        command+=(GGML_CUDA_FORCE_MMVH=1)
    fi
    command+=("$tbo" "$mode" -o MUL_MAT -b CUDA0)
    if [[ "$restricted" == 1 ]]; then
        command+=(-p "$params")
    fi
    {
        printf '%s\n' "$label"
        printf 'command:'
        printf ' %q' "${command[@]}"
        printf '\n'
    } > "$out/$name.txt"
    "${command[@]}" >> "$out/$name.txt" 2>&1 || rc=$?
    if [[ "$mode" == test ]]; then
        # Count only individual MUL_MAT result lines; exclude backend summaries and failure lists.
        counts=$(sed -E 's/\x1B\[[0-9;]*[mK]//g' "$out/$name.txt" | awk '
            /^  MUL_MAT\(/ {
                if ($0 ~ /OK$/) pass++;
                else if ($0 ~ /FAIL$/) fail++;
                else if ($0 ~ /not supported/) unsupported++;
            }
            END { printf "pass=%d fail=%d unsupported=%d", pass, fail, unsupported;
                  if (pass + fail == 0 || fail > 0) exit 1; }') || failed=1
        # A missing CUDA0 backend can exit zero; require its success status as well.
        if ! sed -E 's/\x1B\[[0-9;]*[mK]//g' "$out/$name.txt" | grep -qE '^  Backend CUDA0: OK$'; then
            failed=1
        fi
        printf '%s: exit=%d %s\n' "$name" "$rc" "$counts" | tee -a "$out/run.txt"
    else
        if ! grep -q ' us/run - ' "$out/$name.txt"; then
            failed=1
        fi
        printf '%s: exit=%d (%s)\n' "$name" "$rc" "$label" | tee -a "$out/run.txt"
    fi
    if (( rc != 0 )); then
        failed=1
    fi
}
run_case upstream-test test 0 0
run_case force-test test 1 0
run_case force-targeted-test test 1 1
run_case upstream-targeted-perf perf 0 1
run_case force-targeted-perf perf 1 1
printf 'Saved outputs: %s\n' "$out"
exit "$failed"
