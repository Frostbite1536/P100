#!/bin/bash
# Compile only: no CUDA kernels are executed. Counts include wrapper overhead.
set -euo pipefail
export PATH=/home/p51/Github/P100/.venv/bin:$PATH
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc=${P100_NVCC:-/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc}
cuobjdump=$(dirname "$nvcc")/cuobjdump
llama=/home/p51/Github/P100/third_party/llama.cpp
output=${1:-$root/docs/research/vecdot-sass-2026-10-10.csv}
if [ "$#" -gt 1 ]; then
    echo "usage: $0 [output.csv]" >&2
    exit 2
fi
mkdir -p "$root/build"
work=$(mktemp -d "$root/build/vecdot-sass.XXXXXX")
trap 'rm -rf "$work"' EXIT
for arch in sm_60 sm_61; do
    "$nvcc" -O3 -std=c++17 -cubin -arch="$arch" \
        -I "$llama/ggml/include" -I "$llama/ggml/src" \
        -I "$llama/ggml/src/ggml-cuda" \
        "$root/bench/micro/vecdot_sass.cu" -o "$work/$arch.cubin"
    "$cuobjdump" -sass "$work/$arch.cubin" > "$work/$arch.sass"
done
"${PYTHON:-python}" - "$work" "$output" <<'PY'
import collections
import csv
import pathlib
import re
import sys

types = dict(q4_0=32, q4_1=32, q5_0=32, q5_1=32, q8_0=32,
             q4_K=256, q5_K=256, q6_K=256, iq4_xs=256)
control = set("S2R EXIT BRA RET NOP BAR SYNC SSY PBK BRK CAL".split())
instruction = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9]*)(?:\.[A-Z0-9.]+)?\s*")
rows = []
for arch in ("sm_60", "sm_61"):
    counts = {}
    current = None
    for line in (pathlib.Path(sys.argv[1]) / (arch + ".sass")).read_text().splitlines():
        function = re.search(r"Function\s*:\s*(\S+)", line)
        if function:
            name = function.group(1)
            if not name.startswith("vecdot_") or name[7:] not in types:
                raise SystemExit(f"unexpected SASS function: {name}")
            current = name[7:]
            if current in counts:
                raise SystemExit(f"duplicate SASS function: {name}")
            counts[current] = collections.Counter()
        match = instruction.match(line)
        if match:
            if current is None:
                raise SystemExit("instruction outside a function")
            counts[current][match.group(1)] += 1
        elif re.match(r"^\s*/\*[0-9a-fA-F]+\*/", line):
            raise SystemExit(f"unparsed instruction: {line}")
    if counts.keys() != types.keys():
        raise SystemExit(f"missing SASS functions for {arch}: {types.keys() - counts.keys()}")
    for typ, weights in types.items():
        ops = counts[typ]
        total = sum(ops.values())
        mem = sum(n for op, n in ops.items() if op.startswith(("LD", "ST")))
        ctrl = sum(n for op, n in ops.items() if op in control)
        alu = total - mem - ctrl
        if min(total, mem, alu) <= 0:
            raise SystemExit(f"empty or incomplete counts for {typ}, {arch}")
        top = ";".join(f"{op}:{n}" for op, n in sorted(ops.items(), key=lambda x: (-x[1], x[0]))[:8])
        rows.append((typ, arch, weights, total, alu, mem, ctrl, top))
with open(sys.argv[2], "w", newline="") as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("type", "arch", "weights_per_block", "sass_total", "sass_alu", "sass_mem", "sass_ctrl", "top_opcodes"))
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {sys.argv[2]}")
PY
