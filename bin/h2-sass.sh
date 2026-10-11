#!/bin/bash
# Compile only. Source the classifier expressions from vecdot-sass.sh's Python
# heredoc (via AST), so control/memory categorisation has one maintained source.
set -euo pipefail
export PATH=/home/p51/Github/P100/.venv/bin:$PATH
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc=${P100_NVCC:-/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc}
output=${1:-$root/docs/research/h2-sass-2026-10-10.csv}
if [ "$#" -gt 1 ]; then
    echo "usage: $0 [output.csv]" >&2
    exit 2
fi
mkdir -p "$root/build"
work=$(mktemp -d "$root/build/h2-sass.XXXXXX")
trap 'rm -rf "$work"' EXIT
"$nvcc" -O3 -std=c++17 -cubin -arch=sm_60 -I "$root/include" \
    "$root/bench/micro/h2_mmv.cu" -o "$work/sm_60.cubin"
"$(dirname "$nvcc")/cuobjdump" -sass "$work/sm_60.cubin" > "$work/sm_60.sass"
# Keep the disassembly in build/ for instruction evidence in the ticket report.
cp "$work/sm_60.sass" "$root/build/h2-sm60.sass"
"${PYTHON:-python}" - "$root/bin/vecdot-sass.sh" "$work/sm_60.sass" "$output" <<'PY'
import ast
import collections
import csv
import pathlib
import re
import sys

original = pathlib.Path(sys.argv[1]).read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
tree = ast.parse(original)
# Reuse, without copying or changing, the original classifier's definitions and
# total/mem/ctrl/alu expressions. Call that classifier for each new fixture.
def assignment(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
nodes = []
for name in ('control', 'instruction', 'total', 'mem', 'ctrl', 'alu'):
    matches = [n for n in ast.walk(tree) if assignment(n, name)]
    if len(matches) != 1:
        raise SystemExit(f'original classifier changed: {name}')
    nodes.append(matches[0])
classifier = compile(ast.Module(body=nodes, type_ignores=[]), str(sys.argv[1]), 'exec')
def classify(ops):
    scope = dict(re=re, ops=ops)
    exec(classifier, scope)
    return tuple(scope[n] for n in ('total', 'alu', 'mem', 'ctrl')), scope['instruction']
_, instruction = classify(collections.Counter())
counts = {}
current = None
for line in pathlib.Path(sys.argv[2]).read_text().splitlines():
    match = re.search(r'Function\s*:\s*(\S+)', line)
    if match:
        function = re.fullmatch(r'h2_fixture_(q4_0|q8_0|q4_K)_b(1|2|4|8)', match[1])
        current = (function[1], int(function[2])) if function else None
        if current:
            if current in counts:
                raise SystemExit(f'duplicate fixture: {current}')
            counts[current] = collections.Counter()
    if current:
        match = instruction.match(line)
        if match:
            counts[current][match[1]] += 1
        elif re.match(r'^\s*/\*[0-9a-fA-F]+\*/', line):
            raise SystemExit(f'unparsed instruction: {line}')
expected = {(t, b) for t in ('q4_0', 'q8_0', 'q4_K') for b in (1, 2, 4, 8)}
if counts.keys() != expected:
    raise SystemExit(f'missing fixtures: {expected - counts.keys()}')
with open(sys.argv[3], 'w', newline='') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('type', 'arch', 'batch', 'weights_per_block', 'sass_total', 'sass_alu',
                     'sass_mem', 'sass_ctrl', 'alu_per_weight_per_column', 'top_opcodes'))
    for typ in ('q4_0', 'q8_0', 'q4_K'):
        for batch in (1, 2, 4, 8):
            ops = counts[typ, batch]
            (total, alu, mem, ctrl), _ = classify(ops)
            weights = 256 if typ == 'q4_K' else 32
            if min(total, alu, mem) <= 0:
                raise SystemExit(f'incomplete counts: {typ} b{batch}')
            if ops['HFMA2'] != weights//2*batch:
                raise SystemExit(f'wrong HFMA2 coverage: {typ} b{batch}')
            if typ != 'q8_0' and ops['LOP3'] < weights//2:
                raise SystemExit(f'nibble unpack did not use LOP3: {typ} b{batch}')
            if typ == 'q8_0' and ops['PRMT'] < weights//2:
                raise SystemExit(f'byte unpack did not use PRMT: {typ} b{batch}')
            # The requested opcodes always appear individually, including zeros.
            top = sorted(ops, key=lambda op: (-ops[op], op))[:8]
            top += [op for op in ('HFMA2', 'LOP3', 'PRMT', 'HADD2', 'F2F') if op not in top]
            writer.writerow((typ, 'sm_60', batch, weights, total, alu, mem, ctrl,
                             f'{alu/(weights*batch):.8f}', ';'.join(f'{op}:{ops[op]}' for op in top)))
print(f'wrote {len(counts)} rows to {sys.argv[3]}')
PY
