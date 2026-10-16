#!/usr/bin/env python3
"""MUL_MAT median µs and effective decimal GB/s from SQL groups of rounds.

Usage: python bin/mm-gbps.py a.r1.sql,a.r2.sql b.r1.sql,b.r2.sql --labels a,b
Bytes count one quantised weight read, F32 activations and F32 output.
"""
import argparse
import math
import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
import analyze  # noqa: E402 (share SQL parsing and median-of-rounds semantics)

# Copied from ~/Github/volta-fa/bin/qmatmul-table.py: (bytes/block, elements/block).
BLOCKS = {
    "Q4_0": (18, 32), "Q4_1": (20, 32), "Q5_0": (22, 32),
    "Q5_1": (24, 32), "Q8_0": (34, 32), "Q2_K": (84, 256),
    "Q3_K": (110, 256), "Q4_K": (144, 256), "Q5_K": (176, 256),
    "Q6_K": (210, 256), "IQ4_NL": (18, 32), "IQ4_XS": (136, 256),
    "IQ3_S": (110, 256), "F16": (2, 1), "BF16": (2, 1),
}


def load(group):
    data = {}
    for (op, params), summary in analyze.load_tbo(group).items():
        if op != "MUL_MAT":
            continue
        p = dict(params)
        if (p.get("type_b") != "f32" or p.get("bs", "[1,1]") != "[1,1]"
                or p.get("nr", "[1,1]") != "[1,1]"
                or p.get("per", "[0,1,2,3]") != "[0,1,2,3]"
                or any(p.get(field, "0") != "0" for field in ("v", "k_v", "src_overlap", "m_v", "pad"))
                or p.get("o", "1") != "1"):
            raise ValueError(f"{group}: requires contiguous, unbatched F32 RHS/output: {p}")
        key = (p["type_a"].upper(), int(p["m"]), int(p["k"]), int(p["n"]))
        if min(key[1:]) <= 0:
            raise ValueError(f"{group}: nonpositive shape: {key}")
        if key in data:
            raise ValueError(f"{group}: ambiguous MUL_MAT cases for {key}")
        if summary[0] and (not math.isfinite(summary[1]) or summary[1] <= 0):
            raise ValueError(f"{group}: invalid timing for {key}")
        data[key] = summary
    if not data:
        raise ValueError(f"{group}: no MUL_MAT perf cases")
    return data


def byte_count(kind, m, k, n):
    if kind not in BLOCKS:
        return None
    size, elements = BLOCKS[kind]
    if k % elements:
        raise ValueError(f"{kind}: k={k} is not a multiple of block size {elements}")
    return m * k // elements * size + n * k * 4 + m * n * 4


def report(groups, labels, peak):
    data = [load(group) for group in groups]
    keys = set().union(*data)
    tokens = sorted({key[3] for key in keys})
    print(f"Roofline: {peak:g} GB/s (--peak-gbps); decimal GB/s, percentage of roofline.\n")
    print("Bytes = m·k weights (bytes/block ÷ elements/block) + n·k·4 activations + m·n·4 output.\n")
    headers = ["type", "m", "k", "label"]
    for n in tokens:
        headers.extend((f"n={n} µs", f"n={n} GB/s", f"n={n} % peak"))
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * 4 + ["---:"] * (3 * len(tokens))) + " |")
    for shape in sorted({key[:3] for key in keys}):
        for label, cases in zip(labels, data):
            cells = [*map(str, shape), label]
            for n in tokens:
                summary = cases.get((*shape, n))
                if summary is None or not summary[0]:
                    cells.extend(["not measured" if summary is None else "unsupported"] * 3)
                    continue
                time_us = summary[1]
                size = byte_count(*shape, n)
                cells.append(f"{time_us:.3f}")
                if size is None:
                    cells.extend(["unknown type"] * 2)
                else:
                    gbps = size / (time_us * 1000)
                    cells.extend((f"{gbps:.3f}", f"{100 * gbps / peak:.2f}%"))
            print("| " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+", metavar="SQL_GROUP")
    parser.add_argument("--labels", help="comma-separated labels, one per input group")
    parser.add_argument("--peak-gbps", type=float, default=732, help="roofline in decimal GB/s (default: 732)")
    args = parser.parse_args()
    labels = args.labels.split(",") if args.labels else [pathlib.Path(g.split(",")[0]).stem for g in args.groups]
    if len(labels) != len(args.groups):
        parser.error("--labels must have one label per input group")
    if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0:
        parser.error("--peak-gbps must be finite and positive")
    try:
        report(args.groups, labels, args.peak_gbps)
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
