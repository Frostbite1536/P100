#!/usr/bin/env python3
"""Uncalibrated P100 decode ceilings from weight bytes and static SASS counts."""

import argparse
import csv
import json
import math
from pathlib import Path


# Bytes and weights per block, checked against llama.cpp in vecdot_sass.cu.
BLOCKS = {
    "q4_0": (18, 32), "q4_1": (20, 32), "q5_0": (22, 32),
    "q5_1": (24, 32), "q8_0": (34, 32), "q4_K": (144, 256),
    "q5_K": (176, 256), "q6_K": (210, 256), "iq4_xs": (136, 256),
    "f16": (2, 1),
}
VARIANTS = {"pcie16": (1303, 732), "pcie12": (1303, 549), "sxm2": (1480, 732)}
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "docs/research/vecdot-sass-2026-10-10.csv"
DISCLAIMER = ("datasheet peaks, uncalibrated; ignores activations, KV cache, "
              "attention and launch overhead")


def load_counts(path):
    """Reject missing, duplicate or inconsistent rows instead of guessing costs."""
    counts = {}
    with Path(path).open(newline="") as source:
        for row in csv.DictReader(source):
            typ, arch = row["type"], row["arch"]
            key = (typ, arch)
            if typ not in BLOCKS or typ == "f16" or arch not in ("sm_60", "sm_61"):
                raise ValueError(f"unexpected CSV type/arch: {key}")
            if key in counts:
                raise ValueError(f"duplicate CSV row: {key}")
            weights = int(row["weights_per_block"])
            total, alu, mem, ctrl = (int(row[name]) for name in
                                      ("sass_total", "sass_alu", "sass_mem", "sass_ctrl"))
            if weights != BLOCKS[typ][1] or min(total, alu, mem) <= 0 or ctrl < 0:
                raise ValueError(f"invalid CSV counts: {key}")
            if total != alu + mem + ctrl:
                raise ValueError(f"inconsistent CSV total: {key}")
            counts[key] = alu / weights
    expected = {(typ, arch) for typ in BLOCKS if typ != "f16" for arch in ("sm_60", "sm_61")}
    if counts.keys() != expected:
        raise ValueError(f"missing CSV rows: {sorted(expected - counts.keys())}")
    return counts


def llama_weights():
    layers, hidden, ffn = 32, 4096, 14336
    heads, kv_heads, head_dim, vocab = 32, 8, 128, 128256
    # q/o, k/v, gate/up/down per layer, plus the output head. No embeddings.
    attention = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    return layers * (attention + 3 * hidden * ffn) + hidden * vocab


def prediction(typ, weights, batch, variant, bw_eff, issue_eff, counts):
    clock_mhz, hbm_gbs = VARIANTS[variant]
    issue_rate = 56 * 64 * clock_mhz * 1e6
    block_bytes, block_weights = BLOCKS[typ]
    byte_count = weights * block_bytes / block_weights
    t_bw = byte_count / (hbm_gbs * 1e9 * bw_eff)
    row = {"type": typ, "bytes_per_weight": block_bytes / block_weights,
           "bytes_per_step": byte_count, "bytes_per_token": byte_count / batch,
           "t_bw_s": t_bw}
    for arch in ("sm_60", "sm_61"):
        alu_per_weight = counts.get((typ, arch))
        t_alu = (weights * alu_per_weight * batch / (issue_rate * issue_eff)
                 if alu_per_weight is not None else None)
        ceiling_time = max(t_bw, t_alu) if t_alu is not None else t_bw
        row[arch] = {"alu_per_weight": alu_per_weight, "t_alu_s": t_alu,
                     "bound": ("alu" if t_alu > t_bw else "bw") if t_alu is not None else "bw-only",
                     "t_step_s": ceiling_time, "tokens_per_s": batch / ceiling_time}
    return row


def efficiency(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise argparse.ArgumentTypeError("efficiency must be finite and in (0, 1]")
    return number


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("weights must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llama3.1-8b", "custom"), default="llama3.1-8b")
    parser.add_argument("--weights", type=positive, help="billions of mat-vec weights; requires --model custom")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--type", choices=BLOCKS, default="q8_0")
    selection.add_argument("--all-types", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="pcie16")
    parser.add_argument("--batch", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--bw-eff", type=efficiency, default=1.0)
    parser.add_argument("--issue-eff", type=efficiency, default=1.0)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", action="store_true", help="machine-readable predictions and assumptions")
    args = parser.parse_args()
    if args.model == "custom" and args.weights is None:
        parser.error("--model custom requires --weights <billions>")
    if args.model != "custom" and args.weights is not None:
        parser.error("--weights requires --model custom")
    weights = args.weights * 1e9 if args.model == "custom" else llama_weights()
    if not math.isfinite(weights):
        parser.error("weight count is too large")
    try:
        counts = load_counts(args.csv)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    types = BLOCKS if args.all_types else (args.type,)
    rows = [prediction(typ, weights, args.batch, args.variant, args.bw_eff, args.issue_eff, counts)
            for typ in types]
    assumptions = [DISCLAIMER,
                   "All mat-vec weights use the row's one type; embeddings excluded.",
                   "Weights read once per batch step; bytes/token = bytes/step / batch; tokens/s = batch / step time.",
                   'sm_61 counts use the P100 issue rate as a reference: "if dp4a existed".',
                   "f16: bandwidth-only ceiling; no measured ALU count."]
    if args.json:
        print(json.dumps({"assumptions": assumptions, "model": args.model, "weights": weights,
                          "variant": args.variant, "batch": args.batch, "bw_eff": args.bw_eff,
                          "issue_eff": args.issue_eff, "rows": rows}, allow_nan=False))
        return
    print("PREDICTIONS: " + DISCLAIMER)
    clock, hbm = VARIANTS[args.variant]
    print(f"model={args.model} weights={weights:,.0f} batch={args.batch} variant={args.variant} "
          f"HBM={hbm} GB/s issue={56 * 64 * clock / 1000:.6f} Glane-instructions/s "
          f"bw_eff={args.bw_eff:g} issue_eff={args.issue_eff:g}")
    for assumption in assumptions[1:]:
        print(assumption)
    print("Times in ms; step = max(bw, alu). GB uses decimal units.")
    print(f"{'type':7} {'GB/step':>9} {'GB/token':>9} {'bw_ms':>9} "
          f"{'alu60_ms':>10} {'bound60':>7} {'step60_ms':>10} {'tok60/s':>9} "
          f"{'alu61_ms':>10} {'bound61':>7} {'step61_ms':>10} {'tok61/s':>9}")
    for row in rows:
        line = (f"{row['type']:7} {row['bytes_per_step'] / 1e9:9.6f} "
                f"{row['bytes_per_token'] / 1e9:9.6f} {row['t_bw_s'] * 1000:9.6f}")
        for arch in ("sm_60", "sm_61"):
            result = row[arch]
            alu = f"{result['t_alu_s'] * 1000:.6f}" if result["t_alu_s"] is not None else "n/a"
            line += (f" {alu:>10} {result['bound']:>7} {result['t_step_s'] * 1000:10.6f} "
                     f"{result['tokens_per_s']:9.3f}")
        print(line)


if __name__ == "__main__":
    main()
