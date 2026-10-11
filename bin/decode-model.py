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
VARIANTS = {"pcie16": (1303, 732), "pcie12": (1303, 549), "sxm2": (1480, 732),
            "m2200": None}
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "docs/research/vecdot-sass-2026-10-10.csv"
DEFAULT_DEVICE_PROPS = Path(__file__).resolve().parents[1] / "bench/results/maxwell-decode-2026-10-10/devprops.json"
DISCLAIMER = ("datasheet peaks, uncalibrated; ignores activations, KV cache, "
              "attention and launch overhead")


def load_counts(path):
    """Reject missing, duplicate or inconsistent rows instead of guessing costs."""
    counts = {}
    with Path(path).open(newline="") as source:
        for row in csv.DictReader(source):
            typ, arch = row["type"], row["arch"]
            key = (typ, arch)
            if typ not in BLOCKS or typ == "f16" or arch not in ("sm_60", "sm_61", "sm_52"):
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
    arches = ("sm_60", "sm_61", "sm_52") if any(arch == "sm_52" for _, arch in counts) else ("sm_60", "sm_61")
    expected = {(typ, arch) for typ in BLOCKS if typ != "f16" for arch in arches}
    if counts.keys() != expected:
        raise ValueError(f"missing CSV rows: {sorted(expected - counts.keys())}")
    return counts


def llama_weights():
    layers, hidden, ffn = 32, 4096, 14336
    heads, kv_heads, head_dim, vocab = 32, 8, 128, 128256
    # q/o, k/v, gate/up/down per layer, plus the output head. No embeddings.
    attention = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    return layers * (attention + 3 * hidden * ffn) + hidden * vocab


def variant_arch(variant):
    return "sm_52" if variant == "m2200" else "sm_60"


def load_device_props(path):
    # Maxwell M2200 values must come from this machine, never a datasheet.
    # bench/micro/devprops.cu records multiProcessorCount, clockRate (max SM
    # kHz), memoryClockRate (kHz) and memoryBusWidth via cudaGetDeviceProperties.
    # Cross-check clock rates with nvidia-smi -q -d CLOCK saved beside the JSON.
    # sm_52 has 128 lanes/SM (ticket's architectural assumption).
    # GDDR theoretical GB/s = bus_bits/8 * 2 * memory_clock_khz * 1e3 / 1e9.
    # The sandbox probe failed; no measured device constants are hardcoded.
    try:
        props = json.loads(Path(path).read_text())
        if (props["major"], props["minor"]) != (5, 2) or "M2200" not in props["name"]:
            raise ValueError("requires Maxwell M2200 sm_52 device properties")
        if not props.get("source"):
            raise ValueError("device properties require a source")
        for key in ("sm_count", "clock_khz", "memory_clock_khz", "memory_bus_bits"):
            value = props[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"invalid device property: {key}")
        return props
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Maxwell M2200 measured parameters unavailable at {path}: {error}; "
                         "run build/devprops under flock /tmp/p100-gpu.lock and save its JSON, "
                         "then pass --device-props <file>") from error


def hardware(variant, device_props=None):
    if variant == "m2200":
        if device_props is None:
            raise ValueError("Maxwell M2200 requires measured device properties")
        clock = device_props["clock_khz"] / 1000
        bandwidth = device_props["memory_bus_bits"] / 8 * 2 * device_props["memory_clock_khz"] / 1e6
        return clock, bandwidth, device_props["sm_count"] * 128 * clock * 1e6
    clock, bandwidth = VARIANTS[variant]
    return clock, bandwidth, 56 * 64 * clock * 1e6


def prediction(typ, weights, batch, variant, bw_eff, issue_eff, counts, device_props=None):
    _, hbm_gbs, issue_rate = hardware(variant, device_props)
    block_bytes, block_weights = BLOCKS[typ]
    byte_count = weights * block_bytes / block_weights
    t_bw = byte_count / (hbm_gbs * 1e9 * bw_eff)
    row = {"type": typ, "bytes_per_weight": block_bytes / block_weights,
           "bytes_per_step": byte_count, "bytes_per_token": byte_count / batch,
           "t_bw_s": t_bw}
    for arch in (variant_arch(variant), "sm_61"):
        alu_per_weight = counts.get((typ, arch))
        if typ != "f16" and alu_per_weight is None:
            raise ValueError(f"missing CSV counts for {typ},{arch}")
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


def shape(value):
    try:
        m, k = (int(part) for part in value.split(","))
        if m <= 0 or k <= 0:
            raise ValueError()
        return m, k
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be positive integers m,k") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llama3.1-8b", "custom"), default="llama3.1-8b")
    parser.add_argument("--weights", type=positive, help="billions of mat-vec weights; requires --model custom")
    parser.add_argument("--shape", type=shape, help="one m-by-k weight matrix (m output rows, k reduction width)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--type", choices=BLOCKS, default="q8_0")
    selection.add_argument("--all-types", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="pcie16")
    parser.add_argument("--batch", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--bw-eff", type=efficiency, default=1.0)
    parser.add_argument("--issue-eff", type=efficiency, default=1.0)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--device-props", type=Path, default=DEFAULT_DEVICE_PROPS,
                        help="measured Maxwell M2200 JSON from bench/micro/devprops.cu")
    parser.add_argument("--json", action="store_true", help="machine-readable predictions and assumptions")
    args = parser.parse_args()
    if args.shape and (args.weights is not None or args.model == "custom"):
        parser.error("--shape cannot be combined with --weights or --model custom")
    if args.model == "custom" and args.weights is None:
        parser.error("--model custom requires --weights <billions>")
    if args.model != "custom" and args.weights is not None:
        parser.error("--weights requires --model custom")
    weights = args.shape[0] * args.shape[1] if args.shape else (
        args.weights * 1e9 if args.model == "custom" else llama_weights())
    if not math.isfinite(weights):
        parser.error("weight count is too large")
    try:
        counts = load_counts(args.csv)
        props = load_device_props(args.device_props) if args.variant == "m2200" else None
        if args.variant == "m2200" and any((typ, "sm_52") not in counts for typ in BLOCKS if typ != "f16"):
            raise ValueError("Maxwell M2200 requires sm_52 CSV rows")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    types = BLOCKS if args.all_types else (args.type,)
    if args.shape and any(args.shape[1] % BLOCKS[typ][1] for typ in types):
        parser.error("shape k must be a multiple of each selected type's block size")
    rows = [prediction(typ, weights, args.batch, args.variant, args.bw_eff, args.issue_eff, counts, props)
            for typ in types]
    assumptions = [DISCLAIMER,
                   "All mat-vec weights use the row's one type; embeddings excluded.",
                   "Weights read once per batch step; bytes/token = bytes/step / batch; tokens/s = batch / step time.",
                   'sm_61 counts use the P100 issue rate as a reference: "if dp4a existed".',
                   "f16: bandwidth-only ceiling; no measured ALU count."]
    if args.variant == "m2200":
        assumptions[0] = ("Maxwell M2200 measured device clocks and theoretical GDDR bandwidth; "
                          "uncalibrated unless efficiencies supplied; ignores activations, KV cache, "
                          "attention and launch overhead")
        assumptions[3] = 'sm_61 counts use the Maxwell M2200 issue rate as a reference: "if dp4a existed".'
    if args.shape:
        assumptions[1] = f"Single weight matrix m={args.shape[0]},k={args.shape[1]}; one read per step."
    if args.json:
        output = {"assumptions": assumptions, "model": "shape" if args.shape else args.model, "weights": weights,
                          "variant": args.variant, "batch": args.batch, "bw_eff": args.bw_eff,
                          "issue_eff": args.issue_eff, "rows": rows}
        if args.shape:
            output["shape"] = args.shape
        if props:
            output["device_props"] = props
        print(json.dumps(output, allow_nan=False))
        return
    print("PREDICTIONS: " + assumptions[0])
    clock, hbm, issue_rate = hardware(args.variant, props)
    memory_label = "GDDR" if props else "HBM"
    print(f"model={'shape' if args.shape else args.model} weights={weights:,.0f} batch={args.batch} variant={args.variant} "
          f"{memory_label}={hbm} GB/s issue={issue_rate / 1e9:.6f} Glane-instructions/s "
          f"bw_eff={args.bw_eff:g} issue_eff={args.issue_eff:g}")
    for assumption in assumptions[1:]:
        print(assumption)
    print("Times in ms; step = max(bw, alu). GB uses decimal units.")
    arch_number = variant_arch(args.variant)[3:]
    print(f"{'type':7} {'GB/step':>9} {'GB/token':>9} {'bw_ms':>9} "
          f"{'alu' + arch_number + '_ms':>10} {'bound' + arch_number:>7} "
          f"{'step' + arch_number + '_ms':>10} {'tok' + arch_number + '/s':>9} "
          f"{'alu61_ms':>10} {'bound61':>7} {'step61_ms':>10} {'tok61/s':>9}")
    for row in rows:
        line = (f"{row['type']:7} {row['bytes_per_step'] / 1e9:9.6f} "
                f"{row['bytes_per_token'] / 1e9:9.6f} {row['t_bw_s'] * 1000:9.6f}")
        for arch in (variant_arch(args.variant), "sm_61"):
            result = row[arch]
            alu = f"{result['t_alu_s'] * 1000:.6f}" if result["t_alu_s"] is not None else "n/a"
            line += (f" {alu:>10} {result['bound']:>7} {result['t_step_s'] * 1000:10.6f} "
                     f"{result['tokens_per_s']:9.3f}")
        print(line)


if __name__ == "__main__":
    main()
