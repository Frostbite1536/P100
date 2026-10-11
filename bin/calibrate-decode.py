#!/usr/bin/env python3
"""Compare contiguous decode MUL_MAT SQL rows with the static model (stdlib only).

Multiple SQL files are independent rounds: use the median per shape, and report
the min/max spread. SQL time_us is already us/run, NOT total time / n_runs.
"""

import argparse
import importlib.util
import math
from pathlib import Path
import re
import sqlite3
import statistics


SPEC = importlib.util.spec_from_file_location("decode_model", Path(__file__).with_name("decode-model.py"))
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def parse_params(text):
    return dict(re.findall(r"(\w+)=(\[[^\]]*\]|[^,]+)", text.replace(" ", "")))


def load_round(path, backend="CUDA0"):
    text = Path(path).read_text()
    statements = re.findall(r'^(CREATE TABLE(?: IF NOT EXISTS)? "?test_backend_ops"?\s*\(.*?\);|'
                            r'INSERT INTO "?test_backend_ops"?\s+.*?\);)\s*$', text, re.M | re.S)
    if not statements:
        raise ValueError(f"{path}: no test-backend-ops SQL statements")
    rows = {}
    with sqlite3.connect(":memory:") as db:
        db.executescript("\n".join(statements))
        for row in db.execute("SELECT backend_name, build_commit, op_params, supported, passed, "
                              "time_us, n_runs FROM test_backend_ops "
                              "WHERE op_name='MUL_MAT' AND test_mode='perf'"):
            device, commit, params, supported, passed, time_us, runs = row
            if device != backend or int(supported) != 1 or int(passed) != 1:
                continue
            p = parse_params(params)
            typ = p.get("type_a")
            if typ not in MODEL.BLOCKS or typ == "f16" or p.get("type_b") != "f32":
                continue
            # Single matrix, contiguous inputs, one output: batched tensors,
            # permutations, padded views and repeated graphs need a different model.
            required = {"bs": "[1,1]", "nr": "[1,1]", "per": "[0,1,2,3]"}
            defaults = {"k_v": "0", "o": "1", "src_overlap": "0", "m_v": "0", "pad": "0"}
            if any(p.get(k) != v for k, v in required.items()):
                continue
            if any(p.get(k, v) != v for k, v in defaults.items()):
                continue
            m, k, n = (int(p[name]) for name in ("m", "k", "n"))
            if min(m, k) <= 0 or not 1 <= n <= 8 or k % MODEL.BLOCKS[typ][1]:
                continue
            time_us = float(time_us)
            if not math.isfinite(time_us) or time_us <= 0 or int(runs) <= 0:
                raise ValueError(f"{path}: invalid timing/runs for {typ},{m},{k},{n}")
            key = (typ, m, k, n)
            entry = rows.setdefault(key, {"times": [], "commit": commit})
            if entry["commit"] != commit:
                raise ValueError(f"{path}: mixed llama.cpp commits")
            entry["times"].append(time_us)
    if not rows:
        raise ValueError(f"{path}: no supported contiguous quantised decode rows on {backend}")
    commits = {entry["commit"] for entry in rows.values()}
    if len(commits) != 1:
        raise ValueError(f"{path}: mixed llama.cpp commits")
    return {key: (statistics.median(entry["times"]), entry["commit"]) for key, entry in rows.items()}


def median_rounds(paths, backend):
    rounds = [load_round(path, backend) for path in paths]
    if any(rows.keys() != rounds[0].keys() for rows in rounds):
        raise ValueError("rounds have different selected rows; refusing a partial median")
    commits = {commit for rows in rounds for _, commit in rows.values()}
    if len(commits) != 1:
        raise ValueError("rounds have different llama.cpp commits")
    rows = []
    for key in sorted(rounds[0]):
        times = [r[key][0] for r in rounds]
        rows.append({"key": key, "time_us": statistics.median(times), "min_us": min(times),
                     "max_us": max(times), "rounds": len(times), "commit": next(iter(commits))})
    return rows


def predict(row, variant, counts, props, bw_eff, issue_eff):
    typ, m, k, n = row["key"]
    result = MODEL.prediction(typ, m * k, n, variant, bw_eff, issue_eff, counts, props)
    return result, result[MODEL.variant_arch(variant)]


def fit(rows, variant, counts, props, bw_eff=1.0, issue_eff=1.0):
    ratios = {"bw": [], "alu": []}
    for row in rows:
        _, pred = predict(row, variant, counts, props, bw_eff, issue_eff)
        ratios[pred["bound"]].append(row["time_us"] * 1e-6 / pred["t_step_s"])
    medians = {bound: statistics.median(values) if values else None for bound, values in ratios.items()}
    # Times scale inversely with efficiency: the measured/predicted ratio is
    # the slowdown, so fitted_eff = initial_eff / median(slowdown).
    fitted = (bw_eff / medians["bw"] if medians["bw"] else bw_eff,
              issue_eff / medians["alu"] if medians["alu"] else issue_eff)
    return fitted, medians, {bound: len(values) for bound, values in ratios.items()}


def print_rows(rows, variant, counts, props, bw_eff, issue_eff):
    print("type m k n measured_us/run weight_GB/s bound t_bw_us t_alu_us predicted_us ratio min_us max_us spread_pct rounds")
    for row in rows:
        raw, pred = predict(row, variant, counts, props, bw_eff, issue_eff)
        typ, m, k, n = row["key"]
        time = row["time_us"]
        print(f"{typ} {m} {k} {n} {time:.6f} {raw['bytes_per_step'] / (time * 1000):.6f} "
              f"{pred['bound']} {raw['t_bw_s'] * 1e6:.6f} {pred['t_alu_s'] * 1e6:.6f} "
              f"{pred['t_step_s'] * 1e6:.6f} {time * 1e-6 / pred['t_step_s']:.6f} "
              f"{row['min_us']:.6f} {row['max_us']:.6f} "
              f"{100 * (row['max_us'] - row['min_us']) / time:.6f} {row['rounds']}")


def write_median_sql(path, rows, backend):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE test_backend_ops (backend_name TEXT, build_commit TEXT, op_name TEXT, "
                   "op_params TEXT, test_mode TEXT, supported INTEGER, passed INTEGER, time_us REAL, n_runs INTEGER)")
        for row in rows:
            typ, m, k, n = row["key"]
            params = (f"type_a={typ},type_b=f32,m={m},n={n},k={k},bs=[1,1],nr=[1,1],"
                      "per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0")
            db.execute("INSERT INTO test_backend_ops VALUES (?,?,?,?,?,?,?,?,?)",
                       (backend, row["commit"], "MUL_MAT", params, "perf", 1, 1, row["time_us"], 1))
        # n_runs=1 is a placeholder: time_us remains the median us/run.
        sql = "\n".join(line for line in db.iterdump() if line.startswith(("CREATE TABLE", "INSERT INTO")))
        Path(path).write_text("-- Derived median of selected rounds; n_runs=1 placeholder, time_us is us/run.\n" + sql + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sql", nargs="+", type=Path, help="one file per measurement round")
    parser.add_argument("--variant", choices=MODEL.VARIANTS, default="pcie16")
    parser.add_argument("--backend", default="CUDA0")
    parser.add_argument("--device-props", type=Path, default=MODEL.DEFAULT_DEVICE_PROPS)
    parser.add_argument("--csv", type=Path, default=MODEL.DEFAULT_CSV)
    parser.add_argument("--bw-eff", type=MODEL.efficiency, default=1.0)
    parser.add_argument("--issue-eff", type=MODEL.efficiency, default=1.0)
    parser.add_argument("--no-fit", action="store_true", help="print only the initial predictions")
    parser.add_argument("--median-sql", type=Path, help="save the selected median timings as SQL")
    args = parser.parse_args()
    try:
        counts = MODEL.load_counts(args.csv)
        props = MODEL.load_device_props(args.device_props) if args.variant == "m2200" else None
        if args.variant == "m2200" and any((typ, "sm_52") not in counts for typ in MODEL.BLOCKS if typ != "f16"):
            raise ValueError("Maxwell M2200 requires sm_52 CSV rows")
        rows = median_rounds(args.sql, args.backend)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        parser.error(str(error))
    label = "Maxwell M2200" if args.variant == "m2200" else f"P100 {args.variant} prediction"
    print(f"{label}: backend={args.backend} variant={args.variant} arch={MODEL.variant_arch(args.variant)} "
          f"commit={rows[0]['commit']} selected={len(rows)} rounds={len(args.sql)}")
    print("time_us is already per run; weight_GB/s counts weights once per batch; decimal GB.")
    print("Repeated identical cases within a round use their median; spread is (max-min)/median across rounds.")
    print(f"INITIAL bw_eff={args.bw_eff:.9g} issue_eff={args.issue_eff:.9g}")
    print_rows(rows, args.variant, counts, props, args.bw_eff, args.issue_eff)
    if args.median_sql:
        write_median_sql(args.median_sql, rows, args.backend)
    if args.no_fit:
        return
    (bw, issue), medians, sizes = fit(rows, args.variant, counts, props, args.bw_eff, args.issue_eff)
    for bound, name in (("bw", "bw_eff"), ("alu", "issue_eff")):
        median = f"{medians[bound]:.9g}" if medians[bound] is not None else "n/a (no rows; efficiency unchanged)"
        print(f"FIT {name}: initial_bound={bound} rows={sizes[bound]} median(measured/predicted)={median}")
    print(f"FITTED bw_eff={bw:.9g} issue_eff={issue:.9g}")
    if max(bw, issue) > 1:
        print("Fitted efficiency exceeds 1: empirical fit exceeds the model peak; values were not clipped.")
    print_rows(rows, args.variant, counts, props, bw, issue)
    changed = []
    for row in rows:
        _, before = predict(row, args.variant, counts, props, args.bw_eff, args.issue_eff)
        _, after = predict(row, args.variant, counts, props, bw, issue)
        if before["bound"] != after["bound"]:
            typ, m, k, n = row["key"]
            changed.append(f"{typ} m={m} k={k} n={n}: {before['bound']} -> {after['bound']}")
    print(f"BOUND CHANGES {len(changed)}")
    for change in changed:
        print(change)


if __name__ == "__main__":
    main()
