#!/usr/bin/env python3
"""Check decode CLI predictions or regenerate and validate the static fixture."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BLOCK_WEIGHTS = {"q4_0": 32, "q4_1": 32, "q5_0": 32, "q5_1": 32, "q8_0": 32,
                 "q4_K": 256, "q5_K": 256, "q6_K": 256, "iq4_xs": 256}
CSV = ROOT / "docs/research/vecdot-sass-2026-10-10.csv"


def check_csv(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    expected = {(typ, arch) for typ in BLOCK_WEIGHTS for arch in ("sm_60", "sm_61", "sm_52")}
    keys = [(row["type"], row["arch"]) for row in rows]
    assert len(keys) == len(expected) and set(keys) == expected, "one row per type and arch required"
    costs = {}
    for row in rows:
        typ, arch = row["type"], row["arch"]
        assert int(row["weights_per_block"]) == BLOCK_WEIGHTS[typ]
        total, alu, mem, ctrl = (int(row[name]) for name in
                                  ("sass_total", "sass_alu", "sass_mem", "sass_ctrl"))
        assert total == alu + mem + ctrl and min(alu, mem) > 0 and ctrl >= 0
        assert row["top_opcodes"]
        costs[typ, arch] = alu
    for typ in BLOCK_WEIGHTS:
        assert costs[typ, "sm_61"] < costs[typ, "sm_60"], f"{typ}: sm_61 ALU must be smaller"
    return rows


def run_model(*args):
    return json.loads(subprocess.check_output(
        [sys.executable, str(ROOT / "bin/decode-model.py"), "--json", *args], text=True))


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.001), (actual, expected)


def check_model():
    check_csv(CSV)
    # Independent matrix shapes for q/k/v/o and gate/up/down, plus output head.
    weights = 32 * (4096 * 4096 * 2 + 4096 * 1024 * 2 + 4096 * 14336 * 3) + 4096 * 128256
    for typ, bytes_per_weight in (("q8_0", 34 / 32), ("q4_0", 18 / 32)):
        predictions = {}
        for batch in (1, 4):
            output = run_model("--model", "llama3.1-8b", "--type", typ, "--batch", str(batch))
            assert output["weights"] == weights
            assert "datasheet peaks, uncalibrated" in output["assumptions"][0]
            row, = output["rows"]
            predictions[batch] = row
            close(row["bytes_per_step"], weights * bytes_per_weight)
            close(row["bytes_per_token"], weights * bytes_per_weight / batch)
            close(row["t_bw_s"], weights * bytes_per_weight / 732e9)
            for arch in ("sm_60", "sm_61"):
                result = row[arch]
                close(result["t_alu_s"], weights * result["alu_per_weight"] * batch / (56 * 64 * 1303e6))
                close(result["t_step_s"], max(row["t_bw_s"], result["t_alu_s"]))
                close(result["tokens_per_s"], batch / result["t_step_s"])
                assert result["bound"] == ("alu" if result["t_alu_s"] > row["t_bw_s"] else "bw")
        for arch in ("sm_60", "sm_61"):
            close(predictions[4][arch]["t_alu_s"], 4 * predictions[1][arch]["t_alu_s"])
        close(predictions[1]["bytes_per_token"], weights * bytes_per_weight)
    # Exercise the custom count, variant, efficiency and f16 paths.
    custom = run_model("--model", "custom", "--weights", "1.5", "--all-types", "--batch", "8",
                       "--variant", "pcie12", "--bw-eff", "0.5", "--issue-eff", "0.25")
    assert len(custom["rows"]) == 10 and custom["weights"] == 1.5e9
    for row in custom["rows"]:
        close(row["t_bw_s"], row["bytes_per_step"] / (549e9 * 0.5))
    f16 = custom["rows"][-1]
    assert f16["type"] == "f16" and f16["sm_60"]["t_alu_s"] is None
    assert f16["sm_60"]["bound"] == "bw-only"
    # New shape mode preserves P100 formulas and uses measured-input parameters
    # and sm_52 counts for Maxwell. The device file here is explicitly synthetic.
    shaped = run_model("--shape", "4096,14336", "--all-types", "--batch", "4")
    assert shaped["weights"] == 4096 * 14336 and shaped["shape"] == [4096, 14336]
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as work:
        props = {"name": "SYNTHETIC Maxwell M2200", "major": 5, "minor": 2,
                 "sm_count": 2, "clock_khz": 1000000, "memory_clock_khz": 2000000,
                 "memory_bus_bits": 64, "source": "SYNTHETIC arithmetic test"}
        path = Path(work) / "synthetic-device.json"
        path.write_text(json.dumps(props))
        output = run_model("--shape", "4096,14336", "--all-types", "--variant", "m2200",
                           "--device-props", str(path), "--batch", "4")
        for row in output["rows"]:
            close(row["t_bw_s"], row["bytes_per_step"] / 32e9)
            assert "sm_52" in row and "sm_60" not in row
            if row["type"] != "f16":
                close(row["sm_52"]["t_alu_s"], output["weights"] * row["sm_52"]["alu_per_weight"] * 4 / 256e9)
    failed = subprocess.run([sys.executable, str(ROOT / "bin/decode-model.py"),
                             "--shape", "32,33"], capture_output=True, text=True)
    assert failed.returncode != 0 and "block size" in failed.stderr
    print("decode_model: bytes, batch scaling, ceilings and all per-type arch comparisons passed")


def check_calibration():
    spec = importlib.util.spec_from_file_location("calibrate", ROOT / "bin/calibrate-decode.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    fixture = ROOT / "tests/fixtures/tbo-perf-mulmat-synthetic.sql"
    assert "SYNTHETIC" in fixture.read_text().splitlines()[0]
    rows = tool.median_rounds([fixture], "CUDA0")
    assert len(rows) == 4
    assert {row["key"]: row["time_us"] for row in rows} == {
        ("q8_0", 32, 256, 1): 2, ("q4_1", 32, 256, 1): 2,
        ("q4_0", 32, 256, 1): 5, ("q8_0", 32, 256, 4): 10}
    counts = tool.MODEL.load_counts(CSV)
    (bw, issue), ratios, sizes = tool.fit(rows, "pcie16", counts, None)
    # Independent byte/ALU arithmetic: fixture time_us is already per run.
    bw_ratios = [2e-6 / (8192 * 34 / 32 / 732e9), 2e-6 / (8192 * 20 / 32 / 732e9)]
    alu_ratios = [5e-6 / (8192 * 135 / 32 / (56 * 64 * 1303e6)),
                  10e-6 / (8192 * 125 / 32 * 4 / (56 * 64 * 1303e6))]
    close(bw, 1 / (sum(bw_ratios) / 2))
    close(issue, 1 / (sum(alu_ratios) / 2))
    assert sizes == {"bw": 2, "alu": 2}
    assert ratios["bw"] > 1 and ratios["alu"] > 1
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as work:
        median_path = Path(work) / "median.sql"
        tool.write_median_sql(median_path, rows, "CUDA0")
        assert tool.median_rounds([median_path], "CUDA0") == rows
        second = Path(work) / "second.sql"
        second.write_text(fixture.read_text().replace(',2,100);', ',4,100);'))
        combined = tool.median_rounds([fixture, second, fixture], "CUDA0")
        q8 = next(row for row in combined if row["key"] == ("q8_0", 32, 256, 1))
        assert (q8["time_us"], q8["min_us"], q8["max_us"], q8["rounds"]) == (2, 2, 4, 3)
    output = subprocess.check_output([sys.executable, str(ROOT / "bin/calibrate-decode.py"), str(fixture)], text=True)
    assert "FITTED bw_eff=" in output and "BOUND CHANGES 2" in output
    assert "q4_1 m=32 k=256 n=1: bw -> alu" in output
    assert "q8_0 m=32 k=256 n=1: bw -> alu" in output
    assert "2.000000 4.352000 bw" in output  # 8704 bytes / 2 us = 4.352 GB/s
    print("calibrate_decode: SYNTHETIC filtering, per-run times, GB/s, median fits, round spread and bound changes passed")


def check_fixture():
    # All build artifacts stay in the worktree's build directory; CSV is temporary.
    with tempfile.TemporaryDirectory(prefix="vecdot-check-", dir=ROOT / "build") as work:
        path = Path(work) / "counts.csv"
        subprocess.run(["bash", str(ROOT / "bin/vecdot-sass.sh"), str(path)], check=True)
        generated = check_csv(path)
        assert generated == check_csv(CSV), "regenerated SASS counts differ from the checked-in CSV"
    print("vecdot_sass: 27 real type/arch rows regenerated and match the CSV")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--sass":
        check_fixture()
    elif len(sys.argv) == 2 and sys.argv[1] == "--calibrate":
        check_calibration()
    elif len(sys.argv) == 1:
        check_model()
    else:
        raise SystemExit("usage: decode_model_check.py [--sass|--calibrate]")
