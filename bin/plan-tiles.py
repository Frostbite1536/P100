#!/usr/bin/env python3
"""Planning calculator for fused-attention tile configs on P100 (sm_60). No GPU needed.

For a candidate (BR, BC, D, warps, buffering) it reports shared memory per block, the
occupancy limit from smem / registers / threads, and a roofline floor for one prefill
attention call. Everything is static arithmetic: real register counts come from
bin/sm60-report.py, real timing from the P100 box.

  python bin/plan-tiles.py --d 128 --br 64 --bc 64 --warps 4 --regs 128
  python bin/plan-tiles.py --d 128 --sweep          # table over common configs
"""
import argparse
import math

# Pascal GP100 limits; datasheet peaks from docs/research/volta-fa-lessons.md §1.
SM = dict(
    n_sm=56,
    regs_per_sm=65536,
    max_regs_per_thread=255,
    reg_alloc_unit=256,          # registers are allocated per warp in units of 256
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    smem_per_sm_max=64 * 1024,
    smem_per_block_max=48 * 1024,
)
PEAK = {
    "pcie16": dict(fp32_tflops=9.3, fp16_tflops=18.7, hbm_gbs=732.0),
    "pcie12": dict(fp32_tflops=9.3, fp16_tflops=18.7, hbm_gbs=549.0),
    "sxm2": dict(fp32_tflops=10.6, fp16_tflops=21.2, hbm_gbs=732.0),
}


def smem_bytes(br, bc, dqk, dv, bufs, s_in_smem, pad_halves):
    h = 2
    q = br * (dqk + pad_halves) * h
    k = bufs * bc * (dqk + pad_halves) * h
    v = bufs * bc * (dv + pad_halves) * h
    # Optionally stage scores (fp32) and probabilities (fp16) in shared memory.
    s = br * bc * 4 if s_in_smem else 0
    p = br * (bc + pad_halves) * h if s_in_smem else 0
    return dict(q=q, k=k, v=v, s=s, p=p, total=q + k + v + s + p)


def est_regs(br, bc, dv, warps, s_in_smem):
    """Very rough: O accumulator + S tile in fp32 registers + fixed overhead."""
    threads = warps * 32
    o_acc = br * dv / threads
    s_tile = 0 if s_in_smem else br * bc / threads
    return math.ceil(o_acc + s_tile + 48)


def occupancy(smem, regs, warps):
    threads = warps * 32
    regs_per_warp = math.ceil(regs * 32 / SM["reg_alloc_unit"]) * SM["reg_alloc_unit"]
    by_regs = SM["regs_per_sm"] // (regs_per_warp * warps) if regs else SM["max_blocks_per_sm"]
    by_smem = SM["smem_per_sm_max"] // smem if smem else SM["max_blocks_per_sm"]
    by_warps = SM["max_warps_per_sm"] // warps
    blocks = max(0, min(by_regs, by_smem, by_warps, SM["max_blocks_per_sm"]))
    if smem > SM["smem_per_block_max"] or regs > SM["max_regs_per_thread"] or threads > 1024:
        blocks = 0
    limiter = min(("regs", by_regs), ("smem", by_smem), ("warps", by_warps), key=lambda t: t[1])[0]
    return dict(blocks=blocks, warps=blocks * warps, occ=blocks * warps / SM["max_warps_per_sm"],
                limiter=limiter, threads=threads)


def roofline(sq, skv, dqk, dv, heads, br, causal, variant):
    peak = PEAK[variant]
    frac = 0.5 if causal else 1.0
    flops = 2 * sq * skv * (dqk + dv) * heads * frac
    qo_bytes = sq * (dqk + dv) * 2 * heads
    # DRAM traffic is bracketed: worst case re-reads K/V from DRAM for every Q block (no L2
    # reuse); best case reads each head's K/V once (perfect L2 reuse across Q blocks of a head).
    worst = math.ceil(sq / br) * skv * (dqk + dv) * 2 * heads * frac + qo_bytes
    best = skv * (dqk + dv) * 2 * heads + qo_bytes
    t_c = flops / (peak["fp16_tflops"] * 1e12) * 1e3
    t_worst = worst / (peak["hbm_gbs"] * 1e9) * 1e3
    t_best = best / (peak["hbm_gbs"] * 1e9) * 1e3
    return dict(flops=flops, worst=worst, best=best, t_c=t_c, t_worst=t_worst, t_best=t_best,
                floor_best=max(t_c, t_best), floor_worst=max(t_c, t_worst))


def report(a, br, bc, warps, regs=None, quiet=False):
    s = smem_bytes(br, bc, a.d, a.dv or a.d, a.bufs, a.s_in_smem, a.pad)
    r = regs if regs is not None else est_regs(br, bc, a.dv or a.d, warps, a.s_in_smem)
    o = occupancy(s["total"], r, warps)
    rf = roofline(a.sq, a.skv, a.d, a.dv or a.d, a.heads, br, a.causal, a.variant)
    peak = PEAK[a.variant]
    ok = s["total"] <= SM["smem_per_block_max"] and r <= SM["max_regs_per_thread"] and o["blocks"] > 0
    grid = math.ceil(a.sq / br) * a.heads
    waves = grid / (o["blocks"] * SM["n_sm"]) if o["blocks"] else float("inf")
    if quiet:
        return (br, bc, warps, s["total"], r, o["blocks"], o["occ"], o["limiter"], waves, ok)
    print(f"config  BR={br} BC={bc} D={a.d} Dv={a.dv or a.d} warps={warps} bufs={a.bufs} "
          f"{'S,P in smem' if a.s_in_smem else 'S,P in regs'} pad={a.pad} halves")
    print(f"smem    Q {s['q']}  K {s['k']}  V {s['v']}  S {s['s']}  P {s['p']}  = {s['total']} B "
          f"({s['total']/1024:.1f} KB){'  > 48 KB: exceeds sm_60 per-block limit' if s['total'] > 48*1024 else ''}")
    print(f"regs    {r}/thread {'(given)' if regs is not None else '(ROUGH estimate; use bin/sm60-report.py)'}")
    print(f"occ     {o['blocks']} blocks/SM, {o['warps']}/64 warps = {o['occ']:.0%} (limiter: {o['limiter']})")
    print(f"grid    {grid} blocks over {SM['n_sm']} SMs = {waves:.2f} waves (tail effect if fractional and small)")
    print(f"roof    sq={a.sq} skv={a.skv} heads={a.heads} causal={a.causal}: {rf['flops']/1e9:.1f} GFLOP")
    print(f"        datasheet peaks ({a.variant}): FP32 {peak['fp32_tflops']:.1f} TFLOPS; "
          f"FP16 half2 {peak['fp16_tflops']:.1f} TFLOPS; HBM {peak['hbm_gbs']:.0f} GB/s")
    print(f"        FP16 half2 compute floor {rf['t_c']:.3f} ms; DRAM floor "
          f"{rf['t_best']:.3f} ms (perfect L2 reuse, {rf['best']/1e6:.0f} MB) .. {rf['t_worst']:.3f} ms "
          f"(no reuse, {rf['worst']/1e6:.0f} MB)")
    print(f"        roofline floor = max(compute, DRAM): {rf['floor_best']:.3f} .. "
          f"{rf['floor_worst']:.3f} ms (datasheet peaks; not measured P100 timing)")
    print("valid" if ok else "INVALID: exceeds a hardware limit")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, default=128, help="head dim of Q/K")
    ap.add_argument("--dv", type=int, default=0, help="head dim of V (default = d)")
    ap.add_argument("--br", type=int, default=64)
    ap.add_argument("--bc", type=int, default=64)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--regs", type=int, help="registers/thread from bin/sm60-report.py")
    ap.add_argument("--bufs", type=int, default=1, help="K/V smem buffers (2 = double buffering)")
    ap.add_argument("--pad", type=int, default=8, help="row padding in halves (bank-conflict avoidance)")
    ap.add_argument("--s-in-smem", action="store_true", help="store S/P tiles in shared memory (default: registers)")
    ap.add_argument("--variant", choices=PEAK, default="pcie16", help="P100 datasheet peak variant (default: pcie16)")
    ap.add_argument("--sq", type=int, default=8192)
    ap.add_argument("--skv", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=32, help="q heads x batch")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    for name in ("d", "br", "bc", "warps", "bufs", "sq", "skv", "heads"):
        if getattr(a, name) <= 0:
            ap.error(f"--{name} must be positive")
    if a.dv < 0 or a.pad < 0 or (a.regs is not None and a.regs < 0):
        ap.error("--dv, --pad and --regs must be nonnegative")
    if not a.sweep:
        report(a, a.br, a.bc, a.warps, a.regs)
        return
    print(f"sweep D={a.d} bufs={a.bufs} {'S,P in smem' if a.s_in_smem else 'S,P in regs'}; regs are ROUGH estimates")
    print(f"{'BR':>4} {'BC':>4} {'warps':>5} {'smem':>7} {'regs':>4} {'blk/SM':>6} {'occ':>5} {'limit':>6} {'waves':>6}")
    for br in (16, 32, 64, 128):
        for bc in (32, 64, 128):
            for warps in (2, 4, 8):
                row = report(a, br, bc, warps, quiet=True)
                if not row[-1]:
                    continue
                print(f"{row[0]:>4} {row[1]:>4} {row[2]:>5} {row[3]:>7} {row[4]:>4} {row[5]:>6} {row[6]:>5.0%} {row[7]:>6} {row[8]:>6.2f}")


if __name__ == "__main__":
    main()
