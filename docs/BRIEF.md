# P100 — making llama.cpp faster on the Tesla P100

*Share brief for Pascal-owning communities, sibling of the
[volta-fa brief](https://github.com/Frostbite1536/volta-fa/blob/main/docs/BRIEF.md).
Honest status up front: **nothing in this repo has been run on a P100 yet** — the cards
are ordered (2× PCIe 16 GB). What exists is the measured analysis, six ready patches,
and a runnable measurement kit. Every claim below says what hardware it actually ran on.*

**Repo:** https://github.com/Frostbite1536/P100 (AGPL-3.0)

## What it is

The same method as [volta-fa](https://github.com/Frostbite1536/volta-fa) (our V100
project), applied to the P100 (Pascal GP100, sm_60): find where llama.cpp loses time
*on this specific silicon*, fix it as a small patch series against a pinned commit
(`79e2e74`), and refuse any performance claim that wasn't measured. The P100 is a
different beast from the V100 — no tensor cores, no `dp4a` — so the fixes are different
too.

## The gap that motivated it

From `docs/research/volta-fa-lessons.md` (source-traced at the pinned commit, plus one
static compile):

1. **Quantised decode is likely ALU-bound on the P100.** `dp4a` doesn't exist on
   GP100 (it's sm_61+); llama.cpp emulates it in SASS. Measured disassembly: one q8_0
   block costs **120 instructions on sm_60 vs 36 on sm_61** (~3.75 vs 0.25 per int8
   product). For Llama-3.1-8B, the dot-product ALU work (~6.4 ms/token at full issue)
   is on the order of the time to merely stream the weights (6.3–11.6 ms at 732 GB/s).
   Decode should be memory-bound; on the P100 it plausibly isn't.
2. **FlashAttention accumulates V·KQ in FP16 on Pascal/Volta TILE paths.** With a
   biased V (non-zero mean) at 16K–32K keys, our test cases show the FP16-accumulated
   path drifting badly — the old CPU reference agreed 0/16 with itself before the fix
   (NMSE 0.36–0.60). This is an accuracy bug class, not just speed.
3. **Quantised-KV conversion** (≥3-token batches with a quantised cache) re-converts
   rows with mostly-idle threads — the same problem volta-fa fixed and measured on
   real hardware (V100: 1.7–6.9× faster conversion, bit-identical).
4. **Two-GPU PCIe P2P**: the cards have no NVLink between them (PCIe 3.0 only);
   upstream's tensor-split all-reduce has no direct device-to-device path.

## What exists today (and what it's proven on)

| piece | proven on | P100 |
| --- | --- | --- |
| `0001`/`0002` K/V conversion patches (byte-identical ports of volta-fa's) | V100 1.7–6.9×, Maxwell 2.6–5.2×, bit-identical; FA 4009/4009 on Maxwell | not run |
| `0003` biased-V FA test cases + FP32 CPU reference | Maxwell FP32 build: 16/16 (old FP16 reference: 0/16) | not run |
| `0004` FA TILE flushes FP16 VKQ to FP32 per KV tile | builds sm_60/sm_70, no new spills on sm_60 | not run |
| `0005`/`0006` half2 decode mat-vec (MMVH) for q4_0/q8_0/K-quants, batch 1–8, no dp4a | Maxwell in FP32 emulation: 375/375 correct (n ≤ 8); 0 spills | not run |
| Activation-range MUL_MAT tests | upstream path: 48/80; with `0005`/`0006`: **80/80** | not run |

That last row is an upstream finding independent of the P100: `block_q8_1` stores its
scale and sum in FP16, so activations above 65504 or deep in subnormal territory produce
wrong results on the *current* path. Our half2 mat-vec handles them.

## Why you'd want to try it — or help

The cards land soon and Phase 0 (`docs/ROADMAP.md`) is a ready-to-run kit:
`bench/patches/run.sh` builds every variant, runs the correctness suites and the A/B
timing in one session. If you own a P100 (PCIe or SXM2, any VRAM), you can produce the
first real numbers for any of it — the repo's standing rule is "not run" until
someone runs it, and that someone can be you. The five ranked hypotheses (H1 decode
roofline, H2 FA TILE FP16 budget, H3 FA by head dim, H4 PCIe P2P, H5 conversion gains)
each name the exact command that settles them.

## How to try it

```bash
git clone https://github.com/Frostbite1536/P100
# llama.cpp at bench/LLAMA_COMMIT (79e2e74), then:
git apply third_party/patches/00*.patch   # in order; 0006 requires 0005
```

Build with CUDA 12.8, `-DGGML_CUDA_ARCHITECTURES=60`. Patches are not rebased past the
pinned commit.

## Future work

Everything in `docs/ROADMAP.md` Phase 0+: decode mat-vec without dp4a (H1) is the
expected biggest win if the ALU-bound hypothesis holds; conversion patches upstream
(H5); FA TILE inner loop within the FP16 budget (H2); padded GQA routing for TILE;
two-GPU split and P2P overlap (H4) — the P2P layer is already in the repo, unit-tested
on a Maxwell laptop GPU, waiting for two cards.

Sibling projects: [volta-fa](https://github.com/Frostbite1536/volta-fa) (V100) and
A770 (Intel Arc).
