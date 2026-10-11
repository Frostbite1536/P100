# Ticket p13-mmperf-16: MUL_MAT timing at 1–16 tokens in the day-one A/B kit

Your current directory is a git worktree of the P100 repo (branch `p13-mmperf-16`, based on `main` @ `b13da11`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/sm60-report.py <file.cu>` and paste its table. Spill must be 0.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/p100/attention.h`.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52; its timings are not P100 numbers): anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/volta-fa-p9-lessons-2026-10-11.md` §2, §5. At the pin (`79e2e74`), dense quantised MUL_MAT on sm_60 uses MMVQ up to 8 tokens (`MMVQ_MAX_BATCH_SIZE`, `ggml/src/ggml-cuda/mmvq.cuh:3`). From 9 tokens on it uses dequantise-to-FP16 + cuBLAS (`ggml_cuda_mul_mat` in `ggml-cuda.cu`). The reason is that `ggml_cuda_should_use_mmq` (`mmq.cu`) returns false for dense weights when `ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A` (MMQ is kept only for MoE on Pascal). For Q4_K that is roughly 8× the bytes of one weight read. Patches 0005/0006 (`mmvh`) cover 1–8 tokens.

The kit cannot see the 9–16 range today:
- `bench/patches/run.sh:122` `MM_FILTER` matches `n=[1-8]`.
- Upstream's perf list (`tests/test-backend-ops.cpp`, `make_test_cases_perf`) only has MUL_MAT at n ∈ {1, 2, 3, 4, 5, 8, 512} (4096×14336, all types) and Q4_K at n ∈ {1, 2, 3, 4, 8} for four sizes.

So n = 6, 7 and 9–16 are never timed.

## Design (manager decided, do not change)
1. **New directory `third_party/patches/bench-only/`** for patches that the A/B kit applies identically to **every** variant, including `upstream` (test cases, development switches). They are never part of the upstream series. `.gitignore` already allows `third_party/patches/`. List each bench-only patch in `third_party/patches/README.md` under a new "Bench-only patches" heading (provenance, purpose, "never upstream").
2. **`third_party/patches/bench-only/0001-tests-mul_mat-perf-cases-1-16.patch`** (against the pin; one `git format-patch` commit). In `make_test_cases_perf`, after the existing `for (int bs : {1, 2, 3, 4, 5, 8, 512})` loop, add `test_mul_mat(type_a, GGML_TYPE_F32, m, n, k, {1, 1}, {1, 1})` with:
   - type_a ∈ {Q4_0, Q8_0, Q4_K, Q5_K, Q6_K};
   - (m, k) ∈ Llama-3.1-8B shapes {(4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336)};
   - n = 1..16;
   - skip every combination that already exists in the perf list (4096×14336 at n ∈ {1, 2, 3, 4, 5, 8}, and the existing Q4_K mat-vec cases), so no case name is duplicated;
   - a one-line comment saying the cases are for the local A/B kit.

   The eval list is untouched.
3. **`bench/patches/run.sh`:**
   - Apply every `bench-only/*.patch` that is marked "all variants" in the README to every variant, after its series patches, and log each. `DRYRUN` shows them. (Ticket p14 adds a bench-only patch that is applied to one build only; keep the mechanism able to exclude a file by name.)
   - `MM_FILTER` matches n = 1..16 and, for the new shapes, m/k from the list above.
   - The header comment says the 9–16 range is upstream's cuBLAS path on sm_60, with a pointer to the research doc.
4. **New `bin/mm-gbps.py`** (Python standard library only; reuse `bench/analyze.py`'s SQL parsing).
   - Input: one or more SQL groups (comma-separated rounds) with `--labels`, like `analyze.py tbo`.
   - Output: for each (type, m, k), one row per label with the median µs and effective GB/s at every n.
   - Bytes: m·k weights at the type's bytes per block / elements per block (copy the `BLOCKS` table from `~/Github/volta-fa/bin/qmatmul-table.py`), plus n·k·4 bytes of activations and m·n·4 of output. GB/s is decimal.
   - It prints the roofline used (`--peak-gbps`, default 732) and the percentage of it.
   - `summary.md` calls it after the existing `analyze.py tbo` table.
5. `bench/BASELINE.md`: one line under the day-one A/B paragraph saying what `mmperf` now covers and why 9–16 matters (pointer to the research doc). Facts only.

## Files you may touch
- `third_party/patches/bench-only/0001-tests-mul_mat-perf-cases-1-16.patch` (new), `third_party/patches/README.md` (bench-only section)
- `bench/patches/run.sh`, `bin/mm-gbps.py` (new), `bench/BASELINE.md` (one line)
- `tests/fixtures/shims/*`, `tests/fixtures/mm-gbps/*` (new: a small hand-written perf SQL fixture), `tests/CMakeLists.txt` (dry-run assertions and an `mm_gbps` test)
Nothing else.

## Steps
1. Write the patch in a scratch worktree of `third_party/llama.cpp` at the pin under `build/`, and export it with `git format-patch`.
2. Build that tree's `test-backend-ops` CPU-only (`-DGGML_CUDA=OFF`).
   - Run `test-backend-ops perf -o MUL_MAT -b CPU -p '<filter for q4_K, m=4096, n=16, k=4096>'` to show that one new case exists and runs.
   - Count the new case names (grep the source or a filtered run) and check that none duplicates an existing one.
3. Implement the `run.sh` changes and `bin/mm-gbps.py`, then extend the dry-run test and add the fixture test. The fixture covers two labels, two rounds and n = 8 and 16, and asserts the GB/s values computed by hand.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all)
- `shellcheck -x bench/patches/run.sh bench/phase0.sh bin/*.sh`, `bin/check-forbidden.sh`
- `DRYRUN=1 FORCE=1 OUT=<tmp> bench/patches/run.sh` with the default variants: paste the planned commands that show the bench-only patch and the new filter.

## Done means
The kit times n = 1..16 on the P100 day one, and `summary.md` shows GB/s against 732 GB/s.

Final report:
- files changed;
- each command with its result line;
- the number of new perf cases and the duplicate check;
- the CPU run line for the one case;
- what was NOT verified (everything on a P100).

Do not interpret.
