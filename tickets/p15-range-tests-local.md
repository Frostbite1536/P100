# Ticket p15-range-tests-local: activation-range MUL_MAT tests as a bench-only patch

Your current directory is a git worktree of the P100 repo (branch `p15-range-tests-local`, based on `main` @ `c2692bf`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores (no wmma, no mma.sync of any shape) and NO __dp4a/__dp2a (sm_61+). FP16 math uses half2 intrinsics. Shared memory <= 48 KB per block. Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/sm60-report.py <file.cu>` and paste its table. Spill must be 0.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/p100/attention.h`.
- Do NOT commit, push, or run git commands that write **in the P100 repo**. Leave changes uncommitted; the manager commits. In your private llama.cpp clone(s) (Steps 1–2) you may apply patches, commit and `git format-patch`; they are gitignored scratch.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no P100 on this machine (the local GPU is a Maxwell sm_52; its timings are not P100 numbers): anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/volta-fa-p9-lessons-2026-10-11.md` §4:
- Upstream's `block_q8_1` stores d and the block sum of the raw activations in FP16. The sum overflows at activations around 1e5, and d underflows around 1e-5.
- `mmvh`'s prepass (`x̂ = fp16(x/s)`, `s` = max |x| per 32, sum of x̂ in FP32) cannot overflow, so it should pass where q8_1 fails.

volta-fa's local test is `~/Github/volta-fa/third_party/patches-local/0009-tests-MUL_MAT-with-activations-beyond-FP16-s-range-l.patch` (`test_mul_mat_range`: src1 scaled by `scale / 2^(6·((i/256) % 3))`). It applies on top of volta-fa's 0008, not on the pin. On a V100: current path 18/35, 0008 35/35.

**CPU reference limits at the pin** (`ggml/src/ggml-cpu/ggml-cpu.c`, `type_traits_cpu`):
- Q4_0 and Q8_0 use `vec_dot_type` Q8_0, whose FP16 d underflows at 1e-5, so the reference itself is wrong there.
- Q4_K/Q5_K/Q6_K use Q8_K (FP32 d) and are valid at both scales.

These tests fail on unmodified upstream by design, so they must never enter the upstream series.

## Design (manager decided, do not change)
1. **`third_party/patches/bench-only/0003-tests-mul_mat-activation-range.patch`** (against the pin): port `test_mul_mat_range` (same class, same `vars()` suffix `,src1_scale=`, same initialisation). Cases:
   - type_a ∈ {Q4_0, Q8_0, Q4_K, Q5_K, Q6_K};
   - n ∈ {1, 2, 4, 8} (mmvh's range) and 16 (the cuBLAS path);
   - m ∈ {128, 4096}, K = 1024;
   - scale ∈ {1e5, 1e-5};
   - skip scale 1e-5 for Q4_0 and Q8_0, with a comment giving the reason above.

   Eval list only. p13 and p14 have landed: add a row to `third_party/patches/README.md`'s "Bench-only patches" table (it is the manifest `run.sh` reads) with the applies-to cell exactly `all variants`, marked "fails on upstream by design; never upstream". `run.sh` then applies it after bench-only 0001 in every tree (including p14's `ref` tree).
2. **`bench/patches/run.sh` `test` step:** per variant, one more run, `test-backend-ops test -b CUDA0 -o MUL_MAT -p src1_scale`, into `<arm>.range.txt`. mmvh arms run with and without `GGML_CUDA_DISABLE_MMVH=1`. `summary.md` prints each arm's pass count under a heading that says these cases fail on upstream's q8_1 path by design (not a regression signal).
3. `bench/BASELINE.md`: one line on what the range run answers.

## Files you may touch
- `third_party/patches/bench-only/0003-tests-mul_mat-activation-range.patch` (new), `third_party/patches/README.md` (bench-only section)
- `bench/patches/run.sh`, `bench/BASELINE.md` (one line)
- `tests/fixtures/shims/*`, `tests/CMakeLists.txt` (dry-run assertions)
Nothing else.

## Steps
1. Port the test in a private clone: `git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp build/llama-p15`, checkout `79e2e74eb` (the pin, `bench/LLAMA_COMMIT`). Do not touch `/home/p51/Github/P100/third_party/llama.cpp` itself. Export it with `git format-patch`. Show with `git apply --check` on fresh checkouts that it applies: (a) on the pin alone; (b) after series 0003 + bench-only 0001 (the `upstream` tree as `run.sh` builds it); (c) after series 0001–0006 + bench-only 0001. Both bench-only patches edit `tests/test-backend-ops.cpp`, so (b) and (c) are the orders `run.sh` uses.
2. Build `test-backend-ops` for the two trees below with `-DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=52;60' -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF -DCMAKE_BUILD_RPATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib`, Release (build dirs `build/llama-p15/build-upstream` and `build/llama-p15-mmvh/build-mmvh`; a second private clone is fine). Your sandbox has NO GPU access; do not run them. Write the exact commands (absolute paths, outputs under `build/`) under a heading "Manager GPU commands" in your final report; the manager runs them and records the per-case table. The two runs of the range filter:
   - the pin plus the patch (upstream paths);
   - 0001–0006 plus the patch, with mmvh forced on as `bench/results/maxwell-patches-2026-10-10/README.md` describes (FP32 emulation on Maxwell).

   Expected (unverified): the upstream build fails the 1e5 cases on q8_1, and the mmvh build passes n ≤ 8. The pre-existing q4_K abort (non-contiguous K-quant src0 on the cuBLAS path, `ggml-cuda.cu:1462`) should not hit contiguous range cases; if it does, the manager records where and filters around it.
3. Implement the `run.sh` change and the dry-run assertions.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all; GPU tests may fail or skip in your sandbox because it has no GPU: name them). Add new dry-run assertions; do not remove or weaken existing ones.
- `shellcheck -x bench/patches/run.sh bin/*.sh`, `bin/check-forbidden.sh`
- `DRYRUN=1 FORCE=1 OUT=<tmp> bench/patches/run.sh`: paste the planned range commands.

## Done means
The range cases run per arm on day one, and the Maxwell results of both builds are recorded.

Final report:
- files changed;
- each command with its result line;
- the Manager GPU commands for both builds (the manager produces the per-case Maxwell table);
- what was NOT verified (anything on a P100).

Do not interpret.
