# Ticket p7a-patches-biasv: build the patch series for sm_60, add a biased-V FA correctness case, wire both into Phase 0

Your current directory is a git worktree of the P100 repo (branch `p7a-patches-biasv`, based on `main` @ `6554601`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is NVIDIA Pascal sm_60 (Tesla P100) ONLY, CUDA 12.x. There are NO tensor cores and NO __dp4a/__dp2a (sm_61+). Do NOT use BF16, FP8, cp.async, ldmatrix, TMA, clusters, or FlashAttention-2/3 code. If you think you need one, STOP and explain.
- The toolchain lives in the MAIN checkout's venv. Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest`, `python` and `shellcheck` resolve there. Never install anything.
- Do NOT commit, push, or run git commands that write **in the P100 repo**. In your private llama.cpp clone (step 1) you may `git apply`, `git checkout -- .` and `git diff` freely; that clone is gitignored scratch.
- Never invent benchmark numbers. There is no P100 on this machine. The local GPU is a Maxwell sm_52 (Quadro M2200, no `FAST_FP16_AVAILABLE`, so llama.cpp's FA uses FP32 accumulators there): correctness runs on it are wanted; timings are not.
- GPU use: wrap every command that runs on the GPU in `flock /tmp/p100-gpu.lock <command>` (other workers share it). If CUDA cannot initialise inside your sandbox, say so, leave the exact commands in your report, and continue; the manager will run them.
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
- `third_party/patches/0001-0002` (volta-fa 0005/0006, K/V conversion speed-ups) have never been applied or built at our pin in this repo (`docs/ROADMAP.md`, pre-hardware list; `third_party/patches/README.md`). volta-fa found a pre-existing q4_K MUL_MAT abort on Maxwell in unpatched upstream (`/home/p51/Github/volta-fa/docs/research/` P-8 doc, commit 8533d81); expect it and compare against upstream rather than treating it as a patch failure.
- `docs/research/pre-hardware-findings-2026-10-10.md` §2: llama.cpp's sm_60 TILE kernel keeps the FA output in an FP16 accumulator; the CPU model predicts large errors when a V channel has a non-zero mean at 16K–32K keys. `test-backend-ops` FLASH_ATTN_EXT cases use zero-mean random V, so they would not catch it. Phase 0 needs a case that would.

## Design (manager decided, do not change)
1. Private llama.cpp clone: `git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp third_party/llama.cpp` inside your worktree (gitignored), checkout the commit in `bench/LLAMA_COMMIT`.
2. **Patch 0003** `third_party/patches/0003-tests-biased-V-flash-attention-cases.patch` (against the pin, applies after 0001–0002, touches only `tests/test-backend-ops.cpp`): a `v_bias` float parameter on the FLASH_ATTN_EXT test case (default 0 keeps every existing case and its printed name unchanged; check `vars()` so the name only gains `v_bias=` when non-zero). When non-zero, after the normal initialisation add `v_bias` to V channels 0..7 of every head and key (handle F16 and quantised V types: initialise as llama.cpp already does, then write the biased data the way the test framework writes tensor data). New cases in `make_test_cases_eval`, only F16 K/V: head dim 128 and 256, GQA 4 (4 Q heads, 1 KV head), `v_bias = 2.0`, KV length 16384 and 32768, Q columns 1, 4, 16, 512, no mask except the one llama.cpp requires, `max_bias = 0`, `logit_softcap = 0`. Error metric and threshold: the test case's existing ones (do not change them). The total CPU reference cost must stay reasonable: measure the run time of just these cases on the CPU backend (`-b CPU` is not a comparison; time a GPU run, which includes the CPU reference) and say how long it took.
3. `bench/phase0.sh`: in the `test` step, after the full FLASH_ATTN_EXT run, run the biased-V cases alone into `fa-vbias-test-$dev.txt` with a `-p` filter on `v_bias` (check that `test-backend-ops` at the pin supports `-p`; if not, find its equivalent and say so). Phase 0 builds apply patch 0003 to the tree it builds: add a `PATCHES` variable (default `third_party/patches/0003-*.patch`) applied with `git apply` before the build if not already applied (`git apply --check -R` to detect), logged. Keep `DRYRUN`/fixture behaviour working and extend the existing dry-run ctest for the new step.
4. `bench/patches/run.sh`: every variant also gets 0003 (it only changes tests), so variants stay comparable; the variant list and names do not change.
5. Builds in your clone (Release, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="52;60" -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF`, target `test-backend-ops`): (a) upstream + 0003, (b) 0001+0002+0003. Both must build for sm_60 (report warnings that mention sm_60). Then on Maxwell, under the GPU lock, for (a) and (b): `test-backend-ops test -o FLASH_ATTN_EXT`, `-o MUL_MAT`, `-o CPY`, `-o GET_ROWS`, and the biased-V cases. Save each output under `bench/results/maxwell-patches-<date>/` (text only, gzip anything > 1 MB) with a README listing exact commands, commit of llama.cpp, nvcc version, GPU name.
6. Update `third_party/patches/README.md`: add the 0003 row (provenance: this repo, test-only) and fill the "status on P100" column with "builds for sm_60; Maxwell correctness <summary>; P100 not run". Facts only.

## Files you may touch
- `third_party/patches/0003-tests-biased-V-flash-attention-cases.patch` (new), `third_party/patches/README.md`.
- `bench/phase0.sh`, `bench/patches/run.sh`, `tests/fixtures/shims/*`, `tests/CMakeLists.txt` (dry-run test updates only).
- `bench/results/maxwell-patches-<date>/` (new).
Nothing else. Do not modify 0001/0002. Do not write analysis into docs beyond the README facts.

## Must pass before you finish
- `git -C third_party/llama.cpp apply --check` of 0001, 0002, 0003 in order on a clean pin (paste result)
- both builds succeed (paste the last line of each)
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` in the worktree (all pass)
- `shellcheck -x bench/phase0.sh bench/patches/run.sh bin/*.sh`, `bin/check-forbidden.sh`
- `DRYRUN=1` runs of both scripts (paste the planned commands for the new step)

## Done means
Patch 0003 exists and applies; both trees build for sm_60; Phase 0 runs the biased-V cases; Maxwell results are saved (or handed over with exact commands). Final report: files changed; each command with its result line; per-op pass/fail counts for (a) and (b) on Maxwell and any failure lines verbatim; the biased-V case names and their Maxwell result and run time; what was NOT verified (anything on a P100). Do not interpret or decide whether to ship.
