# Ticket p11-ab-variants: add patches 0004 (FA VKQ flush) and 0005 (mmvh) to the P100 A/B kit

Your current directory is a git worktree of the P100 repo (branch `p11-ab-variants`, based on `main`). Work only here.

## Rules
- Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;`. Never install anything. Build into `build/`.
- Do NOT commit or run git commands that write in the P100 repo. The manager commits.
- Never invent numbers; there is no P100 here. Everything is dry-run/fixture tested.
- No network. Do not read .env, tokens or model weights. Stay inside "Files you may touch".

## Background
`bench/patches/run.sh` builds one llama.cpp tree per variant and runs correctness, `test-backend-ops perf -o FLASH_ATTN_EXT` rounds and llama-bench rounds, interleaved per round (INV-PERF-004). Today: `upstream` (0003), `conv` (0001+0003), `conv8` (0001–0003); `npatch`/`patches_for` assume variant = first N patches plus 0003. Two new patches need day-one A/B on the P100:
- `0004` flushes the TILE FA FP16 output accumulator to FP32 per KV tile (`docs/research/error-budget-split-2026-10-10.txt`): question = does it fix the biased-V cases on the P100, and what does it cost in FA kernel time and llama-bench pp/tg.
- `0005` mmvh, the half2 decode mat-vec (`docs/design/H1-pascal-decode-matvec.md`, its "Accept if" section): question = MUL_MAT correctness and perf at n = 1..8 per type, llama-bench tg, and perplexity. It has a runtime switch `GGML_CUDA_DISABLE_MMVH` (presence-checked) for a same-build A/B.

## Design (manager decided)
1. Variants become explicit patch lists (replace `npatch`): `upstream` = 0003; `conv` = 0001, 0003; `conv8` = 0001, 0002, 0003; `flush` = 0001, 0002, 0003, 0004; `mmvh` = 0001, 0002, 0003, 0005; `all` = 0001–0005. Default `VARIANTS="upstream conv8 flush mmvh"`. Keep the "exactly one file per ordinal" check.
2. Correctness step: unchanged per variant (full FA, biased-V, MUL_MAT). For variants containing 0005, also run MUL_MAT once more with `GGML_CUDA_DISABLE_MMVH=1` into a separate file (proves the switch and gives an in-build baseline).
3. New perf step `mmperf` (in default STEPS after `perf`): per round, per variant, `test-backend-ops perf -o MUL_MAT -b CUDA0 -p <filter> --output sql` with the filter restricted to `type_a` ∈ {q4_0, q8_0, q4_K, q5_K, q6_K}, `type_b=f32`, `n` 1..8, contiguous (reuse the regex from `bench/patches/mmvh-maxwell.sh` / `bench/results/maxwell-decode-2026-10-10/run-maxwell.sh`, extended to these types). For mmvh/all variants, also a second run with `GGML_CUDA_DISABLE_MMVH=1` in the same round, saved as `<variant>-nommvh`. Summaries via `bin/calibrate-decode.py --variant pcie16` where it fits, otherwise `bench/analyze.py tbo` with labels.
4. New optional step `ppl`: if `PPL_TEXT` (path to a text file, e.g. wikitext-2 test) and `MODELS` are set, build `llama-perplexity` too and run `-c 512 --chunks 32 -ngl 99 -fa on` per variant (and mmvh with/without the env switch); print final PPL ± error from each log in the summary. Skip with a log line otherwise.
5. llama-bench step: add `-n 128` decode runs at depth 0 and 4096 (`-p 0`) per model per variant per round, plus the existing kv runs; mmvh variants also with `GGML_CUDA_DISABLE_MMVH=1`.
6. Header comment documents variants, steps, env switches, and which ROADMAP question each answers. `DRYRUN=1` exercises every new step with the fixture shims (extend `tests/fixtures/shims/*` as needed, including a `llama-perplexity` shim printing a fixed FAKE line), and the existing dry-run ctest is extended to assert the new planned commands appear.
7. `bench/BASELINE.md` gets a short "Day-one A/B" paragraph: the command line to run, expected runtime order of magnitude per step (state it as an estimate from case counts, labelled as such), and which files answer which question. Facts only.

## Files you may touch
`bench/patches/run.sh`, `tests/fixtures/shims/*`, `tests/CMakeLists.txt` (dry-run test only), `bench/BASELINE.md` (the new paragraph). Nothing else.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build -R "dry|patches|phase0" --output-on-failure` (all pass), then the full `ctest --test-dir build`
- `shellcheck -x bench/patches/run.sh bench/phase0.sh bin/*.sh`, `bin/check-forbidden.sh`
- `DRYRUN=1 FORCE=1 OUT=<tmp> bench/patches/run.sh` with default variants and all steps (paste the planned-command log)

## Done means
Final report: files changed; each command with its result line; the dry-run log (verbatim, trimmed to the new steps); what was NOT verified (everything on a P100). Do not interpret.
