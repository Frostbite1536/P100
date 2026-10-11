# Ticket p7c-cpu-ref-fp32: make the CPU reference accumulate FA output in FP32 (patch 0003 follow-up)

Your current directory is the p7a worktree (branch `p7a-patches-biasv`) with p7a's uncommitted changes, its private llama.cpp clone and builds. All p7a rules apply (sm_60 only, venv PATH prefix, no git writes in the P100 repo, no tolerance changes, no invented numbers, GPU commands under `flock /tmp/p100-gpu.lock`, stay in file scope).

## What the manager found
On the Maxwell M2200 (whose CUDA FA kernels accumulate in FP32), **all 16 biased-V cases of patch 0003 FAIL** in both builds: NMSE 0.36 at 16K keys and 0.60 at 32K, independent of `nb` and head size. Cause (read the code to confirm, cite lines): `test-backend-ops` puts the CPU backend into reference mode (`ggml_backend_cpu_set_use_ref`, `tests/test-backend-ops.cpp:~12647`), so `ggml_compute_forward_flash_attn_ext_f16` always takes the vec path `ggml_compute_forward_flash_attn_ext_f16_one_chunk` (`ggml/src/ggml-cpu/ops.cpp:~9485-9568`), which for an F16 V accumulates the output in **FP16** (`VKQ16`, `ggml_vec_mad_f16`, `ops.cpp:~8992-9006`) over all keys. With V ≈ 2 and near-uniform attention the FP16 sum stalls once 2·P falls below half an ulp, so the CPU *reference* is the inaccurate side. The cases therefore cannot judge a GPU.

## Design (manager decided)
1. Extend patch `third_party/patches/0003-tests-biased-V-flash-attention-cases.patch` (keep its name; regenerate it with `git format-patch` as one commit, or two commits in one file if cleaner — say which) so that **in reference mode only** (`params->use_ref`), the vec path accumulates V in FP32 for an F16 V: take the existing non-F16 branch (convert V to FP32, `ggml_vec_mad_f32` into `VKQ32`) instead of `VKQ16`. Non-reference CPU behaviour must not change (state how the diff guarantees it). Update the patch body: test-only; why (numbers above).
2. Rebuild both trees in your clone ((a) upstream+0003, (b) 0001+0002+0003; `52;60`, target `test-backend-ops`), check `git apply --check` order 0001→0002→0003 on a clean pin again, and update `bench/results/maxwell-patches-2026-10-10/gpu-commands.txt` if paths changed. Keep the previous Maxwell outputs; the manager will re-run the biased-V and full FA commands into new files with a `-cpuref32` suffix (add those commands to `gpu-commands.txt`).
3. Static evidence in your report: a short arithmetic check of the stall (FP16 ulp at the stalled value vs 2·P) for 16K and 32K keys, uniform attention, so the failure size is explained, not just asserted.

## Files you may touch
`third_party/patches/0003-*.patch`, `third_party/patches/README.md` (the 0003 row), `bench/results/maxwell-patches-2026-10-10/gpu-commands.txt` and `README.md` there. Nothing else.

## Must pass before you finish
- `git apply --check` of 0001, 0002, 0003 in order on a clean pin
- both builds succeed
- `shellcheck -x bench/phase0.sh bench/patches/run.sh bin/*.sh`, `bin/check-forbidden.sh`, `ctest --test-dir build` in the worktree

## Done means
Final report: files changed; each command with its result line; the new ops.cpp hunk verbatim; the stall arithmetic; exact manager commands. Do not interpret beyond the arithmetic.
