# Activation-range MUL_MAT cases on Maxwell (2026-10-16)

**Maxwell Quadro M2200 (sm_52), driver 580.178.04, CUDA 12.8. These are not P100 results; P100: not run.**
Run by the manager outside the worker sandbox, from ticket p15 (`third_party/patches/bench-only/0003-tests-mul_mat-activation-range.patch`).

Trees (private clones at `79e2e74eb`, `-DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=52;60'`, Release, `test-backend-ops` only):
- **upstream**: the pin + bench-only 0003.
- **mmvh**: series 0001–0006 + bench-only 0003. On Maxwell, mmvh runs only when forced (`GGML_CUDA_FORCE_MMVH=1`), and then uses its FP32 emulation, not sm_60's native half2.

Command, per run (outputs with ANSI colour stripped: `upstream.range.txt`, `mmvh.range.txt`, `mmvh-nommvh.range.txt`):

```
flock /tmp/p100-gpu.lock env -u GGML_CUDA_DISABLE_MMVH -u GGML_CUDA_FORCE_MMVH -u GGML_CUDA_F32_REF [GGML_CUDA_FORCE_MMVH=1 | GGML_CUDA_DISABLE_MMVH=1] \
  LD_LIBRARY_PATH=<build>/bin:tools/cuda-12.8/targets/x86_64-linux/lib <build>/bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p src1_scale
```

| run | result | rc |
| --- | --- | --- |
| upstream | 48/80 | 1 |
| mmvh, `GGML_CUDA_FORCE_MMVH=1` | 80/80 | 0 |
| mmvh, `GGML_CUDA_DISABLE_MMVH=1` | 48/80 (same per-case result as upstream) | 1 |

No abort. The pre-existing q4_K abort (non-contiguous K-quant src0, `ggml-cuda.cu:1462`) did not trigger, because the range cases are contiguous. No filter was needed.

Pass counts by type and scale (10 cases each: n = 1, 2, 4, 8, 16 × m = 128, 4096; K = 1024). Q4_0 and Q8_0 have no 1e-5 cases because the CPU reference's Q8_0 vec_dot FP16 d underflows there.

| type @ scale | upstream | mmvh forced | mmvh disabled |
| --- | --- | --- | --- |
| q4_0 @ 1e5 | 2/10 (NaN at n ≤ 8) | 10/10 | 2/10 |
| q8_0 @ 1e5 | 10/10 | 10/10 | 10/10 |
| q4_K @ 1e5 | 10/10 | 10/10 | 10/10 |
| q4_K @ 1e-5 | 2/10 (NMSE 0.075–0.089 > 5e-4 at n ≤ 8) | 10/10 | 2/10 |
| q5_K @ 1e5 | 10/10 | 10/10 | 10/10 |
| q5_K @ 1e-5 | 2/10 (NMSE 0.075–0.089 > 5e-4 at n ≤ 8) | 10/10 | 2/10 |
| q6_K @ 1e5 | 10/10 | 10/10 | 10/10 |
| q6_K @ 1e-5 | 2/10 (NMSE 0.075–0.089 > 5e-4 at n ≤ 8) | 10/10 | 2/10 |

Every upstream failure is at n ≤ 8 (the MMVQ/q8_1 path). Every n = 16 case passes in all three runs.
These are facts from the logs, with two limits:
- The ticket expected all 1e5 cases to fail on q8_1. On Maxwell, only q4_0 failed at 1e5. The K-quants failed at 1e-5 instead.
- The n = 16 rows do not predict the P100. At the pin, `ggml_cuda_mul_mat_cublas` uses F16 compute for quantised src0 only when `fast_fp16_hardware_available(cc)`. That is false on sm_52, so these n = 16 cases ran FP32 SGEMM. On sm_60 it is true, so src1 is converted to FP16 (max 65504). This is a code reading, not a measurement.

## Per case

| type | scale | n | m | upstream | mmvh (forced) | mmvh-nommvh |
|---|---|---|---|---|---|---|
| q4_0 | 100000 | 1 | 128 | FAIL | OK | FAIL |
| q4_0 | 100000 | 1 | 4096 | FAIL | OK | FAIL |
| q4_0 | 100000 | 2 | 128 | FAIL | OK | FAIL |
| q4_0 | 100000 | 2 | 4096 | FAIL | OK | FAIL |
| q4_0 | 100000 | 4 | 128 | FAIL | OK | FAIL |
| q4_0 | 100000 | 4 | 4096 | FAIL | OK | FAIL |
| q4_0 | 100000 | 8 | 128 | FAIL | OK | FAIL |
| q4_0 | 100000 | 8 | 4096 | FAIL | OK | FAIL |
| q4_0 | 100000 | 16 | 128 | OK | OK | OK |
| q4_0 | 100000 | 16 | 4096 | OK | OK | OK |
| q8_0 | 100000 | 1 | 128 | OK | OK | OK |
| q8_0 | 100000 | 1 | 4096 | OK | OK | OK |
| q8_0 | 100000 | 2 | 128 | OK | OK | OK |
| q8_0 | 100000 | 2 | 4096 | OK | OK | OK |
| q8_0 | 100000 | 4 | 128 | OK | OK | OK |
| q8_0 | 100000 | 4 | 4096 | OK | OK | OK |
| q8_0 | 100000 | 8 | 128 | OK | OK | OK |
| q8_0 | 100000 | 8 | 4096 | OK | OK | OK |
| q8_0 | 100000 | 16 | 128 | OK | OK | OK |
| q8_0 | 100000 | 16 | 4096 | OK | OK | OK |
| q4_K | 100000 | 1 | 128 | OK | OK | OK |
| q4_K | 100000 | 1 | 4096 | OK | OK | OK |
| q4_K | 100000 | 2 | 128 | OK | OK | OK |
| q4_K | 100000 | 2 | 4096 | OK | OK | OK |
| q4_K | 100000 | 4 | 128 | OK | OK | OK |
| q4_K | 100000 | 4 | 4096 | OK | OK | OK |
| q4_K | 100000 | 8 | 128 | OK | OK | OK |
| q4_K | 100000 | 8 | 4096 | OK | OK | OK |
| q4_K | 100000 | 16 | 128 | OK | OK | OK |
| q4_K | 100000 | 16 | 4096 | OK | OK | OK |
| q4_K | 1e-05 | 1 | 128 | FAIL | OK | FAIL |
| q4_K | 1e-05 | 1 | 4096 | FAIL | OK | FAIL |
| q4_K | 1e-05 | 2 | 128 | FAIL | OK | FAIL |
| q4_K | 1e-05 | 2 | 4096 | FAIL | OK | FAIL |
| q4_K | 1e-05 | 4 | 128 | FAIL | OK | FAIL |
| q4_K | 1e-05 | 4 | 4096 | FAIL | OK | FAIL |
| q4_K | 1e-05 | 8 | 128 | FAIL | OK | FAIL |
| q4_K | 1e-05 | 8 | 4096 | FAIL | OK | FAIL |
| q4_K | 1e-05 | 16 | 128 | OK | OK | OK |
| q4_K | 1e-05 | 16 | 4096 | OK | OK | OK |
| q5_K | 100000 | 1 | 128 | OK | OK | OK |
| q5_K | 100000 | 1 | 4096 | OK | OK | OK |
| q5_K | 100000 | 2 | 128 | OK | OK | OK |
| q5_K | 100000 | 2 | 4096 | OK | OK | OK |
| q5_K | 100000 | 4 | 128 | OK | OK | OK |
| q5_K | 100000 | 4 | 4096 | OK | OK | OK |
| q5_K | 100000 | 8 | 128 | OK | OK | OK |
| q5_K | 100000 | 8 | 4096 | OK | OK | OK |
| q5_K | 100000 | 16 | 128 | OK | OK | OK |
| q5_K | 100000 | 16 | 4096 | OK | OK | OK |
| q5_K | 1e-05 | 1 | 128 | FAIL | OK | FAIL |
| q5_K | 1e-05 | 1 | 4096 | FAIL | OK | FAIL |
| q5_K | 1e-05 | 2 | 128 | FAIL | OK | FAIL |
| q5_K | 1e-05 | 2 | 4096 | FAIL | OK | FAIL |
| q5_K | 1e-05 | 4 | 128 | FAIL | OK | FAIL |
| q5_K | 1e-05 | 4 | 4096 | FAIL | OK | FAIL |
| q5_K | 1e-05 | 8 | 128 | FAIL | OK | FAIL |
| q5_K | 1e-05 | 8 | 4096 | FAIL | OK | FAIL |
| q5_K | 1e-05 | 16 | 128 | OK | OK | OK |
| q5_K | 1e-05 | 16 | 4096 | OK | OK | OK |
| q6_K | 100000 | 1 | 128 | OK | OK | OK |
| q6_K | 100000 | 1 | 4096 | OK | OK | OK |
| q6_K | 100000 | 2 | 128 | OK | OK | OK |
| q6_K | 100000 | 2 | 4096 | OK | OK | OK |
| q6_K | 100000 | 4 | 128 | OK | OK | OK |
| q6_K | 100000 | 4 | 4096 | OK | OK | OK |
| q6_K | 100000 | 8 | 128 | OK | OK | OK |
| q6_K | 100000 | 8 | 4096 | OK | OK | OK |
| q6_K | 100000 | 16 | 128 | OK | OK | OK |
| q6_K | 100000 | 16 | 4096 | OK | OK | OK |
| q6_K | 1e-05 | 1 | 128 | FAIL | OK | FAIL |
| q6_K | 1e-05 | 1 | 4096 | FAIL | OK | FAIL |
| q6_K | 1e-05 | 2 | 128 | FAIL | OK | FAIL |
| q6_K | 1e-05 | 2 | 4096 | FAIL | OK | FAIL |
| q6_K | 1e-05 | 4 | 128 | FAIL | OK | FAIL |
| q6_K | 1e-05 | 4 | 4096 | FAIL | OK | FAIL |
| q6_K | 1e-05 | 8 | 128 | FAIL | OK | FAIL |
| q6_K | 1e-05 | 8 | 4096 | FAIL | OK | FAIL |
| q6_K | 1e-05 | 16 | 128 | OK | OK | OK |
| q6_K | 1e-05 | 16 | 4096 | OK | OK | OK |
