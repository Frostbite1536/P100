# CPU backend: F16 V attention output accumulates in FP16 (measured, 2026-10-10)

Host: Intel Xeon E3-1505M v6, 8 threads, CPU-only build (`-DGGML_CUDA=OFF`, Release) of llama.cpp
`79e2e74` + patch 0003. No GPU involved.

Question: §2 of `docs/research/pre-hardware-findings-2-2026-10-10.md` read from the code that **normal** (non-reference) CPU
inference accumulates an F16 V in FP16 (`VKQ16`, `ggml_vec_mad_f16`, `ops.cpp` ~8993-9007) whenever the tiled
path is not taken (fewer than 64 query rows), including the decode split-KV path (`ops.cpp:9490`, chunks of
`nek1/nth` keys). With 0003, `test-backend-ops -b CPU` compares exactly that path against the CPU reference
mode, which 0003 makes accumulate in FP32.

Command (both builds): `test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p v_bias` (the 16 biased-V cases of 0003:
V +2 on channels 0..7, D 128/256, 16K/32K keys, nb 1/4/16/512, F16 K/V). Threshold is the test's own NMSE 5e-4.

| build | result |
| --- | --- |
| pin + 0003 (normal CPU path unchanged from upstream) | **4/16 pass** (`cpu-vbias.txt`) |
| same + FP16 branch disabled (`if (false && v->type == GGML_TYPE_F16 && !params->use_ref)` at the three sites) | **16/16 pass** (`cpu-vbias-fp32acc.txt`) |

NMSE in the first build (D 128; D 256 within 5%):

| keys | nb 1 (split-KV, 8 threads) | nb 4 | nb 16 | nb 512 (tiled path) |
| --- | --- | --- | --- | --- |
| 16,384 | 0.022 | 2.39 | 2.22 | pass |
| 32,768 | 0.43 | 11.6 | 12.1 | pass |

The FP16 accumulator is the only difference between the builds, so it causes these failures. Not explained:
why nb 4/16 exceed NMSE 1 (an FP16 sum that only stalls would make the output too small, giving NMSE <= 1).
These are stress inputs (V mean large relative to its spread); real-model impact is not measured (no models on this
machine). Reporting upstream is the owner's decision (llama.cpp does not accept AI-written issue/PR text).
