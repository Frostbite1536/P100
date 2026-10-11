# Maxwell M2200 decode calibration, 2026-10-10

This directory records **Maxwell M2200** work only. No P100 measurements exist.
CUDA initialization in the sandbox failed. Three locked benchmark attempts each
returned exit 0 but produced only the SQL schema, with **zero CUDA0 measurement
rows**. Their stderr records the initialization failure. Clock queries returned
exit 9 (the outer attempted measurement shell also returned 9 from its final
clock query). Raw SQL, stderr, statuses and boundary clock failures are saved.
No median timings or round-to-round spread can be calculated:


```text
cudaGetDeviceProperties: no CUDA-capable device is detected
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

Device parameters, hardware timings, fitted Maxwell efficiencies and round-to-round
spread are unavailable. `devprops.json` is empty because the failed probe produced
no stdout. The Maxwell prediction/calibration text files contain errors, not predictions.
`calibration-SYNTHETIC.txt` is exclusively a parsing/arithmetic test on hand-written
data; its times and fitted parameters are **not hardware results**.

The private, unpatched llama.cpp clone is pinned to
`79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (`bench/LLAMA_COMMIT`).
Compilation is Release, CUDA ON, sm_52, CUDA compiler
`/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc`, LLAMA_CURL OFF.
`nvcc-version.txt` records CUDA 12.8, V12.8.93.
The full source compilation succeeded. The first final link could not locate
indirect libcudart/libcublas dependencies; a build retry with the existing toolkit
library directory on `LD_LIBRARY_PATH` succeeded (exit 0, `[1/1] Linking CXX executable
bin/test-backend-ops`, see `llama-link-retry.log`). No compiler options or llama.cpp
source changed. The handover exports the same library path for linking and runtime:

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; export LD_LIBRARY_PATH=/home/p51/Github/P100/tools/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}; cmake --build build/llama-sm52 --target test-backend-ops -j 8
```

## Files changed

- `bin/vecdot-sass.sh`: same classifier, sm_52 added; prefer the private clone.
- `docs/research/vecdot-sass-2026-10-10.csv`: nine sm_52 rows appended. The original
  18 sm_60/sm_61 rows compare byte-for-byte equal with `diff -u` (exit 0).
- `bin/decode-model.py`: measured-input Maxwell M2200 variant, sm_52 selection,
  `--shape m,k`, sm_61 reference retained; existing P100 text and JSON unchanged.
- `bin/calibrate-decode.py`: stdlib SQL parser, decode filtering, per-run timings,
  weight GB/s, model bounds/times, median efficiency fit, changed bounds, round spread,
  and derived median SQL export.
- `bench/micro/devprops.cu`: CUDA runtime property probe, no kernels or weights.
- `tests/decode_model_check.py`: additional shape/Maxwell and synthetic calibration checks.
- `tests/fixtures/tbo-perf-mulmat-synthetic.sql`: first line explicitly marks synthetic data.
- `tests/CMakeLists.txt`: `calibrate_decode` registered.
- This results directory: commands, failure evidence, toolchain/build/test logs and outputs.

All builds stay in this worktree's `build/`, including `build/llama-sm52` and
`build/devprops`. The clone is gitignored scratch; no llama.cpp source was changed.
No commit, push, installation, network, token, .env or model-weight access was used.

## Device parameters and sources

No measured value is available in this sandbox. The implementation deliberately has
no guessed Maxwell defaults. Run the probe in the handover script to fill these in.

| Parameter | Value here | Source to capture |
| --- | --- | --- |
| GPU name, compute capability | unavailable | `cudaGetDeviceProperties`, `nvidia-smi` |
| SM count | unavailable | `cudaDeviceProp.multiProcessorCount` |
| Maximum SM clock (kHz) | unavailable | `cudaDeviceProp.clockRate`, cross-check `nvidia-smi -q -d CLOCK` |
| Memory bus width (bits) | unavailable | `cudaDeviceProp.memoryBusWidth` |
| Memory clock (kHz) | unavailable | `cudaDeviceProp.memoryClockRate`, cross-check `nvidia-smi -q -d CLOCK` |
| Lanes per SM | 128 | sm_52 architectural assumption specified in the ticket; not a device measurement |
| Theoretical GDDR GB/s | unavailable | `(bus_bits / 8) * 2 * memory_clock_khz / 1e6` |
| Fitted `bw_eff`, `issue_eff` | unavailable | median measured/predicted ratios by **initial** bound; efficiency = initial efficiency / ratio |

## Exact handover

From this worktree, execute:

```bash
export PATH=/home/p51/Github/P100/.venv/bin:$PATH; bench/results/maxwell-decode-2026-10-10/run-maxwell.sh
```

The script checks the clone's commit and clean tracked source, resumes/configures
the build, captures device properties and clock details, and holds
`flock /tmp/p100-gpu.lock` for the whole three-round session. It saves raw
`round-{1,2,3}.sql`, stderr, exit status, clock/temperature/power readings before
the first round and after every round (including between rounds).
It runs exactly `test-backend-ops perf -o MUL_MAT -b CUDA0 --output sql`, with
`-p` selecting modeled quantised types, n=1..8 and contiguous decode shapes.
The regex and full build/measurement commands are in `run-maxwell.sh`.

Upstream's 4096-by-14336 grid includes n=1,2,3,4,5,8; n=6,7 are absent.
Additional Q4_K cases have m=4096,6144,6272,14336 and k=4096.
Unmodeled quantisation types, f16, non-f32 activations, permuted/padded inputs,
extra tensor batches/repeats, and multiple outputs are excluded.

Then the script saves `median.sql`, `uncalibrated.txt`, `calibrated-median.txt`,
`calibrated.txt`, and `maxwell-prediction-batch{1..8}.txt`. `calibrated.txt` fits
the **median time per selected row** while retaining min/max/spread from the three
rounds. `calibrated-median.txt` uses the same medians via the exported SQL file.
The exported `n_runs=1` is a placeholder; SQL `time_us` is already us/run in both
raw and derived files. No further division by `n_runs` is performed.

Spread is reported per row as min/max us/run and `100*(max-min)/median` percent.
Different selected row sets or commits across rounds are rejected.
Repeated identical upstream cases within one round use their median first.
Fits are a single pass grouped by the initial bound; missing bound groups retain
their initial efficiency and report n/a. Fits above 1 are reported without clipping.

## Limitations to carry with measurements

- Laptop clock/power/thermal throttling may vary within and between rounds.
  Boundary clock snapshots cannot establish steady clocks throughout a round.
- Maxwell sm_52 and Pascal sm_60 differ in ISA/scheduling; the regenerated static
  ALU counts are not identical. Wrapper overhead and the classifier's treatment
  of MOV/DEPBAR are retained; static counts do not measure instruction throughput.
- At the pinned commit, both the device and host `get_device_table_id` in
  `ggml/src/ggml-cuda/mmvq.cu` choose `MMVQ_PARAMETERS_GENERIC` (table 0) for
  Maxwell sm_52 and P100 sm_60. The TURING table applies to Volta through pre-Ampere,
  not either of these GPUs. A shared table does not verify every launch/occupancy
  behavior or every other architecture-dependent branch on hardware.
- The model ignores activations, activation quantisation, output writes, launch
  overhead, attention and KV cache. Achieved GB/s here counts only one weight read
  per batch step, not llama.cpp's generic reported bandwidth field.
- Nothing was run, timed, calibrated or validated on a P100. The sandbox did not
  verify the Maxwell crossover or any hardware efficiency/spread.

## Command results and verbatim outputs

Additional result lines and the requested outputs are appended below.
<!-- captured-results -->

Every shell invocation starts with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;`.
The commands below omit that repeated prefix for readability; `run-maxwell.sh`
contains it in both its outer shell and the locked measurement shell.

| Command | Result |
| --- | --- |
| `git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp third_party/llama.cpp` | exit 0, local clone complete |
| `git -C third_party/llama.cpp checkout "$(cat bench/LLAMA_COMMIT)"` | exit 0, HEAD at 79e2e74eb |
| `git -C third_party/llama.cpp status --short --untracked-files=no` | empty; tracked source clean |
| `cmake -S third_party/llama.cpp -B build/llama-sm52 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52 -DCMAKE_CUDA_COMPILER=/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -DLLAMA_CURL=OFF` | exit 0, configured with CUDA 12.8.93, architectures=52 |
| `cmake --build build/llama-sm52 --target test-backend-ops -j 8` with the toolkit library directory on `LD_LIBRARY_PATH` | exit 0, final executable linked; earlier -j 2/-j 4 builds interrupted and resumed; first link failure resolved by the library-path export |
| `bin/vecdot-sass.sh` | exit 0, wrote 27 rows |
| `head -19 docs/research/vecdot-sass-2026-10-10.csv > build/vecdot-existing.csv; diff -u build/vecdot-original.csv build/vecdot-existing.csv` | exit 0, no differences in original sm_60/sm_61 rows |
| `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` | exit 0, 100% tests passed out of 29; 174.00 seconds; CPU reference build plus compile-only CUDA checks |
| `bin/check-forbidden.sh` | exit 0, `ok` |
| `shellcheck -x bin/*.sh` | exit 0, no diagnostics |
| `shellcheck -x bench/results/maxwell-decode-2026-10-10/run-maxwell.sh` | exit 0, no diagnostics |
| `git diff --check` | exit 0, no diagnostics |
| `python bin/decode-model.py --model llama3.1-8b --all-types --batch 1` | exit 0; text and JSON each byte-for-byte equal to the committed script; rounded numbers agree with research findings section 1 |
| Original script versus new script: custom 1.5B/all-types/batch8/pcie12/bw_eff=0.5/issue_eff=0.25 and llama3.1-8b/all-types/batch4/sxm2 | both text outputs byte-for-byte equal |
| `python bin/decode-model.py --csv build/vecdot-original.csv --all-types --batch 1` | exit 0, original 18-row CSV remains accepted and gives identical output |
| `python tests/decode_model_check.py --calibrate` | exit 0, SYNTHETIC filtering, per-run times, GB/s, median fits, round spread and bound changes passed |
| `python bin/calibrate-decode.py tests/fixtures/tbo-perf-mulmat-synthetic.sql` | exit 0; SYNTHETIC output below, no hardware measurements |
| `/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc -O2 -std=c++17 -arch=sm_52 bench/micro/devprops.cu -o build/devprops` | exit 0; expected deprecated-architecture warning |
| `flock /tmp/p100-gpu.lock build/devprops` | exit 1, no CUDA-capable device detected |
| `flock /tmp/p100-gpu.lock nvidia-smi -q -d CLOCK` | driver unavailable; error saved in nvidia-clocks.txt |
| `flock /tmp/p100-gpu.lock nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv` | driver unavailable; error saved in clocks-probe.csv |
| `python bin/decode-model.py --variant m2200 --shape 4096,14336 --all-types --batch 1` | exit 2, measured device parameters unavailable |
| Same prediction with `--batch 4` | exit 2, measured device parameters unavailable |
| `python bin/calibrate-decode.py --variant m2200 --no-fit bench/results/maxwell-decode-2026-10-10/round-{1,2,3}.sql` | exit 2, measured device parameters unavailable |
| Same calibration without `--no-fit` | exit 2, measured device parameters unavailable |
| `flock /tmp/p100-gpu.lock bash` holding all three `test-backend-ops perf -o MUL_MAT -b CUDA0 -p <regex in run-maxwell.sh> --output sql` attempts and clock queries | TBO exits 0/0/0, CUDA initialization fails in every attempt, 0 measurement rows; final clock query/outer shell exit 9 |
| `load_round()` from the calibration tool on each raw SQL file | all three rejected: no supported contiguous quantised decode rows on CUDA0 |

Requested P100 **prediction** (unchanged; not hardware numbers):

`python bin/decode-model.py --model llama3.1-8b --all-types --batch 1`

```text
PREDICTIONS: datasheet peaks, uncalibrated; ignores activations, KV cache, attention and launch overhead
model=llama3.1-8b weights=7,504,658,432 batch=1 variant=pcie16 HBM=732 GB/s issue=4669.952000 Glane-instructions/s bw_eff=1 issue_eff=1
All mat-vec weights use the row's one type; embeddings excluded.
Weights read once per batch step; bytes/token = bytes/step / batch; tokens/s = batch / step time.
sm_61 counts use the P100 issue rate as a reference: "if dp4a existed".
f16: bandwidth-only ceiling; no measured ALU count.
Times in ms; step = max(bw, alu). GB uses decimal units.
type      GB/step  GB/token     bw_ms   alu60_ms bound60  step60_ms   tok60/s   alu61_ms bound61  step61_ms   tok61/s
q4_0     4.221370  4.221370  5.766899   6.779572     alu   6.779572   147.502   2.109200      bw   5.766899   173.403
q4_1     4.690412  4.690412  6.407666   6.026287      bw   6.407666   156.063   2.058981      bw   6.407666   156.063
q5_0     5.159453  5.159453  7.048433  11.500164     alu  11.500164    86.955   5.524096      bw   7.048433   141.876
q5_1     5.628494  5.628494  7.689199  11.198849     alu  11.198849    89.295   5.524096      bw   7.689199   130.053
q8_0     7.973700  7.973700 10.893032   6.277382      bw  10.893032    91.802   1.807886      bw  10.893032    91.802
q4_K     4.221370  4.221370  5.766899   8.819722     alu   8.819722   113.382   4.268620      bw   5.766899   173.403
q5_K     5.159453  5.159453  7.048433   9.937095     alu   9.937095   100.633   5.122344      bw   7.048433   141.876
q6_K     6.156165  6.156165  8.410062  12.328778     alu  12.328778    81.111   6.133002      bw   8.410062   118.905
iq4_xs   3.986850  3.986850  5.446516   9.045707     alu   9.045707   110.550   4.369058      bw   5.446516   183.604
f16     15.009317 15.009317 20.504531        n/a bw-only  20.504531    48.770        n/a bw-only  20.504531    48.770
```

Maxwell M2200 uncalibrated `--shape 4096,14336 --all-types --batch 1` (exit 2)

```text
usage: decode-model.py [-h] [--model {llama3.1-8b,custom}] [--weights WEIGHTS]
                       [--shape SHAPE]
                       [--type {q4_0,q4_1,q5_0,q5_1,q8_0,q4_K,q5_K,q6_K,iq4_xs,f16} | --all-types]
                       [--variant {pcie16,pcie12,sxm2,m2200}]
                       [--batch {1,2,3,4,5,6,7,8}] [--bw-eff BW_EFF]
                       [--issue-eff ISSUE_EFF] [--csv CSV]
                       [--device-props DEVICE_PROPS] [--json]
decode-model.py: error: Maxwell M2200 measured parameters unavailable at /home/p51/Github/worktrees/p100-p7b-decode-maxwell/bench/results/maxwell-decode-2026-10-10/devprops.json: Expecting value: line 1 column 1 (char 0); run build/devprops under flock /tmp/p100-gpu.lock and save its JSON, then pass --device-props <file>
```

Maxwell M2200 uncalibrated `--shape 4096,14336 --all-types --batch 4` (exit 2)

```text
usage: decode-model.py [-h] [--model {llama3.1-8b,custom}] [--weights WEIGHTS]
                       [--shape SHAPE]
                       [--type {q4_0,q4_1,q5_0,q5_1,q8_0,q4_K,q5_K,q6_K,iq4_xs,f16} | --all-types]
                       [--variant {pcie16,pcie12,sxm2,m2200}]
                       [--batch {1,2,3,4,5,6,7,8}] [--bw-eff BW_EFF]
                       [--issue-eff ISSUE_EFF] [--csv CSV]
                       [--device-props DEVICE_PROPS] [--json]
decode-model.py: error: Maxwell M2200 measured parameters unavailable at /home/p51/Github/worktrees/p100-p7b-decode-maxwell/bench/results/maxwell-decode-2026-10-10/devprops.json: Expecting value: line 1 column 1 (char 0); run build/devprops under flock /tmp/p100-gpu.lock and save its JSON, then pass --device-props <file>
```

Maxwell M2200 calibration `--no-fit` (exit 2; no uncalibrated hardware predictions)

```text
usage: calibrate-decode.py [-h] [--variant {pcie16,pcie12,sxm2,m2200}]
                           [--backend BACKEND] [--device-props DEVICE_PROPS]
                           [--csv CSV] [--bw-eff BW_EFF]
                           [--issue-eff ISSUE_EFF] [--no-fit]
                           [--median-sql MEDIAN_SQL]
                           sql [sql ...]
calibrate-decode.py: error: Maxwell M2200 measured parameters unavailable at /home/p51/Github/worktrees/p100-p7b-decode-maxwell/bench/results/maxwell-decode-2026-10-10/devprops.json: Expecting value: line 1 column 1 (char 0); run build/devprops under flock /tmp/p100-gpu.lock and save its JSON, then pass --device-props <file>
```

Maxwell M2200 calibration with fitting (exit 2; no fitted hardware predictions)

```text
usage: calibrate-decode.py [-h] [--variant {pcie16,pcie12,sxm2,m2200}]
                           [--backend BACKEND] [--device-props DEVICE_PROPS]
                           [--csv CSV] [--bw-eff BW_EFF]
                           [--issue-eff ISSUE_EFF] [--no-fit]
                           [--median-sql MEDIAN_SQL]
                           sql [sql ...]
calibrate-decode.py: error: Maxwell M2200 measured parameters unavailable at /home/p51/Github/worktrees/p100-p7b-decode-maxwell/bench/results/maxwell-decode-2026-10-10/devprops.json: Expecting value: line 1 column 1 (char 0); run build/devprops under flock /tmp/p100-gpu.lock and save its JSON, then pass --device-props <file>
```

SYNTHETIC parser/arithmetic calibration only, initial and fitted output verbatim

```text
P100 pcie16 prediction: backend=CUDA0 variant=pcie16 arch=sm_60 commit=synthetic selected=4 rounds=1
time_us is already per run; weight_GB/s counts weights once per batch; decimal GB.
Repeated identical cases within a round use their median; spread is (max-min)/median across rounds.
INITIAL bw_eff=1 issue_eff=1
type m k n measured_us/run weight_GB/s bound t_bw_us t_alu_us predicted_us ratio min_us max_us spread_pct rounds
q4_0 32 256 1 5.000000 0.921600 alu 0.006295 0.007401 0.007401 675.629630 5.000000 5.000000 0.000000 1
q4_1 32 256 1 2.000000 2.560000 bw 0.006995 0.006578 0.006995 285.937500 2.000000 2.000000 0.000000 1
q8_0 32 256 1 2.000000 4.352000 bw 0.011891 0.006852 0.011891 168.198529 2.000000 2.000000 0.000000 1
q8_0 32 256 4 10.000000 0.870400 alu 0.011891 0.027409 0.027409 364.840000 10.000000 10.000000 0.000000 1
FIT bw_eff: initial_bound=bw rows=2 median(measured/predicted)=227.068015
FIT issue_eff: initial_bound=alu rows=2 median(measured/predicted)=520.234815
FITTED bw_eff=0.00440396681 issue_eff=0.00192220892
type m k n measured_us/run weight_GB/s bound t_bw_us t_alu_us predicted_us ratio min_us max_us spread_pct rounds
q4_0 32 256 1 5.000000 0.921600 alu 1.429412 3.850000 3.850000 1.298701 5.000000 5.000000 0.000000 1
q4_1 32 256 1 2.000000 2.560000 alu 1.588235 3.422222 3.422222 0.584416 2.000000 2.000000 0.000000 1
q8_0 32 256 1 2.000000 4.352000 alu 2.700000 3.564815 3.564815 0.561039 2.000000 2.000000 0.000000 1
q8_0 32 256 4 10.000000 0.870400 alu 2.700000 14.259259 14.259259 0.701299 10.000000 10.000000 0.000000 1
BOUND CHANGES 2
q4_1 m=32 k=256 n=1: bw -> alu
q8_0 m=32 k=256 n=1: bw -> alu
```


Final measurement-attempt stderr (identical in all three rounds):

```text
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
```

The round-to-round spread is **unavailable for every row**, because no timing
rows were produced. `run-maxwell.sh` is the exact reproducible handover for a
manager with working CUDA device access; it includes the required library-path
export and refuses to fit missing/partial measurement rounds.

Stderr files use `.stderr.txt` so they are reviewable despite the repository
ignore rule for `*.err`. The handover uses these same filenames.

## Manager runs (2026-10-10)

The worker sandbox had no GPU; the manager ran `run-maxwell.sh` twice. Both runs: Quadro M2200 (8 SMs, 1036 MHz,
128-bit at 2754 MHz from `cudaGetDeviceProperties`), SM clock 1037 MHz during every round, 67–70 °C.

| run | host load | fitted bw_eff | fitted issue_eff | round spread (median / p90 / max) |
| --- | --- | --- | --- | --- |
| this folder | p7a llama.cpp CUDA build running (~16 nvcc processes) | 0.622 | 0.929 | large (several rows > 100%) |
| `../maxwell-decode-2026-10-10-r2` | another session's oneAPI build (load ~7) | 0.628 | 0.891 | 2.4% / 73% / 183% |

Rows with a large spread (mostly n = 5 and 8 in r2) disagree between runs and should not be used. Neither run
had an idle host.
