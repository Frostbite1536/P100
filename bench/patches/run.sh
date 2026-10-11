#!/bin/bash
# A/B the P100 conversion patches, cumulatively over third_party/patches/.
#   bench/patches/run.sh
#   MODELS="/m/llama.gguf /m/gemma.gguf" ROUNDS=5 bench/patches/run.sh
#   STEPS="test perf lb" bench/patches/run.sh   reuse prepared builds
#   DRYRUN=1 OUT=/tmp/patches STEPS="test perf lb" bench/patches/run.sh
#     use test-backend-ops / llama-bench from PATH; skip worktrees and compilation.
# Variants: upstream (0003), conv (0001+0003), conv8 (0001-0003).
# Results: bench/results/patches-<stamp>-<host>/; summary.md uses analyze.py tbo/lb.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
ROOT=$PWD
LLAMA_SRC=${LLAMA_SRC:-$ROOT/third_party/llama.cpp}
WORK=${WORK:-$ROOT/third_party/patches-ab}
VARIANTS=${VARIANTS:-"upstream conv conv8"}
STEPS=${STEPS:-"micro build test perf lb"}
DRYRUN=${DRYRUN:-0}
ROUNDS=${ROUNDS:-3}
LB_ROUNDS=${LB_ROUNDS:-2}
REPS=${REPS:-3}
PATCHES=$ROOT/third_party/patches
OUT=${OUT:-$ROOT/bench/results/patches-$(date +%Y%m%d-%H%M%S)-$(hostname -s 2>/dev/null || hostname)}
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
gpu_state() { nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader 2>/dev/null | head -1 || echo "n/a"; }
export CCACHE_BASEDIR=$WORK CCACHE_NOHASHDIR=1 CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$gpu" == *P100* || "${FORCE:-0}" == 1 ]] || { echo "GPU is '$gpu', not a P100 (FORCE=1 to override)" >&2; exit 2; }
ver=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
[ "$ver" = 12.8 ] || [ "${FORCE:-0}" = 1 ] || { echo "nvcc is '$ver'; CUDA 12.8 required (13 dropped sm_60; 12.9: llama.cpp #28416). FORCE=1 to override" >&2; exit 2; }

npatch() {
    case $1 in
        upstream) echo 0 ;; conv) echo 1 ;; conv8) echo 2 ;;
        *) echo "unknown variant $1" >&2; return 2 ;;
    esac
}
patches_for() {
    local n k
    n=$(npatch "$1") || return 2
    for ((k = 1; k <= n; k++)); do
        # Exactly one patch per ordinal, including in fixture mode.
        local files=("$PATCHES"/"$(printf '%04d' "$k")"-*.patch)
        [ ${#files[@]} = 1 ] && [ -f "${files[0]}" ] || { echo "missing or ambiguous patch $k" >&2; return 2; }
        printf '%s\n' "${files[@]}"
    done
    local tests=("$PATCHES"/0003-*.patch)
    [ ${#tests[@]} = 1 ] && [ -f "${tests[0]}" ] || { echo "missing or ambiguous patch 3" >&2; return 2; }
    printf '%s\n' "${tests[@]}"
}
for v in $VARIANTS; do patches_for "$v" > /dev/null; done

if [ "$DRYRUN" = 1 ]; then
    base="DRYRUN: no llama.cpp checkout or build; all tool outputs are FAKE fixtures"
    DRY_TBO=$(command -v test-backend-ops)
    DRY_LB=$(command -v llama-bench)
else
    base=$(git -C "$LLAMA_SRC" rev-parse HEAD)
fi
( set +e
  echo "llama.cpp base: $base (bench/LLAMA_COMMIT: $(cat "$ROOT/bench/LLAMA_COMMIT" 2>/dev/null))"
  echo "gpu: $gpu"
  nvidia-smi --query-gpu=driver_version,clocks.max.sm,power.limit --format=csv,noheader
  nvcc --version | tail -1
  echo "variants: $VARIANTS"; echo "rounds: $ROUNDS (llama-bench: $LB_ROUNDS x $REPS reps)"
  echo "patches:"; sha256sum "$PATCHES"/*.patch
) > "$OUT/env.txt" 2>&1 || true

if has micro; then
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: skipping microbench compilation and execution"
    else
        mkdir -p "$WORK"
        log "dequant microbench"
        if nvcc -O3 -std=c++17 -arch=sm_60 bench/micro/dequant_nc.cu -o "$WORK/dequant_nc" > "$OUT/dequant_nc.build.txt" 2>&1; then
            "$WORK/dequant_nc" 50 > "$OUT/dequant_nc.txt" 2>&1 || log "  microbench failed (see dequant_nc.txt)"
        else log "  microbench build failed (see dequant_nc.build.txt)"; fi
    fi
fi

# Local perf cases: same shapes and session for quantised K/V and the f16 reference (INV-PERF-004).
PERF_CASES='    for (int hs : {128, 256}) { for (int gqa : {4, 6, 8}) { for (int kv : {4096, 32768}) { for (int nb : {3, 4, 8, 16}) { for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) { test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 4, {gqa, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, t, t)); } } } } } // P100 patches harness'
PERF_FILTER='hsk=(128|256),hsv=(128|256),nh=4,nr23=\[[468],1\],kv=(4096|32768),nb=(3|4|8|16),.*type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0)'

# Select binaries per variant. Fixture mode deliberately never touches WORK.
tbo_for() { if [ "$DRYRUN" = 1 ]; then echo "$DRY_TBO"; else echo "$WORK/$1/build/bin/test-backend-ops"; fi; }
lb_for() { if [ "$DRYRUN" = 1 ]; then echo "$DRY_LB"; else echo "$WORK/$1/build/bin/llama-bench"; fi; }
built=()
if has build && [ "$DRYRUN" != 1 ]; then
    mkdir -p "$WORK"
    git -C "$LLAMA_SRC" worktree prune
fi
for v in $VARIANTS; do
    src=$WORK/$v
    if has build && [ "$DRYRUN" != 1 ]; then
        [ -d "$src" ] || git -C "$LLAMA_SRC" worktree add -q --detach "$src" "$base"
        git -C "$src" checkout -q -- .
        mapfile -t variant_patches < <(patches_for "$v")
        for p in "${variant_patches[@]}"; do
            git -C "$src" apply "$p" || { log "patch $p does not apply for $v"; exit 1; }
        done
        t=$src/tests/test-backend-ops.cpp
        anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
        [ "$(grep -cF "$anchor" "$t")" = 1 ] || { log "perf anchor not unique in $t"; exit 1; }
        L="$PERF_CASES" perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' "$t"
        git -C "$src" diff > "$OUT/$v.patch"
        log "build $v for sm_60"
        if ! { cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
                 -DCMAKE_CUDA_ARCHITECTURES=60 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF > "$OUT/$v.build.log" 2>&1 &&
               cmake --build "$src/build" --target test-backend-ops llama-bench >> "$OUT/$v.build.log" 2>&1; }; then
            log "  BUILD FAILED for $v"; tail -15 "$OUT/$v.build.log" | tee -a "$OUT/run.log"; continue
        fi
    elif has build; then
        while IFS= read -r p; do log "DRYRUN: git -C $src apply $p"; done < <(patches_for "$v")
        log "DRYRUN: skipping worktree preparation and build for $v"
    fi
    if { has test || has perf; } && [ ! -x "$(tbo_for "$v")" ]; then log "missing test-backend-ops for $v; run STEPS=build first"; exit 1; fi
    if has lb && [ ! -x "$(lb_for "$v")" ]; then log "missing llama-bench for $v; run STEPS=build first"; exit 1; fi
    built+=("$v")
    if has test; then
        for op in FLASH_ATTN_EXT MUL_MAT; do
            tag=fa; [ "$op" = MUL_MAT ] && tag=mulmat
            log "correctness $v (full $op CUDA0)"
            "$(tbo_for "$v")" test -b CUDA0 -o "$op" > "$OUT/$v.$tag-test-CUDA0.txt" 2>&1 && rc=0 || rc=$?
            log "  rc=$rc  OK=$(grep -c 'OK' "$OUT/$v.$tag-test-CUDA0.txt" || true)  FAIL=$(grep -c 'FAIL' "$OUT/$v.$tag-test-CUDA0.txt" || true)"
            if [ "$op" = FLASH_ATTN_EXT ]; then
                log "biased-V correctness: $(tbo_for "$v") test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias"
                "$(tbo_for "$v")" test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias > "$OUT/$v.fa-vbias-test-CUDA0.txt" 2>&1 && rc=0 || rc=$?
                log "  rc=$rc  OK=$(grep -c 'OK' "$OUT/$v.fa-vbias-test-CUDA0.txt" || true)  FAIL=$(grep -c 'FAIL' "$OUT/$v.fa-vbias-test-CUDA0.txt" || true)"
            fi
        done
    fi
done
[ ${#built[@]} -gt 0 ] || { log "no variant built"; exit 1; }

has perf || ROUNDS=0
for ((i = 1; i <= ROUNDS; i++)); do
    for v in "${built[@]}"; do
        log "kernel perf $v round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
        stdbuf -oL "$(tbo_for "$v")" perf -b CUDA0 -o FLASH_ATTN_EXT -p "$PERF_FILTER" \
            --output sql > "$OUT/$v.perf.r$i.sql" 2> "$OUT/$v.perf.r$i.err" || log "  perf run failed (see .err)"
    done
done

# MODELS is an optional space-separated override; otherwise use bench/models.txt.
models=()
if [ -n "${MODELS:-}" ]; then read -ra models <<< "$MODELS"
elif [ -f bench/models.txt ]; then
    while IFS= read -r m || [ -n "$m" ]; do
        [[ -z "$m" || "$m" == \#* ]] || models+=("$m")
    done < bench/models.txt
fi
has lb || models=()
for m in "${models[@]}"; do
    n=$(basename "$m" .gguf)
    for ((i = 1; i <= LB_ROUNDS; i++)); do
        for v in "${built[@]}"; do
            log "llama-bench kv $n $v round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
            "$(lb_for "$v")" -m "$m" -dev CUDA0 -ngl 99 -fa on -r "$REPS" -o jsonl \
                -ctk f16,q8_0,q4_0 -ctv f16,q8_0,q4_0 -p 8 -n 64 -d 16384 \
                > "$OUT/$v.$n.kv.r$i.jsonl" 2> "$OUT/$v.$n.kv.r$i.err" || log "  llama-bench failed"
        done
    done
done

{
    echo "# Patch series A/B ($(date +%F), $gpu)"; echo
    if [ "$DRYRUN" = 1 ]; then echo "FAKE fixture outputs only; no hardware measurements."; echo; fi
    if has test; then
        echo "## Correctness"; echo
        grep -E '^\[.*\] (correctness|  rc=)' "$OUT/run.log" | sed 's/^/    /' || true; echo
    fi
    if [ -s "$OUT/dequant_nc.txt" ]; then
        echo "## Conversion microbenchmark (bench/micro/dequant_nc.cu)"; echo
        sed 's/^/    /' "$OUT/dequant_nc.txt"; echo
    fi
    if [ "$ROUNDS" -gt 0 ]; then
        echo "## Kernel timing (median of $ROUNDS interleaved rounds)"; echo
        groups=()
        for v in "${built[@]}"; do
            g=""; for ((i = 1; i <= ROUNDS; i++)); do g+="${g:+,}$OUT/$v.perf.r$i.sql"; done
            groups+=("$g")
        done
        python3 "$ROOT/bench/analyze.py" tbo "${groups[@]}" --labels "$(IFS=,; echo "${built[*]}")" 2>&1 || echo "analyze tbo failed"
    fi
    for m in "${models[@]}"; do
        n=$(basename "$m" .gguf)
        vs=(); gs=()
        for v in "${built[@]}"; do
            g=""; for ((i = 1; i <= LB_ROUNDS; i++)); do
                f=$OUT/$v.$n.kv.r$i.jsonl; [ -s "$f" ] && g+="${g:+,}$f"
            done
            [ -n "$g" ] && { vs+=("$v"); gs+=("$g"); }
        done
        [ ${#gs[@]} -gt 0 ] || continue
        echo; echo "## llama-bench $n kv (median of rounds)"; echo
        python3 "$ROOT/bench/analyze.py" lb "${gs[@]}" --labels "$(IFS=,; echo "${vs[*]}")" 2>&1 || echo "analyze lb failed"
    done
} > "$OUT/summary.md"
log "done: $OUT (summary.md)"
