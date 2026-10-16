#!/bin/bash
# Day-one P100 A/B kit; explicit patch lists from third_party/patches/.
#   bench/patches/run.sh
#   MODELS="/m/llama.gguf /m/gemma.gguf" ROUNDS=5 bench/patches/run.sh
#   MODELS="/m/llama.gguf" PPL_TEXT=/data/wiki.test.txt bench/patches/run.sh
#   STEPS="test perf mmperf lb ppl" bench/patches/run.sh   reuse prepared builds
#   DRYRUN=1 OUT=/tmp/patches MODELS=/FAKE/model.gguf PPL_TEXT=/FAKE/wiki.txt bench/patches/run.sh
#     use bundled FAKE shims; skip worktrees, compilation, model/text access.
# Variants: upstream=0003; conv=0001,0003; conv8=0001,0002,0003;
#   flush=0001,0002,0003,0004; mmvh=0001,0002,0003,0005,0006; all=0001-0006.
# Default variants: upstream conv8 flush mmvh. Default steps:
#   micro (conversion), build, test (full FA, biased-V, MUL_MAT),
#   perf (FA), mmperf (contiguous F32 MUL_MAT n=1..16), lb (kv and tg128), ppl.
# On sm_60, n=9..16 uses upstream's dequantise-to-FP16 + cuBLAS path; see
#   docs/research/volta-fa-p9-lessons-2026-10-11.md sections 2 and 5.
# ROADMAP questions: 0004 fixes biased-V? test + perf/lb measure its FA/pp/tg cost;
#   H1/0005 decode correctness/speed? test + mmperf/lb + ppl cover Accept if.
# GGML_CUDA_DISABLE_MMVH is presence-checked: enabled runs unset it; mmvh/all
#   also run MUL_MAT correctness, mmperf, lb and ppl with it set to 1 (-nommvh).
# PPL requires explicit MODELS and PPL_TEXT; otherwise it logs a skip.
# Opt-in kld (STEPS="build kld", then STEPS=kld to reuse) measures accuracy
#   against FP32, not upstream. Requires MODELS and PPL_TEXT; KLD_CHUNKS=8.
#   KLD_UB=8 keeps n <= 8 in MMVH (n >= 9 routes to cuBLAS on sm_60).
#   Bench-only 0002 is the exception to shared bench patches: ref build only.
#   ref = upstream + shared bench patches + bench-only 0002; never a variant.
# Results: bench/results/patches-<stamp>-<host>/; summary.md uses analyze.py tbo/lb
#   (calibrate-decode.py is absent in this tree) and final PPL +/- error log lines.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2
ROOT=$PWD
LLAMA_SRC=${LLAMA_SRC:-$ROOT/third_party/llama.cpp}
WORK=${WORK:-$ROOT/build/patches-ab}
VARIANTS=${VARIANTS:-"upstream conv8 flush mmvh"}
STEPS=${STEPS:-"micro build test perf mmperf lb ppl"}
DRYRUN=${DRYRUN:-0}
ROUNDS=${ROUNDS:-3}
LB_ROUNDS=${LB_ROUNDS:-2}
REPS=${REPS:-3}
KLD_UB=${KLD_UB:-8}
KLD_CHUNKS=${KLD_CHUNKS:-8}
PATCHES=$ROOT/third_party/patches
OUT=${OUT:-$ROOT/bench/results/patches-$(date +%Y%m%d-%H%M%S)-$(hostname -s 2>/dev/null || hostname)}
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
if [ "$DRYRUN" = 1 ]; then
    export PATH="$ROOT/tests/fixtures/shims:$PATH"
    export SHIM_CALL_LOG=${SHIM_CALL_LOG:-$OUT/shim-calls.jsonl}
fi
gpu_state() { nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
    --format=csv,noheader 2>/dev/null | sed -n '1p' || echo "n/a"; }
export CCACHE_BASEDIR=$WORK CCACHE_NOHASHDIR=1 CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "$gpu" == *P100* || "${FORCE:-0}" == 1 ]] || { echo "GPU is '$gpu', not a P100 (FORCE=1 to override)" >&2; exit 2; }
ver=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
[ "$ver" = 12.8 ] || [ "${FORCE:-0}" = 1 ] || { echo "nvcc is '$ver'; CUDA 12.8 required (13 dropped sm_60; 12.9: llama.cpp #28416). FORCE=1 to override" >&2; exit 2; }

ordinals_for() {
    case $1 in
        upstream) echo 3 ;; conv) echo '1 3' ;; conv8) echo '1 2 3' ;;
        flush) echo '1 2 3 4' ;; mmvh) echo '1 2 3 5 6' ;; all) echo '1 2 3 4 5 6' ;;
        *) echo "unknown variant $1" >&2; return 2 ;;
    esac
}
patches_for() {
    local ordinals k
    ordinals=$(ordinals_for "$1") || return 2
    for k in $ordinals; do
        # Exactly one patch per ordinal, including in fixture mode.
        local files=("$PATCHES"/"$(printf '%04d' "$k")"-*.patch)
        [ ${#files[@]} = 1 ] && [ -f "${files[0]}" ] || { echo "missing or ambiguous patch $k" >&2; return 2; }
        printf '%s\n' "${files[@]}"
    done
}
# README is the manifest: only exact filenames marked "all variants" are shared.
# A future build-specific bench patch is excluded by using another scope there.
bench_patches_for_all() {
    local p
    while IFS= read -r p; do
        [ -f "$PATCHES/$p" ] || { echo "missing bench-only patch $p" >&2; return 2; }
        printf '%s\n' "$PATCHES/$p"
    done < <(awk -F '|' '$2 ~ /`bench-only\/[^`]+\.patch`/ && $3 ~ /^ all variants $/ {
        gsub(/[[:space:]`]/, "", $2); print $2
    }' "$PATCHES/README.md")
}
all_patches_for() {
    patches_for "$1" || return 2
    bench_patches_for_all
}
ref_patches() {
    all_patches_for upstream || return 2
    local p=$PATCHES/bench-only/0002-cuda-f32-reference-matmul-dev-switch.patch
    [ -f "$p" ] || { echo "missing ref-only patch $p" >&2; return 2; }
    printf '%s\n' "$p"
}
has_mmvh() { [[ " $(ordinals_for "$1") " == *' 5 '* ]]; }
# Alternate labels share the same binary. Always unset the presence-checked switch
# first so an inherited value (even 0) cannot silently disable the enabled arm.
arms_for() { echo "$1"; if has_mmvh "$1"; then echo "$1-nommvh"; fi; }
run_arm() {
    local arm=$1; shift
    if [[ "$arm" == *-nommvh ]]; then
        env -u GGML_CUDA_DISABLE_MMVH GGML_CUDA_DISABLE_MMVH=1 "$@"
    else
        env -u GGML_CUDA_DISABLE_MMVH "$@"
    fi
}
planned() {
    [ "$DRYRUN" = 1 ] || return 0
    local arm=$1 command; shift
    printf -v command ' %q' "$@"
    if [[ "$arm" == *-nommvh ]]; then
        log "DRYRUN command: env -u GGML_CUDA_DISABLE_MMVH GGML_CUDA_DISABLE_MMVH=1$command"
    else
        log "DRYRUN command: env -u GGML_CUDA_DISABLE_MMVH$command"
    fi
}
for v in $VARIANTS; do all_patches_for "$v" > /dev/null; done

if [ "$DRYRUN" = 1 ]; then
    base="DRYRUN: no llama.cpp checkout or build; all tool outputs are FAKE fixtures"
    DRY_TBO=$(command -v test-backend-ops)
    DRY_LB=$(command -v llama-bench)
    DRY_PPL=$(command -v llama-perplexity)
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
  sha256sum "$PATCHES"/bench-only/*.patch  # includes the ref-only kld patch
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
MM_FILTER='type_a=(q4_0|q8_0|q4_K|q5_K|q6_K),type_b=f32,(m=(1024|4096|14336),n=([1-9]|1[0-6]),k=4096|m=4096,n=([1-9]|1[0-6]),k=14336),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,src_overlap=0,m_v=0,pad=0(,|$)'
PPL_READY=0
if has ppl; then
    if [ -n "${PPL_TEXT:-}" ] && [ -n "${MODELS:-}" ]; then
        if [ "$DRYRUN" != 1 ] && [ ! -f "$PPL_TEXT" ]; then log "missing PPL_TEXT: $PPL_TEXT"; exit 2; fi
        PPL_READY=1
    else log "ppl skipped: set PPL_TEXT and MODELS"; fi
fi
KLD_READY=0
if has kld; then
    if [ -n "${PPL_TEXT:-}" ] && [ -n "${MODELS:-}" ]; then
        if [ "$DRYRUN" != 1 ] && [ ! -f "$PPL_TEXT" ]; then log "missing PPL_TEXT: $PPL_TEXT"; exit 2; fi
        [[ "$KLD_UB" =~ ^[1-9][0-9]*$ && "$KLD_CHUNKS" =~ ^[1-9][0-9]*$ ]] || {
            log "KLD_UB and KLD_CHUNKS must be positive integers"; exit 2;
        }
        ref_patches > /dev/null
        KLD_READY=1
        # Also remove a partial reference if generation or any arm fails.
        trap 'if [ -n "${kld_base:-}" ]; then rm -f -- "$kld_base"; fi' EXIT
    else log "kld skipped: set PPL_TEXT and MODELS"; fi
fi

# Select binaries per variant. Fixture mode deliberately never touches WORK.
tbo_for() { if [ "$DRYRUN" = 1 ]; then echo "$DRY_TBO"; else echo "$WORK/$1/build/bin/test-backend-ops"; fi; }
lb_for() { if [ "$DRYRUN" = 1 ]; then echo "$DRY_LB"; else echo "$WORK/$1/build/bin/llama-bench"; fi; }
ppl_for() { if [ "$DRYRUN" = 1 ]; then echo "$DRY_PPL"; else echo "$WORK/$1/build/bin/llama-perplexity"; fi; }
targets=(test-backend-ops llama-bench)
[ "$PPL_READY" = 0 ] && [ "$KLD_READY" = 0 ] || targets+=(llama-perplexity)
trees=()
read -ra trees <<< "$VARIANTS"
[ "$KLD_READY" = 0 ] || trees+=(ref)
built=()
if has build && [ "$DRYRUN" != 1 ]; then
    mkdir -p "$WORK"
    git -C "$LLAMA_SRC" worktree prune
fi
for v in "${trees[@]}"; do
    src=$WORK/$v
    variant_targets=("${targets[@]}")
    if [ "$v" = ref ]; then
        mapfile -t variant_patches < <(ref_patches)
        variant_targets=(llama-perplexity)
    else
        mapfile -t variant_patches < <(all_patches_for "$v")
    fi
    if has build && [ "$DRYRUN" != 1 ]; then
        [ -d "$src" ] || git -C "$LLAMA_SRC" worktree add -q --detach "$src" "$base"
        git -C "$src" checkout -q -- .
        for p in "${variant_patches[@]}"; do
            log "apply $p for $v"
            git -C "$src" apply "$p" || { log "patch $p does not apply for $v"; exit 1; }
        done
        if [ "$v" != ref ]; then
            t=$src/tests/test-backend-ops.cpp
            anchor='test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776'
            [ "$(grep -cF "$anchor" "$t")" = 1 ] || { log "perf anchor not unique in $t"; exit 1; }
            L="$PERF_CASES" perl -pi -e 'if (index($_, q{'"$anchor"'}) >= 0) { $_ .= "$ENV{L}\n" }' "$t"
        fi
        git -C "$src" diff > "$OUT/$v.patch"
        log "build $v for sm_60"
        if ! { cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
                 -DCMAKE_CUDA_ARCHITECTURES=60 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF > "$OUT/$v.build.log" 2>&1 &&
               cmake --build "$src/build" --target "${variant_targets[@]}" >> "$OUT/$v.build.log" 2>&1; }; then
            log "  BUILD FAILED for $v"; tail -15 "$OUT/$v.build.log" | tee -a "$OUT/run.log"
            [ "$v" != ref ] || exit 1
            continue
        fi
    elif has build; then
        for p in "${variant_patches[@]}"; do log "DRYRUN: git -C $src apply $p"; done
        log "DRYRUN: skipping worktree preparation and build for $v"
        log "DRYRUN: cmake --build $src/build --target ${variant_targets[*]}"
    fi
    if [ "$v" = ref ]; then
        [ -x "$(ppl_for ref)" ] || { log "missing llama-perplexity for ref; run STEPS='build kld' first"; exit 1; }
        continue
    fi
    if { has test || has perf || has mmperf; } && [ ! -x "$(tbo_for "$v")" ]; then log "missing test-backend-ops for $v; run STEPS=build first"; exit 1; fi
    if has lb && [ ! -x "$(lb_for "$v")" ]; then log "missing llama-bench for $v; run STEPS=build first"; exit 1; fi
    if { [ "$PPL_READY" = 1 ] || [ "$KLD_READY" = 1 ]; } && [ ! -x "$(ppl_for "$v")" ]; then log "missing llama-perplexity for $v; run STEPS=build first"; exit 1; fi
    built+=("$v")
    if has test; then
        for op in FLASH_ATTN_EXT MUL_MAT; do
            tag=fa; [ "$op" = MUL_MAT ] && tag=mulmat
            log "correctness $v (full $op CUDA0)"
            run_arm "$v" "$(tbo_for "$v")" test -b CUDA0 -o "$op" > "$OUT/$v.$tag-test-CUDA0.txt" 2>&1 && rc=0 || rc=$?
            log "  rc=$rc  OK=$(grep -c 'OK' "$OUT/$v.$tag-test-CUDA0.txt" || true)  FAIL=$(grep -c 'FAIL' "$OUT/$v.$tag-test-CUDA0.txt" || true)"
            if [ "$op" = FLASH_ATTN_EXT ]; then
                log "biased-V correctness: $(tbo_for "$v") test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias"
                run_arm "$v" "$(tbo_for "$v")" test -b CUDA0 -o FLASH_ATTN_EXT -p v_bias > "$OUT/$v.fa-vbias-test-CUDA0.txt" 2>&1 && rc=0 || rc=$?
                log "  rc=$rc  OK=$(grep -c 'OK' "$OUT/$v.fa-vbias-test-CUDA0.txt" || true)  FAIL=$(grep -c 'FAIL' "$OUT/$v.fa-vbias-test-CUDA0.txt" || true)"
            fi
        done
        if has_mmvh "$v"; then
            log "correctness $v-nommvh (full MUL_MAT CUDA0)"
            planned "$v-nommvh" "$(tbo_for "$v")" test -b CUDA0 -o MUL_MAT
            run_arm "$v-nommvh" "$(tbo_for "$v")" test -b CUDA0 -o MUL_MAT > "$OUT/$v-nommvh.mulmat-test-CUDA0.txt" 2>&1 && rc=0 || rc=$?
            log "  rc=$rc  OK=$(grep -c 'OK' "$OUT/$v-nommvh.mulmat-test-CUDA0.txt" || true)  FAIL=$(grep -c 'FAIL' "$OUT/$v-nommvh.mulmat-test-CUDA0.txt" || true)"
        fi
    fi
done
[ ${#built[@]} -gt 0 ] || { log "no variant built"; exit 1; }

{ has perf || has mmperf; } || ROUNDS=0
for ((i = 1; i <= ROUNDS; i++)); do
    for v in "${built[@]}"; do
        if has perf; then
            log "kernel perf $v round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
            run_arm "$v" stdbuf -oL "$(tbo_for "$v")" perf -b CUDA0 -o FLASH_ATTN_EXT -p "$PERF_FILTER" \
                --output sql > "$OUT/$v.perf.r$i.sql" 2> "$OUT/$v.perf.r$i.err" || log "  perf run failed (see .err)"
        fi
    done
    if has mmperf; then
        for v in "${built[@]}"; do
            while IFS= read -r arm; do
                log "MUL_MAT perf $arm round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
                planned "$arm" "$(tbo_for "$v")" perf -o MUL_MAT -b CUDA0 -p "$MM_FILTER" --output sql
                run_arm "$arm" stdbuf -oL "$(tbo_for "$v")" perf -o MUL_MAT -b CUDA0 -p "$MM_FILTER" \
                    --output sql > "$OUT/$arm.mmperf.r$i.sql" 2> "$OUT/$arm.mmperf.r$i.err" || log "  mmperf run failed (see .err)"
            done < <(arms_for "$v")
        done
    fi
done

# MODELS is an optional space-separated override; otherwise use bench/models.txt.
models=()
if [ -n "${MODELS:-}" ]; then read -ra models <<< "$MODELS"
elif has lb && [ "$DRYRUN" != 1 ] && [ -f bench/models.txt ]; then
    while IFS= read -r m || [ -n "$m" ]; do
        [[ -z "$m" || "$m" == \#* ]] || models+=("$m")
    done < bench/models.txt
fi
for m in "${models[@]}"; do
    n=$(basename "$m" .gguf)
    if [ "$KLD_READY" = 1 ]; then
        kld_base=$OUT/$n.ref.kld
        kld_args=(-m "$m" -f "$PPL_TEXT" -c 512 -b "$KLD_UB" -ub "$KLD_UB" -ngl 99 -fa on
                  --chunks "$KLD_CHUNKS" --kl-divergence-base "$kld_base")
        log "FP32 reference $n ref"
        planned ref env -u GGML_CUDA_F32_REF GGML_CUDA_F32_REF=1 "$(ppl_for ref)" "${kld_args[@]}"
        run_arm ref env -u GGML_CUDA_F32_REF GGML_CUDA_F32_REF=1 "$(ppl_for ref)" "${kld_args[@]}" \
            > "$OUT/ref.$n.kld.txt" 2> "$OUT/ref.$n.kld.err" || { log "kld reference failed for $n"; exit 1; }
        grep -Fxq 'GGML_CUDA_F32_REF: quantised MUL_MAT via dequantize + SGEMM' "$OUT/ref.$n.kld.err" || {
            log "kld reference switch did not take effect for $n"; exit 1;
        }
        [ -s "$kld_base" ] || { log "kld reference base missing for $n"; exit 1; }
        for v in "${built[@]}"; do
            while IFS= read -r arm; do
                log "KL divergence $n $arm against FP32"
                planned "$arm" env -u GGML_CUDA_F32_REF "$(ppl_for "$v")" "${kld_args[@]}" --kl-divergence
                run_arm "$arm" env -u GGML_CUDA_F32_REF "$(ppl_for "$v")" "${kld_args[@]}" --kl-divergence \
                    > "$OUT/$arm.$n.kld.txt" 2>&1 || { log "kld arm failed for $n $arm"; exit 1; }
                for metric in 'Mean[[:space:]]+KLD:' 'Maximum KLD:' 'RMS Δp[[:space:]]*:' 'Same top p:'; do
                    grep -Eq "$metric" "$OUT/$arm.$n.kld.txt" || { log "kld metric missing for $n $arm: $metric"; exit 1; }
                done
            done < <(arms_for "$v")
        done
        rm -f -- "$kld_base"
        kld_base=
    fi
    if has lb; then
        for ((i = 1; i <= LB_ROUNDS; i++)); do
            for v in "${built[@]}"; do
                while IFS= read -r arm; do
                    log "llama-bench kv $n $arm round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
                    planned "$arm" "$(lb_for "$v")" -m "$m" -dev CUDA0 -ngl 99 -fa on -r "$REPS" -o jsonl \
                        -ctk f16,q8_0,q4_0 -ctv f16,q8_0,q4_0 -p 8 -n 64 -d 16384
                    run_arm "$arm" "$(lb_for "$v")" -m "$m" -dev CUDA0 -ngl 99 -fa on -r "$REPS" -o jsonl \
                        -ctk f16,q8_0,q4_0 -ctv f16,q8_0,q4_0 -p 8 -n 64 -d 16384 \
                        > "$OUT/$arm.$n.kv.r$i.jsonl" 2> "$OUT/$arm.$n.kv.r$i.err" || log "  llama-bench failed"
                    for depth in 0 4096; do
                        log "llama-bench tg128 depth $depth $n $arm round $i  gpu[sm,mem,temp,power,throttle]: $(gpu_state)"
                        planned "$arm" "$(lb_for "$v")" -m "$m" -dev CUDA0 -ngl 99 -fa on -r "$REPS" -o jsonl -p 0 -n 128 -d "$depth"
                        run_arm "$arm" "$(lb_for "$v")" -m "$m" -dev CUDA0 -ngl 99 -fa on -r "$REPS" -o jsonl -p 0 -n 128 -d "$depth" \
                            > "$OUT/$arm.$n.tg128-d$depth.r$i.jsonl" 2> "$OUT/$arm.$n.tg128-d$depth.r$i.err" || log "  llama-bench failed"
                    done
                done < <(arms_for "$v")
            done
        done
    fi
    if [ "$PPL_READY" = 1 ]; then
        for v in "${built[@]}"; do
            while IFS= read -r arm; do
                log "perplexity $n $arm"
                planned "$arm" "$(ppl_for "$v")" -m "$m" -f "$PPL_TEXT" -c 512 --chunks 32 -ngl 99 -fa on
                run_arm "$arm" "$(ppl_for "$v")" -m "$m" -f "$PPL_TEXT" -c 512 --chunks 32 -ngl 99 -fa on \
                    > "$OUT/$arm.$n.ppl.txt" 2>&1 || log "  perplexity failed (see .ppl.txt)"
            done < <(arms_for "$v")
        done
    fi
done

arms=()
for v in "${built[@]}"; do
    while IFS= read -r arm; do arms+=("$arm"); done < <(arms_for "$v")
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
    if has perf && [ "$ROUNDS" -gt 0 ]; then
        echo "## Kernel timing (median of $ROUNDS interleaved rounds)"; echo
        groups=()
        for v in "${built[@]}"; do
            g=""; for ((i = 1; i <= ROUNDS; i++)); do g+="${g:+,}$OUT/$v.perf.r$i.sql"; done
            groups+=("$g")
        done
        python3 "$ROOT/bench/analyze.py" tbo "${groups[@]}" --labels "$(IFS=,; echo "${built[*]}")" 2>&1 || echo "analyze tbo failed"
    fi
    if has mmperf && [ "$ROUNDS" -gt 0 ]; then
        echo; echo "## MUL_MAT timing (median of $ROUNDS interleaved rounds)"; echo
        groups=()
        for arm in "${arms[@]}"; do
            g=""; for ((i = 1; i <= ROUNDS; i++)); do g+="${g:+,}$OUT/$arm.mmperf.r$i.sql"; done
            groups+=("$g")
        done
        python3 "$ROOT/bench/analyze.py" tbo "${groups[@]}" --labels "$(IFS=,; echo "${arms[*]}")" 2>&1 || echo "analyze tbo failed"
        echo
        python3 "$ROOT/bin/mm-gbps.py" "${groups[@]}" --labels "$(IFS=,; echo "${arms[*]}")" 2>&1 || echo "mm-gbps failed"
    fi
    for m in "${models[@]}"; do
        n=$(basename "$m" .gguf)
        if has lb; then
            for kind in kv tg128-d0 tg128-d4096; do
                vs=(); gs=()
                for arm in "${arms[@]}"; do
                    g=""; for ((i = 1; i <= LB_ROUNDS; i++)); do
                        f=$OUT/$arm.$n.$kind.r$i.jsonl; [ -s "$f" ] && g+="${g:+,}$f"
                    done
                    [ -n "$g" ] && { vs+=("$arm"); gs+=("$g"); }
                done
                [ ${#gs[@]} -gt 0 ] || continue
                echo; echo "## llama-bench $n $kind (median of rounds)"; echo
                python3 "$ROOT/bench/analyze.py" lb "${gs[@]}" --labels "$(IFS=,; echo "${vs[*]}")" 2>&1 || echo "analyze lb failed"
            done
        fi
        if [ "$PPL_READY" = 1 ]; then
          echo; echo "## Perplexity $n (512 context, 32 chunks)"; echo
          for arm in "${arms[@]}"; do
              echo "- $arm ($arm.$n.ppl.txt):"
              grep -E 'PPL.*(\+/-|±)' "$OUT/$arm.$n.ppl.txt" | tail -1 || echo "  final PPL +/- error missing; inspect log"
          done
        fi
        if [ "$KLD_READY" = 1 ]; then
            echo; echo "## KL divergence $n against FP32 (512 context, $KLD_CHUNKS chunks, batch/ubatch $KLD_UB)"; echo
            for arm in "${arms[@]}"; do
                echo "- $arm ($arm.$n.kld.txt):"
                grep -E 'Mean[[:space:]]+KLD:|Maximum KLD:|RMS Δp[[:space:]]*:|Same top p:' "$OUT/$arm.$n.kld.txt"
            done
        fi
    done
} > "$OUT/summary.md"
log "done: $OUT (summary.md)"
