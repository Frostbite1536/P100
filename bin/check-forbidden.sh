#!/bin/bash
# Fails if source uses device features unavailable on sm_60 (INV-ARCH-001).
# NVRTC/nvcc at compute_60 rejects most of these anyway; this catches inline PTX
# and code paths hidden behind #if __CUDA_ARCH__ guards.
# Optional paths scan fixtures in isolation; the default scan excludes fixtures
# and never includes third_party/.
cd "$(dirname "$0")/.." || exit 2
PATTERN='__nv_bfloat16|nv_bfloat162?|cuda_bf16\.h|cuda_fp8\.h|__nv_fp8|\.e4m3|\.e5m2|cp\.async|__pipeline_memcpy_async|memcpy_async|mma\.sync\.aligned\.m16n8|m16n8k(8|16|32)|ldmatrix|stmatrix|movmatrix|wgmma|setmaxnreg|cluster_group|cuda/barrier|cp\.async\.bulk|cp\.reduce\.async|nvcuda[[:space:]]*::[[:space:]]*wmma|#[[:space:]]*include[[:space:]]*[<"]mma\.h[>"]|mma\.sync|__dp4a|__dp2a_lo|__dp2a_hi|\bdp(4a|2a)\b|__reduce_(add|min|max|and|or|xor)_sync|__cluster_dims__|cudaLaunchAttributeClusterDimension|\bTMA\b|\btma\b|tensormap|tensor_map'
dirs=()
excludes=()
if [ "$#" -gt 0 ]; then
  dirs=("$@")
else
  for d in src include tests bench; do [ -d "$d" ] && dirs+=("$d"); done
  excludes=(--exclude-dir=fixtures)
fi
[ ${#dirs[@]} -eq 0 ] && { echo "ok (nothing to scan)"; exit 0; }
if grep -rnE "$PATTERN" "${excludes[@]}" --include='*.cu' --include='*.cuh' --include='*.h' --include='*.hpp' --include='*.cpp' -- "${dirs[@]}"; then
  echo "forbidden for sm_60 (INV-ARCH-001)" >&2; exit 1
else
  status=$?
  [ "$status" -eq 1 ] || exit 2
fi
echo "ok"
