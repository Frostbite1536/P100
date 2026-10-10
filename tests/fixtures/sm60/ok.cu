#include <cuda_fp16.h>

extern "C" __global__ void half2_ok(const half2 *a, const half2 *b, half2 *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = __hmul2(a[i], b[i]);
}
