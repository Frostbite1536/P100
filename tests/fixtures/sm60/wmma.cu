#include <mma.h>

extern "C" __global__ void forbidden_matrix(float *out) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    nvcuda::wmma::store_matrix_sync(out, acc, 16, nvcuda::wmma::mem_row_major);
}
