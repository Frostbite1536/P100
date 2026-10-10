// Keep more values live across a barrier than sm_60 can hold in registers.
// Volatile input prevents the compiler from deferring loads until the sum.
extern "C" __global__ void force_spill(const volatile float *input, volatile float *out) {
    float values[512];
    #pragma unroll
    for (int i = 0; i < 512; ++i) {
        values[i] = input[i * blockDim.x + threadIdx.x];
    }
    __syncthreads();
    // The dependent FMA chain cannot be reduced before this volatile read.
    float sum = out[threadIdx.x];
    #pragma unroll
    for (int i = 0; i < 512; ++i) {
        sum = __fmaf_rn(sum, values[i], 1.0f);
    }
    out[threadIdx.x] = sum;
}
