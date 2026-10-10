extern "C" __global__ void too_much_smem(unsigned char *out) {
    __shared__ unsigned char tile[49 * 1024];
    const unsigned int i = threadIdx.x;
    tile[i] = static_cast<unsigned char>(i);
    __syncthreads();
    out[i] = tile[out[i] * 192 + i];
}
