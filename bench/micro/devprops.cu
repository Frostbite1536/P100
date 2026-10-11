// Host-only device probe. No kernel is launched and no model weights are read.
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaDeviceProp p{};
    const cudaError_t status = cudaGetDeviceProperties(&p, 0);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties: %s\n", cudaGetErrorString(status));
        return 1;
    }
    if (p.major != 5 || p.minor != 2) {
        std::fprintf(stderr, "Expected Maxwell M2200 sm_52, got sm_%d%d\n", p.major, p.minor);
        return 1;
    }
    std::printf("{\"label\":\"Maxwell M2200\",\"name\":\"%s\",\"major\":%d,\"minor\":%d,"
                "\"sm_count\":%d,\"clock_khz\":%d,\"memory_clock_khz\":%d,"
                "\"memory_bus_bits\":%d,\"source\":\"cudaGetDeviceProperties(device=0)\"}\n",
                p.name, p.major, p.minor, p.multiProcessorCount, p.clockRate,
                p.memoryClockRate, p.memoryBusWidth);
}
