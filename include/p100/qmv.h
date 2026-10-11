#pragma once

// GGUF layouts: llama.cpp/ggml/src/ggml-common.h:194-199,250-256,323-338.
// Integer half storage keeps the CPU model and CUDA blocks byte-identical.
namespace p100 { namespace qmv {
struct Q4_0 { unsigned short d; unsigned char qs[16]; };
struct Q8_0 { unsigned short d; signed char qs[32]; };
struct Q4_K { unsigned short d, dmin; unsigned char scales[12], qs[128]; };
static_assert(sizeof(Q4_0) == 18, "q4_0 GGUF size");
static_assert(sizeof(Q8_0) == 34, "q8_0 GGUF size");
static_assert(sizeof(Q4_K) == 144, "q4_K GGUF size");
static_assert(alignof(Q4_0) == 2 && alignof(Q8_0) == 2 && alignof(Q4_K) == 2,
              "weight blocks require only two-byte alignment");
} }

#ifndef __CUDACC_RTC__
#include <random>
#include <vector>
namespace p100 { namespace qmv {
enum class Type { q4_0, q8_0, q4_K };
const char* name(Type type);
int block_size(Type type);
int block_bytes(Type type);
unsigned short half_bits(float value);
float half_value(unsigned short bits);
// Full packed rows; scales are finite FP16, all quant/scale fields random.
std::vector<unsigned char> random_row(Type type, int k, std::mt19937& rng);
struct ActivationBlock {
    float scale, sum; // h16: sum of rounded, normalised activations
    float h[32];     // FP16 values represented exactly in FP32, permuted 0,2,1,3
    float qd, qsum;   // q8_1 FP16 scale and ORIGINAL FP32 sum rounded to FP16
    int q[32];
};
std::vector<ActivationBlock> prepare(const float* x, int k);
struct Result { double exact, h16_exact, weight_norm; float q81, h8, h16; };
Result evaluate(Type type, const void* weights, const float* x, int k,
                const std::vector<ActivationBlock>& activations);
// FP64 dot using h16's stored scale, FP16 activations and FP32 sum;
// no rounding after activation preparation (Maxwell indexing/layout reference).
double h16_exact(Type type, const void* weights, int k,
                 const std::vector<ActivationBlock>& activations);
float h16_model(Type type, const void* weights, int k,
                const std::vector<ActivationBlock>& activations, int chain);
template<int CHAIN>
float h16(Type type, const void* weights, int k,
          const std::vector<ActivationBlock>& activations) {
    static_assert(CHAIN == 8 || CHAIN == 16, "supported HFMA2 chain");
    return h16_model(type, weights, k, activations, CHAIN);
}
} }
#endif
