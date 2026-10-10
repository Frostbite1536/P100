// Static one-weight-block, one-output-column fixture; launch with one thread
// if executing it. No GPU execution is needed to collect the SASS counts.
#include "vecdotq.cuh"

static_assert(sizeof(block_q4_0) == 18, "q4_0 block bytes");
static_assert(sizeof(block_q4_1) == 20, "q4_1 block bytes");
static_assert(sizeof(block_q5_0) == 22, "q5_0 block bytes");
static_assert(sizeof(block_q5_1) == 24, "q5_1 block bytes");
static_assert(sizeof(block_q8_0) == 34, "q8_0 block bytes");
static_assert(sizeof(block_q4_K) == 144, "q4_K block bytes");
static_assert(sizeof(block_q5_K) == 176, "q5_K block bytes");
static_assert(sizeof(block_q6_K) == 210, "q6_K block bytes");
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block bytes");
static_assert(sizeof(half) == 2, "f16 bytes");

// MMVQ distributes these iqs values across threads. Here they are summed in
// one thread so static counts cover the entire block. kbx = 0; activations
// point to QK/QK8_1 matching block_q8_1 blocks. Full unrolling makes static
// instruction counts cover every call, with compiler reuse across calls.
#define BLOCK_DOT(type, qi, vdr)                                            \
    static_assert((qi) % (vdr) == 0, "whole block coverage");               \
    extern "C" __global__ void vecdot_##type(                              \
        const void * __restrict__ weights,                                \
        const block_q8_1 * __restrict__ activations,                       \
        float * __restrict__ output) {                                    \
        float sum = 0.0f;                                                 \
        _Pragma("unroll")                                                 \
        for (int iqs = 0; iqs < (qi); iqs += (vdr)) {                      \
            sum += vec_dot_##type##_q8_1(weights, activations, 0, iqs);    \
        }                                                                \
        output[0] = sum;                                                  \
    }

BLOCK_DOT(q4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ)
BLOCK_DOT(q4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ)
BLOCK_DOT(q5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ)
BLOCK_DOT(q5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ)
BLOCK_DOT(q8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ)
BLOCK_DOT(q4_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ)
BLOCK_DOT(q5_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ)
BLOCK_DOT(q6_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ)
BLOCK_DOT(iq4_xs, QI4_XS, VDR_IQ4_XS_Q8_1_MMVQ)

#undef BLOCK_DOT
