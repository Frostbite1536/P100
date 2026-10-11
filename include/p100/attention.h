// Shared contract for every attention kernel, reference and test in the P100 repo.
// Changing anything in this header is a manager decision (see docs/INVARIANTS.md).
#pragma once

#include <string>

namespace p100 {

// Tensor layouts: dense, row-major, contiguous, outermost dimension first.
//   Q   [batch][n_head_q ][seq_q ][d_qk]
//   K   [batch][n_head_kv][seq_kv][d_qk]
//   V   [batch][n_head_kv][seq_kv][d_v ]
//   O   [batch][n_head_q ][seq_q ][d_v ]
//   LSE [batch][n_head_q ][seq_q ]        natural-log sum of exp(logits); -inf for fully masked rows
//   mask (optional) [seq_q][seq_kv]       additive, shared by every batch and head
//
// GQA: query head h reads kv head h / (n_head_q / n_head_kv).
//
// Logit for query row i, key j (matches ggml_flash_attn_ext with max_bias = 0):
//   x = dot(q_i, k_j) * scale
//   if softcap > 0: x = softcap * tanh(x / softcap)
//   x += mask[i][j]                        (if mask given; -inf masks out)
//   if causal and j > i + (seq_kv - seq_q): masked out   (bottom-right alignment, INV-NUM-004)
// O_i = sum_j softmax_j(x) * v_j. A row with every key masked gives O_i = 0, LSE_i = -inf (INV-NUM-002).
//
// Out of scope for now: ALiBi (max_bias), sinks, quantised K/V, dropout, backward.
struct AttnParams {
    int batch = 1;
    int n_head_q = 1;
    int n_head_kv = 1;
    int seq_q = 1;
    int seq_kv = 1;
    int d_qk = 128;
    int d_v = 128;
    float scale = 0.0f;    // 0 means 1/sqrt(d_qk)
    float softcap = 0.0f;  // 0 means off
    bool causal = false;

    float effective_scale() const;
};

// Returns true if p is valid; otherwise false and a reason in *why (if non-null).
bool validate(const AttnParams& p, std::string* why = nullptr);

// Exact reference: inputs are float (the caller rounds them through FP16 if it wants
// FP16-representable data); all accumulation is in double. o and lse must be allocated
// by the caller with the sizes above. mask may be null. lse may be null.
void attention_reference(const AttnParams& p, const float* q, const float* k, const float* v,
                         const float* mask, float* o, float* lse);

// CPU model of the fused-kernel algorithm, used to predict GPU numerics before
// hardware exists: Q rows in blocks of br, K/V in tiles of bc visited in order,
// running max m and sum l in FP32, rescaling of the O accumulator per tile.
struct TiledModelConfig {
    int br = 64;
    int bc = 64;
    bool qk_accum_fp16 = false;  // round the running QK^T dot product to FP16 after every 16-element
                                 // chunk of d_qk (models an FP16-accumulate k-loop, e.g. HFMA2 chains)
    bool p_fp16 = true;          // round P = exp(x - m) to FP16 before P*V (FP16 operand of the P*V product)
    bool o_fp16_out = true;      // round final O to FP16 (kernel output dtype)
    bool qk_scale_in_q = false;  // FP32 scale into Q, then round Q to FP16
    float qk_q_prescale = 1.0f;  // round FP16 Q * prescale; divide the FP32 dot by prescale
    bool qk_products_fp16 = false;  // round each product to FP16; accumulate in FP32
    int qk_half2_chain = 0;      // N FP16 FMAs per even/odd lane, then flush to FP32
    float kq_max_offset = 0.0f;  // running max = max(previous, tile_max + offset)
    bool l_from_unrounded_p = false;  // sum FP32 P before optional FP16 rounding
    bool o_accum_fp16 = false;  // FP16 tile rescale and per-key fused P*V updates
    int kv_parallel_blocks = 1;  // KV tile t belongs to part t % kv_parallel_blocks
    int o_accum_np = 1;          // key j within a tile belongs to accumulator j % o_accum_np
};

void attention_tiled_model(const AttnParams& p, const TiledModelConfig& cfg, const float* q, const float* k,
                           const float* v, const float* mask, float* o, float* lse);

// Round-to-nearest-even float -> IEEE binary16 -> float, with overflow to +/-inf and
// subnormals, matching __float2half_rn.
float round_to_fp16(float x);
// Direct binary64 -> binary16 rounding, avoiding an intermediate binary32 rounding.
float round_to_fp16(double x);

}  // namespace p100
