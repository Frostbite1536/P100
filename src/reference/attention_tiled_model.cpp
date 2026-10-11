#include "p100/attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace p100 {

float round_to_fp16(double x) {
    static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                  std::numeric_limits<double>::is_iec559, "requires IEEE binary64");
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    const std::uint32_t sign = static_cast<std::uint32_t>(bits >> 32) & 0x80000000u;
    const unsigned exponent = static_cast<unsigned>((bits >> 52) & 0x7ffu);
    if (exponent == 0x7ffu) return static_cast<float>(x);
    const int e = static_cast<int>(exponent) - 1023;
    std::uint32_t half = 0;
    if (e > 15) {
        half = 0x7c00u;
    } else if (e >= -25) {
        const std::uint64_t significand = (bits & 0xfffffffffffffu) | (1ull << 52);
        const int shift = e < -14 ? 28 - e : 42;
        std::uint32_t rounded = static_cast<std::uint32_t>(significand >> shift);
        const std::uint64_t remainder = significand & ((1ull << shift) - 1);
        const std::uint64_t midpoint = 1ull << (shift - 1);
        if (remainder > midpoint || (remainder == midpoint && (rounded & 1u))) ++rounded;
        half = e < -14 ? rounded : (static_cast<std::uint32_t>(e + 15) << 10) + rounded - 0x400u;
    }
    const std::uint32_t he = half >> 10;
    std::uint32_t hf = half & 0x3ffu;
    std::uint32_t result = sign;
    if (he == 31) {
        result |= 0x7f800000u;
    } else if (he != 0) {
        result |= ((he + 112u) << 23) | (hf << 13);
    } else if (hf != 0) {
        int normal_exponent = -14;
        while ((hf & 0x400u) == 0) { hf <<= 1; --normal_exponent; }
        result |= (static_cast<std::uint32_t>(normal_exponent + 127) << 23) | ((hf & 0x3ffu) << 13);
    }
    float output;
    std::memcpy(&output, &result, sizeof(output));
    return output;
}

void attention_tiled_model(const AttnParams& p, const TiledModelConfig& cfg, const float* q, const float* k,
                           const float* v, const float* mask, float* o, float* lse) {
    std::string why;
    if (!validate(p, &why)) throw std::invalid_argument(why);
    if (cfg.br <= 0 || cfg.bc <= 0) throw std::invalid_argument("br and bc must be positive");
    if (cfg.kv_parallel_blocks < 1 || cfg.o_accum_np < 1)
        throw std::invalid_argument("KV parallel blocks and O accumulator count must be positive");
    const int qk_modes = static_cast<int>(cfg.qk_accum_fp16) + static_cast<int>(cfg.qk_products_fp16) +
                         static_cast<int>(cfg.qk_half2_chain > 0);
    if (qk_modes > 1) throw std::invalid_argument("QK accumulation modes are mutually exclusive");
    if (cfg.qk_q_prescale <= 0.0f) throw std::invalid_argument("Q prescale must be positive");
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float scale = p.effective_scale();
    const int block_rows = std::min(cfg.br, p.seq_q);
    const std::size_t states = static_cast<std::size_t>(block_rows) * cfg.kv_parallel_blocks;
    const std::size_t row_stride = static_cast<std::size_t>(cfg.o_accum_np) * p.d_v;
    std::vector<float> m(states), l(states);
    std::vector<float> acc(states * row_stride);
    std::vector<float> tile_acc(row_stride);
    std::vector<float> part_scales(cfg.kv_parallel_blocks);
    std::vector<float> prepared_q(static_cast<std::size_t>(block_rows) * p.d_qk);
    const bool prepare_q = cfg.qk_scale_in_q || cfg.qk_q_prescale != 1.0f;
    std::vector<float> logits(std::min(cfg.bc, p.seq_kv));
    for (int b = 0; b < p.batch; ++b) {
        for (int h = 0; h < p.n_head_q; ++h) {
            const int kh = h / (p.n_head_q / p.n_head_kv);
            const std::size_t kv_base = (static_cast<std::size_t>(b) * p.n_head_kv + kh) * p.seq_kv;
            const std::size_t q_base = (static_cast<std::size_t>(b) * p.n_head_q + h) * p.seq_q;
            for (int ib = 0; ib < p.seq_q;) {
                const int nr = std::min(cfg.br, p.seq_q - ib);
                std::fill(m.begin(), m.end(), neg_inf);
                std::fill(l.begin(), l.end(), 0.0f);
                std::fill(acc.begin(), acc.end(), 0.0f);
                if (prepare_q) {
                    for (int r = 0; r < nr; ++r) {
                        for (int d = 0; d < p.d_qk; ++d) {
                            float value = q[(q_base + ib + r) * p.d_qk + d];
                            value = round_to_fp16(cfg.qk_scale_in_q ? value * scale : value);
                            if (cfg.qk_q_prescale != 1.0f) value = round_to_fp16(value * cfg.qk_q_prescale);
                            prepared_q[static_cast<std::size_t>(r) * p.d_qk + d] = value;
                        }
                    }
                }
                for (int jb = 0; jb < p.seq_kv;) {
                    const int part = (jb / cfg.bc) % cfg.kv_parallel_blocks;
                    const int nc = std::min(cfg.bc, p.seq_kv - jb);
                    // Even the last query row in this block cannot see this tile.
                    if (p.causal && jb > ib + nr - 1 + (p.seq_kv - p.seq_q)) break;
                    for (int r = 0; r < nr; ++r) {
                        const std::size_t state = static_cast<std::size_t>(part) * block_rows + r;
                        const int i = ib + r;
                        if (p.causal && jb > i + (p.seq_kv - p.seq_q)) continue;
                        const std::size_t row = q_base + i;
                        float tile_max = neg_inf;
                        for (int c = 0; c < nc; ++c) {
                            const int j = jb + c;
                            float x = neg_inf;
                            if (!p.causal || j <= i + (p.seq_kv - p.seq_q)) {
                                float dot = 0.0f;
                                const float* query = prepare_q ? prepared_q.data() + static_cast<std::size_t>(r) * p.d_qk
                                                               : q + row * p.d_qk;
                                const float* key = k + (kv_base + j) * p.d_qk;
                                if (cfg.qk_half2_chain > 0) {
                                    float lanes[2] = {0.0f, 0.0f};
                                    int steps = 0;
                                    for (int d = 0; d < p.d_qk; d += 2) {
                                        lanes[0] = round_to_fp16(static_cast<double>(query[d]) * key[d] + lanes[0]);
                                        if (d + 1 < p.d_qk)
                                            lanes[1] = round_to_fp16(static_cast<double>(query[d + 1]) * key[d + 1] + lanes[1]);
                                        if (++steps == cfg.qk_half2_chain || d + 2 >= p.d_qk) {
                                            dot += lanes[0] + lanes[1];
                                            lanes[0] = lanes[1] = 0.0f;
                                            steps = 0;
                                        }
                                    }
                                } else {
                                    for (int d = 0; d < p.d_qk; ++d) {
                                        float product = query[d] * key[d];
                                        if (cfg.qk_products_fp16) product = round_to_fp16(product);
                                        dot += product;
                                        if (cfg.qk_accum_fp16 && ((d + 1) % 16 == 0 || d + 1 == p.d_qk))
                                            dot = round_to_fp16(dot);
                                    }
                                }
                                dot /= cfg.qk_q_prescale;
                                x = cfg.qk_scale_in_q ? dot : dot * scale;
                                if (p.softcap > 0.0f) x = p.softcap * std::tanh(x / p.softcap);
                                if (mask) x += mask[static_cast<std::size_t>(i) * p.seq_kv + j];
                            }
                            logits[c] = x;
                            tile_max = std::max(tile_max, x);
                        }
                        const float m_new = std::max(m[state], tile_max + cfg.kq_max_offset);
                        if (m_new == neg_inf) continue;  // No visible key yet; avoid -inf - -inf.
                        const float alpha = m[state] == neg_inf ? 0.0f : std::exp(m[state] - m_new);
                        float tile_sum = 0.0f;
                        float* row_acc = acc.data() + state * row_stride;
                        if (cfg.o_accum_fp16) {
                            const float half_alpha = round_to_fp16(alpha);
                            for (std::size_t d = 0; d < row_stride; ++d)
                                row_acc[d] = round_to_fp16(static_cast<double>(row_acc[d]) * half_alpha);
                        } else {
                            std::fill(tile_acc.begin(), tile_acc.end(), 0.0f);
                        }
                        for (int c = 0; c < nc; ++c) {
                            float weight = std::exp(logits[c] - m_new);
                            if (cfg.l_from_unrounded_p) tile_sum += weight;
                            if (cfg.p_fp16) weight = round_to_fp16(weight);
                            if (!cfg.l_from_unrounded_p) tile_sum += weight;
                            const std::size_t lane = static_cast<std::size_t>(c % cfg.o_accum_np) * p.d_v;
                            const float* value = v + (kv_base + jb + c) * p.d_v;
                            if (cfg.o_accum_fp16) {
                                for (int d = 0; d < p.d_v; ++d)
                                    row_acc[lane + d] = round_to_fp16(static_cast<double>(weight) * value[d] + row_acc[lane + d]);
                            } else {
                                for (int d = 0; d < p.d_v; ++d) tile_acc[lane + d] += weight * value[d];
                            }
                        }
                        if (!cfg.o_accum_fp16)
                            for (std::size_t d = 0; d < row_stride; ++d) row_acc[d] = row_acc[d] * alpha + tile_acc[d];
                        l[state] = l[state] * alpha + tile_sum;
                        m[state] = m_new;
                    }
                    jb += nc;
                }
                for (int r = 0; r < nr; ++r) {
                    const std::size_t row = q_base + ib + r;
                    float global_max = neg_inf;
                    for (int part = 0; part < cfg.kv_parallel_blocks; ++part) {
                        const std::size_t state = static_cast<std::size_t>(part) * block_rows + r;
                        global_max = std::max(global_max, m[state]);
                        // fattn-tile.cuh:1021-1048: add warp partials in order,
                        // rounding each addition in the accumulator precision.
                        float* partial = acc.data() + state * row_stride;
                        for (int lane = 1; lane < cfg.o_accum_np; ++lane) {
                            for (int d = 0; d < p.d_v; ++d) {
                                const float sum = partial[d] + partial[static_cast<std::size_t>(lane) * p.d_v + d];
                                partial[d] = cfg.o_accum_fp16 ? round_to_fp16(sum) : sum;
                            }
                        }
                    }
                    // fattn-common.cuh:963-977: unnormalised FP32 parts,
                    // rebased to their common maximum, then one division.
                    float denominator = 0.0f;
                    for (int part = 0; part < cfg.kv_parallel_blocks; ++part) {
                        const std::size_t state = static_cast<std::size_t>(part) * block_rows + r;
                        part_scales[part] = l[state] == 0.0f ? 0.0f
                            : (cfg.kv_parallel_blocks == 1 ? 1.0f : std::exp(m[state] - global_max));
                        denominator += part_scales[part] * l[state];
                    }
                    for (int d = 0; d < p.d_v; ++d) {
                        float numerator = 0.0f;
                        for (int part = 0; part < cfg.kv_parallel_blocks; ++part) {
                            const std::size_t state = static_cast<std::size_t>(part) * block_rows + r;
                            if (l[state] == 0.0f) continue;
                            numerator += part_scales[part] * acc[state * row_stride + d];
                        }
                        // Preserve the original single-part expression, including signed zero.
                        if (cfg.kv_parallel_blocks == 1) numerator = acc[static_cast<std::size_t>(r) * row_stride + d];
                        float value = denominator == 0.0f ? 0.0f : numerator / denominator;
                        if (cfg.o_fp16_out) value = round_to_fp16(value);
                        o[row * p.d_v + d] = value;
                    }
                    if (lse) lse[row] = denominator == 0.0f ? neg_inf : global_max + std::log(denominator);
                }
                ib += nr;
            }
        }
    }
}

}  // namespace p100
