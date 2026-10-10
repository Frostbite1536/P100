#include "check.h"
#include "p100/attention.h"
#include "p100/tolerance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace p100;

namespace {
// Independent scalar binary16 rounding oracle, including ties to even.
float scalar_half(double x) {
    if (!std::isfinite(x) || x == 0) return static_cast<float>(x);
    const double a = std::abs(x);
    if (a >= 65520) return std::copysign(std::numeric_limits<float>::infinity(), x);
    int exponent;
    std::frexp(a, &exponent);
    const double unit = std::ldexp(1.0, std::max(-24, exponent - 11));
    const double scaled = a / unit;
    double n = std::floor(scaled);
    if (scaled - n > 0.5 || (scaled - n == 0.5 && std::fmod(n, 2.0) != 0)) ++n;
    return static_cast<float>(std::copysign(n * unit, x));
}

void double_rounding() {
    // A binary32 intermediate would erase these displacements from half-way.
    for (double midpoint : {std::ldexp(1.0, -25), 1.0 + std::ldexp(1.0, -11), 65520.0}) {
        for (double x : {midpoint, std::nextafter(midpoint, 0.0),
                         std::nextafter(midpoint, std::numeric_limits<double>::infinity())}) {
            CHECK(float_bits(round_to_fp16(x)) == float_bits(scalar_half(x)));
            CHECK(float_bits(round_to_fp16(-x)) == float_bits(scalar_half(-x)));
        }
    }
    CHECK(float_bits(round_to_fp16(-0.0)) == float_bits(-0.0f));
    CHECK(std::isnan(round_to_fp16(std::numeric_limits<double>::quiet_NaN())));
}

void scale_and_offset() {
    AttnParams p;
    p.seq_q = 3; p.seq_kv = 131; p.d_qk = p.d_v = 64;
    std::mt19937 rng(20261010);
    std::uniform_real_distribution<float> random(-0.5f, 0.5f);
    std::vector<float> q(p.seq_q * p.d_qk), k(p.seq_kv * p.d_qk), v(k.size());
    for (auto* data : {&q, &k, &v}) for (float& x : *data) x = round_to_fp16(random(rng));
    std::vector<float> base(q.size()), out(q.size()), base_lse(p.seq_q), lse(p.seq_q);
    const TiledModelConfig plain{32, 64, false, false, false};
    attention_tiled_model(p, plain, q.data(), k.data(), v.data(), nullptr, base.data(), base_lse.data());
    for (int mode = 0; mode < 3; ++mode) {
        auto cfg = plain;
        if (mode == 0) cfg.qk_q_prescale = 0.25f;
        if (mode == 1) cfg.qk_scale_in_q = true; // scale=1/8 is exact for these inputs
        if (mode == 2) cfg.kq_max_offset = 3 * 0.6931f;
        attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), nullptr, out.data(), lse.data());
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (mode == 0) CHECK(std::abs(static_cast<double>(out[i]) - base[i]) <= 1e-6 * std::abs(base[i]));
            CHECK(out_within_tolerance(out[i], base[i]));
        }
        for (int i = 0; i < p.seq_q; ++i) CHECK(lse_within_tolerance(lse[i], base_lse[i]));
    }
    // Non-power-of-two scale: one visible key exposes the rounded-Q logit in LSE.
    p = AttnParams{}; p.d_qk = p.d_v = 1; p.scale = 0.3f;
    float query = 1.0f, key = 1.0f, value = 2.0f, o, l;
    auto cfg = plain; cfg.qk_scale_in_q = true;
    attention_tiled_model(p, cfg, &query, &key, &value, nullptr, &o, &l);
    CHECK(l == round_to_fp16(p.scale));
    CHECK(lse_within_tolerance(l, p.scale));
    CHECK(o == value);
}

void products_and_chains() {
    AttnParams p;
    p.d_qk = 7; p.d_v = 1; p.scale = 1;
    const float q[] = {-1.0f, 1.0009765625f, 1.0009765625f, 0.00048828125f,
                       0.00048828125f, -0.00048828125f, 0.25f};
    const float k[] = {1, 1.0009765625f, 1.001953125f, 1, 1, 1, 0.5f};
    CHECK(scalar_half(static_cast<double>(q[2]) * k[2] - 1.0) !=
          scalar_half(static_cast<double>(scalar_half(static_cast<double>(q[2]) * k[2])) - 1.0));
    const float v = 3;
    float o, lse;
    TiledModelConfig cfg{32, 64, false, false, false};
    cfg.qk_products_fp16 = true;
    float expected = 0;
    for (int d = 0; d < p.d_qk; ++d) expected += scalar_half(static_cast<double>(q[d]) * k[d]);
    attention_tiled_model(p, cfg, q, k, &v, nullptr, &o, &lse);
    CHECK(lse == expected && o == v);
    cfg.qk_products_fp16 = false;
    for (int n : {1, 2, 4, 8}) {
        cfg.qk_half2_chain = n;
        float dot = 0, lanes[2] = {0, 0};
        int steps = 0;
        for (int d = 0; d < p.d_qk; d += 2) {
            for (int lane = 0; lane < 2 && d + lane < p.d_qk; ++lane)
                lanes[lane] = scalar_half(static_cast<double>(q[d + lane]) * k[d + lane] + lanes[lane]);
            if (++steps == n || d + 2 >= p.d_qk) {
                dot += lanes[0] + lanes[1]; lanes[0] = lanes[1] = 0; steps = 0;
            }
        }
        attention_tiled_model(p, cfg, q, k, &v, nullptr, &o, &lse);
        CHECK(lse == dot && o == v);
    }
}

void sum_and_rescale() {
    AttnParams p;
    p.seq_kv = 2; p.d_qk = p.d_v = 1; p.scale = 1;
    const float q = 1, k[] = {0, 1}, v[] = {2, 3};
    float o, lse;
    TiledModelConfig cfg{32, 1, false, true, false};
    cfg.l_from_unrounded_p = true;
    cfg.o_accum_fp16 = true;
    attention_tiled_model(p, cfg, &q, k, v, nullptr, &o, &lse);
    const float alpha = std::exp(-1.0f);
    const float acc = scalar_half(static_cast<double>(scalar_half(2.0 * scalar_half(alpha))) + 3.0);
    CHECK(o == acc / (alpha + 1));
    CHECK(near(lse, 1.0 + std::log1p(std::exp(-1.0)), analytic_tight_abs_tolerance));
    cfg.bc = 2;
    cfg.o_accum_fp16 = false;
    attention_tiled_model(p, cfg, &q, k, v, nullptr, &o, &lse);
    const float rounded_p = scalar_half(alpha);
    CHECK(o == (rounded_p * 2 + 3) / (alpha + 1));
    cfg.l_from_unrounded_p = false;
    attention_tiled_model(p, cfg, &q, k, v, nullptr, &o, &lse);
    CHECK(o == (rounded_p * 2 + 3) / (rounded_p + 1));
}

void swamping() {
    AttnParams p;
    p.seq_kv = 8192; p.d_qk = p.d_v = 64;
    std::vector<float> q(64, 0), k(8192 * 64, 0), v(k.size(), 1), o(64);
    TiledModelConfig cfg{32, 64, false, true, true};
    attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), nullptr, o.data(), nullptr);
    for (float x : o) CHECK(out_within_tolerance(x, 1.0));
    const float fp32 = o[0];
    cfg.o_accum_fp16 = true;
    attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), nullptr, o.data(), nullptr);
    std::cout << "swamping exact=1 fp32=" << fp32 << " fp16_accumulator=" << o[0] << '\n';
}

void invalid() {
    const AttnParams p;
    float q[128] = {}, k[128] = {}, v[128] = {}, o[128];
    for (int bits = 0; bits < 8; ++bits) {
        TiledModelConfig cfg;
        cfg.qk_accum_fp16 = bits & 1; cfg.qk_products_fp16 = bits & 2; cfg.qk_half2_chain = bits & 4 ? 2 : 0;
        bool threw = false;
        try { attention_tiled_model(p, cfg, q, k, v, nullptr, o, nullptr); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw == ((bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1) > 1));
    }
    for (float prescale : {0.0f, -0.25f}) {
        TiledModelConfig cfg; cfg.qk_q_prescale = prescale;
        bool threw = false;
        try { attention_tiled_model(p, cfg, q, k, v, nullptr, o, nullptr); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
}
} // namespace

int main(int argc, char** argv) {
    return run_tests(argc, argv, {{"double_rounding", double_rounding}, {"scale_and_offset", scale_and_offset},
        {"products_and_chains", products_and_chains}, {"sum_and_rescale", sum_and_rescale},
        {"swamping", swamping}, {"invalid", invalid}});
}
