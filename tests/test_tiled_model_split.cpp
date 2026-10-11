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
void fp32() {
    // Uneven tiles, GQA, masked and causal rows exercise empty parts and rescale.
    AttnParams p;
    p.batch = 2; p.n_head_q = 4; p.n_head_kv = 2;
    p.seq_q = 7; p.seq_kv = 137; p.d_qk = 64; p.d_v = 33;
    std::mt19937 rng(20261010);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<float> q(p.batch * p.n_head_q * p.seq_q * p.d_qk);
    std::vector<float> k(p.batch * p.n_head_kv * p.seq_kv * p.d_qk);
    std::vector<float> v(p.batch * p.n_head_kv * p.seq_kv * p.d_v);
    for (auto* data : {&q, &k, &v}) for (float& x : *data) x = random(rng);
    std::vector<float> mask(p.seq_q * p.seq_kv, 0);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::fill(mask.begin(), mask.begin() + p.seq_kv, neg_inf);
    for (int j = 0; j < p.seq_kv; ++j) mask[p.seq_kv + j] = j == 0 ? 0 : neg_inf;
    std::vector<float> base(p.batch * p.n_head_q * p.seq_q * p.d_v), out(base.size());
    std::vector<float> base_lse(p.batch * p.n_head_q * p.seq_q), lse(base_lse.size());
    TiledModelConfig cfg{3, 16, false, false, false};
    for (bool causal : {false, true}) {
        p.causal = causal;
        cfg.kv_parallel_blocks = cfg.o_accum_np = 1;
        attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), mask.data(), base.data(), base_lse.data());
        cfg.kv_parallel_blocks = cfg.o_accum_np = 4;
        attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), mask.data(), out.data(), lse.data());
        for (std::size_t i = 0; i < out.size(); ++i) CHECK(out_within_tolerance(out[i], base[i]));
        for (std::size_t i = 0; i < lse.size(); ++i) CHECK(lse_within_tolerance(lse[i], base_lse[i]));
    }
}

void swamping() {
    AttnParams p;
    p.seq_kv = 8192; p.d_qk = p.d_v = 64;
    std::vector<float> q(64, 0), k(8192 * 64, 0), v(k.size(), 1), o(64);
    TiledModelConfig cfg{32, 64, false, true, true};
    cfg.o_accum_fp16 = true;
    float baseline_error = 0;
    for (auto split : {std::pair<int, int>{1, 1}, {1, 4}, {8, 1}, {8, 4}}) {
        cfg.kv_parallel_blocks = split.first; cfg.o_accum_np = split.second;
        attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), nullptr, o.data(), nullptr);
        if (split.first == 1 && split.second == 1) baseline_error = std::abs(o[0] - 1.0f);
        for (float x : o) CHECK(std::isfinite(x) && std::abs(x - 1.0f) <= baseline_error);
        std::cout << "swamping blocks=" << split.first << " np=" << split.second << " result=" << o[0] << '\n';
    }
}

void half_reduction() {
    AttnParams p;
    p.seq_kv = 3; p.d_qk = p.d_v = 1;
    const float q = 0, k[] = {0, 0, 0}, v[] = {2048, 1, 1};
    float out;
    TiledModelConfig cfg{1, 3, false, true, false};
    cfg.o_accum_np = 3; cfg.o_accum_fp16 = true;
    attention_tiled_model(p, cfg, &q, k, v, nullptr, &out, nullptr);
    // Each ordered half addition loses the tie: (2048 + 1) + 1 == 2048.
    CHECK(out == 2048.0f / 3.0f);
    cfg.o_accum_fp16 = false;
    attention_tiled_model(p, cfg, &q, k, v, nullptr, &out, nullptr);
    CHECK(out == 2050.0f / 3.0f);
}

void invalid() {
    AttnParams p;
    float q[128] = {}, k[128] = {}, v[128] = {}, o[128];
    for (int value : {0, -1}) for (bool blocks : {false, true}) {
        TiledModelConfig cfg;
        if (blocks) cfg.kv_parallel_blocks = value;
        else cfg.o_accum_np = value;
        bool threw = false;
        try { attention_tiled_model(p, cfg, q, k, v, nullptr, o, nullptr); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
}
} // namespace

int main(int argc, char** argv) {
    return run_tests(argc, argv, {{"fp32", fp32}, {"swamping", swamping}, {"half_reduction", half_reduction}, {"invalid", invalid}});
}
