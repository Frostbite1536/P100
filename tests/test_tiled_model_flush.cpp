#include "check.h"
#include "p100/attention.h"
#include "p100/tolerance.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace p100;

namespace {
void swamping() {
    AttnParams p;
    p.seq_kv = 8192; p.d_qk = p.d_v = 64;
    std::vector<float> q(64, 0), k(8192 * 64, 0), v(k.size(), 1), o(64);
    TiledModelConfig cfg{32, 64, false, true, true};
    cfg.o_accum_fp16 = cfg.o_accum_fp16_tile_flush = true;
    for (auto split : {std::pair<int, int>{1, 1}, {1, 4}, {8, 1}, {8, 4}}) {
        cfg.kv_parallel_blocks = split.first; cfg.o_accum_np = split.second;
        attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), nullptr, o.data(), nullptr);
        for (float x : o) CHECK(out_within_tolerance(x, 1.0));
    }
}

void rescale_and_combine() {
    AttnParams p;
    p.seq_kv = 2; p.d_qk = p.d_v = 1; p.scale = 1;
    const float q = 1, k[] = {0, 1}, v[] = {2, 3};
    float o, lse;
    TiledModelConfig cfg{1, 1, false, true, false};
    cfg.l_from_unrounded_p = true;
    cfg.o_accum_fp16 = cfg.o_accum_fp16_tile_flush = true;
    attention_tiled_model(p, cfg, &q, k, v, nullptr, &o, &lse);
    const float alpha = std::exp(-1.0f);
    CHECK(o == (2 * alpha + 3) / (alpha + 1));
    CHECK(lse_within_tolerance(lse, 1.0 + std::log1p(std::exp(-1.0))));

    p.seq_kv = 3;
    const float zero = 0, keys[] = {0, 0, 0}, values[] = {2048, 1, 1};
    cfg.bc = cfg.o_accum_np = 3;
    attention_tiled_model(p, cfg, &zero, keys, values, nullptr, &o, nullptr);
    CHECK(o == 2050.0f / 3.0f);
}

void flag_off() {
    AttnParams p;
    p.seq_q = 3; p.seq_kv = 137; p.d_qk = p.d_v = 64;
    std::mt19937 rng(20261010);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<float> q(p.seq_q * p.d_qk), k(p.seq_kv * p.d_qk), v(k.size());
    for (auto* data : {&q, &k, &v}) for (float& x : *data) x = round_to_fp16(random(rng));
    std::vector<float> out(q.size()), lse(p.seq_q);
    TiledModelConfig cfg{2, 16, false, true, false};
    cfg.o_accum_fp16 = true; cfg.kv_parallel_blocks = 3; cfg.o_accum_np = 4;
    cfg.kq_max_offset = 3 * 0.6931f; cfg.l_from_unrounded_p = true;
    cfg.o_accum_fp16_tile_flush = false;
    attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), nullptr, out.data(), lse.data());
    // FNV-1a over every binary32 output and LSE word from the pre-flush model.
    std::uint64_t digest = 14695981039346656037ull;
    for (auto* data : {&out, &lse}) for (float x : *data) {
        digest ^= float_bits(x); digest *= 1099511628211ull;
    }
    std::cout << "flag_off_digest=" << digest << '\n';
    CHECK(digest == 12704262495437203684ull);
}
} // namespace

int main(int argc, char** argv) {
    return run_tests(argc, argv, {{"swamping", swamping}, {"rescale_and_combine", rescale_and_combine},
                                 {"flag_off", flag_off}});
}
