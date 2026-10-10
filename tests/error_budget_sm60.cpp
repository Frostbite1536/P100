#include "p100/attention.h"
#include "p100/tolerance.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace p100;

int main(int argc, char** argv) {
    try {
        bool long_run = false;
        if (argc == 2 && std::string(argv[1]) == "--long") long_run = true;
        else if (argc != 1) throw std::invalid_argument("usage: error_budget_sm60 [--long]");
        const auto start = std::chrono::steady_clock::now();
        const char* distributions[] = {"normal", "outlier", "peaked", "vbias"};
        const char* names[] = {"PO", "L60", "L60o32", "C2", "C4", "C8"};
        TiledModelConfig configs[6];
        configs[0] = {32, 64, false, true, true};
        configs[1] = configs[0];
        configs[1].qk_scale_in_q = true;
        configs[1].qk_q_prescale = 0.25f;
        configs[1].qk_products_fp16 = true;
        configs[1].kq_max_offset = 3 * 0.6931f;
        configs[1].l_from_unrounded_p = true;
        configs[1].o_accum_fp16 = true;
        configs[2] = configs[1]; configs[2].o_accum_fp16 = false;
        for (int i = 3; i < 6; ++i) {
            configs[i] = configs[0];
            configs[i].qk_scale_in_q = true;
            configs[i].qk_q_prescale = 0.25f;
            configs[i].qk_half2_chain = 1 << (i - 2);
        }
        struct Summary {
            double max_rel = 0, lse_abs = 0;
            std::size_t out_of_tol = 0, lse_out_of_tol = 0, nonfinite = 0;
        } summaries[6];
        std::cout << "seq_q=16 batch=heads=1 br=32 bc=64; inputs rounded to FP16\n"
                  << "seed=12345+distribution_index*100000+d*100+seq_kv (mt19937)\n"
                  << "normal: scale=1/sqrt(d); outlier: Q,K channels 0..3 x20; "
                     "peaked: scale=64/sqrt(d); vbias: V channels 0..7 +2; V~N(0,1) otherwise\n"
                  << "max_rel=max_abs/max|O_ref|; nonfinite counts both reference and model\n"
                  << "cfg dist d kv causal max_abs max_rel mean_abs lse_abs nonfinite\n";
        std::cout << "kv_shapes=" << (long_run ? "512,4096,32768" : "512,4096") << '\n';
        std::cout << std::scientific << std::setprecision(6);
        for (int distribution = 0; distribution < 4; ++distribution) {
            for (int dim : {64, 128, 256}) {
                for (int nk : {512, 4096, 32768}) {
                    if (nk == 32768 && !long_run) continue;
                    AttnParams p;
                    p.seq_q = 16;
                    p.seq_kv = nk;
                    p.d_qk = p.d_v = dim;
                    if (distribution == 2) p.scale = 64.0f / std::sqrt(static_cast<float>(dim));
                    const unsigned seed = 12345u + static_cast<unsigned>(distribution) * 100000u +
                                          static_cast<unsigned>(dim) * 100u + static_cast<unsigned>(nk);
                    std::mt19937 rng(seed);
                    std::normal_distribution<float> normal(0.0f, 1.0f);
                    std::vector<float> q(static_cast<std::size_t>(p.seq_q) * dim);
                    std::vector<float> k(static_cast<std::size_t>(nk) * dim), v(k.size());
                    for (auto* data : {&q, &k}) {
                        for (std::size_t i = 0; i < data->size(); ++i) {
                            float x = normal(rng);
                            if (distribution == 1 && i % dim < 4) x *= 20.0f;
                            (*data)[i] = round_to_fp16(x);
                        }
                    }
                    for (std::size_t i = 0; i < v.size(); ++i)
                        v[i] = round_to_fp16(normal(rng) + (distribution == 3 && i % dim < 8 ? 2.0f : 0.0f));
                    std::vector<float> ref(q.size()), out(q.size()), ref_lse(p.seq_q), out_lse(p.seq_q);
                    for (bool causal : {false, true}) {
                        p.causal = causal;
                        attention_reference(p, q.data(), k.data(), v.data(), nullptr, ref.data(), ref_lse.data());
                        for (int config = 0; config < 6; ++config) {
                            attention_tiled_model(p, configs[config], q.data(), k.data(), v.data(), nullptr,
                                                  out.data(), out_lse.data());
                            double max_abs = 0, max_ref = 0, sum_abs = 0, lse_abs = 0;
                            std::size_t nonfinite = 0;
                            for (std::size_t i = 0; i < ref.size(); ++i) {
                                nonfinite += !std::isfinite(ref[i]);
                                nonfinite += !std::isfinite(out[i]);
                                const double error = std::isfinite(out[i]) && std::isfinite(ref[i])
                                    ? std::abs(static_cast<double>(out[i]) - ref[i])
                                    : std::numeric_limits<double>::infinity();
                                max_abs = std::max(max_abs, error);
                                max_ref = std::max(max_ref, std::abs(static_cast<double>(ref[i])));
                                sum_abs += error;
                                if (!out_within_tolerance(out[i], ref[i])) ++summaries[config].out_of_tol;
                            }
                            for (int i = 0; i < p.seq_q; ++i) {
                                // Every row here has visible keys, including under causal.
                                nonfinite += !std::isfinite(ref_lse[i]);
                                nonfinite += !std::isfinite(out_lse[i]);
                                const double error = std::isfinite(out_lse[i]) && std::isfinite(ref_lse[i])
                                    ? std::abs(static_cast<double>(out_lse[i]) - ref_lse[i])
                                    : std::numeric_limits<double>::infinity();
                                lse_abs = std::max(lse_abs, error);
                                if (!lse_within_tolerance(out_lse[i], ref_lse[i])) ++summaries[config].lse_out_of_tol;
                            }
                            summaries[config].nonfinite += nonfinite;
                            const double max_rel = !std::isfinite(max_abs) || !std::isfinite(max_ref)
                                ? std::numeric_limits<double>::infinity() : (max_ref == 0 ? 0 : max_abs / max_ref);
                            summaries[config].max_rel = std::max(summaries[config].max_rel, max_rel);
                            summaries[config].lse_abs = std::max(summaries[config].lse_abs, lse_abs);
                            std::cout << names[config] << ' ' << distributions[distribution] << ' ' << dim << ' '
                                      << nk << ' ' << static_cast<int>(causal) << ' ' << max_abs << ' ' << max_rel << ' '
                                      << sum_abs / ref.size() << ' ' << lse_abs << ' ' << nonfinite << '\n';
                        }
                    }
                }
            }
        }
        std::cout << "summary cfg worst_max_rel worst_lse_abs out_outside_tolerance lse_outside_tolerance nonfinite\n";
        for (int i = 0; i < 6; ++i) {
            const auto& summary = summaries[i];
            std::cout << names[i] << ' ' << summary.max_rel << ' ' << summary.lse_abs << ' '
                      << summary.out_of_tol << ' ' << summary.lse_out_of_tol << ' ' << summary.nonfinite << '\n';
        }
        std::cout << "elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << " mode=" << (long_run ? "long" : "short") << '\n';
        const auto& po = summaries[0];
        return po.nonfinite == 0 && po.out_of_tol == 0 && po.lse_out_of_tol == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
