#include "p100/attention.h"
#include "p100/tolerance.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace p100;

namespace {
struct Row {
    int cc, d, bc, np, blocks, nk, keys;
    std::string kernel;
    auto key() const { return std::make_tuple(cc, d, kernel, bc, np, blocks, nk); }
};
struct Metrics {
    double max_abs = 0, max_rel = 0, mean_abs = 0, lse_abs = 0;
    std::size_t outside = 0, nonfinite = 0;
};
Metrics measure(const std::vector<float>& ref, const std::vector<float>& ref_lse,
                const std::vector<float>& out, const std::vector<float>& lse) {
    Metrics m;
    double max_ref = 0, sum_abs = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        m.nonfinite += !std::isfinite(ref[i]); m.nonfinite += !std::isfinite(out[i]);
        const double error = std::isfinite(out[i]) && std::isfinite(ref[i])
            ? std::abs(static_cast<double>(out[i]) - ref[i]) : std::numeric_limits<double>::infinity();
        m.max_abs = std::max(m.max_abs, error);
        max_ref = std::max(max_ref, std::abs(static_cast<double>(ref[i])));
        sum_abs += error;
        m.outside += !out_within_tolerance(out[i], ref[i]);
    }
    for (std::size_t i = 0; i < lse.size(); ++i) {
        m.nonfinite += !std::isfinite(ref_lse[i]); m.nonfinite += !std::isfinite(lse[i]);
        m.lse_abs = std::max(m.lse_abs, std::isfinite(lse[i]) && std::isfinite(ref_lse[i])
            ? std::abs(static_cast<double>(lse[i]) - ref_lse[i]) : std::numeric_limits<double>::infinity());
    }
    m.mean_abs = sum_abs / ref.size();
    m.max_rel = max_ref == 0 ? 0 : m.max_abs / max_ref;
    return m;
}
std::vector<std::string> columns(const std::string& line) {
    std::vector<std::string> result;
    std::istringstream input(line);
    std::string cell;
    while (std::getline(input, cell, ',')) result.push_back(cell);
    return result;
}
std::vector<Row> read_rows(bool long_run) {
    std::ifstream input(std::string(P100_SOURCE_DIR) + "/docs/research/fa-split-sm60-sm70-2026-10-10.csv");
    if (!input) throw std::runtime_error("cannot open split CSV");
    std::string line;
    std::getline(input, line);
    const auto header = columns(line);
    std::map<std::string, std::size_t> indices;
    for (std::size_t i = 0; i < header.size(); ++i) indices.emplace(header[i], i);
    std::vector<Row> rows;
    std::set<decltype(Row{}.key())> seen;
    while (std::getline(input, line)) {
        const auto cells = columns(line);
        auto value = [&](const std::string& name) { return cells.at(indices.at(name)); };
        auto number = [&](const std::string& name) { return std::stoi(value(name)); };
        // Volta MMA rows have no applicable TILE/VEC batch or accumulator model.
        if (value("kernel") == "other") continue;
        Row r{number("cc"), number("d"), number("nbatch_fa"), number("np"),
              number("parallel_blocks"), number("seq_kv"), number("keys_per_fp16_accumulator"), value("kernel")};
        if (!long_run && r.nk > 16384) continue;
        if (r.bc < 1 || r.np < 1 || r.blocks < 1) throw std::runtime_error("invalid split CSV row");
        if (seen.insert(r.key()).second) rows.push_back(r);
    }
    if (rows.empty()) throw std::runtime_error("empty split grid");
    return rows;
}
TiledModelConfig l60(int bc) {
    // Copied L60 definition from error_budget_sm60.cpp:28-35; no inclusion.
    TiledModelConfig cfg{32, 64, false, true, true};
    cfg.qk_scale_in_q = true;
    cfg.qk_q_prescale = 0.25f;
    cfg.qk_products_fp16 = true;
    cfg.kq_max_offset = 3 * 0.6931f;
    cfg.l_from_unrounded_p = true;
    cfg.o_accum_fp16 = true;
    cfg.bc = bc;
    return cfg;
}
struct Inputs {
    AttnParams p;
    std::vector<float> q, k, v, ref, ref_lse;
    Inputs(int d, int nk, int distribution) {
        p.seq_q = 16; p.seq_kv = nk; p.d_qk = p.d_v = d;
        // Exactly error_budget_sm60.cpp:62-78, distribution indices 0 and 3.
        const unsigned seed = 12345u + static_cast<unsigned>(distribution) * 100000u +
                              static_cast<unsigned>(d) * 100u + static_cast<unsigned>(nk);
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        q.resize(static_cast<std::size_t>(p.seq_q) * d);
        k.resize(static_cast<std::size_t>(nk) * d); v.resize(k.size());
        for (auto* data : {&q, &k}) for (float& x : *data) x = round_to_fp16(normal(rng));
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = round_to_fp16(normal(rng) + (distribution == 3 && i % d < 8 ? 2.0f : 0.0f));
        ref.resize(q.size()); ref_lse.resize(p.seq_q);
        attention_reference(p, q.data(), k.data(), v.data(), nullptr, ref.data(), ref_lse.data());
    }
    Metrics run(const TiledModelConfig& cfg) const {
        std::vector<float> out(q.size()), lse(p.seq_q);
        attention_tiled_model(p, cfg, q.data(), k.data(), v.data(), nullptr, out.data(), lse.data());
        return measure(ref, ref_lse, out, lse);
    }
};
void consistency() {
    const Inputs inputs(128, 4096, 0);
    auto cfg = l60(64);
    cfg.kv_parallel_blocks = cfg.o_accum_np = 1;
    const auto m = inputs.run(cfg);
    std::ostringstream row;
    row << std::scientific << std::setprecision(6) << "L60 normal 128 4096 0 " << m.max_abs << ' '
        << m.max_rel << ' ' << m.mean_abs << ' ' << m.lse_abs << ' ' << m.nonfinite;
    // Verbatim unmodified error_budget_sm60 output, built Release (-O3),
    // using attention_tiled_model.cpp from main @ 6554601. Compare the printed
    // precision of that executable; this is a compatibility check, not a tolerance.
    const std::string baseline = "L60 normal 128 4096 0 1.232397e-03 1.425396e-02 1.567871e-04 6.484985e-05 0";
    if (row.str() != baseline) throw std::runtime_error("L60 1/1 consistency failed: " + row.str());
    std::cout << "consistency PASS " << row.str() << '\n';
}
void add_summary(Metrics& summary, const Metrics& m) {
    summary.max_rel = std::max(summary.max_rel, m.max_rel);
    summary.lse_abs = std::max(summary.lse_abs, m.lse_abs);
    summary.outside += m.outside; summary.nonfinite += m.nonfinite;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool long_run = argc == 2 && std::string(argv[1]) == "--long";
        if (argc != 1 && !long_run) throw std::invalid_argument("usage: error_budget_split [--long]");
        const auto start = std::chrono::steady_clock::now();
        const auto rows = read_rows(long_run);
        std::cout << "seq_q=16 batch=heads=1 non-causal; inputs rounded to FP16; L60 br=32 bc=nbatch_fa\n"
                  << "seed=12345+distribution_index*100000+d*100+seq_kv; normal=0 vbias=3\n"
                  << "TILE rows model FP16 O; VEC rows are requested L60 counterfactuals (real NVIDIA VEC O is FP32); other/MMA rows skipped\n"
                  << "keys_per_fp16_accumulator=0 for VEC (inapplicable); max_rel=max_abs/max|O_ref|\n";
        consistency();
        std::cout << "cc,d,kernel,bc,np,parallel_blocks,seq_kv,keys_per_fp16_accumulator,dist,split_max_rel,split_lse_abs,split_outside,unsplit_max_rel,unsplit_lse_abs,unsplit_outside,nonfinite\n"
                  << std::scientific << std::setprecision(6);
        // Reuse metrics only for identical numerical configurations. No KV
        // quantisation is simulated: the ticket requests the same FP16 L60 inputs.
        using Shape = std::tuple<int, int, int>;
        using Run = std::tuple<int, int, int, int, int, int>;
        std::map<Shape, Inputs> input_cache;
        std::map<Run, Metrics> run_cache;
        Metrics split_summary, unsplit_summary, tile_split, tile_unsplit;
        for (const auto& r : rows) for (int distribution : {0, 3}) {
            const Shape shape{r.d, r.nk, distribution};
            auto it = input_cache.find(shape);
            if (it == input_cache.end()) it = input_cache.emplace(shape, Inputs(r.d, r.nk, distribution)).first;
            const auto& inputs = it->second;
            auto run = [&](int blocks, int np) {
                const Run key{r.d, r.nk, distribution, r.bc, blocks, np};
                auto found = run_cache.find(key);
                if (found == run_cache.end()) {
                    auto cfg = l60(r.bc); cfg.kv_parallel_blocks = blocks; cfg.o_accum_np = np;
                    found = run_cache.emplace(key, inputs.run(cfg)).first;
                }
                return found->second;
            };
            const auto split = run(r.blocks, r.np), unsplit = run(1, 1);
            add_summary(split_summary, split); add_summary(unsplit_summary, unsplit);
            if (r.kernel == "TILE") { add_summary(tile_split, split); add_summary(tile_unsplit, unsplit); }
            std::cout << r.cc << ',' << r.d << ',' << r.kernel << ',' << r.bc << ',' << r.np << ','
                      << r.blocks << ',' << r.nk << ',' << r.keys << ',' << (distribution == 0 ? "normal" : "vbias")
                      << ',' << split.max_rel << ',' << split.lse_abs << ',' << split.outside
                      << ',' << unsplit.max_rel << ',' << unsplit.lse_abs << ',' << unsplit.outside
                      << ',' << split.nonfinite + unsplit.nonfinite << '\n';
        }
        std::cout << "summary cfg worst_max_rel worst_lse_abs out_outside_tolerance nonfinite\n";
        for (auto item : {std::make_pair("split", split_summary), std::make_pair("unsplit", unsplit_summary),
                          std::make_pair("TILE_split", tile_split), std::make_pair("TILE_unsplit", tile_unsplit)})
            std::cout << item.first << ' ' << item.second.max_rel << ' ' << item.second.lse_abs << ' '
                      << item.second.outside << ' ' << item.second.nonfinite << '\n';
        std::cout << "elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
                  << " mode=" << (long_run ? "long" : "short") << " distinct_rows=" << rows.size() << '\n';
        return split_summary.nonfinite + unsplit_summary.nonfinite == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
