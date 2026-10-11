#include "p100/qmv.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace p100::qmv;
struct Cell { std::vector<double> norm, relative; int nonfinite=0; };
static double median(std::vector<double> v) {
    std::sort(v.begin(),v.end()); return (v[127]+v[128])*0.5;
}
int main() {
    std::cout << "Seeds: activations=20261010 + K + distribution_index; weights=60060 + K + type_index.\n"
              << "256 rows per cell; same weight rows across distributions; outlier uses four distinct channels.\n"
              << "Weight scales uniform [0.002,0.08], packed quant/scales bytes uniform [0,255].\n"
              << "type path distribution K norm_median norm_max relative_median relative_max nonfinite\n";
    bool bad=false;
    for (int k : {4096,14336}) {
        int ti=0;
        for (Type t : {Type::q4_0,Type::q8_0,Type::q4_K}) {
            std::mt19937 wrng(60060+k+ti++);
            std::vector<std::vector<unsigned char>> rows;
            for(int r=0;r<256;++r) rows.push_back(random_row(t,k,wrng));
            int di=0;
            for(const char* dist : {"normal","outlier","swiglu","large"}) {
                const int index=di++;
                std::mt19937 rng(20261010+k+index); std::normal_distribution<float> normal(0,1);
                std::vector<float> x(k);
                for(auto& v:x) { v=normal(rng); if(index==2) v*=normal(rng)*3.f; if(index==3) v*=1000.f; }
                if(index==1) {
                    std::vector<int> positions(k); for(int i=0;i<k;++i) positions[i]=i;
                    std::shuffle(positions.begin(),positions.end(),rng);
                    for(int i=0;i<4;++i) x[positions[i]]*=50.f;
                }
                double xn=0; for(auto v:x) xn+=double(v)*v; xn=std::sqrt(xn);
                const auto act=prepare(x.data(),k); std::array<Cell,4> cells;
                for(const auto& w:rows) {
                    const auto y=evaluate(t,w.data(),x.data(),k,act);
                    const double paths[]={y.q81,y.h8,y.h16,y.h16_exact};
                    for(int j=0;j<4;++j) {
                        const double error=std::fabs(paths[j]-y.exact), scale=y.weight_norm*xn;
                        cells[j].norm.push_back(error/scale);
                        cells[j].relative.push_back(error/std::max(std::fabs(y.exact),1e-6*scale));
                        if(!std::isfinite(paths[j])) { ++cells[j].nonfinite; bad=true; }
                    }
                }
                for(int j=0;j<4;++j) {
                    const auto& c=cells[j];
                    std::cout << name(t) << ' ' << (j==0?"q81":j==1?"h16<8>":j==2?"h16<16>":"h16_exact") << ' '
                              << dist << ' ' << k << ' ' << std::scientific << std::setprecision(8)
                              << median(c.norm) << ' ' << *std::max_element(c.norm.begin(),c.norm.end()) << ' '
                              << median(c.relative) << ' ' << *std::max_element(c.relative.begin(),c.relative.end())
                              << ' ' << c.nonfinite << '\n';
                }
            }
        }
    }
    return bad ? 1 : 0;
}
