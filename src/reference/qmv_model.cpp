#include "p100/qmv.h"
#include "p100/attention.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace p100 { namespace qmv {
const char* name(Type t) {
    return t == Type::q4_0 ? "q4_0" : t == Type::q8_0 ? "q8_0" : "q4_K";
}
int block_size(Type t) { return t == Type::q4_K ? 256 : 32; }
int block_bytes(Type t) { return t == Type::q4_0 ? 18 : t == Type::q8_0 ? 34 : 144; }
unsigned short half_bits(float x) {
    const float r = round_to_fp16(x);
    unsigned u; std::memcpy(&u, &r, 4);
    const unsigned s = (u >> 16) & 0x8000, e = (u >> 23) & 255, f = u & 0x7fffff;
    if (e == 255) return s | 0x7c00 | (f ? 0x200 : 0);
    if (e == 0) return s;
    if (e < 113) return s | ((f | 0x800000) >> (126-e));
    return s | ((e-112) << 10) | (f >> 13);
}
float half_value(unsigned short h) {
    const float a = (h & 0x7c00) == 0x7c00 ?
        ((h & 1023) ? NAN : INFINITY) :
        (h & 0x7c00) ? std::ldexp(float(1024 + (h & 1023)), ((h >> 10) & 31)-25) :
                       std::ldexp(float(h & 1023), -24);
    return h & 0x8000 ? -a : a;
}
std::vector<unsigned char> random_row(Type t, int k, std::mt19937& rng) {
    if (k <= 0 || k % block_size(t)) throw std::invalid_argument("whole weight blocks required");
    std::vector<unsigned char> v(k/block_size(t)*block_bytes(t));
    std::uniform_real_distribution<float> scale(0.002f, 0.08f);
    std::uniform_int_distribution<int> byte(0,255);
    for (int b=0; b<k/block_size(t); ++b) {
        auto* p = v.data()+b*block_bytes(t);
        for (int i=0; i<block_bytes(t); ++i) p[i] = byte(rng);
        const unsigned short d = half_bits(scale(rng));
        std::memcpy(p, &d, 2);
        if (t == Type::q4_K) { const auto m = half_bits(scale(rng)); std::memcpy(p+2, &m, 2); }
    }
    return v;
}
// Matches common.cuh:468-474 (XOR offsets 16,8,4,2,1), including FP32 rounding.
static float warp_sum(float* a) {
    float tmp[32];
    for (int off=16; off; off/=2) {
        for (int i=0;i<32;++i) tmp[i] = a[i]+a[i^off];
        std::copy(tmp,tmp+32,a);
    }
    return a[0];
}
std::vector<ActivationBlock> prepare(const float* x, int k) {
    if (k <= 0 || k%32) throw std::invalid_argument("whole activation blocks required");
    std::vector<ActivationBlock> v(k/32);
    for (int b=0;b<k/32;++b) {
        auto& a=v[b]; a.scale=0;
        for (int i=0;i<32;++i) a.scale=std::max(a.scale,std::fabs(x[b*32+i]));
        float sum[32], hs[32];
        // quantize.cu:88-110: roundf(x/d), ds = half2(d, original sum).
        const float d=a.scale/127.0f;
        for (int i=0;i<32;++i) {
            const float xi=x[b*32+i]; sum[i]=xi;
            const int perm=(i&~3) | ((i&1)<<1) | ((i&2)>>1);
            a.h[perm]=hs[i]=round_to_fp16(a.scale == 0 ? 0.f : xi/a.scale);
            a.q[i]=a.scale == 0 ? 0 : int(std::round(xi/d));
        }
        a.sum=warp_sum(hs); a.qsum=round_to_fp16(warp_sum(sum)); a.qd=round_to_fp16(d);
    }
    return v;
}
static void scale_min(const Q4_K& b,int j,int& s,int& m) {
    // dequantize.cuh:177-185, get_scale_min_k4, unchanged six-bit packing.
    if (j<4) { s=b.scales[j]&63; m=b.scales[j+4]&63; }
    else { s=(b.scales[j+4]&15)|((b.scales[j-4]>>6)<<4);
           m=(b.scales[j+4]>>4)|((b.scales[j]>>6)<<4); }
}
struct Sub { float d,m; int q[32],sc,mn; };
static Sub unpack(Type t,const unsigned char* p,int sub) {
    Sub s{}; unsigned short d; std::memcpy(&d,p,2); s.d=half_value(d); s.sc=1;
    if (t==Type::q4_0) {
        for (int i=0;i<32;++i) s.q[i]=((p[2+i%16]>>(i/16*4))&15)-8;
    } else if (t==Type::q8_0) {
        for (int i=0;i<32;++i) s.q[i]=p[2+i]<128 ? p[2+i] : int(p[2+i])-256;
    } else {
        Q4_K b; std::memcpy(&b,p,sizeof(b)); s.m=half_value(b.dmin);
        scale_min(b,sub,s.sc,s.mn);
        for (int i=0;i<32;++i) s.q[i]=(b.qs[(sub/2)*32+i]>>((sub%2)*4))&15;
    }
    return s;
}
static double hdot_exact(const Sub& w,const ActivationBlock& a) {
    double dot=0;
    for(int i=0;i<32;++i) {
        const int perm=(i&~3)|((i&1)<<1)|((i&2)>>1);
        dot+=double(w.q[i])*a.h[perm];
    }
    return double(w.d)*w.sc*a.scale*dot-double(w.m)*w.mn*a.scale*a.sum;
}
double h16_exact(Type t,const void* weights,int k,const std::vector<ActivationBlock>& a) {
    if(k<=0 || k%block_size(t) || int(a.size())!=k/32)
        throw std::invalid_argument("invalid h16_exact shape");
    double out=0;
    const auto* p=static_cast<const unsigned char*>(weights);
    for(int b=0;b<k/block_size(t);++b) for(int sub=0;sub<block_size(t)/32;++sub)
        out+=hdot_exact(unpack(t,p+b*block_bytes(t),sub),a[b*block_size(t)/32+sub]);
    return out;
}
static float hdot(const Sub& w,const ActivationBlock& a,int start,int chain) {
        float lo=0,hi=0;
        for (int j=start;j<start+chain;++j) {
            const int i=2*j, p=(i&~3)|((i&1)<<1)|((i&2)>>1);
            const int ip=i+1, pp=(ip&~3)|((ip&1)<<1)|((ip&2)>>1);
            // Each fused half FMA gets exactly ONE FP64-to-FP16 rounding.
            lo=round_to_fp16(double(w.q[p])*a.h[i]+lo);
            hi=round_to_fp16(double(w.q[pp])*a.h[ip]+hi);
        }
        return lo+hi;
}
float h16_model(Type t,const void* weights,int k,const std::vector<ActivationBlock>& a,int chain) {
    if ((chain!=8 && chain!=16) || k%block_size(t) || int(a.size())!=k/32)
        throw std::invalid_argument("invalid h16 shape/chain");
    float out=0;
    const auto* p=static_cast<const unsigned char*>(weights);
    for(int b=0;b<k/block_size(t);++b) for(int sub=0;sub<block_size(t)/32;++sub) {
        const Sub w=unpack(t,p+b*block_bytes(t),sub); const auto& x=a[b*block_size(t)/32+sub];
        const float ds=(w.d*w.sc)*x.scale;
        const float ms=(w.m*w.mn)*x.scale;
        for(int start=0;start<16;start+=chain) out+=hdot(w,x,start,chain)*ds;
        if(t==Type::q4_K) out-=ms*x.sum;
    }
    return out;
}
Result evaluate(Type t,const void* weights,const float* x,int k,const std::vector<ActivationBlock>& a) {
    Result r{}; const auto* p=static_cast<const unsigned char*>(weights);
    for(int b=0;b<k/block_size(t);++b) {
        Sub w[8];
        for(int sub=0;sub<block_size(t)/32;++sub) {
            w[sub]=unpack(t,p+b*block_bytes(t),sub);
            const int base=b*block_size(t)+sub*32;
            r.h16_exact+=hdot_exact(w[sub],a[base/32]);
            for(int i=0;i<32;++i) {
                const double v=double(w[sub].d)*w[sub].sc*w[sub].q[i]-double(w[sub].m)*w[sub].mn;
                r.exact+=v*x[base+i]; r.weight_norm+=v*v;
            }
        }
        if(t!=Type::q4_K) {
            const auto& act=a[b];
            // vecdotq.cuh:118-137,246-260,774-789,851-865: VDR=2.
            // q4_0 calls cover bytes 0..7,8..15 and both nibble planes.
            for(int call=0;call<(t==Type::q4_0 ? 2 : 4);++call) {
                int dot=0;
                for(int i=0;i<8;++i) {
                    const int j=call*8+i;
                    dot+=(w[0].q[j]+(t==Type::q4_0 ? 8 : 0))*act.q[j];
                    if(t==Type::q4_0) dot+=(w[0].q[j+16]+8)*act.q[j+16];
                }
                if(t==Type::q4_0) {
                    // CUDA 12.8 sm_60 SASS also contracts the outer row add.
                    r.q81=std::fma(w[0].d,std::fma(float(dot),act.qd,-4.f*act.qsum),r.q81);
                } else r.q81=std::fma(w[0].d*act.qd,float(dot),r.q81);
            }
        } else {
            // vecdotq.cuh:508-530,918-965: each call consumes 8 values
            // in each of two neighbouring sub-blocks (j..j+3,j+16..j+19).
            for(int group=0;group<4;++group) for(int call=0;call<4;++call) {
                float sd=0,sm=0;
                for(int z=0;z<2;++z) {
                    const int sub=group*2+z; const auto& act=a[b*8+sub];
                    int dot=0,sum=0;
                    for(int i=0;i<8;++i) {
                        const int j=call*4+i%4+(i/4)*16;
                        dot+=w[sub].q[j]*act.q[j]; sum+=act.q[j];
                    }
                    sd=std::fma(act.qd,float(dot*w[sub].sc),sd);
                    sm=std::fma(act.qd,float(sum*w[sub].mn),sm);
                }
                r.q81+=std::fma(w[0].d,sd,-w[0].m*sm);
            }
        }
    }
    r.weight_norm=std::sqrt(r.weight_norm);
    r.h8=h16<8>(t,weights,k,a); r.h16=h16<16>(t,weights,k,a);
    return r;
}
} }
