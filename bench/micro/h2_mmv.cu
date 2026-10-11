// Pascal sm_60 half2 decode prototype. No dynamic or static shared memory.
#include <cuda_fp16.h>
#include "p100/qmv.h"
using p100::qmv::Q4_0;
using p100::qmv::Q8_0;
using p100::qmv::Q4_K;
__device__ __forceinline__ half2 bits_half2(unsigned u) {
    return *reinterpret_cast<const half2*>(&u);
}
__device__ __forceinline__ unsigned mask_or(unsigned w,unsigned m,unsigned c) {
#if __CUDA_ARCH__ < 530
    return (w & m) | c;
#else
    // (w & m) | c, truth table (0xf0 & 0xcc) | 0xaa = 0xea.
    // CUDA 12.8 lowers the C++ expression to two immediate LOP32I ops,
    // even with register constraints; request the same ternary op explicitly.
    unsigned result;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(result) : "r"(w), "r"(m), "r"(c));
    return result;
#endif
}

#if __CUDA_ARCH__ < 530
// Maxwell checks indexing/layout only: half2 arithmetic is FP32, with no
// FP16 rounding of arithmetic results. Stored activations still use FP16.
using Pair = float2;
__device__ __forceinline__ Pair pair_bits(unsigned u) {
    return __half22float2(bits_half2(u));
}
__device__ __forceinline__ Pair zero_pair() { return make_float2(0,0); }
__device__ __forceinline__ Pair pair_sub(Pair a,float b) {
    return make_float2(a.x-b,a.y-b);
}
__device__ __forceinline__ Pair pair_fma(Pair a,Pair b,Pair c) {
    return make_float2(fmaf(a.x,b.x,c.x),fmaf(a.y,b.y,c.y));
}
__device__ __forceinline__ float pair_sum(Pair a) { return a.x+a.y; }
__device__ __forceinline__ Pair load_pair(const half* p) { return __half22float2(*reinterpret_cast<const half2*>(p)); }
#else
using Pair = half2;
__device__ __forceinline__ Pair pair_bits(unsigned u) { return bits_half2(u); }
__device__ __forceinline__ Pair zero_pair() { return __float2half2_rn(0); }
__device__ __forceinline__ Pair pair_sub(Pair a,float b) { return __hsub2(a,__float2half2_rn(b)); }
__device__ __forceinline__ Pair pair_fma(Pair a,Pair b,Pair c) { return __hfma2(a,b,c); }
__device__ __forceinline__ float pair_sum(Pair a) {
    const float2 v=__half22float2(a); return v.x+v.y;
}
__device__ __forceinline__ Pair load_pair(const half* p) { return *reinterpret_cast<const half2*>(p); }
#endif

// GGUF rows can begin at a two-byte boundary (18/34-byte block strides).
__device__ __forceinline__ unsigned word_b2(const void* p) {
    const auto* h=static_cast<const unsigned short*>(p);
    return unsigned(h[0]) | (unsigned(h[1])<<16);
}
__device__ __forceinline__ float scale_half(unsigned short h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ void scale_min(const Q4_K& b,int j,int& s,int& m) {
    // llama.cpp/ggml/src/ggml-cuda/dequantize.cuh:177-185.
    if(j<4) { s=b.scales[j]&63; m=b.scales[j+4]&63; }
    else { s=(b.scales[j+4]&15)|((b.scales[j-4]>>6)<<4);
           m=(b.scales[j+4]>>4)|((b.scales[j]>>6)<<4); }
}
__device__ __forceinline__ void unpack(const Q4_0& b,int,Pair* q,float& d,float& m) {
    d=scale_half(b.d); m=0;
#pragma unroll
    for(int i=0;i<4;++i) {
        const unsigned w=word_b2(b.qs+4*i), v=w>>8;
        q[2*i]   =pair_sub(pair_bits(mask_or(w,0x000f000fU,0x64006400U)),1032);
        q[2*i+1] =pair_sub(pair_bits(mask_or(v,0x000f000fU,0x64006400U)),1032);
        q[8+2*i] =pair_sub(pair_bits(mask_or(w,0x00f000f0U,0x54005400U)),72);
        q[9+2*i] =pair_sub(pair_bits(mask_or(v,0x00f000f0U,0x54005400U)),72);
    }
}
__device__ __forceinline__ void unpack(const Q8_0& b,int,Pair* q,float& d,float& m) {
    d=scale_half(b.d); m=0;
#pragma unroll
    for(int i=0;i<8;++i) {
        const unsigned u=word_b2(b.qs+4*i)^0x80808080U;
        // prmt bytes 0,2 (and 1,3) into the mantissas, 0x64 exponent bytes.
        q[2*i]  =pair_sub(pair_bits(__byte_perm(u,0x64646464U,0x4240)),1152);
        q[2*i+1]=pair_sub(pair_bits(__byte_perm(u,0x64646464U,0x4341)),1152);
    }
}
__device__ __forceinline__ void unpack(const Q4_K& b,int sub,Pair* q,float& d,float& m) {
    int sc,mn; scale_min(b,sub,sc,mn);
    d=__fmul_rn(scale_half(b.d),float(sc)); m=__fmul_rn(scale_half(b.dmin),float(mn));
#pragma unroll
    for(int i=0;i<8;++i) {
        const unsigned w=word_b2(b.qs+(sub/2)*32+4*i), v=w>>8;
        if(sub%2==0) {
            q[2*i]  =pair_sub(pair_bits(mask_or(w,0x000f000fU,0x64006400U)),1024);
            q[2*i+1]=pair_sub(pair_bits(mask_or(v,0x000f000fU,0x64006400U)),1024);
        } else {
            q[2*i]  =pair_sub(pair_bits(mask_or(w,0x00f000f0U,0x54005400U)),64);
            q[2*i+1]=pair_sub(pair_bits(mask_or(v,0x00f000f0U,0x54005400U)),64);
        }
    }
}
extern "C" __global__ void h16_prepass(const float* x,half* h,float2* meta,int k) {
    const int lane=threadIdx.x, block=blockIdx.x, col=blockIdx.y;
    const int i=block*32+lane;
    const float v=x[col*k+i];
    float s=fabsf(v);
#pragma unroll
    for(int off=16;off;off/=2) s=fmaxf(s,__shfl_xor_sync(0xffffffff,s,off));
    const half hv=__float2half_rn(s==0 ? 0.f : __fdiv_rn(v,s));
    const int perm=(lane&~3)|((lane&1)<<1)|((lane&2)>>1);
    h[col*k+block*32+perm]=hv;
    float sum=__half2float(hv);
#pragma unroll
    for(int off=16;off;off/=2) sum+=__shfl_xor_sync(0xffffffff,sum,off);
    if(lane==0) meta[col*(k/32)+block]=make_float2(s,sum);
}

template<class W> struct Layout { static constexpr int size=32; static constexpr bool has_min=false; };
template<> struct Layout<Q4_K> { static constexpr int size=256; static constexpr bool has_min=true; };
// One block unpack per lane, shared across all B columns; no warp shuffles
// in the dot loop. meta.sum corrects q4_K's minimum exactly once per sub-block.
template<class W,int B,int CHAIN>
__device__ __forceinline__ void block_dot(const W& w,const half* h,const float2* meta,
                                         int k,int block,float* result) {
#pragma unroll
    for(int sub=0;sub<Layout<W>::size/32;++sub) {
        Pair q[16]; float d,m; unpack(w,sub,q,d,m);
        const int ab=block*(Layout<W>::size/32)+sub;
        Pair acc[B];
#pragma unroll
        for(int c=0;c<B;++c) acc[c]=zero_pair();
#pragma unroll
        for(int j=0;j<16;++j) {
#pragma unroll
            for(int c=0;c<B;++c) acc[c]=pair_fma(q[j],load_pair(h+c*k+ab*32+j*2),acc[c]);
            if((j+1)%CHAIN==0) {
#pragma unroll
                for(int c=0;c<B;++c) {
                    const float ds=__fmul_rn(d,meta[c*(k/32)+ab].x);
                    result[c]=__fadd_rn(result[c],__fmul_rn(pair_sum(acc[c]),ds));
                    acc[c]=zero_pair();
                }
            }
        }
        if constexpr(Layout<W>::has_min) {
#pragma unroll
            for(int c=0;c<B;++c) {
                const float2 md=meta[c*(k/32)+ab];
                result[c]=__fadd_rn(result[c],-__fmul_rn(__fmul_rn(m,md.x),md.y));
            }
        }
    }
}
template<class W,int B,int CHAIN>
__device__ __forceinline__ void mmv(const W* w,const half* h,const float2* meta,
                                   float* output,int k,int rows) {
    const int row=blockIdx.x,lane=threadIdx.x; float sum[B]={};
    for(int block=lane;block<k/Layout<W>::size;block+=32)
        block_dot<W,B,CHAIN>(w[row*(k/Layout<W>::size)+block],h,meta,k,block,sum);
#pragma unroll
    for(int c=0;c<B;++c) {
#pragma unroll
        for(int off=16;off;off/=2) sum[c]+=__shfl_down_sync(0xffffffff,sum[c],off);
        if(lane==0) output[c*rows+row]=sum[c];
    }
}
#define MMV(TYPE,W,B,C,SUFFIX) \
extern "C" __global__ void mmv_h2_##TYPE##_b##B##SUFFIX(const W* w,const half* h,const float2* m,float* o,int k,int rows) { \
    mmv<W,B,C>(w,h,m,o,k,rows); }
#define FIXTURE(TYPE,W,B) \
extern "C" __global__ void h2_fixture_##TYPE##_b##B(const W* w,const half* h,const float2* m,float* o) { \
    float sum[B]={}; block_dot<W,B,16>(*w,h,m,Layout<W>::size,0,sum); \
    _Pragma("unroll") for(int c=0;c<B;++c) o[c]=sum[c]; }
#define INSTANTIATE(TYPE,W,B) MMV(TYPE,W,B,16,) FIXTURE(TYPE,W,B)
#define BATCHES(TYPE,W) INSTANTIATE(TYPE,W,1) INSTANTIATE(TYPE,W,2) INSTANTIATE(TYPE,W,4) INSTANTIATE(TYPE,W,8)
BATCHES(q4_0,Q4_0)
BATCHES(q8_0,Q8_0)
BATCHES(q4_K,Q4_K)
MMV(q4_0,Q4_0,1,8,_chain8)
