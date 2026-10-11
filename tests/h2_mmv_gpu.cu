// Includes the device implementation so the same translation unit is compiled
// for every configured architecture and by the static sm_60 resource test.
#include "../bench/micro/h2_mmv.cu"
#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
using namespace p100::qmv;
static void check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) {
        std::fprintf(stderr,"%s: %s\n",operation,cudaGetErrorString(status)); std::exit(1);
    }
}
#define CUDA(call) check((call),#call)
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) { CUDA(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
    ~Device() { cudaFree(p); }
};
#define LAUNCH(TYPE,W,B) case B: \
    mmv_h2_##TYPE##_b##B<<<rows,32>>>(reinterpret_cast<const W*>(w),h,m,o,k,rows); break
static void launch(Type type,int batch,const void* w,const half* h,const float2* m,float* o,int k,int rows,bool short_chain) {
    if(short_chain) { mmv_h2_q4_0_b1_chain8<<<rows,32>>>(static_cast<const Q4_0*>(w),h,m,o,k,rows); return; }
    if(type==Type::q4_0) { switch(batch) { LAUNCH(q4_0,Q4_0,1); LAUNCH(q4_0,Q4_0,2); LAUNCH(q4_0,Q4_0,4); LAUNCH(q4_0,Q4_0,8); } }
    if(type==Type::q8_0) { switch(batch) { LAUNCH(q8_0,Q8_0,1); LAUNCH(q8_0,Q8_0,2); LAUNCH(q8_0,Q8_0,4); LAUNCH(q8_0,Q8_0,8); } }
    if(type==Type::q4_K) { switch(batch) { LAUNCH(q4_K,Q4_K,1); LAUNCH(q4_K,Q4_K,2); LAUNCH(q4_K,Q4_K,4); LAUNCH(q4_K,Q4_K,8); } }
}
#define FIX(TYPE,W,B) case B: \
    h2_fixture_##TYPE##_b##B<<<1,1>>>(reinterpret_cast<const W*>(w),h,m,o); break
static void fixture(Type type,int batch,const void* w,const half* h,const float2* m,float* o) {
    if(type==Type::q4_0) { switch(batch) { FIX(q4_0,Q4_0,1); FIX(q4_0,Q4_0,2); FIX(q4_0,Q4_0,4); FIX(q4_0,Q4_0,8); } }
    if(type==Type::q8_0) { switch(batch) { FIX(q8_0,Q8_0,1); FIX(q8_0,Q8_0,2); FIX(q8_0,Q8_0,4); FIX(q8_0,Q8_0,8); } }
    if(type==Type::q4_K) { switch(batch) { FIX(q4_K,Q4_K,1); FIX(q4_K,Q4_K,2); FIX(q4_K,Q4_K,4); FIX(q4_K,Q4_K,8); } }
}
struct Case {
    const char* kind;
    Type type;
    int batch,k,row,col,chain,block;
};
static void print_case(FILE* stream,const Case& c) {
    std::fprintf(stream,"%s %s B=%d K=%d row=%d col=%d chain=%d",c.kind,name(c.type),
                 c.batch,c.k,c.row,c.col,c.chain);
    if(c.block>=0) std::fprintf(stream," block=%d",c.block);
}
struct Checks {
    int checked=0,failed=0;
    double worst=0;
    Case worst_case{};
    void record(Case c,double error,double tolerance) {
        if(!std::isfinite(error)) error=std::numeric_limits<double>::infinity();
        if(checked==0 || error>worst) { worst=error; worst_case=c; }
        ++checked;
        if(error>tolerance) {
            if(failed<20) {
                print_case(stderr,c);
                std::fprintf(stderr," error=%.9g\n",error);
            }
            ++failed;
        }
    }
    void compare(Case c,double output,double target,double norm,double tolerance) {
        const double error=std::fabs(output-target);
        record(c,norm==0?error:error/norm,tolerance);
    }
};
int main() {
    int count=0; CUDA(cudaGetDeviceCount(&count));
    if(count==0) { std::fprintf(stderr,"no CUDA device\n"); return 1; }
    cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop,0));
    const int arch=prop.major*10+prop.minor;
    if(arch!=52 && arch!=60) { std::fprintf(stderr,"test supports sm_52 or sm_60, got sm_%d\n",arch); return 1; }
    std::printf("sm_%d: %s; seed=600602; %s\n",arch,prop.name,
                arch==52?"FP32 emulation checks indexing/layout only":"FP16 arithmetic vs CPU half2 model");
    std::fflush(stdout);
    std::mt19937 rng(600602); std::normal_distribution<float> normal(0,1);
    Checks checks;
    const double tolerance=arch==52?1e-5:1e-3;
    for(Type t:{Type::q4_0,Type::q8_0,Type::q4_K})
    for(int k:{block_size(t),2*block_size(t),4096,14336}) {
        constexpr int rows=17;
        std::vector<unsigned char> weights;
        for(int r=0;r<rows;++r) {
            auto row=random_row(t,k,rng); weights.insert(weights.end(),row.begin(),row.end());
        }
        Device<unsigned char> dw(weights.size()); CUDA(cudaMemcpy(dw.p,weights.data(),weights.size(),cudaMemcpyHostToDevice));
        for(int batch:{1,2,4,8}) {
            std::vector<float> x(batch*k), output(batch*rows);
            for(auto& v:x) v=normal(rng);
            // One all-zero block exercises s=0; other columns remain independent.
            for(int i=0;i<32;++i) x[i]=0;
            if(k>32) x[33]*=50.f;
            Device<float> dx(x.size()), dout(output.size()); Device<half> dh(x.size()); Device<float2> dm(batch*k/32);
            CUDA(cudaMemcpy(dx.p,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
            h16_prepass<<<dim3(k/32,batch),32>>>(dx.p,dh.p,dm.p,k); CUDA(cudaGetLastError());
            CUDA(cudaDeviceSynchronize());
            std::vector<half> host_h(x.size()); std::vector<float2> host_meta(batch*k/32);
            CUDA(cudaMemcpy(host_h.data(),dh.p,x.size()*sizeof(half),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(host_meta.data(),dm.p,host_meta.size()*sizeof(float2),cudaMemcpyDeviceToHost));
            std::vector<std::vector<ActivationBlock>> acts;
            for(int c=0;c<batch;++c) {
                acts.push_back(prepare(x.data()+c*k,k));
                for(int b=0;b<k/32;++b) {
                    const auto& a=acts[c][b]; const float2 md=host_meta[c*k/32+b];
                    // One exact pre-pass check per activation block. There is
                    // no output row or dot chain here (row=-1, chain=0).
                    double error=0;
                    auto difference=[&](double actual,double expected) {
                        const double delta=std::fabs(actual-expected);
                        if(!std::isfinite(delta)) error=std::numeric_limits<double>::infinity();
                        else error=std::fmax(error,delta);
                    };
                    difference(md.x,a.scale); difference(md.y,a.sum);
                    for(int i=0;i<32;++i) {
                        difference(__half2float(host_h[c*k+b*32+i]),a.h[i]);
                    }
                    checks.record({"prepass",t,batch,k,-1,c,0,b},error,0);
                }
            }
            for(int variant=0;variant<((t==Type::q4_0 && batch==1)?2:1);++variant) {
                launch(t,batch,dw.p,dh.p,dm.p,dout.p,k,rows,variant==1);
                CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
                CUDA(cudaMemcpy(output.data(),dout.p,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
                for(int c=0;c<batch;++c) {
                    double xn=0; for(int i=0;i<k;++i) xn+=double(x[c*k+i])*x[c*k+i]; xn=std::sqrt(xn);
                    for(int r=0;r<rows;++r) {
                        const auto ref=evaluate(t,weights.data()+r*(k/block_size(t))*block_bytes(t),x.data()+c*k,k,acts[c]);
                        const double expected=arch==52?ref.h16_exact:variant==1?ref.h8:ref.h16;
                        const double norm=ref.weight_norm*xn;
                        checks.compare({"mmv",t,batch,k,r,c,variant?8:16,-1},
                                       output[c*rows+r],expected,norm,tolerance);
                    }
                }
            }
            if(k==block_size(t)) {
                fixture(t,batch,dw.p,dh.p,dm.p,dout.p);
                CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
                CUDA(cudaMemcpy(output.data(),dout.p,batch*sizeof(float),cudaMemcpyDeviceToHost));
                for(int c=0;c<batch;++c) {
                    const auto ref=evaluate(t,weights.data(),x.data()+c*k,k,acts[c]);
                    double xn=0; for(int i=0;i<k;++i) xn+=double(x[c*k+i])*x[c*k+i]; xn=std::sqrt(xn);
                    const double target=arch==52?ref.h16_exact:ref.h16, norm=ref.weight_norm*xn;
                    checks.compare({"fixture",t,batch,k,0,c,16,-1},output[c],target,norm,tolerance);
                }
            }
        }
    }
    std::printf("checked %d, failed %d, worst error %.9g (",checks.checked,checks.failed,checks.worst);
    print_case(stdout,checks.worst_case);
    std::printf(")\n");
    return checks.failed>0?1:0;
}
#endif
