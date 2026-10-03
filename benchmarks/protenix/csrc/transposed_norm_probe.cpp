// Private FP32 pair-transpose LayerNorm probe; contiguous channels, N=1000/1536.
// Uses the public Welford algorithm with float4 groups and wave64 merging.
#include <hip/hip_runtime.h>
struct Moment { float mean, m2, count; };
__device__ Moment merge(Moment a, Moment b) {
    float total=a.count+b.count;
    if(total==0) return {0,0,0};
    float reciprocal=1.0f/total;
    float wa=a.count*reciprocal, wb=b.count*reciprocal;
    float delta=b.mean-a.mean;
    return {wa*a.mean+wb*b.mean,
            a.m2+b.m2+delta*delta*a.count*wb,total};
}
template<int Waves>
__global__ void transpose_norm(const float* x,const float* weight,const float* bias,
                               float* out,int n,float eps) {
    int lane=threadIdx.x%64, wave=threadIdx.x/64;
    int row=blockIdx.x*Waves+wave;
    if(row>=n*n)return;
    int source=(row%n*n+row/n)*128;
    float4 v=make_float4(0,0,0,0);
    Moment m{0,0,0};
    if(lane<32){
        v=*reinterpret_cast<const float4*>(x+source+lane*4);
        float values[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int i=0;i<4;i++){
            float delta=values[i]-m.mean;
            m.count+=1;
            m.mean+=delta*(1.0f/m.count);
            m.m2+=delta*(values[i]-m.mean);
        }
    }
    #pragma unroll
    for(int offset=32;offset>0;offset/=2){
        Moment other{__shfl_down(m.mean,offset,64),__shfl_down(m.m2,offset,64),
                     __shfl_down(m.count,offset,64)};
        m=merge(other,m);
    }
    float mean=__shfl(m.mean,0,64), var=__shfl(m.m2,0,64)*(1.0f/128);
    float inv=rsqrtf(var+eps);
    if(lane<32){
        float4 w=*reinterpret_cast<const float4*>(weight+lane*4);
        float4 b=*reinterpret_cast<const float4*>(bias+lane*4);
        float4 y=make_float4(w.x*(inv*(v.x-mean))+b.x,w.y*(inv*(v.y-mean))+b.y,
                            w.z*(inv*(v.z-mean))+b.z,w.w*(inv*(v.w-mean))+b.w);
        *reinterpret_cast<float4*>(out+row*128+lane*4)=y;
    }
}
extern "C" int transposed_norm(const float* x,const float* w,const float* b,float* out,
                               int n,int waves,float eps,void* stream){
    if((n!=1000&&n!=1536)||(waves!=4&&waves!=8))return 2;
    if(!x||!w||!b||!out)return 3;
    if(waves==4)hipLaunchKernelGGL(transpose_norm<4>,dim3((n*n+3)/4),dim3(256),0,static_cast<hipStream_t>(stream),x,w,b,out,n,eps);
    else hipLaunchKernelGGL(transpose_norm<8>,dim3((n*n+7)/8),dim3(512),0,static_cast<hipStream_t>(stream),x,w,b,out,n,eps);
    return static_cast<int>(hipGetLastError());
}
