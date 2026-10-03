// Long FP32 packed LayerNorm: vector loads with alternative LDS/output layouts.
#include <hip/hip_runtime.h>
#include <cstdint>
__device__ float sum64(float x) {
    for(int d=32;d;d/=2)x+=__shfl_down(x,d);
    return __shfl(x,0);
}
__device__ float sum32(float x) {
    for(int d=16;d;d/=2)x+=__shfl_down(x,d,32);
    return __shfl(x,0,32);
}
// Emulate the old wave64 tree with four consecutive channels per lane.
template<bool Square>
__device__ float original_tree(float a,float b,float c,float d) {
    float aa=__shfl_down(a,16,32),bb=__shfl_down(b,16,32);
    float cc=__shfl_down(c,16,32),dd=__shfl_down(d,16,32);
    if constexpr(Square) {a=a*a+aa*aa;b=b*b+bb*bb;c=c*c+cc*cc;d=d*d+dd*dd;}
    else {a+=aa;b+=bb;c+=cc;d+=dd;}
    for(int shift=8;shift;shift/=2) {
        a+=__shfl_down(a,shift,32);b+=__shfl_down(b,shift,32);
        c+=__shfl_down(c,shift,32);d+=__shfl_down(d,shift,32);
    }
    return __shfl((a+c)+(b+d),0,32);
}
template<int Rows,int Mode>
__global__ void norm_kernel(const float4* input,const float* weight,const float* bias,
                            float* output,int rows,float eps) {
    __shared__ float tile[Rows][129];
    int base=blockIdx.x*Rows;
    for(int i=threadIdx.x;i<128*(Rows/4);i+=blockDim.x) {
        int c=i/(Rows/4),r=(i%(Rows/4))*4;
        float4 x=input[(c*rows+base+r)/4];
        tile[r][c]=x.x;tile[r+1][c]=x.y;tile[r+2][c]=x.z;tile[r+3][c]=x.w;
    }
    __syncthreads();
    if constexpr(Mode==0) {
        int lane=threadIdx.x%64,wave=threadIdx.x/64;
        for(int r=wave;r<Rows;r+=4) {
            float x=tile[r][lane],y=tile[r][lane+64];
            float mean=sum64(x+y)/128;
            float a=x-mean,b=y-mean;
            float inv=rsqrtf(sum64(a*a+b*b)/128+eps);
            output[(base+r)*128+lane]=a*inv*weight[lane]+bias[lane];
            output[(base+r)*128+lane+64]=b*inv*weight[lane+64]+bias[lane+64];
        }
    } else {
        int lane=threadIdx.x%32,group=threadIdx.x/32,c=lane*4;
        float4 w=reinterpret_cast<const float4*>(weight)[lane];
        float4 b=reinterpret_cast<const float4*>(bias)[lane];
        for(int r=group;r<Rows;r+=8) {
            float a=tile[r][c],d=tile[r][c+1],e=tile[r][c+2],f=tile[r][c+3];
            float mean=(Mode==2 ? original_tree<false>(a,d,e,f) : sum32((a+d)+(e+f)))/128;
            a-=mean;d-=mean;e-=mean;f-=mean;
            float variance=Mode==2 ? original_tree<true>(a,d,e,f) : sum32((a*a+d*d)+(e*e+f*f));
            float inv=rsqrtf(variance/128+eps);
            reinterpret_cast<float4*>(output)[(base+r)*32+lane]=make_float4(
                a*inv*w.x+b.x,d*inv*w.y+b.y,e*inv*w.z+b.z,f*inv*w.w+b.w);
        }
    }
}
extern "C" int packed_norm_vector(float* input,float* weight,float* bias,float* output,
                                  int n,int tile,int vector_output,float eps,void* stream) {
    if((n!=1000 && n!=1536) || (tile!=32 && tile!=64) || (vector_output<0 || vector_output>2) || !(eps>0))return 2;
    if(!input||!weight||!bias||!output||reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(weight)%16||reinterpret_cast<uintptr_t>(bias)%16||reinterpret_cast<uintptr_t>(output)%16)return 3;
    int rows=n*n;
    #define LAUNCH(R,V) hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_kernel<R,V>),dim3(rows/R),dim3(256),0,static_cast<hipStream_t>(stream),reinterpret_cast<float4*>(input),weight,bias,output,rows,eps)
    if(tile==32) {if(vector_output==2) {LAUNCH(32,2);} else if(vector_output==1) {LAUNCH(32,1);} else {LAUNCH(32,0);}}
    else {if(vector_output==2) {LAUNCH(64,2);} else if(vector_output==1) {LAUNCH(64,1);} else {LAUNCH(64,0);}}
    return static_cast<int>(hipGetLastError());
}
