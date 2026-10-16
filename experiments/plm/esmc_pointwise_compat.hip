// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Retain qualified RoPE/GELU and the existing ABI; only ESMC SwiGLU is replaced.
#define fireprot_esm_swiglu esmc_original_swiglu
#include "fireprot_esm.hip"
#undef fireprot_esm_swiglu
extern "C" int esmc_pointwise(int,int,const void*,const void*,void*,int,float,void*);
extern "C" int fireprot_esm_swiglu(const void* input,void* output,int rows,int width,void* stream) {
    if (width == 6912) return esmc_pointwise(1,8,input,nullptr,output,rows,1.0f,stream);
    return esmc_original_swiglu(input,output,rows,width,stream);
}
