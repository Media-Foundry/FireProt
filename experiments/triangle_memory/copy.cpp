// Original FireProt code; see LICENSE for GPL-2.0-only, OpenFold 3 exception
// and the AMD-specific Apache-2.0 grant.
#include <hip/hip_runtime.h>
#include <cstdint>

// Coordinates are reordered on the host: x is source-contiguous, y is
// destination-contiguous, z is the remaining axis. Copy bits, never floats.
__global__ void tiled_copy(const uint32_t* src, uint32_t* dst,
    int nx, int ny, int nz, int64_t sx, int64_t sy, int64_t sz,
    int64_t dx, int64_t dy, int64_t dz) {
    __shared__ uint32_t tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    int z = blockIdx.z;
    #pragma unroll
    for (int i = 0; i < 32; i += 8)
        if (x < nx && y + i < ny)
            tile[threadIdx.y+i][threadIdx.x] = src[x*sx+(y+i)*sy+z*sz];
    __syncthreads();
    x = blockIdx.x * 32 + threadIdx.y;
    y = blockIdx.y * 32 + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 32; i += 8)
        if (x+i < nx && y < ny)
            dst[(x+i)*dx+y*dy+z*dz] = tile[threadIdx.x][threadIdx.y+i];
}

extern "C" int fireprot_triangle_copy(const void* src, void* dst,
    int nx, int ny, int nz, int64_t sx, int64_t sy, int64_t sz,
    int64_t dx, int64_t dy, int64_t dz, void* stream) {
    if (!src || !dst || src == dst || nx <= 0 || ny <= 0 || nz <= 0 ||
        sx != 1 || dy != 1 || sy <= 0 || sz <= 0 || dx <= 0 || dz <= 0)
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(tiled_copy, dim3((nx+31)/32,(ny+31)/32,nz),
        dim3(32,8), 0, reinterpret_cast<hipStream_t>(stream),
        static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst),
        nx,ny,nz,sx,sy,sz,dx,dy,dz);
    return int(hipGetLastError());
}
