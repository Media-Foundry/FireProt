// SPDX-License-Identifier: GPL-2.0-only WITH OpenFold-3-exception
// Bitwise-exact gfx1100 ESM2 GEMM for the qualified FP32 B4/N1000 bucket.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>

namespace {
constexpr int kGroupsPerBlock = 8;
constexpr int kColsPerThread = 4;
// A 32-value K tile preserves the qualified per-accumulator reduction order
// while halving block-wide synchronization frequency versus the v1 kernel.
constexpr int kTileK = 32;
constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;
constexpr int kTileN = kThreadsX * kColsPerThread;

__global__ void periodic_stagger_gemm(const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      float* __restrict__ output,
                                      int columns, int inner) {
    __shared__ float tile_input[kGroupsPerBlock][kThreadsY][kTileK];
    __shared__ float tile_weight[kTileK][kTileN];
    const int lane_x = threadIdx.x;
    const int lane_y = threadIdx.y;
    const int linear_thread = lane_y * kThreadsX + lane_x;
    const int residue = blockIdx.y & 31;
    const int chunk = blockIdx.y >> 5;
    const int column_base = blockIdx.x * kTileN;
    float accumulators[kGroupsPerBlock][kColsPerThread]{};

    // hipBLASLt solutions 2178/2180 stagger every eight-row group by
    // (group & 31) * 64 K values. Groups separated by 256 rows therefore
    // share the exact reduction order and can reuse each weight tile.
    int k_base = residue * 64;
    for (int consumed = 0; consumed < inner; consumed += kTileK) {
        for (int index = linear_thread;
             index < kGroupsPerBlock * kThreadsY * kTileK;
             index += kThreadsX * kThreadsY) {
            int group_slot = index / (kThreadsY * kTileK);
            int remainder = index - group_slot * kThreadsY * kTileK;
            int local_row = remainder / kTileK;
            int local_k = remainder - local_row * kTileK;
            int group_index = residue + (chunk * kGroupsPerBlock + group_slot) * 32;
            int row = group_index * kThreadsY + local_row;
            tile_input[group_slot][local_row][local_k] = row < 4008
                ? input[static_cast<size_t>(row) * inner + k_base + local_k]
                : 0.0f;
        }
        for (int index = linear_thread; index < kTileK * kTileN;
             index += kThreadsX * kThreadsY) {
            int local_k = index / kTileN;
            int local_column = index - local_k * kTileN;
            int column = column_base + local_column;
            tile_weight[local_k][local_column] =
                weight[static_cast<size_t>(column) * inner + k_base + local_k];
        }
        __syncthreads();
        #pragma unroll
        for (int local_k = 0; local_k < kTileK; ++local_k) {
            float right[kColsPerThread];
            #pragma unroll
            for (int j = 0; j < kColsPerThread; ++j)
                right[j] = tile_weight[local_k][lane_x + j * kThreadsX];
            #pragma unroll
            for (int group = 0; group < kGroupsPerBlock; ++group) {
                float left = tile_input[group][lane_y][local_k];
                #pragma unroll
                for (int j = 0; j < kColsPerThread; ++j)
                    accumulators[group][j] =
                        fmaf(left, right[j], accumulators[group][j]);
            }
        }
        __syncthreads();
        k_base += kTileK;
        if (k_base == inner) k_base = 0;
    }
    #pragma unroll
    for (int group = 0; group < kGroupsPerBlock; ++group) {
        int group_index = residue + (chunk * kGroupsPerBlock + group) * 32;
        int row = group_index * kThreadsY + lane_y;
        if (row >= 4008) continue;
        #pragma unroll
        for (int j = 0; j < kColsPerThread; ++j) {
            int column = column_base + lane_x + j * kThreadsX;
            output[static_cast<size_t>(row) * columns + column] =
                accumulators[group][j] + bias[column];
        }
    }
}
}

extern "C" int fireprot_esm2_gemm(const void* input, const void* weight,
                                   const void* bias, void* output, int rows,
                                   int columns, int inner, void* stream) {
    if (!input || !weight || !bias || !output || rows != 4008
            || !((columns == 2560 && inner == 2560)
                 || (columns == 10240 && inner == 2560)
                 || (columns == 2560 && inner == 10240)))
        return int(hipErrorInvalidValue);
    const int groups = (rows + kThreadsY - 1) / kThreadsY;
    const int groups_per_residue = (groups + 31) / 32;
    const int chunks_per_residue =
        (groups_per_residue + kGroupsPerBlock - 1) / kGroupsPerBlock;
    dim3 block(kThreadsX, kThreadsY);
    dim3 grid(columns / kTileN, 32 * chunks_per_residue);
    hipLaunchKernelGGL(periodic_stagger_gemm, grid, block, 0,
        reinterpret_cast<hipStream_t>(stream),
        static_cast<const float*>(input), static_cast<const float*>(weight),
        static_cast<const float*>(bias), static_cast<float*>(output),
        columns, inner);
    return int(hipGetLastError());
}
