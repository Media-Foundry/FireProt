// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-OpenFold-3-exception
// FireProt original code; see LICENSE and licenses/ for additional AMD terms.
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {
constexpr int rows = 1536 * 1536;
template<int TileRows>
__global__ void pack_h16(const uint32_t* __restrict__ input,
                         uint32_t* __restrict__ output) {
    __shared__ uint32_t tile[TileRows * 16];
    const int base = blockIdx.x * TileRows;
    for (int i = threadIdx.x; i < TileRows * 16; i += 256) {
        const int row = i / 16, head = i % 16;
        tile[row * 16 + (head ^ ((row / 2) % 16))] = input[(base + row) * 16 + head];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TileRows * 16; i += 256) {
        const int head = i / TileRows, row = i % TileRows;
        output[head * rows + base + row] = tile[row * 16 + (head ^ ((row / 2) % 16))];
    }
}
}

extern "C" int fireprot_pack_bias_h16_f32(const void* input, void* output,
                                          int tile_rows, void* stream) {
    if (!input || !output || input == output || (tile_rows != 64 && tile_rows != 128))
        return static_cast<int>(hipErrorInvalidValue);
    auto src = static_cast<const uint32_t*>(input);
    auto dst = static_cast<uint32_t*>(output);
    auto s = static_cast<hipStream_t>(stream);
    if (tile_rows == 64)
        hipLaunchKernelGGL(pack_h16<64>, dim3(rows / 64), dim3(256), 0, s, src, dst);
    else
        hipLaunchKernelGGL(pack_h16<128>, dim3(rows / 128), dim3(256), 0, s, src, dst);
    return static_cast<int>(hipGetLastError());
}
