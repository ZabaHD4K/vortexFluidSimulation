#pragma once
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Uniform spatial hash grid for neighbor search
// ---------------------------------------------------------------------------

__device__ inline int3 gridCell(float3 pos, float cellSize, float boxSize) {
    return make_int3(
        (int)floorf((pos.x + boxSize) / cellSize),
        (int)floorf((pos.y + boxSize) / cellSize),
        (int)floorf((pos.z + boxSize) / cellSize)
    );
}

__device__ inline unsigned int gridHash(int3 cell, int gridDim) {
    // Clamp to valid range
    int cx = min(max(cell.x, 0), gridDim - 1);
    int cy = min(max(cell.y, 0), gridDim - 1);
    int cz = min(max(cell.z, 0), gridDim - 1);
    return (unsigned int)(cx + cy * gridDim + cz * gridDim * gridDim);
}

// ---- Kernel: compute cell hash for each particle ----
__global__ void computeGridHashKernel(
    unsigned int* cellHash,
    unsigned int* particleIdx,
    const float4* positions,
    int N, float cellSize, float boxSize, int gridDim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p = positions[i];
    int3 cell = gridCell(make_float3(p.x, p.y, p.z), cellSize, boxSize);
    cellHash[i] = gridHash(cell, gridDim);
    particleIdx[i] = i;
}

// ---- Kernel: find start and end of each cell in sorted array ----
__global__ void findCellStartEndKernel(
    unsigned int* cellStart,
    unsigned int* cellEnd,
    const unsigned int* sortedHash,
    int N, int gridTotal
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned int hash = sortedHash[i];

    // Start of cell: either first particle or hash differs from previous
    if (i == 0 || hash != sortedHash[i - 1]) {
        cellStart[hash] = i;
        if (i > 0) cellEnd[sortedHash[i - 1]] = i;
    }
    // Last particle
    if (i == N - 1) cellEnd[hash] = i + 1;
}

// ---- Kernel: clear cell start/end ----
__global__ void clearGridKernel(unsigned int* cellStart, unsigned int* cellEnd, int gridTotal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gridTotal) return;
    cellStart[i] = 0xFFFFFFFF;
    cellEnd[i] = 0;
}
