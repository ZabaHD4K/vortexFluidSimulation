#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

inline void printDeviceInfo() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA Device: %s (SM %d.%d, %d SMs, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}
