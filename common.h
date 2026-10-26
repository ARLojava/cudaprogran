#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t e = (call);                                     \
    if (e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",            \
                cudaGetErrorString(e), __FILE__, __LINE__);     \
        exit(1);                                                \
    }                                                           \
} while (0)

// 用 cudaEvent 计时：tic() 开始，toc() 返回经过的毫秒数
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void tic()  { CUDA_CHECK(cudaEventRecord(start)); }
    float toc() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};
