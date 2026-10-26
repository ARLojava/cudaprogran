#include "common.h"
#include <vector>

#define BLOCK 256   // 必须是 2 的幂

// v1：交错寻址（幻灯片 Kernel 1）
//   每个线程读 1 个元素到 s[tid]；stride 从 1 开始翻倍，
//   满足 tid % (2*stride) == 0 的线程把 s[tid + stride] 加到 s[tid] 上
__global__ void reduce_v1(const float* in, float* partial, int n)
{
    __shared__ float s[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    s[tid] = (i < n) ? in[i] : 0.0f; 
    __syncthreads();
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1){
        if (tid % (2 * stride) == 0) {
           s[tid] += s[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// v2：顺序寻址（幻灯片 Kernel 3，和 π 那个版本的块内求和一样）
//   每个线程读 1 个元素；stride 从 blockDim.x/2 开始减半，tid < stride 的线程干活
__global__ void reduce_v2(const float* in, float* partial, int n)
{
    __shared__ float s[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
           s[tid] += s[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// v3：先用 grid-stride 循环让每个线程在寄存器里累加多个元素，再做和 v2 一样的块内求和
//   块数固定为 1024（由 main 决定），每个线程大约负责 256 个元素
__global__ void reduce_v3(const float* in, float* partial, int n)
{
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (unsigned int index = blockIdx.x * blockDim.x + tid; index < n; index += gridSize) {
        sum += in[index];
    }
    s[tid] = sum;
    __syncthreads();

    for (unsigned int half = blockDim.x / 2; half > 0; half /= 2) {
       if (tid < half) {
        s[tid] += s[tid + half];
       }
       __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// v4：在 v3 的基础上，块内求和只在 shared memory 里做到剩 32 个数为止（stride >= 32 的那几轮）；
//   然后第 0 个 warp 把 s[0..31] 读进寄存器，最后 5 轮用 warp shuffle 完成：
//       for (int offset = 16; offset > 0; offset /= 2)
//           val += __shfl_down_sync(0xffffffff, val, offset);
//   这几轮不需要 __syncthreads()，因为一个 warp 内的线程本来就是一起执行的
__global__ void reduce_v4(const float* in, float* partial, int n)
{
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (unsigned int index = blockIdx.x * blockDim.x + tid; index < n; index += gridSize) {
        sum += in[index];
    }
    s[tid] = sum;
    __syncthreads();

    for (unsigned int half = blockDim.x / 2; half >= 32; half /= 2) {
       if (tid < half) {
        s[tid] += s[tid + half];
       }
       __syncthreads();
    }
    
    if (tid < 32) {
       float val = s[tid];
       for (int offset = 16; offset > 0; offset /= 2)
           val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0) partial[blockIdx.x] = val;
    }
    
}

using KernelFn = void (*)(const float*, float*, int);

// 跑一次检查结果，然后计时
void run(const char* name, KernelFn kernel, int grid,
         const float* d_in, float* d_partial, int n, double ref)
{
    CUDA_CHECK(cudaMemset(d_partial, 0, grid * sizeof(float)));
    kernel<<<grid, BLOCK>>>(d_in, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> partial(grid);
    CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, grid * sizeof(float), cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float v : partial) sum += v;            // 第二步：在 CPU 上把部分和加起来
    double rel_err = std::fabs(sum - ref) / ref;

    const int iters = 20;
    GpuTimer t;
    t.tic();
    for (int k = 0; k < iters; ++k)
        kernel<<<grid, BLOCK>>>(d_in, d_partial, n);
    float ms = t.toc() / iters;
    CUDA_CHECK(cudaGetLastError());
    double gbps = (double)n * sizeof(float) / (ms * 1e-3) / 1e9;   // 每个元素读一次
    printf("%-3s grid=%-7d rel err: %-9.2e avg: %.3f ms  %.1f GB/s (%.0f%% of 829)\n",
           name, grid, rel_err, ms, gbps, gbps / 829.0 * 100.0);
}

int main(int argc, char** argv)
{
    const int n = (argc > 1) ? atoi(argv[1]) : (1 << 26);
    std::vector<float> h_in(n);
    srand(42);
    for (int i = 0; i < n; ++i) h_in[i] = rand() / (float)RAND_MAX;
    double ref = 0.0;
    for (int i = 0; i < n; ++i) ref += h_in[i];   // CPU 参考答案，用 double 累加

    const int grid_one = (n + BLOCK - 1) / BLOCK;  // v1、v2：每个线程一个元素
    const int grid_fixed = 1024;                   // v3、v4：块数固定
    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, grid_one * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    printf("n = %d, sum = %.1f\n", n, ref);
    run("v1", reduce_v1, grid_one,   d_in, d_partial, n, ref);
    run("v2", reduce_v2, grid_one,   d_in, d_partial, n, ref);
    run("v3", reduce_v3, grid_fixed, d_in, d_partial, n, ref);
    run("v4", reduce_v4, grid_fixed, d_in, d_partial, n, ref);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    return 0;
}
