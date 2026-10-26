#include<iostream>
#include"common.h"

#define TILE 16
__global__
void matrix_multy_GPU(const float* A, const float* B, float* C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] += sum;
    }
}

__global__ void compute_matrix_tile(const float* A, const float* B, float* C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x ,ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int ph = 0; ph < (N  + TILE - 1) / TILE; ++ph) {
      int a_col = ph * TILE + tx;
      int b_row = ph * TILE + ty;
      As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;   
      Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

      __syncthreads();

      for (int k = 0; k < TILE; ++k) {
        sum += As[ty][k] * Bs[k][tx];
      }
    __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}

void matrix_multy_cpu(const float* A, const float* B, float* C, int N)
{

    for (int i = 0; i < N * N; ++i) C[i] = 0.0f;
    for (int i = 0; i < N ; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j]; 
            }
        }
    }
}

using KernelFn = void (*)(const float*, const float*, float*, int);
void run(const char* name, KernelFn kernel, dim3 grid, dim3 block,
         const float* d_A, const float* d_B, float* d_C,
         float* h_C, const float* h_ref, int N)
{
    const size_t bytes = (size_t)N * N * sizeof(float);

    // 先把 C 清零：否则如果这个 kernel 什么都没算，C 里还留着上一个 kernel 的正确结果，检查会误判为通过
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    float max_err = -1.0f;   // -1 表示没有做检查（N 太大）
    if (h_ref) {
        max_err = 0.0f;
        for (int i = 0; i < N * N; ++i)
            max_err = std::max(max_err, std::fabs(h_C[i] - h_ref[i]));
    }

    const int iters = 10;
    GpuTimer t;
    t.tic();
    for (int k = 0; k < iters; ++k)
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    float ms = t.toc() / iters;
    CUDA_CHECK(cudaGetLastError());
    double gflops = 2.0 * N * N * N / (ms * 1e-3) / 1e9;

    printf("%-6s block=%2dx%-2d  max error: %-10g  avg: %.3f ms  %.1f GFLOPS (%.1f%% of 35.6 TFLOPS)\n",
           name, block.x, block.y, max_err, ms, gflops, gflops / 35600.0 * 100.0);
}

int main(int argc, char** argv)
{
    const int N  = (argc > 1) ? atoi(argv[1]) : 1024;
    const int BS = (argc > 2) ? atoi(argv[2]) : 16;   // 只用于朴素版；分块版的块大小固定为 TILE
    const size_t bytes = (size_t)N * N * sizeof(float);

    float* h_A   = new float[N * N];
    float* h_B   = new float[N * N];
    float* h_C   = new float[N * N];
    float* h_ref = new float[N * N];
    srand(42);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    const bool check = (N <= 1024);   // N 太大时 CPU 参考算得太慢，跳过检查
    if (check) matrix_multy_cpu(h_A, h_B, h_ref, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    printf("N = %d\n", N);

    dim3 block1(BS, BS);
    dim3 grid1((N + BS - 1) / BS, (N + BS - 1) / BS);
    run("naive", matrix_multy_GPU, grid1, block1, d_A, d_B, d_C, h_C, check ? h_ref : nullptr, N);

    dim3 block2(TILE, TILE);
    dim3 grid2((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    run("tiled", compute_matrix_tile, grid2, block2, d_A, d_B, d_C, h_C, check ? h_ref : nullptr, N);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_ref;
    return 0;
}
