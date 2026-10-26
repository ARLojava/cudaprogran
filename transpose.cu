#include "common.h"

#define TILE_DIM   32   // 每个块负责一个 32×32 的小块
#define BLOCK_ROWS 8    // 每个块 32×8 个线程，每个线程负责 32/8 = 4 个元素

// 基线：只复制、不转置，访存方式和转置一样
__global__ void copy_kernel(float* out, const float* in, int N)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;   // 列
    int y = blockIdx.y * TILE_DIM + threadIdx.y;   // 行（这个线程负责的第一行）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)  // 每次往下走 8 行，一共 4 次
        out[(y + j) * N + x] = in[(y + j) * N + x];
}

// TODO 1：朴素转置。把 in 的第 (y+j) 行第 x 列，写到 out 的第 x 行第 (y+j) 列
//         结构和 copy_kernel 一样，只改写入的下标
__global__ void transpose_naive(float* out, const float* in, int N)
{
   int x = blockIdx.x * TILE_DIM + threadIdx.x;
   int y = blockIdx.y * TILE_DIM + threadIdx.y;
   for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      out[x * N + (y + j)] = in[(y + j) * N + x];
}

// TODO 2：shared memory 转置
//   第一步：和 copy 一样按行读，把 in 的这个 32×32 小块存进 tile（读是合并的）
//   第二步：__syncthreads()
//   第三步：从 tile 里按列取数，按行写到 out 里（写也是合并的）
//   提示：写出去时，小块本身的位置也要转置：用 blockIdx.y 算列，用 blockIdx.x 算行
__global__ void transpose_smem(float* out, const float* in, int N)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
    __syncthreads();
}

// TODO 3：代码和 TODO 2 完全一样，只是 tile 多了一列 padding
__global__ void transpose_smem_pad(float* out, const float* in, int N)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
    __syncthreads();
}

using KernelFn = void (*)(float*, const float*, int);

// 跑一次并逐个元素检查，然后计时
void run(const char* name, KernelFn kernel, const float* d_in, float* d_out,
         float* h_out, const float* h_expect, int N)
{
    const size_t bytes = (size_t)N * N * sizeof(float);
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(N / TILE_DIM, N / TILE_DIM);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));   // 先清零，防止残留上一个 kernel 的正确结果
    kernel<<<grid, block>>>(d_out, d_in, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    size_t wrong = 0;   // 转置只是搬运数据，结果必须完全相等
    for (size_t i = 0; i < (size_t)N * N; ++i)
        if (h_out[i] != h_expect[i]) ++wrong;

    const int iters = 20;
    GpuTimer t;
    t.tic();
    for (int k = 0; k < iters; ++k)
        kernel<<<grid, block>>>(d_out, d_in, N);
    float ms = t.toc() / iters;
    CUDA_CHECK(cudaGetLastError());
    double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;   // 每个元素读一次、写一次
    printf("%-9s wrong: %-9zu avg: %.3f ms  %.1f GB/s (%.0f%% of 829)\n",
           name, wrong, ms, gbps, gbps / 829.0 * 100.0);
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? atoi(argv[1]) : 4096;
    if (N % TILE_DIM != 0) { printf("N 必须是 %d 的倍数\n", TILE_DIM); return 1; }
    const size_t total = (size_t)N * N;
    const size_t bytes = total * sizeof(float);

    float* h_in  = new float[total];
    float* h_out = new float[total];
    float* h_ref = new float[total];   // 转置的正确答案
    for (size_t i = 0; i < total; ++i) h_in[i] = (float)i;   // 每个元素的值都不同，放错位置一定查得出来
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            h_ref[(size_t)c * N + r] = h_in[(size_t)r * N + c];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("N = %d\n", N);
    run("copy",     copy_kernel,        d_in, d_out, h_out, h_in,  N);   // copy 的正确答案就是输入本身
    run("naive",    transpose_naive,    d_in, d_out, h_out, h_ref, N);
    run("smem",     transpose_smem,     d_in, d_out, h_out, h_ref, N);
    run("smem_pad", transpose_smem_pad, d_in, d_out, h_out, h_ref, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return 0;
}
