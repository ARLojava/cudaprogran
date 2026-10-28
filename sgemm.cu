#include "common.h"
#include <cublas_v2.h>
#include <vector>


#define BK 8
#define BN 64
#define BM 64
#define TM 8
static_assert(BK == BN / TM && BK == BM / TM, "K4 要求每个线程正好搬 As、Bs 各一格");

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t st = (call);                                         \
    if (st != CUBLAS_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st,          \
                __FILE__, __LINE__);                                    \
        exit(1);                                                        \
    }                                                                   \
} while (0)

// ===== K2：合并访存的朴素版（你之前的 matrix_multy_GPU）=====
__global__ void sgemm_naive(const float* A, const float* B, float* C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ===== K3：shared memory 分块（你之前的 compute_matrix_tile）=====
#define TILE 16
__global__ void sgemm_smem(const float* A, const float* B, float* C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int ph = 0; ph < (N + TILE - 1) / TILE; ++ph) {
        int a_col = ph * TILE + tx;
        int b_row = ph * TILE + ty;
        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

// 以后每写一个新版本，就在这里加一个 kernel，再在 main 里加一行 run(...)
__global__ void sgemm_1D_tileing(const float* A, const float* B, float * C, int N)
{
    int tid = threadIdx.x;
    int threadRow = tid / BN;
    int threadCol = tid % BN;
    __shared__ float As[BM][BK];  // 肥
    __shared__ float Bs[BK][BN];  // 高
    float acc[TM] = {0.0f};

    for (unsigned int bk = 0; bk < N; bk += BK) {
        As[tid / BK][tid % BK] = A[(blockIdx.y * BM + tid / BK) * N + bk + tid % BK];
        Bs[tid / BN][tid % BN] = B[(bk + tid / BN) * N + blockIdx.x * BN + tid % BN];
        __syncthreads();
        for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float btemp = Bs[dotIdx][threadCol];
            for (unsigned int resIdx = 0; resIdx < TM; ++resIdx) {
                acc[resIdx] += As[threadRow * TM + resIdx][dotIdx] * btemp;
            }
        }
        __syncthreads();
    }

    for (unsigned int index = 0; index < TM; ++index) {
        C[(blockIdx.y * BM + threadRow * TM + index) * N + blockIdx.x * BN + threadCol] = acc[index];
    }
}

// 跑一次并和 cuBLAS 的结果对比，然后计时。launch 是一个"启动 kernel"的 lambda
template <typename F>
void run(const char* name, F launch, float* d_C, std::vector<float>& h_C,
         const std::vector<float>& h_ref, int N, double gf_ref)
{
    const size_t bytes = (size_t)N * N * sizeof(float);
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0, max_ref = 0.0;   // 相对误差 = 最大误差 / 最大元素
    for (size_t i = 0; i < h_C.size(); ++i) {
        max_err = std::max(max_err, (double)std::fabs(h_C[i] - h_ref[i]));
        max_ref = std::max(max_ref, (double)std::fabs(h_ref[i]));
    }

    const int iters = 10;
    GpuTimer t;
    t.tic();
    for (int k = 0; k < iters; ++k) launch();
    float ms = t.toc() / iters;
    CUDA_CHECK(cudaGetLastError());
    double gf = 2.0 * N * N * N / (ms * 1e-3) / 1e9;
    printf("%-8s rel err: %-9.2e %8.3f ms %9.1f GFLOPS %6.1f%% of cuBLAS\n",
           name, max_err / max_ref, ms, gf, gf / gf_ref * 100.0);
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? atoi(argv[1]) : 4096;
    const size_t bytes = (size_t)N * N * sizeof(float);

    std::vector<float> h_A((size_t)N * N), h_B((size_t)N * N), h_C((size_t)N * N), h_ref((size_t)N * N);
    srand(42);
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // ===== 基线：cuBLAS =====
    // cuBLAS 默认按列主序存储。行主序的 C = A·B，在列主序看来就是 Cᵀ = Bᵀ·Aᵀ，
    // 所以调用时把 B 放在前面、A 放在后面，算出来的正好是行主序的 C
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    auto launch_cublas = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                 &alpha, d_B, N, d_A, N, &beta, d_ref, N));
    };
    launch_cublas();   // 预热，同时得到参考答案
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, bytes, cudaMemcpyDeviceToHost));
    const int iters = 10;
    GpuTimer t;
    t.tic();
    for (int k = 0; k < iters; ++k) launch_cublas();
    float ms_ref = t.toc() / iters;
    double gf_ref = 2.0 * N * N * N / (ms_ref * 1e-3) / 1e9;
    printf("N = %d\n", N);
    printf("%-8s %-19s %8.3f ms %9.1f GFLOPS %6.1f%% of peak 35.6 TFLOPS\n",
           "cuBLAS", "(基线)", ms_ref, gf_ref, gf_ref / 35600.0 * 100.0);

    // ===== 你的版本 =====
    dim3 block2(16, 16), grid2((N + 15) / 16, (N + 15) / 16);
    run("K2", [&]() { sgemm_naive<<<grid2, block2>>>(d_A, d_B, d_C, N); },
        d_C, h_C, h_ref, N, gf_ref);

    dim3 block3(TILE, TILE), grid3((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    run("K3", [&]() { sgemm_smem<<<grid3, block3>>>(d_A, d_B, d_C, N); },
        d_C, h_C, h_ref, N, gf_ref);
    
    if (N % BM == 0 && N % BN == 0 && N % BK == 0) {
        dim3 block4(BM * BN / TM);      // 一维线程块：64 × 64 ÷ 8 = 512 个线程
        dim3 grid4(N / BN, N / BM);     // x 方向有几个列块，y 方向有几个行块
        run("K4", [&]() { sgemm_1D_tileing<<<grid4, block4>>>(d_A, d_B, d_C, N); },
            d_C, h_C, h_ref, N, gf_ref);
    } else {
        printf("K4       跳过：N 必须是 64 的倍数\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
