#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>


#define CUDA_CHECK(call) do {                                   \
    cudaError_t e = (call);                                     \
    if (e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",            \
                cudaGetErrorString(e), __FILE__, __LINE__);     \
        exit(1);                                                \
    }                                                           \
} while (0)


__global__
void saxpy(int n, float a, const float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = a * x[index] + y[index];
    }
}

int main()
{
   int N = 1 << 20;
   float *x, *y, *d_x, *d_y;
   int blockSize = 512;
   int numBlock = (N + blockSize - 1) / blockSize;
   x = new float[N];
   y = new float[N];

   for (int i = 0; i < N; ++i) {
     x[i] = 1.0f;
     y[i] = 2.0f;
   }


   CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
   CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

   CUDA_CHECK(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

   //1.验证正确性
   saxpy<<<numBlock, blockSize>>>(N, 2.0f, d_x, d_y);
   CUDA_CHECK(cudaGetLastError());
   CUDA_CHECK(cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
   float max_error = 0;
   for (int i = 0; i < N; ++i) {
    max_error = std::fmax(max_error, std::abs(y[i] - 4.0f));
   }
   printf("max_error is %f\n", max_error);

   //2.计时
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   const int iters = 20;
   CUDA_CHECK(cudaEventRecord(start));
   for (int k = 0; k < iters; ++k) {
    saxpy<<<numBlock, blockSize>>>(N, 2.0f, d_x, d_y);
   }
   CUDA_CHECK(cudaEventRecord(stop));
   CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double avg_ms = ms / iters;
  double gbps = (3.0 * N * 4) / (avg_ms * 1e-3) / 1e9;
  printf("avg time: %.3f us, Effective Bandwidth: %.1f GB/s\n", avg_ms * 1000.0, gbps);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  delete x;
  delete y;
  return 0;
}
