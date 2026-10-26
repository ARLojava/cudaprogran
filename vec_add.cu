#include<iostream>
#include<math.h>

__global__ void add_vec_Kernal(float* C, float* A, float* B, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


void add_vec(float* C, float* A, float* B, int N)
{
    float* d_c, *d_a, *d_b;

    size_t bytes = N * sizeof(float);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

    int blockSize = 512;
    int numBlock = (N + blockSize - 1) / blockSize;
    for (int k = 0; k < 200000; ++k) {
    	add_vec_Kernal<<<numBlock, blockSize>>>(d_c, d_a, d_b, N);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

int main()
{
    const int N = 1000;
    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];

    // init
    for (int i = 0; i < N ; ++i) {
        A[i] = float(i);
        B[i] = A[i] / 1000.0f;
    }

    add_vec(C, A, B, N);
    for (int i = 0; i < N; i += 1) {
            if (i > 0) {
            printf(", ");
            if (i % 10 == 0) std::cout<<std::endl;
        }
        printf("%8.3f", C[i]);
    }
  printf("\n");
  delete A;
  delete B;
  delete C;
  return 0;
}


