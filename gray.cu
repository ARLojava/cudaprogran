#include "common.h"

// TODO：你来实现
// 输入 in ：RGB 交错存放（RGBRGB...），共 width*height*3 字节
// 输出 out：灰度图，共 width*height 字节
// 公式：gray = 0.21*r + 0.72*g + 0.07*b
// 提示：用 .x 算列号 col，用 .y 算行号 row；别忘了边界检查
__global__ void rgb_to_gray(const unsigned char* in, unsigned char* out, int width, int height)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (col < width && row < height) {
        int i = row * width + col;
        float r = in[3 * i], g = in[3 * i + 1], b = in[3 * i + 2];
        out[i] = (unsigned char)(0.21f*r + 0.72f*g + 0.07f*b);
    }    
}

// CPU 参考答案
void rgb_to_gray_cpu(const unsigned char* in, unsigned char* out, int width, int height)
{
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            int i = row * width + col;
            float r = in[3 * i], g = in[3 * i + 1], b = in[3 * i + 2];
            out[i] = (unsigned char)(0.21f * r + 0.72f * g + 0.07f * b);
        }
}

int main()
{
    const int width = 1920, height = 1080;   // 1080 不是 16 的倍数，正好检验边界处理
    const int npix = width * height;

    // 1. CPU 端：造一张随机"假图"，算出参考答案
    unsigned char* h_in  = new unsigned char[npix * 3];
    unsigned char* h_out = new unsigned char[npix];
    unsigned char* h_ref = new unsigned char[npix];
    srand(42);
    for (int k = 0; k < npix * 3; ++k) h_in[k] = rand() % 256;
    rgb_to_gray_cpu(h_in, h_ref, width, height);

    // 2. GPU 端：分配、拷贝
    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, npix * 3));
    CUDA_CHECK(cudaMalloc(&d_out, npix));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, npix * 3, cudaMemcpyHostToDevice));

    // 3. 启动配置：每块 16×16 个线程，块数向上取整
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    printf("grid = (%d, %d), block = (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // 4. 跑一次（兼作预热），拷回来和 CPU 对比
    rgb_to_gray<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, npix, cudaMemcpyDeviceToHost));
    int max_diff = 0;
    for (int i = 0; i < npix; ++i)
        max_diff = std::max(max_diff, std::abs(h_out[i] - h_ref[i]));
    printf("max diff: %d  (0 或 1 都算对)\n", max_diff);

    // 5. 计时：连续跑 20 次取平均
    const int iters = 20;
    GpuTimer t;
    t.tic();
    for (int k = 0; k < iters; ++k)
        rgb_to_gray<<<grid, block>>>(d_in, d_out, width, height);
    float ms = t.toc() / iters;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double gbps = 4.0 * npix / (ms * 1e-3) / 1e9;   // 每个像素读 3 字节、写 1 字节
    printf("avg time: %.1f us, bandwidth: %.1f GB/s\n", ms * 1000, gbps);

    // 6. 释放
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return 0;
}
