# SGEMM 优化（RTX 3090，N = 4096，FP32）

| Kernel | GFLOPS | % of cuBLAS | 关键改动 |
|---|---|---|---|
| cuBLAS | 25307 | 100% | 基线 |
| K2 naive | 2269 | 9.0% | |
| K3 shared memory | 3037 | 12.0% | 分块 |
| K4 1D blocktiling | 8757 | 34.6% | 每线程算 8 个结果 |
| K5 2D blocktiling | 17401 | 68.8% | 每线程算 8×8 外积；stride 用编译期常量 |
| K6 vectorized | 18832 | 74.4% | float4 读显存，As 转置 |


## 备忘录

### K5：两个循环都没有展开，性能只有一半（2026-10-29）

K5 初版只有 9262 GFLOPS（cuBLAS 的 37.8%）。在同一张 3090 上跑 siboehm 的参考实现，发现对方快了 1.78 倍。用 `cuobjdump -sass` 数指令，发现加载循环只有 2 条 LDG、dotIdx 循环只有 64 条 FFMA，说明两个循环都没有展开。原因是 stride 用了运行时才知道的 `blockDim.x`。改成编译期常量后，LDG 变成 8 条、FFMA 变成 512 条，性能提到 17401 GFLOPS（cuBLAS 的 68.8%，3 次中位数）。

### K6：向量化访存（2026-10-30）

读显存改用 float4，As 转置存放。LDG 从 8 条 32 位变成 2 条 128 位，寄存器从 128 降到 102，比 K5 快 8.2%（18832 GFLOPS，cuBLAS 的 74.4%）。LDS 仍是 32 条，因为 K5 里编译器已经自动把读 As 合并成了 128 位。
