# LeetgpuSolution

Personal Triton GPU kernels that solve a handful of LeetGPU practice problems. Each solution keeps the problem-focused entry point exposed as a `solve(...)` function so you can plug the kernels into the LeetGPU grader or reuse them locally for benchmarking.

# Perf benchmark

## NVIDIA A100-80GB

### vector-addition
![vector-addition](benchmarks/result/vector-addition.png)

### matrix-multiplication
![matrix-multiplication](benchmarks/result/matrix-multiplication.png)

### matrix-transpose
![matrix-transpose](benchmarks/result/matrix-transpose.png)

### color-inversion
![color-inversion](benchmarks/result/color-inversion.png)

### 1d-convolution
![1d-convolution](benchmarks/result/1d-convolution.png)

### reverse-array
![reverse-array](benchmarks/result/reverse-array.png)

### relu
![relu](benchmarks/result/relu.png)

### leaky-relu
![leaky-relu](benchmarks/result/leaky-relu.png)

### rainbow-table
![rainbow-table](benchmarks/result/rainbow-table.png)
