__global__ void matmul(int n, const float *A, const float *B, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    C[row * n + col] = 0.0;
    for (int i = 0; i < n; ++i) {
      C[row * n + col] += A[row * n + i] * B[n * i + col];
    }
  }
}