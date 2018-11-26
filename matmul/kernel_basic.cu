__global__ void matmul(const int n, const int m, const int p, const float *A, const float *B, float *C) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x,
    y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < p) {
    C[x * p + y] = 0.0;
    for (int i = 0; i < m; ++i) {
      C[x * p + y] += A[x * m + i] * B[p * i + y];
    }
  }
}