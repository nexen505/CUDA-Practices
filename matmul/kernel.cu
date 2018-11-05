__global__ void multiply(int n, int m, int p, float *a, float *b, float *c)
{
    int idx = p * threadIdx.x + threadIdx.y;
    
    c[idx] = 0.0;
    for(int k = 0; k < m; k++)
        c[idx] += a[m * threadIdx.x + k] * b[threadIdx.y + k * p];
}