#define TILE_WIDTH 32

__global__ void sgemm(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint bx = blockIdx.x; const uint by = blockIdx.y;
  const uint tx = threadIdx.x; const uint ty = threadIdx.y;

  const uint row = by * blockDim.y + ty;
  const uint col = bx * blockDim.x + tx;

  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  float Cvalue = 0.0;

  for (int t = 0; t < (K-1)/TILE_WIDTH+1; ++t) {
    if (row < M && t*TILE_WIDTH+tx < K)
        As[ty][tx] = A[row*K + t*TILE_WIDTH+tx];
    else
        As[ty][tx] = 0.0;
    if (col < N && t*TILE_WIDTH+ty < K)
        Bs[ty][tx] = B[(t*TILE_WIDTH+ty)*N + col];
    else
        Bs[ty][tx] = 0.0;

    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; ++i)
        Cvalue += As[ty][i] * Bs[i][tx];

    __syncthreads();
  }

  if (row < M && col < N)
    C[row*N + col] = alpha * Cvalue + beta * C[row*N + col];
}