#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void run_sgemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}