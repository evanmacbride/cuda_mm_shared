#include <stdio.h>
#include <time.h>

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
        msg, cudaGetErrorString(__err), \
        __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  } while (0)

const int default_dim = 8192;
const int block_size = 32; // The CUDA max is 1024 threads per block
const float A_val = 3.0f;
const float B_val = 2.0f;
const float tol = 1e-8;

//__global__ void mmul(const float *A, const float *B, float *C, int ds) {
__global__ void mmul(const float *A, const float *B, float *C, int n, int m, int q) {
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  //if ((idx < ds) && (idy < ds)) {
  if ((idx < n) && (idy < m)) {
    float temp = 0;
    //for (int i = 0; i < ds/block_size; i++) {
    for (int i = 0; i < q; i++) {
      //As[threadIdx.y][threadIdx.x] = A[idy*ds + (i*block_size + threadIdx.x)];
      As[threadIdx.y][threadIdx.x] = A[idy * q + i + threadIdx.x];
      //Bs[threadIdx.y][threadIdx.x] = B[(i*block_size + threadIdx.y) * ds + idx];
      Bs[threadIdx.y][threadIdx.x] = B[i * n + threadIdx.y * n + idx];
      __syncthreads();

      //for (int k = 0; k < block_size; k++) {
      for (int k = 0; k < blockDim.x; k++) {
	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
      }
      __syncthreads();
    }
    //C[idy*ds+idx] = temp;
    C[idy * n + idx] = temp;
  }
}

int main(int argc, char* argv) {
  float *h_A, *h_B, *h_C, *h_check, *d_A, *d_B, *d_C;
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;
  int m, n, p, q;

  if (argc > 1 && argc < 6) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);
    q = atoi(argv[4]);
  } else if (argc >= 6){
    printf("Usage: ./executable M N P Q\n");
    return -1;
  } else {
    m = default_dim;
    n = default_dim;
    p = default_dim;
    q = default_dim;
  }
  if (n != p) {
    printf("Dimension N (%d) != dimension P (%d). Operation undefined.\n",
		    n, p); 
    return -1;
  }

  t0 = clock();

  // Allocate host matrices
  h_A = (float*)malloc(m*n*sizeof(float));
  h_B = (float*)malloc(p*q*sizeof(float));
  h_C = (float*)malloc(m*q*sizeof(float));
  h_check = (float*)malloc(m*q*sizeof(float));
  
  // Init matrices
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      h_A[i*n+j] = A_val;
  for (int i = 0; i < p; i++)
    for (int j = 0; j < q; j++)
      h_B[i*q+j] = B_val;

  // Calculate AxB=C on the host
  float temp = 0.0;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < q; j++) {
      temp = 0.0;
      for (int k = 0; k < p; k++) {
        temp += h_A[i*p+k] * h_B[k*q+j];
      }
      h_check[i*q+j] = temp;
      h_C[i*q+j] = 0.0;
      }

  /*
  // printout for debugging
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < q; j++) 
      printf("%8.3f", h_check[i*q+j]);
    printf("\n");
  }
  */

  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds. Begin compute.\n", t1sum);

  // Allocate device matrices
  cudaMalloc(&d_A, m*n*sizeof(float));
  cudaMalloc(&d_B, p*q*sizeof(float));
  cudaMalloc(&d_C, m*q*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, p*q*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");

  //int x_blocks = n / block_size;
  //int y_blocks = m / block_size;

  dim3 block(block_size, block_size);
  //dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  dim3 grid((n+block.x-1)/block.x, (m+block.y-1)/block.y);
  // Calcuate AxB=C on the device
  //mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  mmul<<<grid, block>>>(d_A, d_B, d_C, n, m, q);
  cudaCheckErrors("kernel launch failure");

  cudaMemcpy(h_C, d_C, m*q*sizeof(float), cudaMemcpyDeviceToHost);

  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // Check for correctness
  for (int i = 0; i < m*q; i++) { 
    if (abs(h_C[i] - h_check[i]) > tol) {
      printf("mismatch at index %d, was: %f, should be: %f\n", 
		      i, h_C[i], h_check[i]); 
      return -1;
    }
  }
  printf("Success!\n");

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaCheckErrors("cudaFree failure");
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_check);
  return 0;
}
