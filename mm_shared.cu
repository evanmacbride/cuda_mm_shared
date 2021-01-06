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

//const int DSIZE = 32;
const int DSIZE = 8192;  // Want this to work for any size, not just 2^n
const int block_size = 32; // The CUDA max is 1024 threads per block
const float A_val = 3.0f;
const float B_val = 2.0f;
const float tol = 1e-8;

__global__ void mmul(const float *A, const float *B, float *C, int ds) {
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;

  if ((idx < ds) && (idy < ds)) {
    float temp = 0;
    for (int i = 0; i < ds/block_size; i++) {
      As[threadIdx.y][threadIdx.x] = A[idy*ds + (i*block_size + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = B[(i*block_size + threadIdx.y) * ds + idx];
      __syncthreads();

      for (int k = 0; k < block_size; k++) {
	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
      }
      __syncthreads();
    }
    C[idy*ds+idx] = temp;
  }
}

int main() {
  float *h_A, *h_B, *h_C, *h_check, *d_A, *d_B, *d_C;
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;
  int m = DSIZE, n = DSIZE, p = DSIZE, q = DSIZE;

  if (n != p) {
    printf("Dimension N (%d) != dimension P (%d). Operation undefined.\n",
		    n, p); 
    return -1;
  }

  t0 = clock();

  /*
  // Old code to init matrices
  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  for (int i = 0; i < DSIZE*DSIZE; i++) {
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0.0;
  }
  */
  
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

  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");

  dim3 block(block_size, block_size);
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");
  /*
  for (int i = 0; i < DSIZE*DSIZE; i++) { 
    if (h_C[i] != A_val*B_val*DSIZE) {
      printf("mismatch at index %d, was: %f, should be: %f\n", 
		      i, h_C[i], A_val*B_val*DSIZE); 
      return -1;
    }
  }
  */

  // Check for correctness
  for (int i = 0; i < DSIZE*DSIZE; i++) { 
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
