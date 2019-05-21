#include <stdio.h>
#include <math.h>
#include "read_file.h"
#include "utils.h"

#define COL 128


__global__ void diff_kernel(double *x, double *y_diff, double *theta, double *h, int m, int n) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(x_idx < m && y_idx < n) {
        h[x_idx * n + y_idx] = x[x_idx * n + y_idx] * theta[y_idx];
    }
}

__global__ void grad_kernel(double *h, double *x, double *grad, int m, int n) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if(x_idx < m && y_idx < n) {
        __syncthreads();
        grad[y_idx] += h[x_idx * n] * x[x_idx * n + y_idx];
    }
}

__global__ void update_kernel(double *theta, double *grad, double alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        // printf("\n\n%lf\n\n", grad[idx]);
        theta[idx] += alpha * grad[idx];
        grad[idx] = 0.0;
    }
}

__global__ void reduction_kernel2(double* h, double *y, int m, int n){
  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if(x_idx >= m || y_idx >= n) return;

  int idx = x_idx * n + y_idx;

  __syncthreads();
  if (y_idx < 64 && y_idx + 64 < n) h[idx] += h[idx +  64];
  __syncthreads();
  if (y_idx <  32) {
    h[idx] += h[idx +  32];
    __syncthreads();
    h[idx] += h[idx +  16];
    __syncthreads();
    h[idx] += h[idx + 8];
    __syncthreads();
    h[idx] += h[idx +  4];
    __syncthreads();
    h[idx] += h[idx +   2];
    __syncthreads();
    if (y_idx == 0) h[idx] = y[x_idx] - (h[idx] + h[idx + 1]);
  }
}

int main() {
    Timer t;
    int n = 100;
    int m = 10000;
    int iters = 100;

    double tt_compute = 0.0;
    double tt_transfer = 0.0;
    double cost = 0.0;
    double alpha = 1.0 / m ;

    double *x, *y, *theta;
    double *c;

    cudaMallocHost((void**)&x, m * n * sizeof(double));
    cudaMallocHost((void**)&y, m * sizeof(double));
    cudaMallocHost((void**)&theta, n * sizeof(double));
    cudaMallocHost((void**)&cost, sizeof(double));
    cudaMallocHost((void**)&c, m * n * sizeof(double));

    t.tic();
    read_x(x, m, n);
    read_y(y, m);

    for(int i = 0; i < n; i++) {
        theta[i] = drand48();
    }
    printf("Time to read data :: %f\n", t.toc());

    int n_m_blocks = ceil(((double)m)/((double)1024/COL));
    int n_n_blocks = ceil(((double)n)/((double)COL));

    printf("%d\n", n_m_blocks);
    printf("%d\n", n_n_blocks);

    dim3 dimBlock(1024/COL, COL);
    dim3 dimGrid(n_m_blocks, n_n_blocks);

    double *x_d, *y_d, *theta_d, *h, *grad_d;

    cudaMalloc(&x_d, m * n * sizeof(double));
    cudaMalloc(&y_d, m * sizeof(double));
    cudaMalloc(&theta_d, n * sizeof(double));
    cudaMalloc(&h, m * n * sizeof(double));
    cudaMalloc(&grad_d, n * sizeof(double));

    cudaMemcpyAsync(x_d, x, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(y_d, y, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(theta_d, theta, n * sizeof(double), cudaMemcpyHostToDevice);

    for(int i = 0; i < iters; i++) {

        t.tic();
        diff_kernel<<<dimGrid, dimBlock>>>(x_d, y_d, theta_d, h, m, n);

        reduction_kernel2<<<dimGrid, dimBlock>>>(h, y_d, m, n);
        tt_compute += t.toc();
        if(i % 10 == 0) {
            cudaMemcpyAsync(c, h, m * n * sizeof(double), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            #ifdef _OPENMP
            #pragma omp parallel for (+:reduction)
            #endif
            for(int i = 0; i < m; i++) {
                cost += c[n*i] * c[n*i];
            }
            printf("Iter :: %d, Cost :: %lf\n", i, cost);
            cost = 0.0;
        }
        t.tic();
        grad_kernel<<<dimGrid, dimBlock>>>(h, x_d, grad_d, m, n);

        update_kernel<<<1, COL>>>(theta_d, grad_d, alpha, n);
        tt_compute += t.toc();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    printf("Time to compute :: %lf\n", tt_compute);


    cudaDeviceSynchronize();
}
