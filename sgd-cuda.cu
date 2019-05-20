#include <stdio.h>
#include <math.h>
#include "read_file.h"
#include "utils.h"


__global__ void diff_kernel(double *x, double *y, double *theta, int m, int n) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // if(x_idx < m && y_idx < n)
        printf("%d %d\n", x_idx, y_idx);
}

int main() {
    Timer t;
    int n = 100;
    int m = 1000;
    int thread_num = 32;

    double *x, *y, *theta;

    cudaMallocHost((void**)&x, m * n * sizeof(double));
    cudaMallocHost((void**)&y, m * sizeof(double));
    cudaMallocHost((void**)&theta, n * sizeof(double));

    t.tic();
    read_x(x, m, n);
    read_y(y, m);

    for(int i = 0; i < n; i++) {
        theta[i] = drand48();
    }
    printf("Time to read data :: %f\n", t.toc());

    int n_m_blocks = ceil(((double)m)/((double)thread_num));
    int n_n_blocks = ceil(((double)n)/((double)thread_num));

    printf("%d\n", n_m_blocks);
    printf("%d\n", n_n_blocks);

    dim3 dimBlock(thread_num, thread_num);
    dim3 dimGrid(n_m_blocks, n_n_blocks);

    double *x_d, *y_d, *theta_d;

    cudaMalloc(&x_d, m * n * sizeof(double));
    cudaMalloc(&y_d, m * sizeof(double));
    cudaMalloc(&theta_d, n * sizeof(double));

    cudaMemcpyAsync(x_d, x, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(y_d, y, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(theta_d, theta, n * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();


    diff_kernel<<<dimGrid, dimBlock>>>(x_d, y_d, theta_d, m, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));


    cudaDeviceSynchronize();
}
