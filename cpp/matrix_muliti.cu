#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel for matrix multiplication
_global_ void matMul(double *a, double *b, double *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        double sum = 0.0;
        for (int k = 0; k < n; k++)
        {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main()
{
    int n = 5;
    size_t bytes = n * n * sizeof(double);

    // Host matrices
    double *h_a = (double *)malloc(bytes), *h_b = (double *)malloc(bytes), *h_c = (double *)malloc(bytes);
    // Device matrices
    double *d_a, *d_b, *d_c;

    // Allocate memory for each matrix on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize matrices on host
    for (int i = 0; i < n * n; i++)
    {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    // Copy host matrices to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Set up kernel execution parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Execute the kernel
    matMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result matrix back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Result matrix:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f\t", h_c[i * n + j]);
        }
        printf("\n");
    }
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}