#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}

int main()
{
    int n = 10;
    size_t bytes = n * sizeof(double);
    double *h_a = (double *)malloc(bytes), *h_b = (double *)malloc(bytes), *h_c = (double *)malloc(bytes);
    double *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    for (int i = 0; i < n; i++)
    {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vecAdd<<<(int)ceil((float)n / 1024), 1024>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += h_c[i];
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    printf("final result: %.2f\n", sum);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}