// This implementation 0 pads the array to the nearest power of 2.
// It's inefficient but it works! And if the array is a power of two
// it works perfectly.

#include <cuda_runtime.h>
#include <iostream>

// Launch half as many threads as n
__global__ void all_reduce(int n, float *a) {
    extern __shared__ float a_shared[];

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    a_shared[idx] = a[idx];
    a_shared[idx + 1] = a[idx + 1];
    a_shared[n - 1] = a[n - 1];

    int logn = floor(log2f(n));
    // printf("idx: %d\n", threadIdx.x);
    int window_size = 1;
    for (int i = 0; i < logn; i++) {
        window_size *= 2;
        int idx2 = (idx + window_size - 1) % n;

        a_shared[idx] = a_shared[idx] + a_shared[idx2];
        a_shared[idx2] = a_shared[idx];
        __syncthreads();
        // printf("%d, %d, || %f, %f\n", idx, idx2, a[idx], a[idx2]);
    }
    a[idx] = a_shared[idx];
    a[idx + 1] = a_shared[idx + 1];
    a[n - 1] = a_shared[n - 1];
}

int main() {
    int n = 16;
    int actual_n = 3;

    float a[] = {1.f, 2.f, 3.f};

    float *d;
    cudaMalloc((void **)&d, sizeof(float) * n);
    cudaMemset(d, 0, sizeof(float) * n);
    cudaMemcpy(d, a, sizeof(float) * actual_n, cudaMemcpyHostToDevice);

    all_reduce<<<1, n / 2, n * sizeof(float)>>>(n, d);
    cudaDeviceSynchronize();
    float *h;
    cudaMallocHost((void **)&h, sizeof(float) * n);
    cudaMemcpy(h, d, sizeof(float) * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < actual_n; i++)
        printf("%f\n", h[i]);
    printf("%d extraneous values", n - actual_n);
    cudaFree(d);
    cudaFree(h);
}