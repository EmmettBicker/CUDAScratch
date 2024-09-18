#include <iostream>
#include <cuda_runtime.h>

__global__ void myKernel(volatile int* dummy)
{
    long long sum = 0;
    for (long long i = 0; i < 100000001; i++)
    {

        sum += i % 2 == 0 ? i : 0;
    }
    *dummy = sum;  // Prevent optimization
}

int main()
{
    cudaStream_t stream, stream2;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&stream2);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));

    myKernel<<<1,1>>>(d_dummy);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    
    myKernel<<<1,1,0,stream>>>(d_dummy);
    myKernel<<<1,1, 0, stream2>>>(d_dummy);

   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Wait for stop event to complete

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Elapsed time: %f ms\n", ms);

    cudaFree(d_dummy);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return 0;
}