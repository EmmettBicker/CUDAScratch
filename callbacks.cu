#include <cuda_runtime.h>
#include <iostream>
#include <limits.h> // For INT_MAX

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void k1(int *x)
{
    int b = 4/0;
    for (int i = 0; i < INT_MAX+1; i++) {*x += i % 2 == 0 ? 1 : 0;}
    printf("Hello from %d\n",*x);
}

__global__ void k2(int *x)
{
    for (int i = 0; i < INT_MAX; i++) {*x += i % 2 == 0 ? 1 : 0;}
    printf("Hello from %d\n",*x);
}

void CUDART_CB cb(void* userData)
{
    printf("Callback\n");
}

int main()
{
    cudaStream_t st_high, st_low;
    cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, -5);
    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, 1);

    int one_val = 1, two_val = 2, three_val = 3, four_val = 4;
    int *one, *two, *three, *four;
    
    cudaMalloc(&one, sizeof(int));
    cudaMalloc(&two, sizeof(int));
    cudaMalloc(&three, sizeof(int));
    cudaMalloc(&four, sizeof(int));

    cudaMemcpy(one, &one_val, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(two, &two_val, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(three, &three_val, sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy(four, &four_val, sizeof(int), cudaMemcpyHostToDevice);

    k1<<<1,1,0,st_low>>>(three);
    cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    std::cerr << "Kernel launch error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
}
    k2<<<1,1,0,st_low>>>(four);
    error = cudaGetLastError();
if (error != cudaSuccess) {
    std::cerr << "Kernel launch error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
}
    cudaLaunchHostFunc(st_low, cb, nullptr);


    cudaDeviceSynchronize();

    k1<<<1,1,0,st_high>>>(one);
    error = cudaGetLastError();
if (error != cudaSuccess) {
    std::cerr << "Kernel launch error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
}
    k2<<<1,1,0,st_high>>>(two);
    error = cudaGetLastError();
if (error != cudaSuccess) {
    std::cerr << "Kernel launch error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
}
    cudaLaunchHostFunc(st_high, cb, nullptr);
    cudaDeviceSynchronize();
    


}