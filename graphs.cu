#include <cuda_runtime.h>
#include <iostream>

#define CUDA_ERROR_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

__global__ void kernel_A(int *data) 
{
    printf("hellur A %d\n",*data);
}
__global__ void kernel_B(int *data) 
{
    printf("hellur B %d\n", *data);
}
__global__ void kernel_C() {}
__global__ void kernel_D() {}
__global__ void kernel_E() {}

int main()
{
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaGraphNode_t nodeA, nodeB;
    cudaKernelNodeParams k_params = {0}; 

    int a = 1, b = 2;
    int *d_a_ptr, *d_b_ptr; 
    cudaMalloc(&d_a_ptr, sizeof(int));
    cudaMalloc(&d_b_ptr, sizeof(int));
    cudaMemcpy(d_a_ptr, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_ptr, &b, sizeof(int), cudaMemcpyHostToDevice);


    void **kernel_args = (void **)malloc(sizeof(void*));
    
    k_params.func = kernel_A; 
    k_params.gridDim = dim3 (1,1,1);
    k_params.blockDim = dim3(1,1,1);
    k_params.sharedMemBytes = 0;
    k_params.kernelParams = kernel_args;
    kernel_args[0] = &d_a_ptr;
    k_params.extra = nullptr; // No extra parameters

    cudaGraphCreate(&graph, 0);
    
    cudaGraphAddKernelNode(&nodeA, graph, NULL, 0, &k_params);
    
    k_params.func = kernel_B; 
    kernel_args[0] = &d_b_ptr;
    cudaGraphAddKernelNode(&nodeB, graph, NULL, 0, &k_params);
    
    
    cudaGraphInstantiate(&graph_exec, graph);
    cudaGraphLaunch(graph_exec,0);
    
    cudaFree(d_a_ptr);
    cudaFree(d_b_ptr);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);

}