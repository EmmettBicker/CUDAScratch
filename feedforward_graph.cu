#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "mat_mult.cuh"

int main() {
    int w1 = 2, h1 = 3, w2 = 2, h2 = 2;

    const int num_streams = 2;

    cudaStream_t streams[num_streams];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float arr1[] = {1, 2, 3, 4, 5, 6};
    float arr2[] = {1, 2, 3, 4};

    float *d_mat1, *d_mat2, *d_out;
    size_t sz_mat1 = sizeof(float) * w1 * h1;
    size_t sz_mat2 = sizeof(float) * w2 * h2;
    size_t sz_out = sizeof(float) * h1 * w2;

    cudaMalloc((void**)&d_mat1, sz_mat1);
    cudaMalloc((void**)&d_mat2, sz_mat2);
    cudaMalloc((void**)&d_out, sz_out);

    float* h_out;

    cudaMallocHost((void**)&h_out, sz_out);

    cudaMemset(d_out, 0, sz_out);

    // cudaEventRecord(start);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // Use cudaMemcpyAsync with specified stream
    cudaMemcpyAsync(d_mat1, arr1, sz_mat1, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_mat2, arr2, sz_mat2, cudaMemcpyHostToDevice);

    // float ms = 0;
    // cudaEventElapsedTime(&ms, start, stop);
    // printf("%f ms\n", ms);

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaGraphNode_t node_A;
    cudaGraphCreate(&graph, 0);

    cudaKernelNodeParams k_params;
    dim3 gridDim(1, 1);
    dim3 blockDim(h1, w2);

    k_params.func = (void*)mat_mult;
    k_params.gridDim = gridDim;
    k_params.blockDim = blockDim;
    void* k_args[] = {&w1, &h1, &w2, &h2, &d_mat1, &d_mat2, &d_out};
    k_params.kernelParams = k_args;
    k_params.sharedMemBytes = 0;

    cudaGraphAddKernelNode(&node_A, graph, NULL, 0, &k_params);

    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

    cudaDeviceSynchronize();  // Ensure all previous operations are done
    cudaGraphLaunch(graph_exec, 0);
    // mat_mult<<<1,dim3(h1, w2)>>>(w1, h1, w2, h2, d_mat1, d_mat2, d_out);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sz_out, cudaMemcpyDeviceToHost);

    printf("\n");
    for (int i = 0; i < h1 * w2; i++) {
        printf("%d: %f\n", i, h_out[i]);
    }

    for (int i = 0; i < num_streams; i++)
        cudaStreamDestroy(streams[i]);
    free(h_out);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_out);
}