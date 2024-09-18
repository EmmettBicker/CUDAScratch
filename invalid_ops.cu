#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel() {
    printf("Hello from kernel\n");
}

void checkCudaErrors(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    cudaStream_t stream1, stream2;
    cudaGraph_t graph;

    // Create streams
    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2));

    // Begin capture in stream1
    checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Launch kernel in stream1
    kernel<<<1, 1, 0, stream1>>>();
    checkCudaErrors(cudaGetLastError());

    // Record an event in stream1
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));
    checkCudaErrors(cudaEventRecord(event, stream1));

    // Wait for the event in stream2
    checkCudaErrors(cudaStreamWaitEvent(stream2, event, 0));

   

    // End capture in stream1
    checkCudaErrors(cudaStreamEndCapture(stream1, &graph));
 // Launch kernel in stream2
    kernel<<<1, 1, 0, stream2>>>();
    checkCudaErrors(cudaGetLastError());
    // Synchronize streams to ensure all operations are completed
    checkCudaErrors(cudaStreamSynchronize(stream1));
    checkCudaErrors(cudaStreamSynchronize(stream2));

    // Cleanup
    checkCudaErrors(cudaStreamDestroy(stream1));
    checkCudaErrors(cudaStreamDestroy(stream2));
    checkCudaErrors(cudaEventDestroy(event));

    return 0;
}
