#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <iostream>
#include <bitset>
__global__ void add(int n, float* f1, float* out) {
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	atomicAdd(out,(float)x[row * n + column]);
	
}



int main() {
	dim3 blocksDims(1,1,1);
	dim3 blocksSize(2,2,1); 	
	
	int n = 2;

	float h_x[] = {
		1.0f, 2.0f, 3.0f, 4.0f
	};

	float* d_x;
	float* d_out;

	cudaMalloc((void**) &d_x, sizeof(h_x));
	cudaMalloc((void**) &d_out, sizeof(float));
	cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);


    add<<<blocksDims, blocksSize>>>(n, d_x, d_out);

	cudaDeviceSynchronize();
    
	int h_out;
	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);


	cudaFree(d_out);
	cudaFree(d_x);

	return 0;
}