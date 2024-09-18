#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <iostream>
#include <bitset>
__global__ void mat_mult_2d_square(int n, float* f1, float* f2, float* out) {
    int col = blockDim.x * blockIdx.x + threadIdx.x; 
	int row = blockDim.y * blockIdx.y + threadIdx.y;


    float dot_prod = 0.0f;

    for (int i = 0; i < n; i++){

        dot_prod += f1[row * n + i] * f2[i * n + col];
    }
    printf("%d, %d", row, col);
    out[row * n + col] = dot_prod;

    
}



int main() {

    int n = 3;


	dim3 blocksDims(n,n,1);
	dim3 blocksSize(n,2,1); 	
	
	

	float h_x1[] = {
		1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
	};

    float h_x2[] = {
		1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
	};
	float* d_x1;
    float* d_x2;
	float* d_out;

	cudaMalloc((void**) &d_x1, sizeof(h_x1));
    cudaMalloc((void**) &d_x2, sizeof(h_x2));
	cudaMalloc((void**) &d_out, sizeof(h_x1));

	cudaMemcpy(d_x1, h_x1, sizeof(h_x1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, sizeof(h_x2), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(h_x1));

    mat_mult_2d_square<<<blocksDims, blocksSize>>>(n, d_x1, d_x2, d_out);


	cudaDeviceSynchronize();

	float *h_out = (float*)malloc(n * n * sizeof(float));  // Allocate dynamically
    // cuda memcpy replaces the stuff @ the location of the reference, so 
    // if your output is int, then do &(your int) so that it replaces the value.
    // If you're doing a pointer, just put in the pointer so that it replaces the contents
    // and the material wil stay the same
	cudaMemcpy(h_out, d_out, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n*n; i++)
    {
        printf("Element %d, %d: %f\n", ((i) / n), (i) % n, h_out[i]);
    }

    free(h_out);
	cudaFree(d_out);
	cudaFree(d_x1);
    cudaFree(d_x2);

	return 0;
}