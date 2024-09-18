#include <tuple>
#include <cuda_runtime.h>
#include <stdio.h>
struct Shape {
    int x;
    int y;
    int z;
};

__global__ void broadcast_addition(
    Shape shape, 
    float* arr1, 
    float* arr2, 
    float* arr3, 
    float* arr_out) 
{
    int x = shape.x,y = shape.y,z = shape.z;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (col < x && row < y && depth < z){

        int idx_1 = depth * x * y + row * x + col; // arr1 e (X x Y x Z)
        int idx_2 = row * x + col; // arr1 e (X x Y)
        int idx_3 = col;

        
        arr_out[idx_1] = arr1[idx_1] + arr2[idx_2] + arr3[idx_3];
    
    }
    

}

int main() {
    dim3 blocksDims(1,1,1);
    dim3 blocksSize(3,2,2);

    // 3 x 2 x 2

    float f1[] = {
        1.0f, 2.0f, 3.0f, 
        4.0f, 5.0f, 6.0f, 


        7.0f, 8.0f, 9.0f, 
        10.0f, 11.0f, 12.0f
    };

    float f2[] = {
        1.0f, 2.0f, 3.0f, 
        4.0f, 5.0f, 6.0f, 
    };

    float f3[] = {
        1.0f, 2.0f, 3.0f, 
    };
    
    float *d_arr1, *d_arr2, *d_arr3, *d_out;

    cudaMalloc((void**) &d_arr1, sizeof(f1));
    cudaMalloc((void**) &d_arr2, sizeof(f2));
    cudaMalloc((void**) &d_arr3, sizeof(f3));
    cudaMalloc((void**) &d_out, sizeof(f1));
    cudaMemcpy(d_arr1, f1, sizeof(f1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, f2, sizeof(f2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr3, f3, sizeof(f3), cudaMemcpyHostToDevice);
    cudaot(d_out, 0, sizeof(f1));


    int x = 3, y= 2 ,z= 2;
    Shape shape = {x, y, z};



    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);



    broadcast_addition<<<blocksDims, blocksSize>>>(shape, d_arr1, d_arr2, d_arr3, d_out);
    // Record stop event
    cudaEventRecord(stop, 0);
    
    // Wait for the events to complete
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f ms\n", elapsedTime);
    float* h_out;
    cudaMallocHost((void**) &h_out, sizeof(f1));
    cudaMemcpy(h_out, d_out, sizeof(f1), cudaMemcpyDeviceToHost);
    for (int i = 0; i < x*y*z; i++){ 
        printf("Element %d, %d, %d: %f\n", i/z/y,i/z % y, i % y*z, h_out[i]);
    }

    cudaFreeHost(h_out);
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_arr3);
    cudaFree(d_out);

    return 0;
}