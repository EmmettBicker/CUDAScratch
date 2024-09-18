
#include <cuda_bf16.h>
#include <curand_kernel.h>

#include <cassert>
#include <iostream>
// #include <stdfloat>
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
            std::cerr << " - " << cudaGetErrorString(err) << std::endl;   \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void float_to_b16(int n, float* f_arr, float* b_arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b_arr[idx] = __float2bfloat16(f_arr[idx]);
    }
}

__global__ void device_initialize_arr(int total_size, int layer_size, float* f, int rand_offset=0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        curandState_t state;
        curand_init(0, idx+rand_offset, 0, &state);
        float stddev = sqrtf(2.0f / layer_size);
        f[idx] = curand_normal(&state) * stddev;
    }
}

__global__ void device_initialize_arr_with_ones(int total_size, float* f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        f[idx] = 1.0f;
    }
}

__device__ __nv_bfloat16 kSqrt2OverPi;  // sqrt(2/pi)
__device__ __nv_bfloat16 kCoefficient;
__device__ __nv_bfloat16 one_half_bfloat;
__device__ __nv_bfloat16 one_bfloat;
__global__ void setup_gelu_constants()
{
    kSqrt2OverPi = __float2bfloat16(0.7978845608028654f);
    kCoefficient = __float2bfloat16(0.044715f);
    one_half_bfloat = __float2bfloat16(0.5f);
    one_bfloat = __float2bfloat16(1.0f);


}

__device__ __nv_bfloat16 gelu(__nv_bfloat16 x) {
    return x * 
        (one_half_bfloat * (one_bfloat +  
        __float2bfloat16(tanhf(__bfloat162float(
            kSqrt2OverPi * (x + kCoefficient * x * x * x)
        )))));

}

__global__ void h_gelu(__nv_bfloat16 x, __nv_bfloat16 *out)
{
    
    *out = gelu(x);
}



__global__ void fwd_pass(
    int h1,
    int w1,
    int h2,
    int w2,
    float* mat1,
    float* mat2,
    float* mat_out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (row < h1 && col < w2) {
        float dot_prod = 0.f;
        for (int i = 0; i < w1; i++) {
            dot_prod += __bfloat162float(__float2bfloat16(mat1[row * w1 + i]) * __float2bfloat16(mat2[i * w2 + col]));
        }
        mat_out[row * w2 + col] = gelu(dot_prod);
    }


}

__global__ void element_wise_sum(int n, float* arr_1, float* arr2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        printf("Value 1: %f, Value 2: %f\n", arr_1[idx], arr2[idx]);
        arr_1[idx] += arr2[idx];
    }
}

__global__ void mat_mult(
    int h1,
    int w1,
    int h2,
    int w2,
    float* mat1,
    float* mat2,
    float* mat_out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // ending size is h1 x w2
    // printf("%d, %d, %s", row, col, row < h1 && col < w2 ? "true" : "false");
    if (row < h1 && col < w2) {
        float dot_prod = 0.f;
        for (int i = 0; i < w1; i++) {
            dot_prod += mat1[row * w1 + i] * mat2[i * w2 + col];
        }
        mat_out[row * w2 + col] = dot_prod;
    }
}


__host__ void set_up_all_constants()
{
    setup_gelu_constants<<<1,1>>>();
}


__host__ float* initialize_array(int n_elements, int layer_size, int rand_offset=0) {
    float *d_array;
    cudaMalloc(&d_array, sizeof(float) * n_elements);
    device_initialize_arr<<<(n_elements / 32) + 1, 32>>>(n_elements, layer_size, d_array, rand_offset);
    return d_array;
}

__host__ float* initailize_array_of_ones(int n)
{
    float *d_array;
    cudaMalloc(&d_array, sizeof(float) * n);
    device_initialize_arr_with_ones<<<(n / 32) + 1, 32>>>(n, d_array);
    return d_array;
}

template <typename T>
__host__ T* move_to_host(int n, T* d_array) {
    T* h_array;
    cudaMallocHost(&h_array, sizeof(T) * n);
    cudaMemcpy(h_array, d_array, sizeof(T) * n, cudaMemcpyDeviceToHost);
    return h_array;
}

__host__ void h_mat_mult(
    int h1,
    int w1,
    int h2,
    int w2,
    float* mat1,  // host arr
    float* mat2,  // host arr
    float* mat_out,
    cudaStream_t stream
    ) {
    dim3 gridDim((h1 / 32) + 1, (w2 / 32) + 1);
    dim3 blockDim(32, 32);
    fwd_pass<<<gridDim, blockDim, 0, stream>>>(h1, w1, h2, w2, mat1, mat2, mat_out);
}

// This method of sharding isn't analagous to sharding across devices, but I wonder
// if it's faster than just regular computation because it's still splitting the
// array up and the memory allocation only happens once rather than N_d times?
// probably very minimal improvements. Also, it doesn't shard it "horizontally",
// but all the dimensionality in memory is a construct so it doesn't matter
__host__ void shard_array(int n, int layer_size, int shards, float** shard_locations) {
    assert(n % shards == 0);
    int len = n/shards;
    for (int i = 0; i < shards; i++) {
        shard_locations[i] = initialize_array(len, layer_size, i*len);
    }
}


void gelu_test()
{
    set_up_all_constants();

    for (float f = -10.f; f < 10.0f; f+= 1.0f)
    {
        __nv_bfloat16 b = f;

        __nv_bfloat16 *c;
        cudaMalloc(&c, sizeof(__nv_bfloat16));
        h_gelu<<<1,1>>>(b, c);
        cudaDeviceSynchronize();
        __nv_bfloat16 h;
        cudaMemcpy(&h, c, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        printf("x: %f, gelu(x): %f\n",b, __bfloat162float(h));
    }
    
}

int main() {
    set_up_all_constants();
    const int batch_size = 1;
    const int n_x = 2;
    const int n_a_1 = 3;
    const int n_params_in_w1 = n_x * n_a_1;
    const int shards = 3;
    const int params_per_shard = n_params_in_w1 / shards;
    const int shard_width = n_a_1 / shards;
    const int sharded_mat_mult_output_params = batch_size * shard_width;

    const size_t sz_x = sizeof(float) * n_x;
    const size_t sz_layer1 = sizeof(float) * n_params_in_w1;
    const size_t sz_layer1_shard = sz_layer1 / shards;

    
    const size_t sz_shard_output = sizeof(float) * sharded_mat_mult_output_params;
    float** shard_locations = new float*[shards];
    shard_array(n_params_in_w1, n_a_1, shards, shard_locations);
    
    cudaStream_t *streams = new cudaStream_t[shards];
    for (int i = 0; i < shards; i++)
    {
        cudaStreamCreate(&(streams[i]));
    }
    
    // const int n_a_2 = 100;
    // const int n_a_3 = 10;

    // 784 x 1

  
    float* d_x = initailize_array_of_ones(n_x);  
    float* h_x = move_to_host(n_x, d_x);

    
    float **outs = new float*[shards];
    for (int i = 0; i < shards; i++)
    {
        cudaMallocAsync(&(outs[i]), sz_shard_output, streams[i]);
    }
    for (int i = 0; i < shards; i++) 
    {
        cudaStreamSynchronize(streams[i]);
    }
    for (int i = 0; i < shards; i++) 
    {
        h_mat_mult(batch_size, n_x, params_per_shard, n_a_1/shards, d_x, shard_locations[i], outs[i], streams[i]);
    }
    for (int i = 0; i < shards; i++) 
    {
        cudaStreamSynchronize(streams[i]);
    }


    // Concatenate the outputs to pass into the next layer
        float* a_next;
        cudaMalloc(&a_next, sz_shard_output * shards);
        for (int i = 0; i < shards; i++)
        {
            cudaMemcpyAsync(a_next + i*sharded_mat_mult_output_params, outs[i], sz_shard_output, cudaMemcpyDeviceToDevice, streams[i]);
        }
        cudaDeviceSynchronize();

    float **h_outs = new float*[shards];
    for (int i = 0; i < shards; i++)
    {
        h_outs[i] = move_to_host(sharded_mat_mult_output_params, outs[i]);
    }

    printf("x:");
    for (int i = 0; i < n_x; i++)
    {
        printf("%f | ", h_x[i]);
    }
    printf("\n");

    printf("shards:");
    for (int s = 0; s < shards; s++) {
        printf("shard %d:\n", s);
        float *temp = move_to_host(params_per_shard, shard_locations[s]);
        for (int i = 0; i < params_per_shard; i++) {
            
            printf("%f\n", temp[i]);
            
        }
        cudaFreeHost(temp);
    }
    printf("\n");

    for (int s = 0; s < shards; s++) {
        printf("shard %d output:\n", s);
        for (int i = 0; i < sharded_mat_mult_output_params; i++) {
            printf("%f\n", h_outs[s][i]);
        }
    }

    printf("\na_next:\n");
    float* h_a_next = move_to_host(n_a_1, a_next);
    for (int i = 0; i < n_a_1; i++)
    {
        printf("%f\n", h_a_next[i]);
    }

    cudaFree(d_x);
    cudaFree(a_next);
    cudaFreeHost(h_x);
    cudaFreeHost(h_a_next);
    for (int i = 0; i < shards; i++)
    {
        cudaFree(shard_locations[i]);
        cudaFree(outs[i]);
        cudaFreeHost(h_outs[i]);
    }

    return 0;
}