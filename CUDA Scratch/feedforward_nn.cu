// just handles one input, I didn't end up making it handle multiple at a time. But it's all here!

// input is 784 x 1
// hidden layers are (300 x 1), (100 x 1)
// output softmax is (10 x 1) 

#include <iostream>
#include <cuda_runtime.h>

__global__ void mat_mult(
    int n,
    int n_next,
    float *a,
    float *w,
    float *out
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < n_next) {
        float dot_prod = 0.0f;
        for (int i = 0; i < n; i++)
        {
            dot_prod += w[row*n + i] * a[i];
        }
        out[row] = dot_prod;
    }
}

__global__ void add(int n, float *f1, float *f_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        f_out[idx] += f1[idx];
    }
}

__global__ void relu(int n, float *f_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        f_out[idx] = fmaxf(f_out[idx], 0.0f);   
    }
}


// could be parallized but I should move on
void initialize_array(float **f, float **d_f, size_t size) {
    cudaMallocHost((void**) f, size);
    int elements = size / sizeof(float);
    for (int i = 0; i < elements; i++) {
        (*f)[i] = (((float)(rand())) / (float)(RAND_MAX)) * 2 - 1;
    }
    cudaMalloc( (void**) d_f, size);
    cudaMemcpy(*d_f, *f, size, cudaMemcpyHostToDevice);
}

__global__ void softmax(int n, float *f_in, float *f_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float max_val = f_out[0];
        float divisor = 0.0f;
        for (int i = 0; i < n; i++)
        {
            max_val = max(max_val, f_in[i]);
        }
        for (int i = 0; i < n; i++)
        {
            divisor += exp(f_in[i] - max_val);
        }

        f_out[idx] = exp((f_in[idx]-max_val)) / divisor;
    }
}

int main()
{
    srand(time(0));

    int n_x = 784;
    int n_h1 = 300;
    int n_h2 = 100;
    int n_h3 = 10;
    float *h_x, *h_w_1, *h_wb_1;
    float *h_w_2, *h_wb_2;
    float *h_w_3, *h_wb_3;

    float *d_x, *d_w_1, *d_wb_1;
    float *d_out_1, *d_out_2, *d_out_3;
    float *d_y_hat;
    float *d_w_2, *d_wb_2;
    float *d_w_3, *d_wb_3;


    size_t s_h_x =  sizeof(float)*n_x;
    size_t s_h_w_1 =  sizeof(float)*n_x*n_h1;
    size_t s_h_wb_1 =  sizeof(float)*n_h1;
    size_t s_h_out_1 =  sizeof(float)*n_h1;
    size_t s_h_out_2 =  sizeof(float)*n_h2;
    size_t s_h_out_3 =  sizeof(float)*n_h3;
    size_t s_y_hat =  sizeof(float)*n_h3;

    size_t s_h_w_2 =  sizeof(float)*n_h1*n_h2;
    size_t s_h_wb_2 =  sizeof(float)*n_h2;
    
    size_t s_h_w_3 =  sizeof(float)*n_h2*n_h3;
    size_t s_h_wb_3 =  sizeof(float)*n_h3;
    

    initialize_array(&h_x, &d_x, s_h_x);
    initialize_array(&h_w_1, &d_w_1, s_h_w_1);
    initialize_array(&h_wb_1, &d_wb_1, s_h_wb_1);

    initialize_array(&h_w_2, &d_w_2, s_h_w_2);
    initialize_array(&h_wb_2, &d_wb_2, s_h_wb_2);
    initialize_array(&h_w_3, &d_w_3, s_h_w_3);
    initialize_array(&h_wb_3, &d_wb_3, s_h_wb_3);

    cudaMalloc( (void**) &d_out_1, s_h_out_1);
    cudaMemset(d_out_1, 0, s_h_out_1);

    cudaMalloc( (void**) &d_out_2, s_h_out_2);
    cudaMemset(d_out_2, 0, s_h_out_2);

    cudaMalloc( (void**) &d_out_3, s_h_out_3);
    cudaMemset(d_out_3, 0, s_h_out_3);

    cudaMalloc( (void**) &d_y_hat, s_y_hat);
    cudaMemset(d_y_hat, 0, s_y_hat);
 
    int threads = 1024;

    dim3 blockGridDims(1,ceil(n_h1/((float)threads)),1);
    dim3 blockDims(1,threads,1);

    mat_mult<<<blockGridDims, blockDims>>>(n_x, n_h1, d_x, d_w_1, d_out_1);
    add<<<ceil(n_h1/((float)threads)),threads>>>(n_h1, d_wb_1, d_out_1);
    relu<<<ceil(n_h1/((float)threads)),threads>>>(n_h1, d_out_1);

    
    float *d_z1 = d_out_1;

    blockGridDims = dim3(1,ceil(n_h2/((float)threads)),1);
    blockDims = dim3(1,threads,1);

    mat_mult<<<blockGridDims, blockDims>>>(n_h1, n_h2, d_z1, d_w_2, d_out_2);
    add<<<ceil(n_h2/((float)threads)),threads>>>(n_h2, d_wb_2, d_out_2);
    relu<<<ceil(n_h2/((float)threads)),threads>>>(n_h2, d_out_2);

    float *d_z2 = d_out_2;


    blockGridDims = dim3(1,ceil(n_h3/((float)threads)),1);
    blockDims = dim3(1,threads,1);

    mat_mult<<<blockGridDims, blockDims>>>(n_h2, n_h3, d_z2, d_w_3, d_out_3);
    add<<<ceil(n_h3/((float)threads)),threads>>>(n_h3, d_wb_3, d_out_3);
    relu<<<ceil(n_h3/((float)threads)),threads>>>(n_h3, d_out_3);

    softmax<<<ceil(n_h3/((float)threads)), threads>>>(n_h3, d_out_3, d_y_hat);

    cudaDeviceSynchronize();

    float *h_out;
    cudaMallocHost((void**) &h_out, s_y_hat);
    cudaMemcpy(h_out, d_y_hat, s_y_hat, cudaMemcpyDeviceToHost);

    float *h_out_pre_sm;
    cudaMallocHost((void**) &h_out_pre_sm, s_h_out_3);
    cudaMemcpy(h_out_pre_sm, d_out_3, s_h_out_3, cudaMemcpyDeviceToHost);
   
    for (int i = 0; i < n_h3; i++)
    {
       printf("%d, not_softmax: %f, softmax: %f\n", i, h_out_pre_sm[i], h_out[i]);
    }
  


    cudaFreeHost(h_out);
    cudaFreeHost(h_x);
    cudaFreeHost(h_w_1);
    cudaFree(d_x);
    cudaFree(d_w_1);
    cudaFree(d_wb_1);
    cudaFree(d_out_1);

    cudaFree(d_w_2);
    cudaFree(d_wb_2);
    cudaFree(d_out_2);

    cudaFree(d_w_3);
    cudaFree(d_wb_3);
    cudaFree(d_out_3);

    return 0;

}