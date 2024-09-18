#ifndef ALL_GATHER
#define ALL_GATHER
__global__ void all_gather(int n_devices, int shard_length, float **list_of_arrays, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < n_devices; i++) {
        float *curr_array = list_of_arrays[i];
        out[i * shard_length + idx] = curr_array[idx];
    }
}
#endif

#ifndef ARBITRARY_OPERATION
#define ARBITRARY_OPERATION
__global__ void arbitrary_operation(int n, float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + 1;
    }
}
#endif

#ifndef REDUCE_SCATTER
#define REDUCE_SCATTER
// Would love shared memory for all the devices for the gradients for reading, maybe it should
// be moved to constant memory or to the l2 cache
__global__ void reduce_scatter(int n_devices, int shard_length, int shardIdx, float **gradients, float *out) {
    // col should have max length = shard_length
    // row should have max length = n_devices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // if first in the shard, start at 0, if second, start at 4, if third, start at 8
    int grad_idx = col + shardIdx * shard_length;

    if (row < n_devices && col < shard_length) {
        atomicAdd(&out[col], gradients[row][grad_idx]);
    }
}
#endif