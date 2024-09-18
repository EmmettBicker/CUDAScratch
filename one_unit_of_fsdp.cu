#include <cuda_runtime.h>

#include <iostream>

#include "fsdp_lib.cuh"

const int n_devices = 3;
const int shard_length = 4;
const int params_in_unit = n_devices * shard_length;

int main() {
    size_t sz_device_pointers = sizeof(float *) * n_devices;
    size_t sz_shard = sizeof(float) * shard_length;
    size_t sz_unit = sizeof(float) * params_in_unit;
    dim3 blockDim(shard_length, n_devices);

    cudaStream_t stream_per_device[n_devices];

    for (int i = 0; i < n_devices; i++)
        cudaStreamCreate(&stream_per_device[i]);

    float **device_weights = new float *[n_devices];

    // This is esentially one unit of FSDP computation loosely simulated with cuda code.
    for (int i = 0; i < n_devices; i++) {
        device_weights[i] = new float[shard_length];
        // Shard i on Device i
        device_weights[i][0] = 1.0f;
        device_weights[i][1] = 2.0f;
        device_weights[i][2] = 3.0f;
        device_weights[i][3] = 4.0f;
    }

    // ****************** MEMORY ALLOCATION ****************** //
    float **d_dev_ptrs = new float *[n_devices];
    float **local_d_dev_ptrs = new float *[n_devices];

    float **device_arrs;
    cudaMalloc(&device_arrs, sz_device_pointers);

    for (int i = 0; i < n_devices; i++) {
        cudaMallocAsync(&d_dev_ptrs[i], sz_shard, stream_per_device[i]);
        cudaMallocAsync(&local_d_dev_ptrs[i], sz_shard, stream_per_device[i]);
    }

    float **d_out_ptrs = new float *[n_devices];
    for (int i = 0; i < n_devices; i++) {
        cudaMallocAsync(&d_out_ptrs[i], sizeof(float) * params_in_unit, stream_per_device[i]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < n_devices; i++) {
        cudaMemcpyAsync(d_dev_ptrs[i], device_weights[i], sz_shard, cudaMemcpyHostToDevice, stream_per_device[i]);
        cudaMemcpyAsync(local_d_dev_ptrs[i], device_weights[0], sz_shard, cudaMemcpyHostToDevice, stream_per_device[i]);
    }

    cudaMemcpy(device_arrs, d_dev_ptrs, sz_device_pointers, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // ****************** OPERATIONS  ****************** //

    for (int i = 0; i < n_devices; i++) {
        all_gather<<<1, shard_length, 0, stream_per_device[i]>>>(n_devices, shard_length, device_arrs, d_out_ptrs[i]);
    }

    // ****************** MEMORY ALLOCATION PT 2 ****************** //
    // Should reduce time by doing this after launching prev operations on the GPU?
    // Also maybe all these mallocs should be done on seperate streams but that seems like too much
    float **d_activations = new float *[n_devices];

    for (int i = 0; i < n_devices; i++) {
        cudaMalloc(&d_activations[i], sz_unit);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < n_devices; i++) {
        arbitrary_operation<<<1, params_in_unit, 0, stream_per_device[i]>>>(
            params_in_unit, d_out_ptrs[i], d_activations[i]);
    }

    cudaDeviceSynchronize();

    // im aware that using float ** is less efficient than just contiguous memory
    //  but rewriting everything would be tedious.
    float **d_list_of_device_activations;
    cudaMalloc(&d_list_of_device_activations, sizeof(float *) * n_devices);
    cudaMemcpy(d_list_of_device_activations, d_activations, sizeof(float *) * n_devices, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    for (int i = 0; i < n_devices; i++) {
        reduce_scatter<<<1, blockDim, 0, stream_per_device[i]>>>(
            n_devices, shard_length, i, d_list_of_device_activations, local_d_dev_ptrs[i]);
    }

    cudaDeviceSynchronize();

    size_t sz_local_dev_out = sz_shard;
    float *h_out;
    cudaMallocHost(&h_out, sz_local_dev_out);

    cudaMemcpy(h_out, local_d_dev_ptrs[0], sz_local_dev_out, cudaMemcpyDeviceToHost);

    for (int i = 0; i < shard_length; i++) {
        printf("%f  |  ", h_out[i]);
        if ((i + 1) % 4 == 0) {
            printf("\n");
        }
    }

    cudaFree(device_arrs);
    for (int i = 0; i < n_devices; i++) {
        cudaStreamDestroy(stream_per_device[i]);
        cudaFree(d_out_ptrs[i]);
        cudaFree(d_dev_ptrs[i]);
    }
    cudaFreeHost(h_out);
    cudaFree(d_list_of_device_activations);
}