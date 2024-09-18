__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    int N = 256;
    dim3 threadsPerBlock(16,16,1);
    dim3 numsBlocks(int(N/threadsPerBlock.x), int(N/ threadsPerBlock.y),1);

    cluster_kernel<<<numsBlocks, threadsPerBlock>>>(input, output);
}   