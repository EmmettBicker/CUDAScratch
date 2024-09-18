#ifndef MAT_MULT
#define MAT_MULT
__global__ void mat_mult(
    int w1,
    int h1,
    int w2,
    int h2,
    float* mat1,
    float* mat2,
    float* out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < h1 && col < w2) {
        float dot_prod = 0.0f;
        for (int i = 0; i < w1; i++) {
            dot_prod += mat1[row * w1 + i] * mat2[i * w2 + col];
        }

        out[row * w2 + col] = dot_prod;
    }
}

#endif 