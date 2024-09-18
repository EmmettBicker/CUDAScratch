#include <cuda_runtime.h>

#include <iostream>

extern "C" __global__ void kernel() {
    int x;
    asm volatile(
        ".reg .pred p;\n"                     // Declare a predicate register
        "setp.ne.s32 p, 1000000, 1000032;\n"  // Set predicate
        "selp.b32 %0, 1, 0, p;\n"             // Select based on predicate
        : "=r"(x)                             // Output operand

    );
    printf("This is x: %d\n", x);
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
