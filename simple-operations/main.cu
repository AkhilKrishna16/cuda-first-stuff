#include <iostream>
using namespace std;

__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    int N = 5;
    size_t size = N * sizeof(float);

    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];
    for (int i = 0; i < N; i++) { // this allocates the data on the CPU!
        A[i] = i * 2;
        B[i] = i;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); // this allocates the data on the GPU!
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice); // host to device == CPU->GPU
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); // host to device == GPU->CPU

    for (int i = 0; i < N; i++) {
        cout << C[i] << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}