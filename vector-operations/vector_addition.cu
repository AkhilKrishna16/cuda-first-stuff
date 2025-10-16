#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // gets the global index for a thread
    if(i < N) {
        C[i] = A[i] + B[i]; 
    }
}

int main() {
    const int N = 1 << 20; // you can only have a max of 1024 or 1 << 10 per block. 
    // if over limit, need more blocks.
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t size = N * sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];
    srand(time(NULL)); // ?

    for(int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); // this allocates the data on the GPU!
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);  // host to device == CPU->GPU
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    VecAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N); // `blocks` block, each with `threadsPerBlock`

    float *h_C = new float[N];
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "[";
    for(int i = 0; i < N; i++) {
        cout << h_C[i] << ", ";
    }
    cout << "]";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}