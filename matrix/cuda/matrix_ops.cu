#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for matrix addition
__global__ void matrixAdd(double* A, double* B, double* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrixMul(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        double sum = 0.0;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// Wrapper function for launching matrixAdd kernel
extern "C" void launchMatrixAdd(double* d_A, double* d_B, double* d_C, int rows, int cols) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching matrixAdd kernel: blocksPerGrid = %d, threadsPerBlock = %d\n", blocksPerGrid, threadsPerBlock);

    // Launch the kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronize to ensure the kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Wrapper function for launching matrixMul kernel
extern "C" void launchMatrixMul(double* d_A, double* d_B, double* d_C, int rowsA, int colsA, int colsB) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Launching matrixMul kernel: blocksPerGrid = (%d, %d), threadsPerBlock = (%d, %d)\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    // Launch the kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronize to ensure the kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
}