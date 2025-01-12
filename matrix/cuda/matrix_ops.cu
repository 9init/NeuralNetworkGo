#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for matrix addition
__global__ void matrixAdd(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// CUDA kernel for element-wise multiplication (Hadamard product)
__global__ void matrixHadamard(float* A, float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        C[idx] = A[idx] * B[idx];
    }
}

// CUDA kernel for scalar multiplication
__global__ void matrixScalarMul(float* A, float scalar, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        C[idx] = A[idx] * scalar;
    }
}