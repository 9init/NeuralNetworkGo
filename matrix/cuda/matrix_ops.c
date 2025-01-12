#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

// Define DEBUG flag at compile time (or pass it via the compiler)
// #define DEBUG

// Wrapper for matrix addition
void cudaMatrixAdd(double* A, double* B, double* C, int rows, int cols) {
    double *d_A, *d_B, *d_C;
    size_t size = rows * cols * sizeof(double);

    #ifdef DEBUG
    printf("Allocating memory for matrixAdd: size = %zu\n", size);
    #endif

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_A: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy data to device
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_A: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Call CUDA kernel launch wrapper
    launchMatrixAdd(d_A, d_B, d_C, rows, cols);

    // Copy result back to host
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_C: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cudaMatrixSub(double* A, double* B, double* C, int rows, int cols) {
    double *d_A, *d_B, *d_C;
    size_t size = rows * cols * sizeof(double);

    #ifdef DEBUG
    printf("Allocating memory for matrixSub: size = %zu\n", size);
    #endif

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_A: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy data to device
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_A: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Call CUDA kernel launch wrapper
    launchMatrixSub(d_A, d_B, d_C, rows, cols);

    // Copy result back to host
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_C: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Wrapper for matrix multiplication
void cudaMatrixMul(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
    double *d_A, *d_B, *d_C;
    size_t sizeA = rowsA * colsA * sizeof(double);
    size_t sizeB = colsA * colsB * sizeof(double);
    size_t sizeC = rowsA * colsB * sizeof(double);


    #ifdef DEBUG
    printf("Allocating memory for matrixMul: sizeA = %zu, sizeB = %zu, sizeC = %zu\n", sizeA, sizeB, sizeC);
    #endif

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_A, sizeA);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_A: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&d_B, sizeB);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc((void**)&d_C, sizeC);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy data to device
    err = cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_A: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Call CUDA kernel launch wrapper
    launchMatrixMul(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Copy result back to host
    err = cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_C: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Wrapper for element-wise matrix multiplication (Hadamard product)
void cudaMatrixHadamard(double* A, double* B, double* C, int rows, int cols) {
    double *d_A, *d_B, *d_C;
    size_t size = rows * cols * sizeof(double);

    #ifdef DEBUG
    printf("Allocating memory for matrixHadamard: size = %zu\n", size);
    #endif

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_A: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy data to device
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_A: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Call CUDA kernel launch wrapper
    launchMatrixHadamard(d_A, d_B, d_C, rows, cols);

    // Copy result back to host
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for d_C: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}