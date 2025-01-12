#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

// Declare C wrapper functions for CUDA kernels

// Wrapper for matrix addition
void cudaMatrixAdd(float* A, float* B, float* C, int rows, int cols);

// Wrapper for matrix multiplication
void cudaMatrixMul(float* A, float* B, float* C, int rowsA, int colsA, int colsB);

// Wrapper for element-wise multiplication (Hadamard product)
void cudaMatrixHadamard(float* A, float* B, float* C, int rows, int cols);

// Wrapper for scalar multiplication
void cudaMatrixScalarMul(float* A, float scalar, float* C, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_OPS_H