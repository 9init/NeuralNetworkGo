#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

// Declare C wrapper functions for CUDA kernels
void cudaMatrixRand(double* A, int rows, int cols);
void cudaMatrixAdd(double* d_A, double* d_B, double* d_C, int rows, int cols);
void cudaMatrixSub(double* d_A, double* d_B, double* d_C, int rows, int cols);
void cudaMatrixMul(double* d_A, double* d_B, double* d_C, int rowsA, int colsA, int colsB);
void cudaMatrixHadamard(double* A, double* B, double* C, int rows, int cols);
void cudaMatrixTranspose(double* A, double* C, int rows, int cols);
void cudaMatrixScalarMul(double* A,  double* C, double scalar, int rows, int cols);
void cudaMatrixSigmoid(double* A, double* C, int rows, int cols);
void cudaMatrixDSigmoid(double* A, double* C, int rows, int cols);


#ifdef __cplusplus
}
#endif

#endif // MATRIX_OPS_H