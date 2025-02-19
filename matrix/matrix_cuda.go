//go:build cuda

package matrix

/*
#cgo LDFLAGS: -L../build/lib -lmatrix_ops
#include "cuda/matrix_ops.h"
*/
import "C"

import (
	"neuraln/errors"
	"unsafe"
)

// AddFromMatrix adds another Matrix to the current Matrix using CUDA.
func (m *Matrix) AddFromMatrix(sMatrix *Matrix) (*Matrix, error) {
	if m.Col != sMatrix.Col || m.Row != sMatrix.Row {
		return nil, errors.ErrMatricesDimensionsMustMatch
	}

	result := New(m.Row, m.Col)

	// Flatten matrices
	a := m.Flatten()
	b := sMatrix.Flatten()
	c := make([]float64, len(a))

	// Call CUDA wrapper
	C.cudaMatrixAdd(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&c[0])),
		C.int(m.Row),
		C.int(m.Col),
	)

	// Reshape result
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = c[i*m.Col+j]
		}
	}

	return result, nil
}

// SubtractMatrix subtracts another Matrix from the current Matrix using CUDA.
func (m *Matrix) SubtractMatrix(sMatrix *Matrix) (*Matrix, error) {
	if m.Col != sMatrix.Col || m.Row != sMatrix.Row {
		return nil, errors.ErrMatricesDimensionsMustMatch
	}

	result := New(m.Row, m.Col)

	// Flatten matrices
	a := m.Flatten()
	b := sMatrix.Flatten()
	c := make([]float64, len(a))

	// Call CUDA wrapper
	C.cudaMatrixSub(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&c[0])),
		C.int(m.Row),
		C.int(m.Col),
	)

	// Reshape result
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = c[i*m.Col+j]
		}
	}

	return result, nil
}

// DotProduct performs matrix multiplication using CUDA.
func (m *Matrix) DotProduct(sMatrix *Matrix) (*Matrix, error) {
	if m.Col != sMatrix.Row {
		return nil, errors.ErrRowsMustEqualColumns
	}

	result := New(m.Row, sMatrix.Col)

	// Flatten matrices
	a := m.Flatten()
	b := sMatrix.Flatten()
	c := make([]float64, m.Row*sMatrix.Col)

	// Call CUDA wrapper
	C.cudaMatrixMul(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&c[0])),
		C.int(m.Row),
		C.int(m.Col),
		C.int(sMatrix.Col),
	)

	// Reshape result
	for i := 0; i < m.Row; i++ {
		for j := 0; j < sMatrix.Col; j++ {
			result.Matrix[i][j] = c[i*sMatrix.Col+j]
		}
	}

	return result, nil
}

// HadProduct performs element-wise multiplication (Hadamard product) using CUDA.
func (m *Matrix) HadProduct(sMatrix *Matrix) (*Matrix, error) {
	if m.Row != sMatrix.Row || m.Col != sMatrix.Col {
		return nil, errors.ErrRowsColsMustEqual
	}

	result := New(m.Row, m.Col)

	// Flatten matrices
	a := m.Flatten()
	b := sMatrix.Flatten()
	c := make([]float64, len(a))

	// Call CUDA wrapper
	C.cudaMatrixHadamard(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&c[0])),
		C.int(m.Row),
		C.int(m.Col),
	)

	// Reshape result
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = c[i*m.Col+j]
		}
	}

	return result, nil
}

// Randomize fills the Matrix with random values using CUDA.
func (m *Matrix) Randomize() *Matrix {
	result := New(m.Row, m.Col)

	// Flatten matrix
	a := result.Flatten()

	// Call CUDA wrapper
	C.cudaMatrixRand(
		(*C.double)(unsafe.Pointer(&a[0])),
		C.int(m.Row),
		C.int(m.Col),
	)

	// Reshape result
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = a[i*m.Col+j]
		}
	}

	return result
}

// Transpose returns the transpose of the Matrix using CUDA.
func (m *Matrix) Transpose() *Matrix {
	result := New(m.Col, m.Row)

	// Flatten matrices
	a := m.Flatten()
	b := make([]float64, len(a))

	// Call CUDA wrapper
	C.cudaMatrixTranspose(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		C.int(m.Row),
		C.int(m.Col),
	)

	// Reshape result
	for i := 0; i < m.Col; i++ {
		for j := 0; j < m.Row; j++ {
			result.Matrix[i][j] = b[i*m.Row+j]
		}
	}

	return result
}

// ScalerMul multiplies the Matrix by a scalar value using CUDA.
func (m *Matrix) ScalerMul(n float64) *Matrix {
	result := New(m.Row, m.Col)

	// Flatten matrix
	a := m.Flatten()
	b := make([]float64, len(a))

	// Call CUDA wrapper
	C.cudaMatrixScalarMul(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		C.double(n),
		C.int(m.Row),
		C.int(m.Col),
	)

	// Reshape result
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = b[i*m.Col+j]
		}
	}

	return result
}

// Sigmoid applies the sigmoid function to the Matrix using CUDA.
func (m *Matrix) Sigmoid() *Matrix {
	result := New(m.Row, m.Col)

	// Flatten matrix
	a := m.Flatten()
	b := make([]float64, len(a))

	// Call CUDA wrapper
	C.cudaMatrixSigmoid(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		C.int(m.Row),
		C.int(m.Col),
	)

	// Reshape result
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = b[i*m.Col+j]
		}
	}

	return result
}

// DSigmoid applies the derivative of the sigmoid function to the Matrix using CUDA.
func (m *Matrix) DSigmoid() *Matrix {
	result := New(m.Row, m.Col)

	// Flatten matrix
	a := m.Flatten()
	b := make([]float64, len(a))

	// Call CUDA wrapper
	C.cudaMatrixDSigmoid(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		C.int(m.Row),
		C.int(m.Col),
	)

	// Reshape result
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = b[i*m.Col+j]
		}
	}

	return result
}
