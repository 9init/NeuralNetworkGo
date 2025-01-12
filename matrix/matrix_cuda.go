//go:build cuda

package matrix

/*
#cgo CFLAGS: -I./cuda
#cgo LDFLAGS: -L./cuda -lmatrix_ops
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
