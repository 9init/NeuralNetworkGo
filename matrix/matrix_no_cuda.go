//go:build !cuda

package matrix

import (
	"fmt"
	"neuraln/errors"
)

// AddFromMatrixGPU adds another Matrix to the current Matrix using the CPU fallback.
func (m *Matrix) AddFromMatrix(sMatrix *Matrix) (*Matrix, error) {
	if m.Col != sMatrix.Col || m.Row != sMatrix.Row {
		return nil, errors.ErrMatricesDimensionsMustMatch
	}

	a := m.Flatten()
	b := sMatrix.Flatten()

	// print first 10 elements of a and b
	if len(a) >= 10 && len(b) >= 10 {
		fmt.Println("First 10 elements of a and b before CUDA call:")
		fmt.Println(a[:10])
		fmt.Println(b[:10])
	}

	result := New(m.Row, m.Col)
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = m.Matrix[i][j] + sMatrix.Matrix[i][j]
		}
	}

	if len(a) >= 10 && len(b) >= 10 {
		fmt.Println("First 10 elements of result after CPU call:")
		fmt.Println(result.Flatten())
	}

	return result, nil
}

// DotProductGPU performs matrix multiplication using the CPU fallback.
func (m *Matrix) DotProduct(sMatrix *Matrix) (*Matrix, error) {
	if m.Col != sMatrix.Row {
		return nil, errors.ErrRowsMustEqualColumns
	}

	result := New(m.Row, sMatrix.Col)
	for i := 0; i < m.Row; i++ {
		for j := 0; j < sMatrix.Col; j++ {
			for k := 0; k < sMatrix.Row; k++ {
				result.Matrix[i][j] += m.Matrix[i][k] * sMatrix.Matrix[k][j]
			}
		}
	}

	return result, nil
}
