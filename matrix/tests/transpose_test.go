package matrix_test

import (
	"neuraln/matrix"
	"testing"
)

func TestTranspose(t *testing.T) {
	a := matrix.New(2, 3)
	a.Matrix = [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}

	result := a.Transpose()

	expected := [][]float64{
		{1, 4},
		{2, 5},
		{3, 6},
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			if result.Matrix[i][j] != expected[i][j] {
				t.Errorf("Expected %v, got %v", expected, result.Matrix)
				return
			}
		}
	}
}
