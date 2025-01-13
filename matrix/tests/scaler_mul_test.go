package matrix_test

import (
	"neuraln/matrix"
	"testing"
)

func TestScalerMul(t *testing.T) {
	a := matrix.New(2, 3)
	a.Matrix = [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}

	result := a.ScalerMul(2)

	expected := [][]float64{
		{2, 4, 6},
		{8, 10, 12},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if result.Matrix[i][j] != expected[i][j] {
				t.Errorf("Expected %v, got %v", expected, result.Matrix)
				return
			}
		}
	}
}
