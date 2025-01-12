package matrix_test

import (
	"neuraln/matrix"
	"testing"
)

func TestMatrixOperations(t *testing.T) {
	// Test matrix addition
	a := matrix.New(2, 2)
	a.Matrix = [][]float64{
		{1, 2},
		{3, 4},
	}

	b := matrix.New(2, 2)
	b.Matrix = [][]float64{
		{5, 6},
		{7, 8},
	}

	resultAdd, err := a.AddFromMatrix(b)
	if err != nil {
		t.Fatalf("AddFromMatrix failed: %v", err)
	}

	expectedAdd := [][]float64{
		{6, 8},
		{10, 12},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if resultAdd.Matrix[i][j] != expectedAdd[i][j] {
				t.Errorf("Expected %v, got %v", expectedAdd, resultAdd.Matrix)
				goto out
			}
		}
	}
out:

	// Test matrix multiplication
	c := matrix.New(2, 3)
	c.Matrix = [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}

	d := matrix.New(3, 2)
	d.Matrix = [][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	}

	resultMul, err := c.DotProduct(d)
	if err != nil {
		t.Fatalf("DotProduct failed: %v", err)
	}

	expectedMul := [][]float64{
		{58, 64},
		{139, 154},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if resultMul.Matrix[i][j] != expectedMul[i][j] {
				t.Errorf("Expected %v, got %v", expectedMul, resultMul.Matrix)
				return
			}
		}
	}
}
