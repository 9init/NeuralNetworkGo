package matrix_test

import (
	"neuraln/matrix"
	"testing"
)

func TestLargeMatrix(t *testing.T) {
	width := 5000
	height := 5000

	a := matrix.New(width, height)
	b := matrix.New(width, height)

	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			a.Matrix[i][j] = 1
			b.Matrix[i][j] = 2
		}
	}

	_, err := a.AddFromMatrix(b)
	if err != nil {
		t.Fatalf("AddFromMatrix failed: %v", err)
	}

	_, err = a.DotProduct(b)
	if err != nil {
		t.Fatalf("DotProduct failed: %v", err)
	}

	_, err = a.HadProduct(b)
	if err != nil {
		t.Fatalf("HadProduct failed: %v", err)
	}

	_, err = a.SubtractMatrix(b)
	if err != nil {
		t.Fatalf("SubtractMatrix failed: %v", err)
	}

	a.ScalerMul(2)

	a.Transpose()

	a.Randomize()

	a.Sigmoid()

	a.DSigmoid()
}
