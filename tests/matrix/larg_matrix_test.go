package matrix_test

import (
	"neuraln/matrix"
	"testing"
)

func TestLargeMatrix(t *testing.T) {
	width := 10000
	height := 10000

	a := matrix.New(width, height)
	b := matrix.New(width, height)

	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			a.Matrix[i][j] = 1
			b.Matrix[i][j] = 2
		}
	}

	resultAdd, err := a.AddFromMatrix(b)
	if err != nil {
		t.Fatalf("AddFromMatrix failed: %v", err)
	}

	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			if resultAdd.Matrix[i][j] != 3 {
				t.Errorf("Expected 3, got %v", resultAdd.Matrix[i][j])
			}
		}
	}

	_, err = a.DotProduct(b)
	if err != nil {
		t.Fatalf("DotProduct failed: %v", err)
	}

	_, err = a.HadProduct(b)
	if err != nil {
		t.Fatalf("HadProduct failed: %v", err)
	}

	t.Log("TestLargeMatrix passed")
}
