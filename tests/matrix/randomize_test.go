package matrix_test

import (
	"neuraln/matrix"
	"testing"
)

func TestRandomize(t *testing.T) {
	width := 1000
	height := 1000

	a := matrix.New(width, height)
	r := a.Randomize()

	for i := 0; i < width; i++ {
		for j := 0; j < height; j++ {
			if r.Matrix[i][j] == 0 {
				t.Errorf("Expected a random value, got 0")
				return
			}
		}
	}
}
