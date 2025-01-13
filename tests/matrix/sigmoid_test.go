package matrix_test

import (
	"math"
	"neuraln/matrix"
	"testing"
)

func TestSigmoid(t *testing.T) {
	a := matrix.New(2, 3)
	a.Matrix = [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}

	result := a.Sigmoid()

	expected := [][]float64{
		{0.7310585786300049, 0.8807970779778823, 0.9525741268224334},
		{0.9820137900379085, 0.9933071490757153, 0.9975273768433653},
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

func TestDSigmoid(t *testing.T) {
	a := matrix.New(2, 3)
	a.Matrix = [][]float64{
		{0.7310585786300049, 0.8807970779778823, 0.9525741268224334},
		{0.9820137900379085, 0.9933071490757153, 0.9975273768433653},
	}

	result := a.DSigmoid()

	expected := [][]float64{
		{0.19661193324148185, 0.10499358540350662, 0.045176659730912},
		{0.017662706213291107, 0.006648056670790033, 0.002466509291608128},
	}

	// Round results and expected values to 5 decimal places to avoid floating point errors
	round := func(x float64) float64 {
		return math.Round(x*1e6) / 1e6
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			resultRounded := round(result.Matrix[i][j])
			expectedRounded := round(expected[i][j])
			if resultRounded != expectedRounded {
				t.Errorf("At (%d, %d): Expected %v, got %v", i, j, expectedRounded, resultRounded)
			}
		}
	}
}
