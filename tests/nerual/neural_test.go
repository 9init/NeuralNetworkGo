package neural_test

import (
	"math"
	"neuraln"
	"testing"
)

func TestCreate(t *testing.T) {
	if neuraln.New(2, 2, 1) == nil {
		t.Error("TestCreate failed")
		return
	}
	t.Log("TestCreate passed")
}

func TestTrain(t *testing.T) {
	nn := neuraln.New(2, 2, 1)

	inputsData := [][]float64{
		{1, 0}, {0, 1}, {1, 1}, {0, 0},
	}

	outputsData := [][]float64{
		{1}, {1}, {0}, {0},
	}

	nn.Train(inputsData, outputsData, 100)

	t.Log("TestTrain passed")
}

func TestFeedForword(t *testing.T) {
	nn := neuraln.New(2, 5000, 1)

	inputsData := [][]float64{
		{1, 0}, {0, 1}, {1, 1}, {0, 0},
	}

	outputsData := [][]float64{
		{1}, {1}, {0}, {0},
	}

	err := nn.Train(inputsData, outputsData, 50)
	if err != nil {
		t.Errorf("TestFeedForword failed: %v", err)
		return
	}

	testingData := [][][]float64{
		{{1, 0}, {1}},
		{{0, 1}, {1}},
		{{1, 1}, {0}},
		{{0, 0}, {0}},
	}

	for _, data := range testingData {
		predictions, err := nn.Predict(data[0])
		if err != nil {
			t.Errorf("TestFeedForword failed: %v", err)
			return
		}

		roundedPrediction := math.Round(predictions[0])
		if roundedPrediction != data[1][0] {
			t.Errorf("TestFeedForword failed on %v: expected %v, got %v, predictions: %v", data[0], data[1][0], roundedPrediction, predictions)
			return
		}
	}

	t.Log("TestFeedForword passed")
}
