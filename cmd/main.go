package main

import (
	"fmt"
	"math"
	"neuraln"
)

func main() {
	nn := neuraln.New(2, 500, 1)

	inputsData := [][]float64{
		{1, 0}, {0, 1}, {1, 1}, {0, 0},
	}

	outputsData := [][]float64{
		{1}, {1}, {0}, {0},
	}

	err := nn.Train(inputsData, outputsData, 50)
	if err != nil {
		fmt.Printf("TestFeedForword failed: %v\n", err)
		return
	}

	// clear all last prints
	fmt.Print("\n")

	testingData := [][][]float64{
		{{1, 0}, {1}},
		{{0, 1}, {1}},
		{{1, 1}, {0}},
		{{0, 0}, {0}},
	}

	for _, data := range testingData {
		predictions, err := nn.Predict(data[0])
		if err != nil {
			fmt.Printf("TestFeedForword failed: %v\n", err)
			return
		}

		fmt.Printf("Predicted: %f, Expected: %f\n", predictions[0], data[1][0])

		roundedPrediction := math.Round(predictions[0])
		if roundedPrediction != data[1][0] {
			fmt.Printf("TestFeedForword failed on %v: expected %v, got %v, predictions: %v\n", data[0], data[1][0], roundedPrediction, predictions)
			return
		}

		fmt.Printf("Predected: %v, Expected: %v\n", roundedPrediction, data[1][0])
	}

}
