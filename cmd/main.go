/**
 * This is a simple example of how to use the neuraln package.
 * It trains a neural network to learn the XOR function.
**/

package main

import (
	"fmt"
	"math"
	"neuraln"
)

func main() {
	// Create a new neural network with 2 input neurons, 500 hidden neurons, and 1 output neuron
	nn := neuraln.New(2, 500, 1)

	// Define the XOR problem training data
	inputs := [][]float64{
		{1, 0}, {0, 1}, {1, 1}, {0, 0},
	}

	targets := [][]float64{
		{1}, {1}, {0}, {0},
	}

	// Train the neural network
	fmt.Println("Training the neural network...")
	err := nn.Train(inputs, targets, 50) // 50 epochs
	if err != nil {
		fmt.Printf("Error during training: %v\n", err)
		return
	}
	fmt.Print("Training complete! Testing the network...\n\n")

	// Test the neural network with the same inputs
	testingData := [][][]float64{
		{{1, 0}, {1}}, // Input: [1, 0], Expected Output: 1
		{{0, 1}, {1}}, // Input: [0, 1], Expected Output: 1
		{{1, 1}, {0}}, // Input: [1, 1], Expected Output: 0
		{{0, 0}, {0}}, // Input: [0, 0], Expected Output: 0
	}

	for _, data := range testingData {
		input := data[0]
		expected := data[1][0]

		// Get the neural network's prediction
		predictions, err := nn.Predict(input)
		if err != nil {
			fmt.Printf("Error during prediction: %v\n", err)
			return
		}

		// Round the prediction to the nearest integer (0 or 1)
		roundedPrediction := math.Round(predictions[0])

		// Print the results
		fmt.Printf("Input: %v\n", input)
		fmt.Printf("  - Expected Output: %v\n", expected)
		fmt.Printf("  - Raw Prediction:  %.7f\n", predictions[0])
		fmt.Printf("  - Rounded Prediction: %v\n", roundedPrediction)

		// Check if the prediction matches the expected output
		if roundedPrediction != expected {
			fmt.Printf("  ❌ Mismatch! Expected %v, but got %v\n", expected, roundedPrediction)
		} else {
			fmt.Printf("  ✅ Correct! Prediction matches expected output.\n")
		}
		fmt.Println()
	}

	fmt.Println("Testing complete!")
}
