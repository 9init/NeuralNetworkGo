package neural

import (
	"math/rand/v2"
	"neuraln/errors"
	"neuraln/matrix"
)

// Train trains the neural network using the provided input and target arrays for a specified number of epochs.
//
// Parameters:
//   - inputArray: A slice of float64 representing the input data.
//   - targetArray: A slice of float64 representing the target data.
//   - epochs: An integer specifying the number of training iterations.
//
// Returns:
//   - error: An error if the input and target arrays do not match the expected dimensions, otherwise nil.
func (neural *Neural) Train(inputArray, targetArray [][]float64, epochs int) error {
	if err := neural.validate(inputArray, targetArray); err != nil {
		return err
	}

	for i := 0; i < epochs; i++ {
		// Shuffle the input and target arrays
		shuffledInputs, shuffledTargets := shuffleArrays(inputArray, targetArray)

		for j := 0; j < len(shuffledInputs); j++ {
			// Convert the shuffled input and target arrays to matrices
			inputs := matrix.NewFromArray(shuffledInputs[j])
			targets := matrix.NewFromArray(shuffledTargets[j])

			// Perform backpropagation
			err := neural.backPropagate(inputs, targets)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// shuffleArrays shuffles the input and target arrays while maintaining their correspondence.
// Both inputArray and targetArray are 2D slices ([][]float64).
func shuffleArrays(inputArray, targetArray [][]float64) ([][]float64, [][]float64) {
	// Create a slice of indices
	indices := make([]int, len(inputArray))
	for i := range indices {
		indices[i] = i
	}

	// Shuffle the indices
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Use the shuffled indices to reorder the input and target arrays
	shuffledInputs := make([][]float64, len(inputArray))
	shuffledTargets := make([][]float64, len(targetArray))
	for i, idx := range indices {
		shuffledInputs[i] = inputArray[idx]
		shuffledTargets[i] = targetArray[idx]
	}

	return shuffledInputs, shuffledTargets
}

// validate verifies that the dimensions of the input and target arrays match the expected
// number of input and output nodes of the neural network.
//
// Parameters:
// - inputArray: A 2D slice of float64 representing the input data.
// - targetArray: A 2D slice of float64 representing the target data.
//
// Returns:
// - error: An error if the input and target arrays do not match the expected dimensions, otherwise nil.
func (neural *Neural) validate(inputArray, targetArray [][]float64) error {

	if len(inputArray) == 0 || len(targetArray) == 0 {
		return errors.ErrEmptyInputOutput
	}

	if len(inputArray) != len(targetArray) {
		return errors.ErrInputOutputMismatch
	}

	for _, input := range inputArray {
		if len(input) != neural.InputNodes {
			return errors.ErrInputNodesMismatch
		}
	}

	for _, target := range targetArray {
		if len(target) != neural.OutputNodes {
			return errors.ErrOutputNodesMismatch
		}
	}

	return nil
}
