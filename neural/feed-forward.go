package neural

import (
	"neuraln/errors"
	"neuraln/matrix"
)

// FeedForword performs the forward pass of the neural network, computing the output
// for a given input array.
//
// Parameters:
//   - inputArray: A slice of float64 representing the input values to the neural network.
//
// Returns:
//   - *matrix.Matrix: A pointer to the output matrix.
//   - error: An error if the input array length does not match the number of input nodes, otherwise nil.
func (neural *Neural) FeedForword(inputArray []float64) (*matrix.Matrix, error) {
	// Check if the input array length matches the number of input nodes
	if len(inputArray) != neural.InputNodes {
		return nil, errors.ErrInputNodesMismatch
	}

	// Convert the input array to a matrix
	inputs := matrix.NewFromArray(inputArray)

	// Compute hidden layer activations
	hidden, err := neural.WeightIH.DotProduct(inputs)
	if err != nil {
		return nil, err
	}
	hidden, err = hidden.AddFromMatrix(neural.BiasH)
	if err != nil {
		return nil, err
	}
	hidden = hidden.Sigmoid()

	// Compute output layer activations
	outputs, err := neural.WeightHO.DotProduct(hidden)
	if err != nil {
		return nil, err
	}
	outputs, err = outputs.AddFromMatrix(neural.BiasO)
	if err != nil {
		return nil, err
	}
	outputs = outputs.Sigmoid()

	return outputs, nil
}
