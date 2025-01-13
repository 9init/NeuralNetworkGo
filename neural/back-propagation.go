package neural

import (
	"neuraln/matrix"
)

// backPropagate performs the backpropagation algorithm using Gradient Descent to adjust
// the weights and biases of the neural network based on the provided inputs and target outputs.
//
// The function computes the error between the predicted outputs and the target outputs,
// then propagates this error backward through the network to calculate gradients.
// These gradients are used to update the weights and biases, minimizing the error.
//
// Parameters:
//   - inputs: A matrix representing the input values to the neural network.
//   - targets: A matrix representing the target output values for the given inputs.
//
// If any matrix operations fail, the function will return an error.
func (neural *Neural) backPropagate(inputs *matrix.Matrix, targets *matrix.Matrix) error {
	// Forward pass
	hidden, err := neural.WeightIH.DotProduct(inputs)
	if err != nil {
		return err
	}
	hidden, err = hidden.AddFromMatrix(neural.BiasH)
	if err != nil {
		return err
	}
	hidden = hidden.Sigmoid()
	outputs, err := neural.WeightHO.DotProduct(hidden)
	if err != nil {
		return err
	}
	outputs, err = outputs.AddFromMatrix(neural.BiasO)
	if err != nil {
		return err
	}
	outputs = outputs.Sigmoid()

	// Calculate output errors
	outputErrors, err := targets.SubtractMatrix(outputs)
	if err != nil {
		return err
	}

	// Calculate output gradient
	outputGradients := outputs.DSigmoid()
	outputGradients, err = outputGradients.HadProduct(outputErrors)
	if err != nil {
		return err
	}
	outputGradients = outputGradients.ScalerMul(neural.LearningRate)

	// Calculate delta for weights between hidden and output layers
	hiddenTransposed := hidden.Transpose()
	weightsHODelta, err := outputGradients.DotProduct(hiddenTransposed)
	if err != nil {
		return err
	}

	// Adjust weights and biases for the output layer
	neural.WeightHO, err = neural.WeightHO.AddFromMatrix(weightsHODelta)
	if err != nil {
		return err
	}
	neural.BiasO, err = neural.BiasO.AddFromMatrix(outputGradients)
	if err != nil {
		return err
	}

	// Calculate hidden layer errors
	weightHOTransposed := neural.WeightHO.Transpose()
	hiddenErrors, err := weightHOTransposed.DotProduct(outputErrors)
	if err != nil {
		return err
	}

	// Calculate hidden gradient
	hiddenGradients := hidden.DSigmoid()
	hiddenGradients, err = hiddenGradients.HadProduct(hiddenErrors)
	if err != nil {
		return err
	}
	hiddenGradients = hiddenGradients.ScalerMul(neural.LearningRate)

	// Calculate delta for weights between input and hidden layers
	inputsTransposed := inputs.Transpose()
	weightsIHDelta, err := hiddenGradients.DotProduct(inputsTransposed)
	if err != nil {
		return err
	}

	// Adjust weights and biases for the hidden layer
	neural.WeightIH, err = neural.WeightIH.AddFromMatrix(weightsIHDelta)
	if err != nil {
		return err
	}
	neural.BiasH, err = neural.BiasH.AddFromMatrix(hiddenGradients)
	if err != nil {
		return err
	}

	return nil
}
