package NeuralNetworkGo

import (
	Matrix "github.com/9init/NeuralNetworkGo/Matrix"
)

func (neural *NeuralN) Train(inputArray, targetArray []float64) {
	neural.check(inputArray, targetArray)

	targets := Matrix.NewFromArray(targetArray)
	inputs := Matrix.NewFromArray(inputArray)

	hidden, _ := neural.weightIH.StaticDotProduct(inputs)
	hidden.AddFromMatrix(neural.biasH)
	hidden.Map(sigmoid)
	outputs, err := neural.weightHO.StaticDotProduct(hidden)
	outputs.AddFromMatrix(neural.biasO)
	outputs.Map(sigmoid)

	// calculate weights between hidden and outputs
	output_errors := targets
	output_errors.SuptractMatrix(outputs)

	// Calculate output gradient
	// X * (1 - X) -> dsigmoid
	outputs_G := outputs
	outputs_G.Map(dsigmoid)
	_, err = outputs_G.HadProduct(output_errors)
	if err != nil {
		panic(err)
	}

	outputs_G.Multiply(neural.learning_Rate)

	// Calculate delta
	// Learning rate * Error *
	hidden_T := hidden
	hidden_T.Transpose()
	weights_HO_G, err := outputs_G.StaticDotProduct(hidden_T)
	if err != nil {
		panic(err)
	}

	// Adjust the weight by delta
	neural.weightHO.AddFromMatrix(weights_HO_G)
	// Adjust the bias by gradient
	neural.biasO.AddFromMatrix(outputs_G)

	// Calculate hidden layer error
	whoT := neural.weightHO
	whoT.Transpose()
	hidden_errors, err := whoT.StaticDotProduct(output_errors)
	if err != nil {
		panic(err)
	}

	// Calculate hidden gradient
	hidden_G := hidden
	hidden_G.Map(dsigmoid)
	//fmt.Println(hidden_G, "\n", hidden_errors)
	_, err = hidden_G.HadProduct(hidden_errors)
	if err != nil {
		panic(err)
	}
	hidden_G.Multiply(neural.learning_Rate)

	// Calculate input->hidden deltas
	input_T := inputs
	input_T.Transpose()
	weight_HI_Delta, _ := hidden_G.StaticDotProduct(input_T)
	if err != nil {
		panic(err)
	}

	// Adjust the weight by delta
	neural.weightIH.AddFromMatrix(weight_HI_Delta)
	// Adjust the bias by grediant
	neural.biasH.AddFromMatrix(hidden_G)

}
