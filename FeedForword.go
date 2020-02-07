package NeuralNetworkGo

import (
	"errors"

	Matrix "github.com/9init/NeuralNetworkGo/Matrix"
)

func (neural *NeuralN) FeedForword(inputArray []float64) Matrix.Matrix {
	if len(inputArray) != neural.inputNodes {
		err := errors.New("Number of \"Input Nodes\" must equal the length of \"Inputed Array\" ")
		panic(err)
	}

	inputs := Matrix.NewFromArray(inputArray)
	hidden, _ := neural.weightIH.StaticDotProduct(inputs)
	hidden.AddFromMatrix(neural.biasH)
	hidden.Map(sigmoid)
	outputs := neural.weightHO
	outputs.DotProduct(hidden)
	outputs.AddFromMatrix(neural.biasO)
	outputs.Map(sigmoid)
	return outputs
}
