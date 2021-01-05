package NeuralNetworkGo

import (
	"errors"
	"log"

	Matrix "github.com/9init/NeuralNetworkGo/Matrix"
)

func (neural *NeuralN) FeedForword(inputArray []float64) Matrix.Matrix {
	if len(inputArray) != neural.InputNodes {
		err := errors.New("Number of \"Input Nodes\" must equal the length of \"Inputed Array\" ")
		log.Fatal(err)
	}

	inputs := Matrix.NewFromArray(inputArray)
	hidden, _ := neural.WeightIH.StaticDotProduct(inputs)
	hidden.AddFromMatrix(neural.BiasH)
	hidden.Map(sigmoid)
	outputs := neural.WeightHO
	outputs.DotProduct(hidden)
	outputs.AddFromMatrix(neural.BiasO)
	outputs.Map(sigmoid)
	return outputs
}
