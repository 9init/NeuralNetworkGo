package neural

import (
	"fmt"
	"neuraln/matrix"
)

func (neural *Neural) Create(inputNodes, hiddenNodes, outputNodes int) *Neural {

	neural.WeightIH = matrix.New(hiddenNodes, inputNodes).Randomize()
	neural.WeightHO = matrix.New(outputNodes, hiddenNodes).Randomize()
	neural.BiasH = matrix.New(hiddenNodes, 1).Randomize()
	neural.BiasO = matrix.New(outputNodes, 1).Randomize()

	// print all weights and biases
	fmt.Println("WeightIH:")
	fmt.Println(neural.WeightIH.Matrix)
	fmt.Println("WeightHO:")
	fmt.Println(neural.WeightHO.Matrix)
	fmt.Println("BiasH:")
	fmt.Println(neural.BiasH.Matrix)
	fmt.Println("BiasO:")
	fmt.Println(neural.BiasO.Matrix)

	neural.LearningRate = 1
	neural.InputNodes = inputNodes
	neural.OutputNodes = outputNodes

	return neural
}
