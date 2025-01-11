package neural

import "neuraln/matrix"

func (neural *Neural) Create(inputNodes, hiddenNodes, outputNodes int) *Neural {

	neural.WeightIH = matrix.New(hiddenNodes, inputNodes).Randomize()
	neural.WeightHO = matrix.New(outputNodes, hiddenNodes).Randomize()
	neural.BiasH = matrix.New(hiddenNodes, 1).Randomize()
	neural.BiasO = matrix.New(outputNodes, 1).Randomize()

	neural.LearningRate = 1
	neural.InputNodes = inputNodes
	neural.OutputNodes = outputNodes

	return neural
}
