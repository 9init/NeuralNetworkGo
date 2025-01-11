package neural

import "neuraln/matrix"

func (neural *Neural) Create(inputNodes, hiddenNodes, outputNodes int) *Neural {

	neural.WeightIH = matrix.New(hiddenNodes, inputNodes)
	neural.WeightIH.Randomize()
	neural.WeightHO = matrix.New(outputNodes, hiddenNodes)
	neural.WeightHO.Randomize()
	neural.BiasH = matrix.New(hiddenNodes, 1)
	neural.BiasH.Randomize()
	neural.BiasO = matrix.New(outputNodes, 1)
	neural.BiasO.Randomize()
	neural.LearningRate = 1
	neural.InputNodes = inputNodes
	neural.OutputNodes = outputNodes

	return neural
}
