package NeuralNetworkGo

func (neural *NeuralN) Create(InputNodes, hiddenNodes, OutputNodes int) NeuralN {
	neural.WeightIH.Create(hiddenNodes, InputNodes)
	neural.WeightIH.Randomize()
	neural.WeightHO.Create(OutputNodes, hiddenNodes)
	neural.WeightHO.Randomize()
	neural.BiasH.Create(hiddenNodes, 1)
	neural.BiasH.Randomize()
	neural.BiasO.Create(OutputNodes, 1)
	neural.BiasO.Randomize()
	neural.LearningRate = 1
	neural.InputNodes = InputNodes
	neural.OutputNodes = OutputNodes
	return *neural
}
