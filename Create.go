package NeuralNetworkGo

func (neural *NeuralN) Create(inputNodes, hiddenNodes, outputNodes int) NeuralN {
	neural.weightIH.Create(hiddenNodes, inputNodes)
	neural.weightIH.Randomize()
	neural.weightHO.Create(outputNodes, hiddenNodes)
	neural.weightHO.Randomize()
	neural.biasH.Create(hiddenNodes, 1)
	neural.biasH.Randomize()
	neural.biasO.Create(outputNodes, 1)
	neural.biasO.Randomize()
	neural.learning_Rate = 1
	neural.inputNodes = inputNodes
	neural.outputNodes = outputNodes
	return *neural
}
