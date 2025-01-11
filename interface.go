package neuraln

import "neuraln/neural"

type NeuralNetwork struct {
	neural *neural.Neural
}

func New(inputNodes, hiddenNodes, outputNodes int) *NeuralNetwork {
	neural := neural.Neural{}
	return &NeuralNetwork{
		neural.Create(inputNodes, hiddenNodes, outputNodes),
	}
}

func ImportJSON(data []byte) (*neural.Neural, error) {
	return neural.ImportJSON(data)
}

func (n *NeuralNetwork) ExportJSON() ([]byte, error) {
	return n.neural.ExportJSON()
}

func (n *NeuralNetwork) Train(inputArray, targetArray [][]float64, epochs int) error {
	return n.neural.Train(inputArray, targetArray, epochs)
}

func (n *NeuralNetwork) Predict(inputArray []float64) ([]float64, error) {
	predictions, err := n.neural.FeedForword(inputArray)
	if err != nil {
		return nil, err
	}

	return predictions.Flatten(), nil
}
