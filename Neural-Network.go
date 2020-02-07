package NeuralNetworkGo

import (
	Matrix "github.com/9init/NeuralNetworkGo/Matrix"
)

//NeuralN is a Neural Network
type NeuralN struct {
	inputNodes    int
	outputNodes   int
	weightIH      Matrix.Matrix
	weightHO      Matrix.Matrix
	biasH         Matrix.Matrix
	biasO         Matrix.Matrix
	learning_Rate float64
}
