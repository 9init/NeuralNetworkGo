package NeuralNetworkGo

import (
	"encoding/json"
	"log"

	Matrix "github.com/9init/NeuralNetworkGo/Matrix"
)

//NeuralN is a Neural Network
type NeuralN struct {
	InputNodes   int
	OutputNodes  int
	WeightIH     Matrix.Matrix
	WeightHO     Matrix.Matrix
	BiasH        Matrix.Matrix
	BiasO        Matrix.Matrix
	LearningRate float64
}

func (n *NeuralN) ExportJSON() ([]byte, error) {
	b, err := json.Marshal(n)
	if err != nil {
		return []byte{}, err
	}
	return b, err
}

func ImportJSON(data []byte) (NeuralN, error) {
	var m NeuralN
	err := json.Unmarshal(data, &m)
	if err != nil {
		log.Fatal(err)
	}
	return m, err
}
