package neural

import (
	"encoding/json"
	"log"
	"neuraln/matrix"
)

type Neural struct {
	InputNodes   int
	OutputNodes  int
	WeightIH     matrix.Matrix
	WeightHO     matrix.Matrix
	BiasH        matrix.Matrix
	BiasO        matrix.Matrix
	LearningRate float64
}

func (n *Neural) ExportJSON() ([]byte, error) {
	b, err := json.Marshal(n)
	if err != nil {
		return []byte{}, err
	}
	return b, err
}

func ImportJSON(data []byte) (*Neural, error) {
	m := &Neural{}
	err := json.Unmarshal(data, &m)
	if err != nil {
		log.Fatal(err)
	}
	return m, err
}
