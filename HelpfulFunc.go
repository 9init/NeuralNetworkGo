package NeuralNetworkGo

import (
	"errors"
	"math"
)

//"check" is a function that check the input nodes and output nodes to avoid out of range error and some dump mistaks
func (neural *NeuralN) check(inputArray, targetArray []float64) {
	if len(inputArray) != neural.inputNodes || len(targetArray) != neural.outputNodes {
		err := errors.New("Number of (Input Nodes / Output Nodes) must equal the length of(Inputed Array / Trageted Array)")
		panic(err)
	}
}

//helpfull functions
func sigmoid(n float64) float64 {
	return 1 / (1 + math.Exp(-n))
}

func dsigmoid(n float64) float64 {
	return n * (1 - n)
}
