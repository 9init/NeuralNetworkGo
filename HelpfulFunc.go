package NeuralNetworkGo

import (
	"errors"
	"log"
	"math"
)

//"check" is a function that check the input nodes and output nodes to avoid out of range error and some dump mistaks
func (neural *NeuralN) check(inputArray, targetArray []float64) {
	if len(inputArray) != neural.InputNodes || len(targetArray) != neural.OutputNodes {
		err := errors.New("Number of (Input Nodes / Output Nodes) must equal the length of(Inputed Array / Trageted Array)")
		log.Fatal(err)
	}
}

//helpfull functions
func sigmoid(n float64) float64 {
	return 1 / (1 + math.Exp(-n))
}

func dsigmoid(n float64) float64 {
	return n * (1 - n)
}
