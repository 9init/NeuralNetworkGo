package NeuralNetworkGo

import (
	"errors"
	"fmt"
	"log"
	"math"

	Matrix "github.com/9init/Matrix/matrix"
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

//"check" is a function that check the input nodes and output nodes to avoid out of range error and some dump mistaks
func (neural *NeuralN) check(inputArray, targetArray []float64) {
	if len(inputArray) != neural.inputNodes || len(targetArray) != neural.outputNodes {
		err := errors.New("Number of (Input Nodes / Output Nodes) must equal the length of(Inputed Array / Trageted Array)")
		log.Fatal(err)
	}
}

func (neural *NeuralN) FeedForword(inputArray []float64) Matrix.Matrix {
	if len(inputArray) != neural.inputNodes {
		err := errors.New("Number of \"Input Nodes\" must equal the length of \"Inputed Array\" ")
		log.Fatal(err)
	}

	inputs := Matrix.NewFromArray(inputArray)
	hidden, _ := neural.weightIH.StaticDotProduct(inputs)
	hidden.AddFromMatrix(neural.biasH)
	hidden.Map(sigmoid)
	outputs := neural.weightHO
	outputs.DotProduct(hidden)
	outputs.AddFromMatrix(neural.biasO)
	outputs.Map(sigmoid)
	return outputs
}

func (neural *NeuralN) Train(inputArray, targetArray []float64) {
	neural.check(inputArray, targetArray)

	targets := Matrix.NewFromArray(targetArray)
	inputs := Matrix.NewFromArray(inputArray)

	hidden, _ := neural.weightIH.StaticDotProduct(inputs)
	hidden.AddFromMatrix(neural.biasH)
	hidden.Map(sigmoid)
	outputs, err := neural.weightHO.StaticDotProduct(hidden)
	outputs.AddFromMatrix(neural.biasO)
	outputs.Map(sigmoid)

	// calculate weights between hidden and outputs
	output_errors := targets
	output_errors.SuptractMatrix(outputs)

	// Calculate output gradient
	// X * (1 - X) -> dsigmoid
	outputs_G := outputs
	outputs_G.Map(dsigmoid)
	_, err = outputs_G.HadProduct(output_errors)
	if err != nil {
		fmt.Println(outputs_G, "\n", output_errors)
		log.Fatal(err)
	}

	outputs_G.Multiply(neural.learning_Rate)

	// Calculate delta
	// Learning rate * Error *
	hidden_T := hidden
	hidden_T.Transpose()
	weights_HO_G, err := outputs_G.StaticDotProduct(hidden_T)
	if err != nil {
		log.Fatal(err)
	}

	// Adjust the weight by delta
	neural.weightHO.AddFromMatrix(weights_HO_G)
	// Adjust the bias by gradient
	neural.biasO.AddFromMatrix(outputs_G)

	// Calculate hidden layer error
	whoT := neural.weightHO
	whoT.Transpose()
	hidden_errors, err := whoT.StaticDotProduct(output_errors)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate hidden gradient
	hidden_G := hidden
	hidden_G.Map(dsigmoid)
	//fmt.Println(hidden_G, "\n", hidden_errors)
	_, err = hidden_G.HadProduct(hidden_errors)
	if err != nil {
		log.Fatal(err)
	}
	hidden_G.Multiply(neural.learning_Rate)

	// Calculate input->hidden deltas
	input_T := inputs
	input_T.Transpose()
	weight_HI_Delta, _ := hidden_G.StaticDotProduct(input_T)
	if err != nil {
		log.Fatal(err)
	}

	// Adjust the weight by delta
	neural.weightIH.AddFromMatrix(weight_HI_Delta)
	// Adjust the bias by grediant
	neural.biasH.AddFromMatrix(hidden_G)

}

//helpfull functions
func sigmoid(n float64) float64 {
	return 1 / (1 + math.Exp(-n))
}

func dsigmoid(n float64) float64 {
	return n * (1 - n)
}
