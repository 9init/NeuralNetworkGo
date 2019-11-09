package NeuralNetwork

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

//Matrix is a Matrix from math
type Matrix_s struct {
	Matrix [][]float64
	col    int
	row    int
}

func (m *Matrix_s) fromArray(array []float64) Matrix_s {
	m.create(len(array), 1)
	for i, v := range array {
		m.Matrix[i][0] = v
	}
	return *m
}

func (m *Matrix_s) create(col int, row int) Matrix_s {
	m.col = col
	m.row = row
	m.Matrix = make([][]float64, col)
	for i := range m.Matrix {
		m.Matrix[i] = make([]float64, row)
	}

	return *m
}

func (m *Matrix_s) randomize() {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			rand.Seed(time.Now().UnixNano())
			time.Sleep(1)
			n := rand.Float64()*(1-(-1)) - 1
			m.Matrix[i][j] = n
		}
	}
}

func (m *Matrix_s) addFromMatrix(sMatrix Matrix_s) Matrix_s {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			m.Matrix[i][j] += sMatrix.Matrix[i][j]
		}
	}
	return *m
}

//Map takes a function and preform the function for every single value in the matrix
func (m *Matrix_s) Map(f func(float64) float64) Matrix_s {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			m.Matrix[i][j] = f(m.Matrix[i][j])
		}
	}
	return *m
}

func (m *Matrix_s) dotProduct(sMatrix Matrix_s) (Matrix_s, error) {
	if m.row != sMatrix.col {
		err := errors.New("rows must equal colomns")
		return *m, err
	}

	nMatrix := make([][]float64, m.col)
	for i := range nMatrix {
		nMatrix[i] = make([]float64, sMatrix.row)
	}

	for i := 0; i < m.col; i++ {
		for j := 0; j < sMatrix.row; j++ {
			for k := 0; k < sMatrix.col; k++ {
				nMatrix[i][j] += m.Matrix[i][k] * sMatrix.Matrix[k][j]
			}
		}
	}

	m.Matrix = nMatrix
	m.row = sMatrix.row
	return *m, nil
}

func (m *Matrix_s) HarProduct(sMatrix Matrix_s) (Matrix_s, error) {
	if m.row != sMatrix.row || m.col != sMatrix.col {
		err := errors.New("rows&cols must equal")
		return *m, err
	}

	nMatrix := new(Matrix_s).create(m.col, m.row)

	for i := 0; i < m.col; i++ {
		for j := 0; j < sMatrix.row; j++ {
			nMatrix.Matrix[i][j] = m.Matrix[i][j] * sMatrix.Matrix[i][j]
		}
	}

	*m=nMatrix
	return *m, nil
}


func (m *Matrix_s) multiply(n float64) Matrix_s {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
				m.Matrix[i][j] = m.Matrix[i][j] * n
			}
	}
	return *m
}

func (m *Matrix_s) suptractMatrix(sMatrix Matrix_s) Matrix_s {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			m.Matrix[i][j] -= sMatrix.Matrix[i][j]
		}
	}
	return *m
}

func (m *Matrix_s) transpose() (Matrix_s, error) {
	result := new(Matrix_s).create(m.row, m.col)
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			result.Matrix[j][i] = m.Matrix[i][j]
		}
	}
	*m = result
	return result, nil
}

//NeuralN is a Neural Network
type NeuralN struct {
	inputNodes    int
	outputNodes   int
	weightIH      Matrix_s
	weightHO      Matrix_s
	biasH         Matrix_s
	biasO         Matrix_s
	learning_Rate float64
}

func (neural *NeuralN) Create(inputNodes, hiddenNodes, outputNodes int) NeuralN {
	neural.weightIH.create(hiddenNodes, inputNodes)
	neural.weightIH.randomize()
	neural.weightHO.create(outputNodes, hiddenNodes)
	neural.weightHO.randomize()
	neural.biasH.create(hiddenNodes, 1)
	neural.biasH.randomize()
	neural.biasO.create(outputNodes, 1)
	neural.biasO.randomize()
	neural.learning_Rate = 0.2
	neural.inputNodes=inputNodes
	neural.outputNodes=outputNodes
	return *neural
}

//"check" is a function that check the input nodes and output nodes to avoid out of range error and some dump mistaks
func (neural *NeuralN) check(inputArray, targetArray []float64){
	if len(inputArray) != neural.inputNodes || len(targetArray) != neural.outputNodes{
		err := errors.New("Number of (Input Nodes / Output Nodes) must equal the length of(Inputed Array / Trageted Array)")
		log.Fatal(err)
	}
}

func (neural *NeuralN) FeedForword(inputArray []float64) Matrix_s{
	if len(inputArray) != neural.inputNodes{
		err := errors.New("Number of \"Input Nodes\" must equal the length of \"Inputed Array\" ")
		log.Fatal(err)
	}
	
	inputs := new(Matrix_s).fromArray(inputArray)
	hidden := neural.weightIH
	hidden.dotProduct(inputs)
	hidden.addFromMatrix(neural.biasH)
	hidden.Map(sigmoid)
	outputs := neural.weightHO
	outputs.dotProduct(hidden)
	outputs.addFromMatrix(neural.biasO)
	outputs.Map(sigmoid)
	return outputs
}

func (neural *NeuralN) Train(inputArray, targetArray []float64) {
	neural.check(inputArray, targetArray)
	
	targets := new(Matrix_s).fromArray(targetArray)
	inputs := new(Matrix_s).fromArray(inputArray)

	hidden := neural.weightIH
	hidden.dotProduct(inputs)
	hidden.addFromMatrix(neural.biasH)
	hidden.Map(sigmoid)
	outputs := neural.weightHO
	outputs.dotProduct(hidden)
	outputs.addFromMatrix(neural.biasO)
	outputs.Map(sigmoid)

	// calculate weights between hidden and outputs
	output_errors := targets
	output_errors.suptractMatrix(outputs)

	// Calaculat output grediants
	// X * (1 - X) -> dsigmoid
	outputs_G := outputs
	outputs_G.Map(dsigmoid)
	_, err := outputs_G.HarProduct(output_errors)
	if err != nil {
		fmt.Println(outputs_G, "\n", output_errors)
		fmt.Println(0)
		log.Fatal(err)
	}

	outputs_G.multiply(neural.learning_Rate)

	// Calculat delta
	// Learning rate * Error *
	hidden_T := hidden
	hidden_T.transpose()
	weights_HO_G, err := outputs_G.dotProduct(hidden_T)
	if err != nil {
		log.Fatal(err)
	}

	// Adjust the weight by delta
	neural.weightHO.addFromMatrix(weights_HO_G)
	// Adjust the bias by grediant
	neural.biasO.addFromMatrix(outputs_G)

	// Calculate hidden layer error
	whoT := neural.weightHO
	whoT.transpose()
	hidden_errors, err := whoT.dotProduct(output_errors)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate hidden grediant
	hidden_G := hidden
	hidden_G.Map(dsigmoid)
	//fmt.Println(hidden_G, "\n", hidden_errors)
	_, err = hidden_G.HarProduct(hidden_errors)
	if err != nil {
		log.Fatal(err)
	}
	hidden_G.multiply(neural.learning_Rate)

	// Calculate input->hidden deltas
	input_T := inputs
	input_T.transpose()
	weight_HI_Delta := hidden_G
	_, err = weight_HI_Delta.dotProduct(input_T)
	if err != nil {
		log.Fatal(err)
	}

	// Adjust the weight by delta
	neural.weightIH.addFromMatrix(weight_HI_Delta)
	// Adjust the bias by grediant
	neural.biasH.addFromMatrix(hidden_G)

}

//helpfull functions
func sigmoid(n float64) float64 {
	return 1 / (1 + math.Exp(-n))
}

func dsigmoid(n float64) float64 {
	return n * (1 - n)
}
