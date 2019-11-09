package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

//Matrix is a Matrix from math
type Matrix struct {
	matrix [][]float64
	col    int
	row    int
}

func (m *Matrix) fromArray(array []float64) Matrix {
	m.create(len(array), 1)
	for i, v := range array {
		m.matrix[i][0] = v
	}
	return *m
}

func (m *Matrix) create(col int, row int) Matrix {
	m.col = col
	m.row = row
	m.matrix = make([][]float64, col)
	for i := range m.matrix {
		m.matrix[i] = make([]float64, row)
	}

	return *m
}

func (m *Matrix) randomize() {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			rand.Seed(time.Now().UnixNano())
			time.Sleep(1)
			n := rand.Float64()*(1-(-1)) - 1
			m.matrix[i][j] = n
		}
	}
}

func (m *Matrix) addFromMatrix(sMatrix Matrix) Matrix {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			m.matrix[i][j] += sMatrix.matrix[i][j]
		}
	}
	return *m
}

//Map takes a function and preform the function for every single value in the matrix
func (m *Matrix) Map(f func(float64) float64) Matrix {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			m.matrix[i][j] = f(m.matrix[i][j])
		}
	}
	return *m
}

func (m *Matrix) dotProduct(sMatrix Matrix) (Matrix, error) {
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
				nMatrix[i][j] += m.matrix[i][k] * sMatrix.matrix[k][j]
			}
		}
	}

	m.matrix = nMatrix
	m.row = sMatrix.row
	return *m, nil
}

func (m *Matrix) HarProduct(sMatrix Matrix) (Matrix, error) {
	if m.row != sMatrix.row || m.col != sMatrix.col {
		err := errors.New("rows&cols must equal")
		return *m, err
	}

	nMatrix := new(Matrix).create(m.col, m.row)

	for i := 0; i < m.col; i++ {
		for j := 0; j < sMatrix.row; j++ {
			nMatrix.matrix[i][j] = m.matrix[i][j] * sMatrix.matrix[i][j]
		}
	}

	*m=nMatrix
	return *m, nil
}


func (m *Matrix) multiply(n float64) Matrix {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
				m.matrix[i][j] = m.matrix[i][j] * n
			}
	}
	return *m
}

func (m *Matrix) suptractMatrix(sMatrix Matrix) Matrix {
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			m.matrix[i][j] -= sMatrix.matrix[i][j]
		}
	}
	return *m
}

func (m *Matrix) transpose() (Matrix, error) {
	result := new(Matrix).create(m.row, m.col)
	for i := 0; i < m.col; i++ {
		for j := 0; j < m.row; j++ {
			result.matrix[j][i] = m.matrix[i][j]
		}
	}
	*m = result
	return result, nil
}

//NeuralN is a Neural Network
type NeuralN struct {
	inputNodes    int
	hiddenNodes   int
	outputNodes   int
	weightIH      Matrix
	weightHO      Matrix
	biasH         Matrix
	biasO         Matrix
	learning_Rate float64
}

func (neural *NeuralN) create(inputNodes, hiddenNodes, outputNodes int) NeuralN {
	neural.weightIH.create(hiddenNodes, inputNodes)
	neural.weightIH.randomize()
	neural.weightHO.create(outputNodes, hiddenNodes)
	neural.weightHO.randomize()
	neural.biasH.create(hiddenNodes, 1)
	neural.biasH.randomize()
	neural.biasO.create(outputNodes, 1)
	neural.biasO.randomize()
	neural.learning_Rate = 0.4
	return *neural
}

func (neural *NeuralN) feedForword(inputArray []float64) Matrix {
	inputs := new(Matrix).fromArray(inputArray)
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

func (neural *NeuralN) train(inputArray, targetArray []float64) {


	targets := new(Matrix).fromArray(targetArray)
	inputs := new(Matrix).fromArray(inputArray)

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
		fmt.Println(1)
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
		fmt.Println(2)
		log.Fatal(err)
	}

	// Calculate hidden grediant
	hidden_G := hidden
	hidden_G.Map(dsigmoid)
	//fmt.Println(hidden_G, "\n", hidden_errors)
	_, err = hidden_G.HarProduct(hidden_errors)
	if err != nil {
		fmt.Println(3)
		log.Fatal(err)
	}
	hidden_G.multiply(neural.learning_Rate)

	// Calculate input->hidden deltas
	input_T := inputs
	input_T.transpose()
	weight_HI_Delta := hidden_G
	_, err = weight_HI_Delta.dotProduct(input_T)
	if err != nil {
		fmt.Println(4)
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

type objects struct {
	inputs  []float64
	outputs []float64
}

func shuffle(l *[]objects) {
	rand.Seed(time.Now().UnixNano())
	time.Sleep(1)
	rand.Shuffle(len(*l), func(i, j int) { (*l)[i], (*l)[j] = (*l)[j], (*l)[i] })
}

func main() {
	nn := new(NeuralN)
	nn.create(2, 50, 1)
	list := []objects{
		objects{[]float64{1,0}, []float64{1}},
		objects{[]float64{0,1}, []float64{1}},
		objects{[]float64{1,1}, []float64{0}},
		objects{[]float64{0,0}, []float64{0}},
	}
	

	for i := 0; i < 3000; i++ {
		for _, v := range list {
			shuffle(&list)
			nn.train(v.inputs, v.outputs)
		}
	}

	test:=[][]float64{{0,1},{0,0},{1,0},{1,1}}
	for _, f := range test{
		m := nn.feedForword(f)
		for _, v := range m.matrix {
			for _,v2:=range v{
				fmt.Println(v2)
			}
		}
		fmt.Print("\n")
	}

}
