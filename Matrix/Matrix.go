package Matrix

import (
	"errors"
	"math/rand"
	"time"
)

/*Matrix it works only with float64 type*/
type Matrix struct {
	Matrix [][]float64
	Col    int
	Row    int
}

// NewMatrix function for creating a Matrix
// after import that package you can use it by type........
// myMatrix := Matrix.NewMatrix(1, 3)
func NewMatrix(Row, Col int) Matrix {
	return new(Matrix).Create(Row, Col)
}

// FromArray it used to convert matrix completely to fit the array
func (m *Matrix) FromArray(array []float64) {
	m.Create(len(array), 1)
	for i, v := range array {
		m.Matrix[i][0] = v
	}
}

// NewFromArray hmm
func NewFromArray(array []float64) Matrix {
	nMatrix := NewMatrix(len(array), 1)
	for i, v := range array {
		nMatrix.Matrix[i][0] = v
	}
	return nMatrix
}

// Create is a function to create new matrix....if it already created the matrix gonna change completley
func (m *Matrix) Create(Row, Col int) Matrix {
	m.Col = Col
	m.Row = Row
	m.Matrix = make([][]float64, Row)
	for i := range m.Matrix {
		m.Matrix[i] = make([]float64, Col)
	}

	return *m
}

// Randomize is function to randomize the matrix
func (m *Matrix) Randomize() {
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			rand.Seed(time.Now().UnixNano())
			time.Sleep(1)
			n := rand.Float64()*(1-(-1)) - 1
			m.Matrix[i][j] = n
		}
	}
}

//StaticRandomize is a static version of Randomize()
func (m *Matrix) StaticRandomize() Matrix {
	nMatrix := *m
	nMatrix.Randomize()
	return nMatrix
}

// AddFromMatrix is a function to sum two matrices
func (m *Matrix) AddFromMatrix(sMatrix Matrix) (Matrix, error) {
	if m.Col != sMatrix.Col || m.Row != sMatrix.Row {
		err := errors.New("Matrices dimensions must match")
		return *m, err
	}

	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			m.Matrix[i][j] += sMatrix.Matrix[i][j]
		}
	}
	return *m, nil
}

// StaticAddFromMatrix is a static version of SuptractMatrix()
func (m *Matrix) StaticAddFromMatrix(sMatrix Matrix) (Matrix, error) {
	nMatrix := *m
	return nMatrix.AddFromMatrix(sMatrix)
}

// SuptractMatrix is a function that subtract two matrices from each other
func (m *Matrix) SuptractMatrix(sMatrix Matrix) (Matrix, error) {
	if m.Col != sMatrix.Col || m.Row != sMatrix.Row {
		err := errors.New("Matrices dimensions must match")
		return *m, err
	}

	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			m.Matrix[i][j] -= sMatrix.Matrix[i][j]
		}
	}
	return *m, nil
}

// StaticSuptractMatrix is a static version of SuptractMatrix()
func (m *Matrix) StaticSuptractMatrix(sMatrix Matrix) (Matrix, error) {
	nMatrix := *m
	return nMatrix.SuptractMatrix(sMatrix)
}

//Map takes a function and preform the function for every single value in the matrix
func (m *Matrix) Map(f func(float64) float64) Matrix {
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			m.Matrix[i][j] = f(m.Matrix[i][j])
		}
	}
	return *m
}

//StaticMap static version of Map(float64)
func (m *Matrix) StaticMap(f func(float64) float64) Matrix {
	nMatrix := *m
	return nMatrix.Map(f)
}

// DotProduct is a dot product function =
func (m *Matrix) DotProduct(sMatrix Matrix) (Matrix, error) {
	if m.Col != sMatrix.Row {
		err := errors.New("rows must equal columns")
		return *m, err
	}

	nMatrix := NewMatrix(m.Row, sMatrix.Col)
	for i := 0; i < m.Row; i++ {
		for j := 0; j < sMatrix.Col; j++ {
			for k := 0; k < sMatrix.Row; k++ {
				nMatrix.Matrix[i][j] += m.Matrix[i][k] * sMatrix.Matrix[k][j]
			}
		}
	}

	*m = nMatrix
	return *m, nil
}

//StaticDotProduct it's static version of DotProduct
func (m *Matrix) StaticDotProduct(sMatrix Matrix) (Matrix, error) {
	nMatrix := *m
	return nMatrix.DotProduct(sMatrix)
}

// HadProduct is hadamard product
func (m *Matrix) HadProduct(sMatrix Matrix) (Matrix, error) {
	if m.Row != sMatrix.Row || m.Col != sMatrix.Col {
		err := errors.New("rows&cols must equal")
		return *m, err
	}

	for i := 0; i < m.Row; i++ {
		for j := 0; j < sMatrix.Col; j++ {
			m.Matrix[i][j] = m.Matrix[i][j] * sMatrix.Matrix[i][j]
		}
	}

	return *m, nil
}

// StaticHadProduct is a static version of HadProduct()
func (m *Matrix) StaticHadProduct(sMatrix Matrix) (Matrix, error) {
	nMatrix := *m
	return nMatrix.HadProduct(sMatrix)
}

//Multiply is a function to multiply a value by every single value in the matrix
func (m *Matrix) Multiply(n float64) Matrix {
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			m.Matrix[i][j] = m.Matrix[i][j] * n
		}
	}
	return *m
}

// Transpose is a function that transpose the matrix
func (m *Matrix) Transpose() (Matrix, error) {
	result := new(Matrix).Create(m.Col, m.Row)
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[j][i] = m.Matrix[i][j]
		}
	}
	*m = result
	return result, nil
}

// StaticTranspose is a static version of Transpose()
func (m *Matrix) StaticTranspose() (Matrix, error) {
	nMatrix := *m
	return nMatrix.Transpose()
}
