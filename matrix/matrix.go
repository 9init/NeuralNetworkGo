package matrix

/*Matrix it works only with float64 type*/
type Matrix struct {
	Matrix [][]float64
	Col    int
	Row    int
}

// NewMatrix creates a new Matrix with the specified number of rows and columns.
func NewMatrix(Row, Col int) *Matrix {
	return New(Row, Col)
}

// NewFromArray creates a new Matrix from a given slice of float64 values.
func NewFromArray(array []float64) *Matrix {
	nMatrix := NewMatrix(len(array), 1)
	for i, v := range array {
		nMatrix.Matrix[i][0] = v
	}
	return nMatrix
}

// New creates a new Matrix with the specified number of rows and columns.
func New(Row, Col int) *Matrix {
	m := Matrix{
		Col:    Col,
		Row:    Row,
		Matrix: make([][]float64, Row),
	}
	for i := range m.Matrix {
		m.Matrix[i] = make([]float64, Col)
	}
	return &m
}

// Map applies a function to each element of the Matrix and returns a new Matrix.
func (m *Matrix) Map(f func(float64) float64) *Matrix {
	result := New(m.Row, m.Col)
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			result.Matrix[i][j] = f(m.Matrix[i][j])
		}
	}
	return result
}

// Flatten converts the Matrix into a 1D slice.
func (m *Matrix) Flatten() []float64 {
	flat := make([]float64, m.Row*m.Col)
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			flat[i*m.Col+j] = m.Matrix[i][j]
		}
	}
	return flat
}
