package neural

import (
	"neuraln/errors"
	"neuraln/matrix"
)

func (neural *Neural) FeedForword(inputArray []float64) (*matrix.Matrix, error) {
	if len(inputArray) != neural.InputNodes {
		return nil, errors.ErrInputNodesMismatch
	}

	inputs := matrix.NewFromArray(inputArray)
	hidden, _ := neural.WeightIH.StaticDotProduct(inputs)
	hidden.AddFromMatrix(neural.BiasH)
	hidden.Map(sigmoid)
	outputs := neural.WeightHO
	outputs.DotProduct(hidden)
	outputs.AddFromMatrix(neural.BiasO)
	outputs.Map(sigmoid)

	return &outputs, nil
}
