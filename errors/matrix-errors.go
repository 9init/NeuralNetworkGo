package errors

import "errors"

var (
	ErrMatricesDimensionsMustMatch = errors.New("matrices dimensions must match")
	ErrRowsMustEqualColumns        = errors.New("rows must equal columns")
	ErrRowsColsMustEqual           = errors.New("rows and columns must be equal")
)
