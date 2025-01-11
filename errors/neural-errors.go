package errors

import "errors"

var (
	ErrEmptyInputOutput    = errors.New("empty input/output array")
	ErrInputOutputMismatch = errors.New("input/output mismatch: number of input nodes must equal input array length")
	ErrInputNodesMismatch  = errors.New("input nodes must match input array length")
	ErrOutputNodesMismatch = errors.New("output nodes must match target array length")
	ErrInputOutputNodes    = errors.New("input/output nodes mismatch: number of input nodes must equal input array length, and output nodes must equal target array length")
)
