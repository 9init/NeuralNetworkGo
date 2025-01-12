# Variables
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CUDA_LIB_PATH := /usr/local/cuda/lib64
CUDA_SRC := $(ROOT_DIR)/matrix/cuda/matrix_ops.cu
C_SRC := $(ROOT_DIR)/matrix/cuda/matrix_ops.c
CUDA_LIB := $(ROOT_DIR)/matrix/cuda/libmatrix_ops.so
GO_SRC := $(wildcard matrix/*.go)
GO_TAGS := cuda

# Default target
all: build

# Compile CUDA and C code into a shared library
$(CUDA_LIB): $(CUDA_SRC) $(C_SRC)
	nvcc -o $(CUDA_LIB) -shared -Xcompiler -fPIC $(CUDA_SRC) $(C_SRC)

# Build Go code with CUDA support
build-cuda: $(CUDA_LIB)
	export LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$(ROOT_DIR)/matrix/cuda:$$LD_LIBRARY_PATH; \
	go build -tags $(GO_TAGS) -o cmd/bin/main cmd/main.go

# Run the Go program
run-cuda: build-cuda
	export LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$(ROOT_DIR)/matrix/cuda:$$LD_LIBRARY_PATH; \
	./cmd/bin/main

# Run the Go program without CUDA support
run: build
	./cmd/bin/main

# Build Go code without CUDA support
build:
	go build -o cmd/bin/main cmd/main.go

# Run tests without CUDA support
test:
	go test -v -count=1 ./tests/nerual

# Run tests with CUDA support
test-cuda: $(CUDA_LIB)
	export LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$(ROOT_DIR)/matrix/cuda:$$LD_LIBRARY_PATH; \
	go test -count=1 -tags $(GO_TAGS) ./tests/nerual

# Run matrix tests without CUDA support
test-matrix:
	go test -v -count=1 ./tests/matrix

# Run matrix tests with CUDA support
test-matrix-cuda: $(CUDA_LIB)
	export LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$(ROOT_DIR)/matrix/cuda:$$LD_LIBRARY_PATH; \
	go test -v -count=1 -tags $(GO_TAGS) ./tests/matrix

# Clean up build artifacts
clean:
	rm -f $(CUDA_LIB) cmd/bin/main

# Phony targets
.PHONY: all build build-cuda run run-cuda test test-cuda clean test-matrix test-matrix-cuda