# Variables
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CUDA_LIB_PATH := /usr/local/cuda/lib64
CUDA_SRC := $(ROOT_DIR)/matrix/cuda/matrix_ops.cu
CUDA_LIB := $(ROOT_DIR)/matrix/cuda/libmatrix_ops.so
GO_SRC := $(wildcard matrix/*.go)
GO_TAGS := cuda

# Default target
all: build

# Compile CUDA code into a shared library
$(CUDA_LIB): $(CUDA_SRC)
	nvcc -o $(CUDA_LIB) -shared -Xcompiler -fPIC $(CUDA_SRC)

# # Build Go code with CUDA support
# build: $(CUDA_LIB)
# 	export LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$$LD_LIBRARY_PATH; \
# 	go build -tags $(GO_TAGS) -o my_program 

# Run tests with CUDA support
test-cuda: $(CUDA_LIB)
	export LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$$LD_LIBRARY_PATH; \
	go test -tags $(GO_TAGS) -v -count=1 ./tests

# Run tests without CUDA support
test:
	go test -v -count=1 ./tests

# Clean up build artifacts
clean:
	rm -f $(CUDA_LIB) cmd/main

# Build Go code with CUDA support
build-cuda: $(CUDA_LIB)
	export LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$$LD_LIBRARY_PATH; \
	go build -tags $(GO_TAGS) -o cmd/bin/main cmd/main.go

# Build Go code without CUDA support
build:
	go build -o cmd/bin/main cmd/main.go

cuda: $(CUDA_LIB)

# Phony targets
.PHONY: all build test test-no-cuda clean build build-cuda cuda