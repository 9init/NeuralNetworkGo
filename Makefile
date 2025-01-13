# Variables
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CUDA_LIB_PATH := /usr/local/cuda/lib64
CUDA_MATRIX_SRC := $(ROOT_DIR)/matrix/cuda/matrix_ops.cu
C_MATRIX_SRC := $(ROOT_DIR)/matrix/cuda/matrix_ops.c
BUILD_DIR := $(ROOT_DIR)/build
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
CUDA_LIB := $(LIB_DIR)/libmatrix_ops.so
GO_TAGS := cuda

# Debug flag (set to 1 to enable debug prints)
DEBUG ?= 0

# Compiler flags
CFLAGS := -I./cuda
NVCCFLAGS := -shared -Xcompiler -fPIC

# Add debug flag if enabled
ifeq ($(DEBUG), 1)
    CFLAGS += -DDEBUG
    NVCCFLAGS += -DDEBUG
endif

# Ensure build directories exist
$(BUILD_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

# Compile CUDA and C code into a shared library
$(CUDA_LIB): $(CUDA_MATRIX_SRC) $(C_MATRIX_SRC) | $(LIB_DIR)
	nvcc $(NVCCFLAGS) -o $(CUDA_LIB) $(CUDA_MATRIX_SRC) $(C_MATRIX_SRC)

# Default target
all: build

# Build CUDA shared library
cuda: $(CUDA_LIB)

# Build Go code with CUDA support (embed RPATH using -extldflags)
build-cuda: $(CUDA_LIB) | $(BIN_DIR)
	go build -tags $(GO_TAGS) -ldflags "-extldflags '-Wl,-rpath,$(LIB_DIR)'" -o $(BIN_DIR)/main cmd/main.go

# Run the Go program (no LD_LIBRARY_PATH needed)
run-cuda: build-cuda
	$(BIN_DIR)/main

# Run the Go program without CUDA support
run: build
	$(BIN_DIR)/main

# Build Go code without CUDA support
build: | $(BIN_DIR)
	go build -o $(BIN_DIR)/main cmd/main.go

# Run tests all without CUDA support
test:
	go test -timeout 0 -v -count=1 ./neural/tests ./matrix/tests

# Run tests all with CUDA support (embed RPATH using -extldflags)
test-cuda: $(CUDA_LIB)
	go test -v -count=1 -tags $(GO_TAGS) -ldflags "-extldflags '-Wl,-rpath,$(LIB_DIR)'" ./neural/tests ./matrix/tests

# Run tests without CUDA support
test-nerual:
	go test -timeout 0 -v -count=1 ./neural/tests

# Run tests with CUDA support (embed RPATH using -extldflags)
test-nerual-cuda: $(CUDA_LIB)
	go test -count=1 -tags $(GO_TAGS) -ldflags "-extldflags '-Wl,-rpath,$(LIB_DIR)'" ./neural/tests

# Run matrix tests without CUDA support
test-matrix:
	go test -timeout 0 -v -count=1 ./matrix/tests

# Run matrix tests with CUDA support (embed RPATH using -extldflags)
test-matrix-cuda: $(CUDA_LIB)
	go test -v -count=1 -tags $(GO_TAGS) -ldflags "-extldflags '-Wl,-rpath,$(LIB_DIR)'" ./matrix/tests

# Clean up build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all build build-cuda run run-cuda test test-cuda clean test-matrix test-matrix-cuda test-nerual test-nerual-cuda