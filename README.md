# NeuralN: A Simple Artificial Neural Network in Go

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Why NeuralN?](#why-neuraln)
4. [Prerequisites](#prerequisites)
   - [General Requirements](#general-requirements)
   - [Environment Variables (for CUDA)](#environment-variables-for-cuda)
5. [Installation](#installation)
6. [Usage](#usage)
   - [Example](#example)
7. [Performance Considerations](#performance-considerations)
   - [Test Results](#test-results)
   - [Recommendations](#recommendations)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## Introduction

**NeuralN** is a lightweight and educational implementation of an artificial neural network (ANN) in Go. Designed with simplicity and clarity in mind, this project demonstrates the core concepts of neural networks, including forward propagation, backpropagation, and gradient descent. Whether you're new to machine learning or looking to explore how neural networks work under the hood, NeuralN provides a hands-on way to understand these concepts in a clean and efficient Go environment.

### Key Features:
- **Educational Focus**: A clear and concise implementation of neural networks, perfect for learning and experimentation.
- **Gradient Descent Optimization**: Demonstrates how gradient descent is used to train the network and minimize error.
- **GPU Acceleration**: Includes CUDA support to explore how GPUs can accelerate large-scale computations, such as matrix operations and complex neural network tasks. Learn how GPUs work and how they can be leveraged to improve performance in machine learning.
- **Future Support for Image Processing**: While image processing is not yet implemented, it is planned for future updates.

### Why NeuralN?
- **Learn by Doing**: NeuralN is designed to help you understand the inner workings of neural networks by building and training them from scratch.
- **Go-Powered**: Leverages Go's simplicity and performance to create a clean and efficient implementation.
- **GPU Exploration**: Dive into the world of GPU computing and see how it can dramatically improve performance for large-scale tasks. NeuralN includes CUDA integration, allowing you to compare CPU and GPU performance and understand when to use each for optimal results.

Whether you're a student, a hobbyist, or a developer curious about machine learning, NeuralN is your gateway to understanding artificial neural networks and their real-world applications. Explore the power of GPU acceleration and learn how GPUs work to transform computational performance in machine learning tasks.

## Prerequisites

### General Requirements
- Go (version 1.20 or later)
- CUDA Toolkit (for GPU support)
- GCC or Clang (for building CUDA and C components)
- Linux-based system (recommended for CUDA integration)

### Environment Variables (for CUDA)
The following environment variables can be optionally configured:
- `LD_LIBRARY_PATH`: Include the directory for the shared CUDA library, e.g., `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/9init/neuraln.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd neuraln
   ```

3. **Build the project**:
   - For a standard build (CPU-only):
     ```bash
     make build
     ```
   - For a CUDA-enabled build (GPU acceleration):
     ```bash
     make build-cuda
     ```
     *Note: Ensure that the CUDA Toolkit and GCC/Clang are installed on your system before running `make build-cuda`.*

---

### Example of Using CUDA

If you want to leverage CUDA for faster matrix operations, ensure that your system meets the prerequisites (CUDA Toolkit, GCC/Clang, and a compatible GPU). Then, build the project using the `make build-cuda` command. This will compile the CUDA-enabled components and link them with the Go code.

After building with CUDA, you can run the neural network as usual, and it will automatically use the GPU for matrix operations where applicable.

---

### Additional Notes

- **CUDA Environment Setup**: Ensure that your environment variables (e.g., `LD_LIBRARY_PATH`) are correctly configured to include the CUDA library paths. For example:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
  ```

- **Testing CUDA Integration**: After building with CUDA, you can verify that the CUDA-enabled components are working by running the example code provided in the `main.go` file. The performance improvement should be noticeable for larger matrices and more complex networks.

---

This addition ensures that users are aware of the CUDA build option and how to use it effectively.

## Usage

The main components of the neural network are implemented in the `neural` package. You can create and train a neural network by interacting with this package.

### Example

Here’s an example of how to create and train a neural network to solve the XOR problem, a classic problem in neural network training. This example demonstrates the usage of the `neuraln` package with clear output formatting and feedback.

The example is located in `cmd/main.go`. You can build and run it using the following commands:

```bash
make build  # Build the project, use `make build-cuda` for enable gpu computitional power 
make run    # Run the example
```

#### Code Example (`cmd/main.go`):
```go
package main

import (
	"fmt"
	"math"
	"neuraln"
)

func main() {
	// Create a new neural network with 2 input neurons, 500 hidden neurons, and 1 output neuron
	nn := neuraln.New(2, 500, 1)

	// Define the XOR problem training data
	inputs := [][]float64{
		{1, 0}, {0, 1}, {1, 1}, {0, 0},
	}

	targets := [][]float64{
		{1}, {1}, {0}, {0},
	}

	// Train the neural network
	fmt.Println("Training the neural network...")
	err := nn.Train(inputs, targets, 50) // 50 epochs
	if err != nil {
		fmt.Printf("Error during training: %v\n", err)
		return
	}

	fmt.Print("Training complete! Testing the network...\n\n")

	// Test the neural network with the same inputs
	testingData := [][][]float64{
		{{1, 0}, {1}}, // Input: [1, 0], Expected Output: 1
		{{0, 1}, {1}}, // Input: [0, 1], Expected Output: 1
		{{1, 1}, {0}}, // Input: [1, 1], Expected Output: 0
		{{0, 0}, {0}}, // Input: [0, 0], Expected Output: 0
	}

	for _, data := range testingData {
		input := data[0]
		expected := data[1][0]

		// Get the neural network's prediction
		predictions, err := nn.Predict(input)
		if err != nil {
			fmt.Printf("Error during prediction: %v\n", err)
			return
		}

		// Round the prediction to the nearest integer (0 or 1)
		roundedPrediction := math.Round(predictions[0])

		// Print the results
		fmt.Printf("Input: %v\n", input)
		fmt.Printf("  - Expected Output: %v\n", expected)
		fmt.Printf("  - Raw Prediction:  %.4f\n", predictions[0])
		fmt.Printf("  - Rounded Prediction: %v\n", roundedPrediction)

		// Check if the prediction matches the expected output
		if roundedPrediction != expected {
			fmt.Printf("  ❌ Mismatch! Expected %v, but got %v\n", expected, roundedPrediction)
		} else {
			fmt.Printf("  ✅ Correct! Prediction matches expected output.\n")
		}
		fmt.Println()
	}

	fmt.Println("Testing complete!")
}
```

#### Output Example:
```
Training the neural network...
Training complete! Testing the network...

Input: [1 0]
  - Expected Output: 1
  - Raw Prediction:  0.9939548
  - Rounded Prediction: 1
  ✅ Correct! Prediction matches expected output.

Input: [0 1]
  - Expected Output: 1
  - Raw Prediction:  0.9933063
  - Rounded Prediction: 1
  ✅ Correct! Prediction matches expected output.

Input: [1 1]
  - Expected Output: 0
  - Raw Prediction:  0.0083347
  - Rounded Prediction: 0
  ✅ Correct! Prediction matches expected output.

Input: [0 0]
  - Expected Output: 0
  - Raw Prediction:  0.0027525
  - Rounded Prediction: 0
  ✅ Correct! Prediction matches expected output.

Testing complete!
```

#### Performance Considerations

The performance of the neural network and matrix operations depends on the problem size and the hardware being used. Here are the key insights:

1. **Neural Network Performance**:
   - **Small-Scale Problems (e.g., XOR)**:
     - The CPU outperforms the GPU for small-scale tasks.
     - **CPU Time**: `0.246s`
     - **GPU Time**: `10.555s`
     - This is because the overhead of GPU context switching and data transfer outweighs the benefits of parallel computation for small datasets and simple architectures.

   - **Large-Scale Problems (e.g., Image Classification)**:
     - GPU acceleration is expected to provide significant performance improvements for larger datasets and more complex tasks.

2. **Matrix Operations Performance**:
   - **Large Matrix Multiplication (5000x5000)**:
     - The GPU significantly outperforms the CPU for large matrix operations.
     - **CPU Time**: `759.185s`
     - **GPU Time**: `5.597s`
     - This demonstrates the GPU's ability to handle massive parallel computations efficiently.

3. **Real-World Applications**:
   - **CPU**: Ideal for small-scale problems or when GPU resources are limited.
   - **GPU**: Recommended for large-scale problems, such as image classification or large matrix operations, where parallel computation can be fully utilized.

#### Test Results

- **Neural Network Tests**:
  - **Without CUDA (CPU)**: `$ make test-neural`
    ```plaintext
    Testing Neural Network package without CUDA support
    === RUN   TestCreate
        neural_test.go:14: TestCreate passed
    --- PASS: TestCreate (0.00s)
    === RUN   TestTrain
        neural_test.go:30: TestTrain passed
    --- PASS: TestTrain (0.00s)
    === RUN   TestFeedForword
        neural_test.go:71: TestFeedForword passed
    --- PASS: TestFeedForword (0.24s)
    PASS
    ok      neuraln/neural/tests    0.246s
    ```

  - **With CUDA (GPU)**: `$ make test-neural-cuda`
    ```plaintext
    Testing Neural Network package with CUDA support
    === RUN   TestCreate
        neural_test.go:14: TestCreate passed
    --- PASS: TestCreate (0.30s)
    === RUN   TestTrain
        neural_test.go:30: TestTrain passed
    --- PASS: TestTrain (6.08s)
    === RUN   TestFeedForword
        neural_test.go:71: TestFeedForword passed
    --- PASS: TestFeedForword (4.14s)
    PASS
    ok      neuraln/neural/tests    10.555s
    ```

- **Matrix Tests**:
  - **Without CUDA (CPU)**: `$ make test-matrix`
    ```plaintext
    Testing Matrix package without CUDA support
    === RUN   TestLargeMatrix
    --- PASS: TestLargeMatrix (759.17s)
    === RUN   TestMatrixOperations
    --- PASS: TestMatrixOperations (0.00s)
    === RUN   TestRandomize
    --- PASS: TestRandomize (0.01s)
    === RUN   TestScalerMul
    --- PASS: TestScalerMul (0.00s)
    === RUN   TestSigmoid
    --- PASS: TestSigmoid (0.00s)
    === RUN   TestDSigmoid
    --- PASS: TestDSigmoid (0.00s)
    === RUN   TestTranspose
    --- PASS: TestTranspose (0.00s)
    PASS
    ok      neuraln/matrix/tests    759.185s
    ```

  - **With CUDA (GPU)**: `$ make test-matrix-cuda`
    ```plaintext
    Testing Matrix package with CUDA support
    === RUN   TestLargeMatrix
    --- PASS: TestLargeMatrix (5.52s)
    === RUN   TestMatrixOperations
    --- PASS: TestMatrixOperations (0.00s)
    === RUN   TestRandomize
    --- PASS: TestRandomize (0.03s)
    === RUN   TestScalerMul
    --- PASS: TestScalerMul (0.00s)
    === RUN   TestSigmoid
    --- PASS: TestSigmoid (0.00s)
    === RUN   TestDSigmoid
    --- PASS: TestDSigmoid (0.00s)
    === RUN   TestTranspose
    --- PASS: TestTranspose (0.00s)
    PASS
    ok      neuraln/matrix/tests    5.597s
    ```

#### Recommendations
- Use **CPU** for small-scale problems or when GPU resources are limited.
- Use **GPU** for large-scale problems or when performance is critical (e.g., image classification, large matrix operations).

*Note: All tests were performed on the following hardware: NVIDIA RTX 3090 (12GB) and AMD Ryzen 5 5600.*

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality or fix bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by the desire to provide a simple, educational example of how neural networks can be implemented from scratch in Go. It also serves as an exploration of GPU acceleration and its impact on computational performance, particularly for large-scale problems.

### Key Inspirations:
1. **Neural Network Fundamentals**:
   - The project draws from foundational concepts in neural networks, such as forward propagation, backpropagation, and gradient descent, to create a clear and accessible implementation.

2. **GPU Acceleration**:
   - A significant part of this project involves exploring how GPUs can be leveraged to accelerate computations, especially for large matrix operations and complex neural network tasks.
   - By integrating CUDA support, the project demonstrates the power of parallel processing and how it can drastically improve performance for tasks like large matrix multiplications.

3. **Real-World Applications**:
   - The project highlights the importance of choosing the right hardware (CPU vs. GPU) based on the problem size and complexity. This exploration is crucial for understanding how to optimize performance in real-world applications, such as image classification or natural language processing.

### Special Thanks:
- To the developers of Go for creating a language that is both simple and powerful, making it an excellent choice for educational projects.
- To NVIDIA for providing CUDA, a robust platform for GPU computing, which enabled the exploration of GPU acceleration in this project.
- To the open-source community for their contributions to machine learning and GPU computing, which provided valuable insights and resources.

This project is a testament to the power of combining theoretical knowledge with practical implementation, and it aims to inspire others to explore the fascinating world of neural networks and GPU computing.
---

*Note: This project is intended for educational purposes and is not optimized for production use.* 