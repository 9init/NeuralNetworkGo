#include <cuda_runtime.h>
#include <curand_kernel.h> // Include cuRAND header
#include <stdio.h>

// Kernel to initialize cuRAND states
__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// CUDA kernel for Randomize matrix using cuRAND
__global__ void matrixRandomize(double* A, int rows, int cols, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        // Generate random number using cuRAND
        A[idx] = curand_uniform(&states[idx]);
    }
}

// CUDA kernel for matrix addition
__global__ void matrixAdd(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel for matrix subtraction
__global__ void matrixSub(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        C[idx] = A[idx] - B[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrixMul(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int rowsA, int colsA, int colsB) {
    // Tile size
    const int TILE_SIZE = 16;

    // Shared memory for tiles of A and B
    __shared__ double sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ double sharedB[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    // Loop over tiles
    for (int t = 0; t < (colsA + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < rowsA && t * TILE_SIZE + threadIdx.x < colsA) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * colsA + t * TILE_SIZE + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < colsB && t * TILE_SIZE + threadIdx.y < colsA) {
            sharedB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * colsB + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads(); // Synchronize to ensure all threads have loaded their tiles

        // Compute partial sum for the tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads(); // Synchronize before loading the next tile
    }

    // Write the result to global memory
    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
    }
}

// CUDA kernel for matrix Hadamard product
__global__ void matrixHadamard(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        C[idx] = A[idx] * B[idx];
    }
}

// Wrapper function for launching matrixRandomize kernel
extern "C" void launchMatrixRandomize(double* d_A, int rows, int cols) {
    int threadsPerBlock = 256; // Optimal for most GPUs
    int numElements = rows * cols;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for cuRAND states
    curandState* d_states;
    cudaMalloc((void**)&d_states, numElements * sizeof(curandState));

    // Initialize cuRAND states
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, time(0));

    #ifdef DEBUG
    printf("Launching matrixRandomize kernel: blocksPerGrid = %d, threadsPerBlock = %d, numElements = %d\n",
           blocksPerGrid, threadsPerBlock, numElements);
    #endif

    // Launch the kernel
    matrixRandomize<<<blocksPerGrid, threadsPerBlock>>>(d_A, rows, cols, d_states);

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronize to ensure the kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Free cuRAND states
    cudaFree(d_states);
}

// Wrapper function for launching matrixAdd kernel
extern "C" void launchMatrixAdd(double* d_A, double* d_B, double* d_C, int rows, int cols) {
    int threadsPerBlock = 256; // Optimal for most GPUs
    int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

    #ifdef DEBUG
    printf("Launching matrixAdd kernel: blocksPerGrid = %d, threadsPerBlock = %d\n", blocksPerGrid, threadsPerBlock);
    #endif

    // Launch the kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronize to ensure the kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Wrapper function for launching matrixSub kernel
extern "C" void launchMatrixSub(double* d_A, double* d_B, double* d_C, int rows, int cols) {
    int threadsPerBlock = 256; // Optimal for most GPUs
    int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

    #ifdef DEBUG
    printf("Launching matrixSub kernel: blocksPerGrid = %d, threadsPerBlock = %d\n", blocksPerGrid, threadsPerBlock);
    #endif

    // Launch the kernel
    matrixSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronize to ensure the kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Wrapper function for launching matrixMul kernel
extern "C" void launchMatrixMul(double* d_A, double* d_B, double* d_C, int rowsA, int colsA, int colsB) {
    dim3 threadsPerBlock(16, 16); // Optimal for shared memory tiles
    dim3 blocksPerGrid((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    #ifdef DEBUG
    printf("Launching matrixMul kernel: blocksPerGrid = (%d, %d), threadsPerBlock = (%d, %d)\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    #endif
    
    // Launch the kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronize to ensure the kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Wrapper function for launching matrixHadamard kernel
extern "C" void launchMatrixHadamard(double* d_A, double* d_B, double* d_C, int rows, int cols) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;

    #ifdef DEBUG
    printf("Launching matrixHadamard kernel: blocksPerGrid = %d, threadsPerBlock = %d\n", blocksPerGrid, threadsPerBlock);
    #endif

    // Launch the kernel
    matrixHadamard<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronize to ensure the kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
}