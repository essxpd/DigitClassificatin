#include <iostream>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "MatrixMultiply.hpp"


using namespace std;
using namespace chrono;

FloatMatrix matrixMultiply(FloatMatrix& a, FloatMatrix& b)
{
	if(a.size2() != b.size1())
	{
		cerr << "Invalid Matrix dimensions" << endl;
		return FloatMatrix();
	}

	FloatMatrix result(a.size1(), b.size2());
	
	//////////////////////////////////// Use CUBLAS to perform matrix multiplication
	
	// Block size
	int block_size = 32;
	
	// Allocate Device Memory
	float *d_A, *d_B, *d_C;
	unsigned int mem_size_A = sizeof(float) * a.size1() * a.size2();
	unsigned int mem_size_B = sizeof(float) * b.size1() * b.size2();
	unsigned int mem_size_C = sizeof(float) * result.size1() * result.size2();
	
	float *h_A = &(a.data()[0]);
	float *h_B = &(b.data()[0]);
	float *h_C = &(result.data()[0]);

	cudaMalloc((void **) &d_A, mem_size_A);
	cudaMalloc((void **) &d_B, mem_size_B);
	cudaMalloc((void **) &d_C, mem_size_C);
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(result.size1() / threads.x, result.size2() / threads.y);

	//cout << "Computing the result using CUBLAS..." << endl;

	// CUBLAS version 2.0
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		cublasHandle_t handle;

		cublasCreate(&handle);

		// Warmup operation with cublas
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
			b.size2(), a.size1(), a.size2(), 
			&alpha, 
			d_B, b.size2(), 
			d_A, a.size2(), 
			&beta, 
			d_C, b.size2());

		// Copy the result from device to host
		cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
		
		// Destroy handle
		cublasDestroy(handle);
	}

	// clean up memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaDeviceReset();

	return result;
}

void matrixMultiply(FloatMatrix& a, FloatMatrix& b, FloatMatrix& result)
{
	if(a.size2() != b.size1())
	{
		cerr << "Invalid Matrix dimensions" << endl;
		return;
	}

	//////////////////////////////////// Use CUBLAS to perform matrix multiplication

	// Block size
	int block_size = 32;
	
	// Allocate Device Memory
	float *d_A, *d_B, *d_C;
	unsigned int mem_size_A = sizeof(float) * a.size1() * a.size2();
	unsigned int mem_size_B = sizeof(float) * b.size1() * b.size2();
	unsigned int mem_size_C = sizeof(float) * result.size1() * result.size2();
	
	float *h_A = &(a.data()[0]);
	float *h_B = &(b.data()[0]);
	float *h_C = &(result.data()[0]);

	cudaMalloc((void **) &d_A, mem_size_A);
	cudaMalloc((void **) &d_B, mem_size_B);
	cudaMalloc((void **) &d_C, mem_size_C);
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(result.size1() / threads.x, result.size2() / threads.y);

	// cout << "Computing the result using CUBLAS..." << endl;

	// CUBLAS version 2.0
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		cublasHandle_t handle;

		auto q = cublasCreate(&handle);
		if(q == CUBLAS_STATUS_NOT_INITIALIZED)
			cout << "Not initalized" << endl;

		// Warmup operation with cublas
		auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
			b.size2(), a.size1(), a.size2(), 
			&alpha, 
			d_B, b.size2(), 
			d_A, a.size2(), 
			&beta, 
			d_C, b.size2());

		
		// Copy the result from device to host
		cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	
		// Destroy handle
		cublasDestroy(handle);
	}

	// clean up memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaDeviceReset();
}


void dotProduct(FloatMatrix& a, FloatMatrix& b, FloatMatrix& c)
{

}

