#include "GradientDescent.hpp"
#include "CostFunctions.hpp"
#include <iostream>
#include <iomanip>
#include <random>

void gradientDescent(vector<float>& theta, const int& visibleSize, const int& hiddenSize, const float& lambda, const float& sparsity, const float& beta, FloatMatrix& data, int maxIter)
{
	// Hyper Parameters
	float eta = 2.5e-5;	// Learning rate
	float rho = .99;  	// Decay for gradient

	// ADAGRAD variables
	vector<float> grad(theta.size());
	vector<float> E(theta.size());
	vector<float> s(theta.size());
	float delta;

	// SGD variables
	int batchSize = 256;
	FloatMatrix miniBatch(data.size1(), batchSize);
	default_random_engine generator;
	uniform_int_distribution<int> distribution(0, data.size2() - 1);

	vector<float> bestTheta(theta.size());
	float bestCost = 1e20;
	float cost;
	float rand;

	for(int i = 0; i < maxIter; ++i)
	{
		// Create random mini batch of the data
		for(int j = 0; j < batchSize; ++j)
		{
			rand = distribution(generator);	
			for(int k = 0; k < data.size1(); ++k)
				miniBatch(k, j) = data(k, rand);
		}

		// Get the gradient
		sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsity, beta, miniBatch, cost, grad);

		if(cost < bestCost)
		{
			bestCost = cost;
			for(int j = 0; j < theta.size(); ++j)
				bestTheta[j] = theta[j];
		}

		cout << "Interation: " << i << "   \tCost: " << cost << endl;
		
		// ADADELTA calculations
		for(int j = 0; j < theta.size(); ++j)
		{
			E[j] = rho * E[j] + (1 - rho) * grad[j] * grad[j];
			delta = sqrt(s[j] + eta) / sqrt(E[j] + eta) * -grad[j];
			s[j] = (1 - rho) * delta * delta + rho * s[j];
			theta[j] += delta;
		}
	}

	for(int i = 0; i < theta.size(); ++i)
		theta[i] = bestTheta[i];
}


void gradientDescentSoftmax(vector<float>& theta, const int& numClasses, const int& inputSize, const float& lambda, FloatMatrix& data, vector<float> labels, int maxIter)
{
	// Hyper Parameters
	float eta = 10;	// Learning rate
	vector<float> grad(theta.size());
	vector<float> bestTheta(theta.size());
	float bestCost = 1e20;
	float cost;

	for(int i = 0; i < maxIter; ++i)
	{
		// Get the gradient
		softmaxCost(theta, numClasses, inputSize, lambda, data, labels, cost, grad);

		if(cost < bestCost)
		{
			bestCost = cost;
			for(int j = 0; j < theta.size(); ++j)
				bestTheta[j] = theta[j];
		}

		cout << "Interation: " << i << "   \tCost: " << cost << endl;
		
		// ADAGRAD calculations
		for(int j = 0; j < theta.size(); ++j)
		{
			theta[j] -= eta * grad[j];
		}
	}

	for(int i = 0; i < theta.size(); ++i)
		theta[i] = bestTheta[i];
}

