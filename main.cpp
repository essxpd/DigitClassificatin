#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <limits>
#include <boost/lexical_cast.hpp>
#include "helper/helper_image.h"
#include "MatrixMultiply.hpp"
#include "FloatMatrix.hpp"
#include "CostFunctions.hpp"
#include "ImageLoader.hpp"
#include "GradientDescent.hpp"

using namespace std;
using namespace chrono;

void initializeParameters(vector<float>&, const int&, const int&);
void initializeParametersSoftmax(vector<float>&);
void normalize(FloatMatrix&, int);
void saveParams(vector<float>&);
vector<float> computeNumericalGradient(vector<float>&, const int&, const int&, const float&, const float&, const float&, FloatMatrix&);
void separateData(FloatMatrix&, FloatMatrix&, FloatMatrix&, FloatMatrix&, vector<float>&, vector<float>&, vector<float>&, int);
void printNetwork(FloatMatrix&);

int main()
{
	const int imageSize = 28;
	const int inputSize = imageSize * imageSize;
	const int hiddenSize = 196;
	const int numClasses = 10;
	const float sparsityParam = 0.1f;
	float lambda = 3E-3f;
	const float beta = 3;
	vector<float> optTheta(inputSize * hiddenSize * 2 + hiddenSize + inputSize);
	vector<float> softmaxTheta(hiddenSize * 5);
	initializeParameters(optTheta, inputSize, hiddenSize);
	initializeParametersSoftmax(softmaxTheta);

	// Load the images and labels
	ImageLoader MNISTLoader;

	FloatMatrix images;
	vector<float> labels;
	
	MNISTLoader.loadImages(images, "mnist/train-images.idx3-ubyte");
	MNISTLoader.loadLabels(labels, "mnist/train-labels.idx1-ubyte");
	normalize(images, 255);
	
	// Simulate a Labeled and unlabeled set (0 - 4 will be labeled but not 5 - 9)
	FloatMatrix unlabeledData;
	FloatMatrix trainData;
	FloatMatrix testData;
	vector<float> trainLabels;
	vector<float> testLabels;
	separateData(images, trainData, testData, unlabeledData, labels, trainLabels, testLabels, 5);
	
	// Train autoencoder
	gradientDescent(optTheta, inputSize, hiddenSize, lambda, sparsityParam, beta, unlabeledData, 3000);
	/*ifstream savedWeights("correctlyTrained.csv");
	string value;
	for(int i = 0; i < optTheta.size(); ++i)
	{
		getline(savedWeights, value, ',');
		optTheta[i] = atof(value.c_str());
	}
	savedWeights.close();
	*/
	// Get features
	FloatMatrix trainFeatures = feedForwardAutoencoder(optTheta, hiddenSize, inputSize, trainData);
	FloatMatrix testFeatures = feedForwardAutoencoder(optTheta, hiddenSize, inputSize, testData);

	// Train softmax
	lambda = 1e-4;
	gradientDescentSoftmax(softmaxTheta, 5, hiddenSize, lambda, trainFeatures, trainLabels, 1000);

	// Uncomment and comment out gradientDescentSoftmax to skip training if it has already been done
/*
	ifstream saved("trainedSoftmax.csv");
	string v;
	for(int i = 0; i < softmaxTheta.size(); ++i)
	{
		getline(saved, v);
		softmaxTheta[i] = boost::lexical_cast<float>(v.c_str());
	}
	saved.close();
*/

	// Reshape softmaxTheta
	FloatMatrix theta(5, hiddenSize); 
	for(int i = 0; i < theta.size1(); ++i)
		for(int j = 0; j < theta.size2(); ++j)
			theta(i, j) = softmaxTheta[j*theta.size1() + i];

	vector<float> prediction;
	softmaxPredict(theta, testFeatures, prediction);
	
	
	// Calculate accuracy
	int correct = 0;
	for(int i = 0; i < prediction.size(); ++i)
	{
		if(prediction[i] == testLabels[i])
			++correct;
	}	

	cout << "Accuracy: " << 100 * (float)correct / prediction.size() << endl;

	auto it = optTheta.begin();
	FloatMatrix W1(inputSize, hiddenSize); 
	copy(it, it + hiddenSize * inputSize, &(W1.data()[0]));
	printNetwork(W1);

	return 0;
}

// Initializes weights in the network
void initializeParameters(vector<float>& theta, const int& visibleSize, const int& hiddenSize)
{
	srand(1000);

	int sizeW1 = hiddenSize * visibleSize;
	int sizeW2 = sizeW1;
	float r = sqrt(6) / sqrt(hiddenSize + visibleSize + 1);

	for(int i = 0; i < sizeW1 + sizeW2; ++i)
		theta[i] = ((float) rand() / (RAND_MAX)) * 2 * r - r;

	// Optional load from file
/*
	// Save to a file 
	ofstream oFile("parameters.csv");
	for(int i = 0; i < theta.size(); ++i)
	{
		oFile << theta[i];
		if( i != theta.size() - 1 )
			oFile << ',';
	}
	oFile.close();
*/
}

// Initializes weights for softmax
void initializeParametersSoftmax(vector<float>& theta)
{
	default_random_engine generator;
	uniform_real_distribution<float> distribution(0.0f, 1.0f);

	for(int i = 0; i < theta.size(); ++i)
	{
		theta[i] = 0.005f * distribution(generator);
	}
}

// Normalizes a matrix
void normalize(FloatMatrix& m, int max)
{
	float* m_ptr = &(m.data()[0]);
	const int l = m.size1() * m.size2();
	const float fraction = 1/(float)max;

	#pragma omp parallel for
	for(int i = 0; i < l; ++i)
		m_ptr[i] *= fraction;
}

// Checks that calculations are correct by manually computing the gradient
vector<float> computeNumericalGradient(vector<float>& theta, const int& visibleSize, const int& hiddenSize, const float& lambda, const float& sparsityParam, const float& beta, FloatMatrix& data)
{
	vector<float> numGrad(theta.size());
	const float EPSILON = .1;

	//#pragma omp parallel for
	for(int i = 0; i < 10; ++i) // can be while i < theta.size() but that takes a very long time
	{
		float costMinus;
		float costPlus;
		vector<float> gradMinus;
		vector<float> gradPlus;

		theta[i] -= EPSILON;
		sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data, costMinus, gradMinus);
		theta[i] += 2*EPSILON;
		sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data, costPlus, gradPlus);
		theta[i] -= EPSILON;

		cout << "+: " << costPlus << endl;
		cout << "-: " << costMinus << endl;

		numGrad[i] = (costPlus - costMinus) / (2 * EPSILON);

		cout << "Finished: " << i << ' ' << setprecision(15) << numGrad[i] << endl;
	}
	
	return numGrad;
}

// Separates data into labeled/unlabeled training and 
// 		Param data: 			the data to be separate
// 		Param trainLabelSet: 	a set of data that will have labels associated with it and used to train with 
//  	Param testLabelSet: 	a set of data that will have labels associated with it and used to test with
//		Param unlabeledSet:		a set of data with no labels
//		Param labels:			labels for the param data
//		Param trainLabels:		a set of labels for param trainLabelSet
// 		Param testLabels:		a set of labels for param testLabelSet
//		Param dividor:			a value used to decide where to separate labeled from unlabeled data
void separateData(FloatMatrix& data, FloatMatrix& trainLabelSet, FloatMatrix& testLabelSet, FloatMatrix& unlabeledSet, vector<float>& labels, vector<float>& trainLabels, vector<float>& testLabels, int dividor
{
	vector<int> labeledSetIdx;
	vector<int> unlabeledSetIdx;

	// Separate the indexes for each set
	for(int i = 0; i < labels.size(); ++i)
	{
		if(labels[i] < dividor)
			labeledSetIdx.push_back(i);
		else
			unlabeledSetIdx.push_back(i);
	}

	// Create the unlabeled set
	unlabeledSet = FloatMatrix(data.size1(), unlabeledSetIdx.size());
	for(int i = 0; i < unlabeledSetIdx.size(); ++i)
	{
		for(int j = 0; j < data.size1(); ++j)
		{
			unlabeledSet(j, i) = data(j, unlabeledSetIdx[i]);
		}
	}

	// Create the labeled test/training sets
	const int numTrain = labeledSetIdx.size() / 2;
	trainLabelSet = FloatMatrix(data.size1(), numTrain);
	testLabelSet = FloatMatrix(data.size1(), labeledSetIdx.size() - numTrain);
	trainLabels = vector<float>(numTrain);
	testLabels = vector<float>(labeledSetIdx.size() - numTrain);
	for(int i = 0; i < labeledSetIdx.size(); ++i)
	{
		if(i < numTrain)
		{
			trainLabels[i] = labels[labeledSetIdx[i]];
			for(int j = 0; j < data.size1(); ++j)
			{
				trainLabelSet(j, i) = data(j, labeledSetIdx[i]);
			}
		}
		else
		{
			testLabels[i - numTrain] = labels[labeledSetIdx[i]];
			for(int j = 0; j < data.size1(); ++j)
			{
				testLabelSet(j, i - numTrain) = data(j, labeledSetIdx[i]);
			}
		}
	}

	cout << "# Examples in ulabeled set: " << unlabeledSet.size2() << endl;
	cout << "# Examples in supervised training set: " << trainLabelSet.size2() << endl;
	cout << "# Examples in supervised testing set: " << testLabelSet.size2() << endl << endl;
}

// Uses the weights of the first layer to produce an image of the weights in the network
void printNetwork(FloatMatrix& A)
{
	// Zero mean
	float mean = 0;
	for(int i = 0; i < A.size1(); ++i)
		for(int j = 0; j < A.size2(); ++j)
			mean += A(i,j);
	mean /= A.size1() * A.size2();

	#pragma omp parallel for
	for(int i = 0; i < A.size1(); ++i)
		for(int j = 0; j < A.size2(); ++j)
			A(i, j) -= mean;

	// Determine dimensions
	const int L = A.size1();
	const int M = A.size2();
	int sz = sqrt(L);
	int buf = 1;
	int n,m;

	if(pow(floor(sqrt(M)), 2) != M)
	{
		n = ceil(sqrt(M));
		for(; M % n != 0 && n < 1.2*sqrt(M); ++n);
		m = ceil(M/n);
	}
	else
	{
		n = sqrt(M);
		m = ceil(M/n);
	}

	// Initialize image array
	FloatMatrix array(buf+m*(sz+buf),buf+n*(sz+buf));
	for(int i = 0; i < array.size1(); ++i)
		for(int j = 0; j < array.size2(); ++j)
			array(i,j) = 1;

	int k = 0;
	float clim;
	vector<float> idx1(sz);
	vector<float> idx2(sz);
	FloatMatrix reshaped(sz, sz);
	for(int i = 0; i < m; ++i)
	{
		for(int j = 0; j < n; ++j)
		{
			if(k <= M)
			{
				clim = numeric_limits<float>::min();
				for(int q = 0; q < A.size1(); ++q)
					clim = max(clim, abs(A(q, k)));

				for(int q = 0; q < sz; ++q)
				{
					idx1[q] = buf+i*(sz+buf)+q;
					idx2[q] = buf+j*(sz+buf)+q;
				}

				for(int q = 0; q < A.size1(); ++q)
					reshaped(q % sz, q / sz) = A(q, k) / clim;

				for(int q = 0; q < idx1.size(); ++q)
				{
					for(int r = 0; r < idx2.size(); ++r)
						array(idx1[q], idx2[r]) = reshaped(q, r);
				}

				++k; 
			}
		}
	}

	// Recenter values between 0 and 1
	int newMin = 0;
	int newMax = 1;
	int min = -1;
	int max = 1;

	for(int i = 0; i < array.size1(); ++i)
	{
		for(int j = 0; j < array.size2(); ++j)
		{
			array(i, j) = (newMax - newMin) * (array(i, j) - min) / (max - min) + newMin;
		}
	}

	unsigned int s1 = array.size1(), s2 = array.size2();
	sdkSavePGM("weights.pgm", &(array.data()[0]), s1, s2); // Cuda utility function in 'helper/helper_image'
}

