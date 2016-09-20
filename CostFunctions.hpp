#include <vector>
#include "FloatMatrix.hpp"

using namespace std;

// Gets cost and gradient of network given theta as paramters
//		Param theta:			network parameters
//		Param visibleSize:		the number of input values
//		Param hiddenSize:		the number of hidden neurons
//		Param lambda:			weight decay parameter
//		Param sparsityParam:	sparsity parameter
//		Param beta:				KL divergence param
//		Param data:				input
//		Param cost:				calculated error (Return value)
//		Param grad:				calculated gradient (Return value)
void sparseAutoencoderCost(vector<float> &theta, const int &visibleSize, const int &hiddenSize, const float &lambda, const float &sparsityParam, const float &beta, FloatMatrix &data, float &cost, vector<float> &grad);

// Gets cost and gradient of network given theta as paramters
//		Param theta:			network parameters
//		Param numClasses:		the number of ouput values
//		Param visibleSize:		the number of input values
//		Param lambda:			weight decay parameter
//		Param data:				input
//		Param cost:				calculated error (Return value)
//		Param grad:				calculated gradient (Return value)
void softmaxCost(vector<float>& theta, const int& numClasses, const int& inputSize, const float& lambda, FloatMatrix& data, vector<float>& labels, float& cost, vector<float>& grad);

// Calculates the activations for the network
//		Param theta:			network parameters
//		Param hiddneSize:		number of hidden neurons
//		Param visibleSize:		number of input values
//		Param data:				input
//		Return Value:			Activations of the output layer
FloatMatrix feedForwardAutoencoder(vector<float>& theta, const int& hiddenSize, const int& visibleSize, FloatMatrix& data);

// Determines which class have been selected for each data item
void softmaxPredict(FloatMatrix& theta, FloatMatrix& data, vector<float>& pred);

// Activation function (stores activation of m in a)
void activate(FloatMatrix& a, FloatMatrix& m);

// Adds a vector to each column in a matrix and stores in target
void addVector(FloatMatrix& target, FloatMatrix& source, vector<float>& v);

// Adds a vector to each column in a matrix 
inline void addVector(FloatMatrix& m, vector<float>& v);

// Adds a vector to each column in a matrix and calls the activation function on it
void addVectorAndActivate(FloatMatrix& m, vector<float>& v);

// Calculates teh squared difference between two matricies
float diffSqrd(FloatMatrix& h, FloatMatrix& y);

// Calculates the sum of squared values in w
float sumSqrd(FloatMatrix& w);

// Activation function (currently sigmoid)
inline float f(const float& x)
{
	return 1.0f / (1.0f + exp(-x));
	//return x / (1 + abs(x));
	//return tanh(x) * .5 + .5;
	//return tanh(x);
}

// Gradient of activation function
inline float fGrad(const float& x)
{
	return f(x) * (1 - f(x));
	//return .5*(1 - (tanh(x) * tanh(x)));
}
