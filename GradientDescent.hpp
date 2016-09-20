#include <vector>
#include "FloatMatrix.hpp"

using namespace std;

// Uses gradient descent to train the network using sparseAutoEncoderCost
//      Param theta:        The parameters of the network
//      Param visibleSize:  The number of visible neurons (input size)
//      Param hiddenSize:   The number of hidden neurons
//      Param lambda:       Weight decay parameter
//      Param sparsity:     Sparsity parameter
//      Param beta:         Beta parameter
//      Param data:         Input data
//      Param maxIter:      Maximum number of iterations allowed
void gradientDescent(vector<float>& theta, const int& visibleSize, const int& hiddenSize, const float& lambda, const float& sparsity, const float& beta, FloatMatrix& data, int maxIter);

// Uses gradient descent to train the network using softmaxAutoEncoderCost
//      Param theta:        The parameters of the network
//      Param numClasses:   The number of neurons in the output layer
//      Param visibleSize:  The number of visible neurons
//      Param lambda:       Weight decay parameter
//      Param data:         Input data
//      Param maxIter:      Maximum number of iterations allowed
void gradientDescentSoftmax(vector<float>& theta, const int& numClasses, const int& visibleSize, const float& lambda, FloatMatrix& data, vector<float> labels, int maxIter);

// Would like to create a generalized function where I can pass it a cost function