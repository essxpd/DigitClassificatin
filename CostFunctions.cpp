#include <algorithm>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <limits>

#include "CostFunctions.hpp"
#include "MatrixMultiply.hpp"

using namespace chrono;

void sparseAutoencoderCost(vector<float>& theta, 
			const int& visibleSize, const int& hiddenSize, 
			const float& lambda, const float& sparsityParam, const float& beta, 
			FloatMatrix& data, 
			float& cost, vector<float>& grad)
{
	auto t1 = high_resolution_clock::now();
	FloatMatrix W1(visibleSize, hiddenSize);
	FloatMatrix W2(hiddenSize, visibleSize);
	
	vector<float> b1(hiddenSize);
	vector<float> b2(visibleSize);

	// roll theta into variables
	auto it = theta.begin();
	copy(it, it + hiddenSize * visibleSize, &(W1.data()[0]));
	copy(it + hiddenSize * visibleSize, it + hiddenSize * visibleSize * 2, &(W2.data()[0]));
	copy(it + hiddenSize * visibleSize * 2, it + hiddenSize * visibleSize * 2 + hiddenSize, b1.begin());
	copy(it + hiddenSize * visibleSize * 2 + b1.size(), theta.end(), b2.begin());

	W1 = boost::numeric::ublas::trans(W1);
	W2 = boost::numeric::ublas::trans(W2);

	// Cost and gradient variables
	cost = 0;
	FloatMatrix W1grad(W1.size1(), W1.size2());
	FloatMatrix W2grad(W2.size1(), W2.size2());
	vector<float> b1grad(b1.size());
	vector<float> b2grad(b2.size());

	// Get number of cases
	int m = data.size2();

	// Calculate activations
	FloatMatrix a2(W1.size1(), data.size2());
	FloatMatrix z2(a2.size1(), a2.size2());
	matrixMultiply(W1, data, z2);				// Cuda operation
	addVector(z2, b1);
	activate(a2, z2);
	
	FloatMatrix a3(W2.size1(), a2.size2());
	FloatMatrix z3(a3.size1(), a3.size2());
	matrixMultiply(W2, a2, z3);					// Cuda operation
	addVector(z3, b2);
	activate(a3, z3);
	
	auto t2 = high_resolution_clock::now();

	// Calculate cost
	// Averate sum-of-squares error term
	float ssq = (1.0f/(float)m) * (1.0f/2.0f) * diffSqrd(data, a3); 
	cost += ssq;

	// Regularization term (weight decay)
	float decay = ((lambda/2.0f) * (sumSqrd(W1) + sumSqrd(W2)));
	cost += decay;

	// KL Divergence term (sparsity)
	vector<float> rho(a2.size1());
	float* a2_ptr = &(a2.data()[0]);	
	
	for(int i = 0; i < rho.size(); ++i)
	{
		for(int j = 0; j < a2.size2(); ++j)
			rho[i] += a2_ptr[j + a2.size2() * i];

		rho[i] /= (float)m;
	}

	float KL_divergence_1 = 0;
	float KL_divergence_2 = 0;
	float t = 0;

	for(int i = 0; i < rho.size(); ++i)
	{	
		KL_divergence_1 += (sparsityParam*log(sparsityParam / rho[i]));
		KL_divergence_2 += ((1 - sparsityParam)*log((1-sparsityParam)/(1-rho[i])));

		t += (sparsityParam*log(sparsityParam / rho[i]) + (1 - sparsityParam)*log((1 - sparsityParam) / (1-rho[i])));
	}

	float KL = beta*(KL_divergence_1 + KL_divergence_2);
	cost += KL;

	// Calculate d3 and d2
	FloatMatrix d3(data.size1(), data.size2());
	float* d_ptr = &(data.data()[0]);
	float* d3_ptr = &(d3.data()[0]);
	float* a3_ptr = &(a3.data()[0]);
	float* z3_ptr = &(z3.data()[0]);


	int l = data.size1() * data.size2();
	#pragma omp parallel for
	for(int i = 0; i < l; ++i)
	{
		d3_ptr[i] = -(d_ptr[i] - a3_ptr[i]) * fGrad(z3_ptr[i]);
	}

	FloatMatrix W2t = boost::numeric::ublas::trans(W2);
	FloatMatrix d2(W2t.size1(), d3.size2());
	float* d2_ptr = &(d2.data()[0]);
	float* z2_ptr = &(z2.data()[0]);
	matrixMultiply(W2t, d3, d2);		// Cuda operation

	l = d2.size1() * d2.size2();

	#pragma omp parallel for
	for(int i = 0; i < d2.size1(); ++i)
	{
		for(int j = 0; j < d2.size2(); ++j)
		{
			d2(i, j) += beta * (-sparsityParam / rho[i] + (1 - sparsityParam) / (1 - rho[i])); 
			d2(i, j) *= fGrad(z2(i, j));
		}
	}

	// Calculate gradients
	FloatMatrix W1delta(W1.size1(), W1.size2());
	FloatMatrix W2delta(W2.size1(), W2.size2());
	vector<float> b1delta(b1.size());
	vector<float> b2delta(b2.size());
	FloatMatrix dataTranspose = boost::numeric::ublas::trans(data);
	FloatMatrix a2Transpose = boost::numeric::ublas::trans(a2);

	matrixMultiply(d2, dataTranspose, W1delta);		// Cuda operation
	matrixMultiply(d3, a2Transpose, W2delta);		// Cuda operation
	
	#pragma omp parallel for
	for(int i = 0; i < d2.size1(); ++i)
	{
		for(int j = 0; j < d2.size2(); ++j)
			b1delta[i] += d2_ptr[i * d2.size2() + j];
	}

	#pragma omp parallel for
	for(int i = 0; i < d3.size1(); ++i)
	{
		for(int j = 0; j < d3.size2(); ++j)
			b2delta[i] += d3_ptr[i * d3.size2() + j];
	}

	float* W1grad_ptr = &(W1grad.data()[0]);
	float* W2grad_ptr = &(W2grad.data()[0]);
	float* W1delta_ptr = &(W1delta.data()[0]);
	float* W2delta_ptr = &(W2delta.data()[0]);
	float* W1_ptr = &(W1.data()[0]);
	float* W2_ptr = &(W2.data()[0]);

	#pragma omp parallel for
	for(int i = 0; i < W1grad.size1(); ++i)
	{
		for(int j = 0; j < W1grad.size2(); ++j)
			W1grad_ptr[i * W1grad.size2() + j] = W1delta_ptr[i * W1grad.size2() + j] / m + lambda * W1_ptr[i * W1grad.size2() + j];
		
		b1grad[i] = b1delta[i] / m;
	}

	#pragma omp parallel for
	for(int i = 0; i < W2grad.size1(); ++i)
	{
		for(int j = 0; j < W2grad.size2(); ++j)
			W2grad_ptr[i * W2grad.size2() + j] = W2delta_ptr[i * W2grad.size2() + j] / m + lambda * W2_ptr[i * W2grad.size2() + j];
		
		b2grad[i] = b2delta[i] / m;
	}

	W1grad = boost::numeric::ublas::trans(W1grad);
	W2grad = boost::numeric::ublas::trans(W2grad);
	W1grad_ptr = &(W1grad.data()[0]);
	W2grad_ptr = &(W2grad.data()[0]);

	// Unroll into a vector
	grad = vector<float>(W1grad.size1()*W1grad.size2() + W2grad.size1()*W2grad.size2() + b1grad.size() + b2grad.size());

	auto grad_it = grad.begin();
	copy(W1grad_ptr, W1grad_ptr + W1grad.size1()*W1grad.size2(), grad_it);
	copy(W2grad_ptr, W2grad_ptr + W2grad.size1()*W2grad.size2(), grad_it + W1grad.size1()*W1grad.size2());
	copy(b1grad.begin(), b1grad.end(), grad_it + W1grad.size1()*W1grad.size2() + W2grad.size1()*W2grad.size2());
	copy(b2grad.begin(), b2grad.end(), grad_it + W1grad.size1()*W1grad.size2() + W2grad.size1()*W2grad.size2() + b1.size());

/*
	cout << "Cost:" << endl;
	cout << "\tSSQ test: " << ssq << endl;
	cout << "\tDecay test: " << decay << endl;
	cout << "\tDiv test: " << KL << endl;

	auto t3 = high_resolution_clock::now();
	auto d1 = duration_cast<duration<float>>(t3 - t2);
	auto d = duration_cast<duration<float>>(t3 - t1);
	cout << "Full Time: " << d.count() << endl;
	cout << "Test time: " << d1.count() << endl;

	cin.ignore();
*/
}

// Calculate activations of network given data as input
FloatMatrix feedForwardAutoencoder(vector<float>& theta, const int& hiddenSize, const int& visibleSize, FloatMatrix& data)
{
	FloatMatrix W1(visibleSize, hiddenSize);
	FloatMatrix W2(hiddenSize, visibleSize);
	
	vector<float> b1(hiddenSize);
	vector<float> b2(visibleSize);

	// unroll theta into variables
	auto it = theta.begin();
	copy(it, it + hiddenSize * visibleSize, &(W1.data()[0]));
	copy(it + hiddenSize * visibleSize, it + hiddenSize * visibleSize * 2, &(W2.data()[0]));
	copy(it + hiddenSize * visibleSize * 2, it + hiddenSize * visibleSize * 2 + hiddenSize, b1.begin());
	copy(it + hiddenSize * visibleSize * 2 + b1.size(), theta.end(), b2.begin());
	W1 = boost::numeric::ublas::trans(W1);
	
	// Calculate activation
	FloatMatrix a2(W1.size1(), data.size2());
	FloatMatrix z2(a2.size1(), a2.size2());
	matrixMultiply(W1, data, z2);			// Cuda operation
	addVector(z2, b1);
	activate(a2, z2);

	return a2;
}


void softmaxCost(vector<float>& theta, const int& numClasses, const int& inputSize, const float& lambda, FloatMatrix& data, vector<float>& labels, float& cost, vector<float>& grad)
{
	// Unroll theta into parameters
	FloatMatrix t(numClasses, inputSize);
	for(int i = 0; i < t.size1(); ++i)
		for(int j = 0; j < t.size2(); ++j)
			t(i, j) = theta[i + t.size1() * j];

	// Calculate the target output (groundTruth)
	FloatMatrix groundTruth(numClasses, data.size2());
	for(int i = 0; i < groundTruth.size2(); ++i)
	{
		for(int j = 0; j < groundTruth.size1(); ++j)
		{
			if(labels[i] == j)
				groundTruth(labels[i], i) = 1.0f;
			else
				groundTruth(j, i) = 0.0f;
		}
	}

	FloatMatrix thetaGrad(numClasses, inputSize);
	FloatMatrix h_denominator_exp(t.size1(), data.size2());

	matrixMultiply(t, data, h_denominator_exp);		// Cuda operation
	vector<float> denominator(data.size2());
	FloatMatrix h(numClasses, data.size2());

	// Reduce exponent to prevent overflow
	for(int i = 0; i < h_denominator_exp.size2(); ++i)
	{
		float max1 = -1e10;
		for(int j = 0; j < h_denominator_exp.size1(); ++j)
			max1 = max(h_denominator_exp(j, i), max1);

		for(int j = 0; j < h_denominator_exp.size1(); ++j)
		{
			h_denominator_exp(j, i) -= max1;
			denominator[i] += exp(h_denominator_exp(j,i));
		}

		for(int j = 0; j < h.size1(); ++j)
		{
			h(j, i) = exp(h_denominator_exp(j, i)) / denominator[i];
			if(h(j, i) <= 0)
				h(j, i) = numeric_limits<float>::min();
		}
	}

	// Calculate cost 
	double cost_indicator = 0.0f;
	for(int i = 0; i < groundTruth.size1(); ++i)
		for(int j = 0; j < groundTruth.size2(); ++j)
			cost_indicator += -(1.0f/data.size2()) * groundTruth(i, j) * log(h(i, j));

	float cost_decay = 0.0f;
	for(int i = 0; i < t.size1(); ++i)
		for(int j = 0; j < t.size2(); ++j)
			cost_decay += pow(t(i, j), 2);
	cost_decay *= (lambda / 2.0f);
	
	cost = cost_indicator + cost_decay;
	
	// Calculate gradient
	FloatMatrix diff(groundTruth.size1(), groundTruth.size2());
	for(int i = 0; i < diff.size1(); ++i)
		for(int j = 0; j < diff.size2(); ++j)
			diff(i, j) = groundTruth(i, j) - h(i, j);

	FloatMatrix costGrad1(diff.size1(), data.size1());
	FloatMatrix dataT = boost::numeric::ublas::trans(data);
	matrixMultiply(diff, dataT, costGrad1);			// Cuda operation
	
	for(int i = 0; i < costGrad1.size1(); ++i)
		for(int j = 0; j < costGrad1.size2(); ++j)
			costGrad1(i,j) *= -1.0f/data.size2();

	FloatMatrix costGrad2(t.size1(), t.size2());
	for(int i = 0; i < costGrad2.size1(); ++i)
		for(int j = 0; j < costGrad2.size2(); ++j)
			costGrad2(i, j) = lambda * t(i, j);

	for(int i = 0; i < thetaGrad.size1(); ++i)
		for(int j = 0; j < thetaGrad.size2(); ++j)
			thetaGrad(i, j) = costGrad1(i, j) + costGrad2(i, j);


	grad = vector<float>(thetaGrad.size1() * thetaGrad.size2());
	thetaGrad = boost::numeric::ublas::trans(thetaGrad);
	float* tg_ptr = &(thetaGrad.data()[0]);
	auto grad_it = grad.begin();
	copy(tg_ptr, tg_ptr + thetaGrad.size1()*thetaGrad.size2(), grad_it);

/*
	cout << "Cost indicator: " << cost_indicator << endl;
	cout << "Cost decay: " << cost_decay << endl;
	cin.ignore();
*/
}


// Uses theta to predict output when shown data
void softmaxPredict(FloatMatrix& theta, FloatMatrix& data, vector<float>& pred)
{
	FloatMatrix prob(theta.size1(), data.size2());
	matrixMultiply(theta, data, prob);
	pred = vector<float>(data.size2());

	for(int i = 0; i < prob.size2(); ++i)
	{
		float maxValue = -1e10;
		int maxIdx = 0;
		for(int j = 0; j < prob.size1(); ++j)
		{
			if(prob(j, i) > maxValue)
			{
				maxValue = prob(j, i);
				maxIdx = j;
			}
		}
		pred[i] = maxIdx;
	}
}


void activate(FloatMatrix& a, FloatMatrix& m)
{
	float* a_ptr = &(a.data()[0]);
	float* m_ptr = &(m.data()[0]);
	
	#pragma omp parallel for
	for(int i = 0; i < m.size1() * m.size2(); ++i)
		a_ptr[i] = f(m_ptr[i]);
}

void addVector(FloatMatrix& target, FloatMatrix& source, vector<float>& v)
{
	float* t_ptr = &(target.data()[0]);
	float* s_ptr = &(source.data()[0]);

	#pragma omp parallel for
	for(int i = 0; i < source.size1() * source.size2(); ++i)
		t_ptr[i] = s_ptr[i] + v[i % source.size1()];
}

void addVector(FloatMatrix& m, vector<float>& v)
{
	#pragma omp parallel for
	for(int i = 0; i < m.size1(); ++i)
		for(int j = 0; j < m.size2(); ++j)
			m(i, j) += v[i];
}

// Does both in one for loop to reduce overhead
void addVectorAndActivate(FloatMatrix& m, vector<float>& v)
{
	float* m_ptr = &(m.data()[0]);
	#pragma omp parallel for
	for(int i = 0; i < m.size1() * m.size2(); ++i)
	{
		m_ptr[i] = f(m_ptr[i] + v[i % m.size1()]);
	}
}

float diffSqrd(FloatMatrix& h, FloatMatrix& y)
{
	float* h_ptr = &(h.data()[0]);
	float* y_ptr = &(y.data()[0]);
	const int l = h.size1() * h.size2();
	vector<float> total(h.size2());
	float t = 0;

	#pragma omp parallel for reduction(+ : t)
	for(int _y = 0; _y < h.size2(); ++_y)
	{
		for(int _x = 0; _x < h.size1(); ++_x)
			total[_y] += pow(h_ptr[_x*h.size2() + _y] - y_ptr[_x*h.size2() + _y], 2);

		t += total[_y];
	}

	return t;
}


float sumSqrd(FloatMatrix& w)
{
	float* w_ptr = &(w.data()[0]);
	const int l = w.size1() * w.size2();
	float total = 0;	

	#pragma omp parallel for reduction(+ : total)
	for(int i = 0; i < l; ++i)
	{
		total += (w_ptr[i] * w_ptr[i]);
	}

	return total;
}

