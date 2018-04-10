#include "Matrix.h"
#include "linAlgebra.h"
#include "fnn.h"
#include "Util.h"

#include <assert.h>
#include <cmath>
#include <random>

using namespace std;

NeuralNetwork::NeuralNetwork(int input_dim, std::vector<int> hidden_layers, int output_dim){

	hidden_layers.push_back(output_dim);
	int h0 = input_dim, h1 = hidden_layers[0];

	std::unique_ptr<Matrix> A(new Matrix(h0+1, h1, true));
	this->weights.emplace_back(std::move(A));

	for(unsigned int i = 1; i < hidden_layers.size(); i++){
		h0 = h1;
		h1 = hidden_layers[i];
		std::unique_ptr<Matrix> A(new Matrix(h0+1, h1, true));
		this->weights.emplace_back(std::move(A));
	}
}

std::vector<std::unique_ptr<Matrix> > NeuralNetwork::forward_propagate(std::unique_ptr<Matrix> &input){
	/*	Forward propagate the provided inputs through the Neural Network 
	 *	and return the outputs of each layer as vector of Matrices
	 *
	 * */
	std::vector<std::unique_ptr<Matrix> > activations;
	// append column of 1s to inputs and to output of every layer
	std::vector<double> onevec(input->getRows(), (double)1);
	
	std::unique_ptr<Matrix> z ;
	z = np::concatenate(input, onevec);
	activations.emplace_back(std::move(z));
	for(unsigned int i = 0; i < weights.size() - 1; i++){
		z = np::dot(activations[i], weights[i]);
		z = np::applyFunction(z, fns::relu);
		std::vector<double> onevec(z->getRows(), (double) 1);
		z = np::concatenate(z, onevec);
		activations.emplace_back(std::move(z));
	}
	z = np::dot(activations[activations.size()-1], weights[weights.size()-1]);
	// softmax function provided in Util.cpp is un-normalized
	// So we need to row wise normalize the output of below applyFunction
	z = np::applyFunction(z, fns::softmax);
	z = np::normalize(z); // current implementation only normalizes such that each row sums to 1
	activations.emplace_back(std::move(z));
	return activations;
}

double NeuralNetwork::cross_entropy(std::unique_ptr<Matrix> &ypred, std::unique_ptr<Matrix> &ytrue){
	/*	Calculate cross entropy loss if the predictions and true values are given
	 */
	assert(ypred->getRows() == ytrue->getRows() && ypred->getColumns() == ytrue->getColumns());
	unsigned int batch_size = ypred->getRows();
	std::unique_ptr<Matrix> z = np::applyFunction(ypred,log);
	z = np::multiply(z,ytrue);
	double error = np::element_sum(z);
	return (-error/batch_size);
}

std::vector<std::unique_ptr<Matrix> > NeuralNetwork::back_propagate(std::unique_ptr<Matrix> &delta_L, 
					std::vector<std::unique_ptr<Matrix> > &activations, double (*active_fn_der)(double)){
	/*	Compute deltas of each layer and return the same.
	 *	delta_L: delta of the final layer, computed and passed as argument
	 *	activations: Output of each layer after applying activation function
	 *	Assume that all layers have same activation function except that of final layer.
	 *	active_fn_der: function pointer for the derivative of activation function, 
	 *					which takes activation of the layer as input
	 * */
	assert(activations.size() == weights.size() + 1);

	std::vector<std::unique_ptr<Matrix> > deltas;
	deltas.emplace_back(std::move(delta_L));
	std::unique_ptr<Matrix> z, y;
	unsigned int nr_layers = weights.size();

	for(int i = nr_layers-1; i >= 0; i--) {
		z = np::transpose(weights[i]);
		z = np::dot(deltas[nr_layers-1 - i], z, 1); // don't compute the last column of delta as that
													//	belongs to bias
		y = np::applyFunction(activations[i], active_fn_der);
		z = np::multiply(z, y, 1);	// same here, don't compute the last column
		deltas.emplace_back(std::move(z));
	}

	std::reverse(deltas.begin(),deltas.end());
	return deltas;
}

void NeuralNetwork::update_weights(std::vector<std::unique_ptr<Matrix> > &activations, 
		std::vector<std::unique_ptr<Matrix> > &deltas, double learning_rate, unsigned int batch_size){
	/* 	Mini-Batch gradient descent with batch_size as number of training examples
	 *	Now it is simple gradient descent, but in future Momentum etc can be added
	 *	With activations, deltas given gradient of error w.r.t each weight matrix 
	 *	can be computed easily
	 */
	assert(deltas.size() == weights.size() + 1 && activations.size() == deltas.size());
	std::unique_ptr<Matrix> z;
	for(unsigned int i = 0; i < weights.size(); i++){
		z = np::transpose(activations[i]);

		z = np::dot(z, deltas[i+1]);	// don't compute the last column of delta
		z = np::multiply(z,(learning_rate * (1.0/batch_size)));

		this->weights[i] = np::subtract(weights[i], z);
	}
}

int NeuralNetwork::train(std::vector<std::vector<double> > &Xtrain, std::vector<std::vector<double> > &Ytrain, 
		double learning_rate=0.01, unsigned int batch_size=32, unsigned int epochs=10){
	/*	Train the Neural Network aka change weights such that error between 
	 * the forward propagated Xtrain and Ytrain is reduced.
	 * Break (Xtrain, ytrain) into batches of size batch_size and run 3 following
	 * methods for all batches:
	 *		1) forward_propagate
	 *		2) error calculation: cross_entropy
	 *		3) back_propagate
	 * 		4) update_weights
	 * Do this procedure 'epochs' number of times
	 */
	assert(Xtrain.size() == Ytrain.size());

	unsigned int e = 1;
	while(e <= epochs){
		// Split (Xtrain, Ytrain) into batches	
		// This is expensive because it is making sub vectors by copying
		// Have to think of a better way
		unsigned int it = 0, nr_batches = 1;
		double error = 0;
		while(it + batch_size < Xtrain.size()){
			std::vector<std::vector<double> > Xbatch(&Xtrain[it], &Xtrain[it+batch_size]);
			std::vector<std::vector<double> > Ybatch(&Ytrain[it], &Ytrain[it+batch_size]);

			auto random_engine2 = random_engine;
			std::shuffle(std::begin(Xbatch), std::end(Xbatch), random_engine);
			std::shuffle(std::begin(Ybatch), std::end(Ybatch), random_engine2);
			
			std::unique_ptr<Matrix> X(new Matrix(Xbatch));
			std::unique_ptr<Matrix> Y(new Matrix(Ybatch));

			std::vector<std::unique_ptr<Matrix> > activations = forward_propagate(X);
			error += cross_entropy(activations.back(), Y);
			
			// The below calculation of delta for final layer - delta_L is
			// only applicable when last layer has softmax activation and
			// the loss function used is cross entropy
			std::unique_ptr<Matrix> delta_L = np::subtract(activations.back(), Y);
		
			std::vector<std::unique_ptr<Matrix> > deltas = back_propagate(delta_L, activations, fns::relu_gradient);
			update_weights(activations, deltas, learning_rate, batch_size);


			nr_batches += 1;
			it += batch_size;
		}
		cout << "epoch: " << e << " error: " << (error/nr_batches) << endl ;
		e += 1;
	}
	return 0;
}

double NeuralNetwork::validate(std::vector<std::vector<double> > &Xval, std::vector<std::vector<double> > &Yval,
				unsigned int batch_size){
	/*	Calculate the Validation error over the validation set.
	 *	So only do forward_propagate for each batch without updating weights
	 *	each iteration
	*/
	assert(Xval.size() == Yval.size());

	// Split (Xval, Yval) into batches	
	// This is expensive because it is making sub vectors by copying
	// Have to think of a better way
	unsigned int it = 0, nr_batches = 1;
	double error = 0;
	while(it + batch_size < Xval.size()){
		std::vector<std::vector<double> > Xbatch(&Xval[it], &Xval[it+batch_size]);
		std::vector<std::vector<double> > Ybatch(&Yval[it], &Yval[it+batch_size]);

		auto random_engine2 = random_engine;
		std::shuffle(std::begin(Xbatch), std::end(Xbatch), random_engine);
		std::shuffle(std::begin(Ybatch), std::end(Ybatch), random_engine2);
		
		std::unique_ptr<Matrix> X(new Matrix(Xbatch));
		std::unique_ptr<Matrix> Y(new Matrix(Ybatch));

		std::vector<std::unique_ptr<Matrix> > activations = forward_propagate(X);
		error += cross_entropy(activations.back(), Y);
	
		nr_batches += 1;
		it += batch_size;
	}
	cout << "validation error: " << (error/nr_batches) << endl ;
	return (error/nr_batches);
}

void NeuralNetwork::info(bool verbose=false){
	for(unsigned int i = 0; i < weights.size(); i++){
		int rows = weights[i] -> getRows();
		int columns = weights[i] -> getColumns();
		std::cout << "matrix " << i << " dimension: (" << rows << "," << columns << ")" << std::endl ;
	}
	if(verbose){
		for(unsigned int i = 0; i < weights.size(); i++){
			std::cout << "matrix " << i << endl; 
			weights[i]->pretty_print() ;
			cout << endl ;
		}
	}
}
