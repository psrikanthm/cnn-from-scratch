#include "Matrix.h"
#include "linAlgebra.h"
#include "cnn.h"
#include "Util.h"

#include <assert.h>
#include <cmath>
#include <random>

using namespace std;

CNN::CNN(Shape input_dim, Shape kernel_size, Shape pool_size,  unsigned int hidden_layer_nodes, 
			unsigned int output_dim){
	/*	Initialize the weight matrices and kernel matrix
	 *	Since there is only one hidden_layer, there are two weight matrices -
	 *	One between Pool layer (flattened output of pool) and one that maps 
	 *	Hidden layer to outputs.
	 *	Also there is only one convolution layer, so only one kernel.
	 *	There is no learning with pool layer, therefore no weights associated
	 *	with them.
	 * */
	assert(input_dim.rows > kernel_size.rows && input_dim.columns > kernel_size.columns);
	assert(input_dim.rows - kernel_size.rows + 1 > pool_size.rows && 
			input_dim.columns - kernel_size.columns + 1 > pool_size.columns);

	//std::unique_ptr<Matrix> K(new Matrix(kernel_size.rows, kernel_size.columns, true));
	kernel = std::make_unique<Matrix>(kernel_size.rows, kernel_size.columns, true);

	//set pool size
	pool_window = pool_size;

	int x = ((input_dim.rows - kernel_size.rows + 1)/pool_size.rows);
	int y = ((input_dim.columns - kernel_size.columns + 1)/pool_size.columns);

	std::unique_ptr<Matrix> W0(new Matrix((x*y)+1, hidden_layer_nodes, true));
	this->weights.emplace_back(std::move(W0));
	std::unique_ptr<Matrix> W1(new Matrix(hidden_layer_nodes+1, output_dim, true));
	this->weights.emplace_back(std::move(W1));
}

int CNN::forward_propagate(std::unique_ptr<Matrix> &input,
	std::vector<std::unique_ptr<Matrix> > &conv_activations, 
	std::vector<std::unique_ptr<std::vector<double> > > &activations){
	/*	Forward propagate the provided inputs through the Convolution Neural Network 
	 *	and the outputs of each Dense layer is appended as vector to activations.
	 *	Output of convolution layer(matrix) is appended to conv_activations	
	 * */
	assert(weights.size() == 2); // flatten -> hidden, hidden -> output
	std::unique_ptr<Matrix> conv = std::make_unique<Matrix>(input->getRows() - kernel->getRows() + 1,
							input->getColumns() - kernel->getColumns() + 1, true);	

	for(unsigned int i = 0; i < conv->getRows(); i++){
		for(unsigned int j = 0; j < conv->getColumns(); j++){
			conv->set(i,j,np::multiply(kernel, input, i, j));				
		}
	}
	conv = np::applyFunction(conv, fns::relu);

	unsigned int x = (conv->getRows()/pool_window.rows);
	unsigned int y = (conv->getColumns()/pool_window.columns);
	
	std::unique_ptr<Matrix>	pool = std::make_unique<Matrix>(conv->getRows(), conv->getColumns(), false);
	std::unique_ptr<std::vector<double> > pool_flatten = std::make_unique<std::vector<double> >();

	unsigned int xptr=0, yptr=0;
	auto max_index = std::make_unique<Shape>(Shape{0,0});
	for(unsigned int i=0; i < x; i++){
		xptr = (i * pool_window.rows);
		for(unsigned int j=0; j < y; j++){
			yptr = (j * pool_window.columns);
			double max = np::maximum(conv, xptr, yptr, pool_window, max_index);
			pool_flatten->push_back(max);
			pool->set(max_index->rows, max_index->columns, 1);
		}
	}
	
	conv_activations[0] = std::move(pool);
	
	//	append 1s to inputs and to output of every layer (for bias)
	pool_flatten->push_back(1);


	//	hidden layer
	std::unique_ptr<Matrix> W0 = np::transpose(weights[0]);
	std::unique_ptr<std::vector<double> > hidden = np::dot(W0, pool_flatten);
	hidden = np::applyFunction(hidden, fns::relu);
	hidden->push_back(1);

	activations[0] = std::move(pool_flatten);
	// output layer
	std::unique_ptr<Matrix> W1 = np::transpose(weights[1]);
	std::unique_ptr<std::vector<double> > output = np::dot(W1, hidden);
	output = np::applyFunction(output, fns::softmax);
	output = np::normalize(output); 
	
	activations[1] = std::move(hidden);
	activations[2] = std::move(output);
	return 0;
}

double CNN::cross_entropy(std::unique_ptr<std::vector<double> > &ypred, 
					std::unique_ptr<std::vector<double> > &ytrue){
	/*	Calculate cross entropy loss if the predictions and true values are given
	 */
	assert(ypred->size() == ytrue->size());
	std::unique_ptr<std::vector<double> > z = np::applyFunction(ypred,log);
	z = np::multiply(z,ytrue);
	double error = np::element_sum(z);
	return (-error);
}

int CNN::back_propagate(std::unique_ptr<std::vector<double> > &delta_L, 
		std::vector<std::unique_ptr<Matrix> > &conv_activations, 
		std::vector<std::unique_ptr<std::vector<double> > > &activations,
		std::unique_ptr<Matrix> &input, double (*active_fn_der)(double), double learning_rate){
	/*	Compute deltas of each layer and return the same.
	 *	delta_L: delta of the final layer, computed and passed as argument
	 *	activations: Output of each layer after applying activation function
	 *	Assume that all layers have same activation function except that of final layer.
	 *	active_fn_der: function pointer for the derivative of activation function, 
	 *					which takes activation of the layer as input
	 * */
	
	std::unique_ptr<std::vector<double> > delta_h = np::dot(weights[1], delta_L);
	std::unique_ptr<std::vector<double> > active = np::applyFunction(activations[1], 
																active_fn_der);
	delta_h = np::multiply(delta_h, active);

	std::unique_ptr<std::vector<double> > delta_x = np::dot(weights[0], delta_h, 1); 
									// don't compute last layer
	active = np::applyFunction(activations[0], active_fn_der);
	delta_x = np::multiply(delta_x, active);

	std::unique_ptr<Matrix> delta_conv = 
		std::make_unique<Matrix>(conv_activations[0]->getRows(), conv_activations[0]->getColumns(), false);

	unsigned int counter = 0;
	for(unsigned int r=0; r<conv_activations[0]->getRows(); r++){
		for(unsigned int c=0; c<conv_activations[0]->getColumns(); c++){
			if(conv_activations[0]->get(r,c) == 1.0) {
				delta_conv->set(r, c, delta_x->at(counter));
				counter ++;
			}
		}
	}

	// update weights
	std::unique_ptr<Matrix> dW0 = np::dot(activations[0], delta_h, 1);
		// last column has to be sliced off	
	std::unique_ptr<Matrix> dW1 = np::dot(activations[1], delta_L);
	dW0 = np::multiply(dW0,(learning_rate));
	dW1 = np::multiply(dW1,(learning_rate));

	weights[0] = np::subtract(weights[0], dW0);
	weights[1] = np::subtract(weights[1], dW1);

	for(unsigned int i = 0; i < kernel->getRows(); i++){
		for(unsigned int j = 0; j < kernel->getColumns(); j++){
			kernel->set(i,j,np::multiply(delta_conv, input, i, j));				
		}
	}
	return 0;
}

int CNN::train(std::vector<std::unique_ptr<Matrix> > &Xtrain, 
				std::vector<std::unique_ptr<std::vector<double> > > &Ytrain, 
				double learning_rate = 0.01, unsigned int epochs=10){
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
		unsigned int it = 0;
		double error = 0;
		while(it < Xtrain.size()){
			std::vector<std::unique_ptr<Matrix> > conv_activations(2);
			std::vector<std::unique_ptr<std::vector<double> > > activations(3);

			forward_propagate(Xtrain[it], conv_activations, activations);
			error += cross_entropy(activations.back(), Ytrain[it]);
			
			std::unique_ptr<std::vector<double> > delta_L = np::subtract(activations.back(), Ytrain[it]);
			
			back_propagate(delta_L, conv_activations, activations, Xtrain[it], 
									fns::relu_gradient, learning_rate);

			it += 1;
		}
		cout << "epoch: " << e << " error: " << (error/Xtrain.size()) << endl ;
		e += 1;
	}
	return 0;
}

double CNN::validate(std::vector<std::unique_ptr<Matrix> > &Xval, 
						std::vector<std::unique_ptr<std::vector<double> > > &Yval){
	/*	Calculate the Validation error over the validation set.
	 *	So only do forward_propagate for each batch without updating weights
	 *	each iteration
	*/
	assert(Xval.size() == Yval.size());
	unsigned int it = 1;
	double error = 0;
	while(it <= Xval.size()){
		std::vector<std::unique_ptr<Matrix> > conv_activations(2);
		std::vector<std::unique_ptr<std::vector<double> > > activations(3);

		forward_propagate(Xval[it], conv_activations, activations);
		error += cross_entropy(activations.back(), Yval[it]);
		
		it += 1;
	}
	cout << " error: " << (error/Xval.size()) << endl ;
	return (error/Xval.size());
}

void CNN::info(){
	cout << "Kernel size: (" << kernel->getRows() << "," << kernel->getColumns() << ")" << endl;
	for(unsigned int i = 0; i < weights.size(); i++){
		cout << "Weight "<< i << " size: (" << weights[i]->getRows() << "," << weights[i]->getColumns() << ")" << endl;
	}
}
