#ifndef CNN_H
#define CNN_H

#include "Matrix.h"
#include "Util.h"

/* The CNN architecture is fixed, in the sense that it has one convolution layer 
 * (with single channel, stride = 1)followed by one max pool layer and then one Dense layer. 
 * However, filter size in convolution layer,pool size in max pool layer, hidden units 
 * in Dense layer are all tunable.
 * Since this project is more for demonstrative purpose and to understand the 
 * workings of CNN in detail and doesn't aim for performance. 
 * Only Stochastic Gradient Descent is supported for minimizing the CNN error function,
 * Since the chosen data structure "Matrix.h" is only 2D and supporting higher dimensional
 * Tensors is probably out of scope of this project.
 */

class CNN{
public:
	// each image is a Matrix
	int train(std::vector<std::unique_ptr<Matrix> > &Xtrain, 
			std::vector<std::unique_ptr<std::vector<double> > > &Ytrain, 
			double learning_rate, unsigned int epochs);
	double validate(std::vector<std::unique_ptr<Matrix> > &Xval, 
						std::vector<std::unique_ptr<std::vector<double> > > &Yval);
	CNN(Shape input_dim, Shape kernel_size, Shape pool_size,  unsigned int hidden_layer_nodes, 
				unsigned int output_dim);
	void info();
private:
	std::vector<std::unique_ptr<Matrix> > weights; 
		//weights.size() = 2: max-pool(flatten) to hidden, hidden to output
	std::unique_ptr<Matrix> kernel;
	Shape pool_window;
	int back_propagate(std::unique_ptr<std::vector<double> > &delta_L,
			std::vector<std::unique_ptr<Matrix> > &conv_activations, 
			std::vector<std::unique_ptr<std::vector<double> > > &activations,
			std::unique_ptr<Matrix> &input, double (*active_fn_der)(double), double learning_rate);
		//each column of activations belongs to 1 hidden layer
		//conv_activations contains activations of convolutional layer
		// and also "winning units" index after max pool layer
	int forward_propagate(std::unique_ptr<Matrix> &input,
				std::vector<std::unique_ptr<Matrix> > &conv_activations, 
				std::vector<std::unique_ptr<std::vector<double> > > &activations);
	double cross_entropy(std::unique_ptr<std::vector<double> > &ypred, 
					std::unique_ptr<std::vector<double> > &ytrue);
};
#endif
