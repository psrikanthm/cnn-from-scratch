#ifndef FNN_H
#define FNN_H

#include "Matrix.h"

class NeuralNetwork{
public:
	int train(std::vector<std::vector<double> > &Xtrain, std::vector<std::vector<double> > &Ytrain, 
			double learning_rate, unsigned int batch_size, unsigned int epochs);
	double validate(std::vector<std::vector<double> > &Xval, std::vector<std::vector<double> > &Yval,
				unsigned int batch_size);
	NeuralNetwork(int input_dim, std::vector<int> hidden_layers, int output_dim);
	void info(bool verbose);
private:
	std::vector<std::unique_ptr<Matrix> > weights;
	std::vector<std::unique_ptr<Matrix> > back_propagate(std::unique_ptr<Matrix> &delta_L, 
					std::vector<std::unique_ptr<Matrix> > &activations, double (*active_fn_der)(double));
	std::vector<std::unique_ptr<Matrix> > forward_propagate(std::unique_ptr<Matrix> &input);
	void update_weights(std::vector<std::unique_ptr<Matrix> > &activations, 
			std::vector<std::unique_ptr<Matrix> > &deltas, double learning_rate, unsigned int batch_size);
	double cross_entropy(std::unique_ptr<Matrix> &ypred, std::unique_ptr<Matrix> &ytrue);
};

#endif
