#ifndef UTIL_H
#define UTIL_H

#include "Matrix.h"
#include <random>

// utility variables and methods

// random double generator
extern std::default_random_engine random_engine; 

// struct datastructure used for storing 2D shape
struct Shape{
	int rows;
	int columns;
};

namespace fns{
	double relu(double x);
	double sigmoid(double x);
	double tan(double x);
	double relu_gradient(double x);
	double sigmoid_gradient(double x);
	double tan_gradient(double x);
	double softmax(double x);
}

namespace pre_process{
	int process_mnist_images(const char* path, std::vector<std::unique_ptr<Matrix> > &Xtrain, 
		std::vector<std::unique_ptr<std::vector<double> > > &Ytrain, unsigned int nr_images=100);
	int process_mnist_csv(const char* filename, std::vector<std::vector<double> > &Xtrain, 
		std::vector<std::vector<double> > &Ytrain) ;
	void process_image(const char* filename);
}

#endif
