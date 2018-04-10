#include "linAlgebra.h"
#include "Util.h"
#include "cnn.h"

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
using namespace std;

int test1(){
	Shape input_dim={28,28};
	Shape kernel_dim={6,6};
	Shape pool_size={4,4};
	std::unique_ptr<CNN> cnn(new CNN(input_dim, kernel_dim, pool_size, 30, 10));
	
	const char* path = "data/mnist_png/training/";
	std::vector<std::unique_ptr<Matrix> > Xtrain;
	std::vector<std::unique_ptr<std::vector<double> > > Ytrain;
	pre_process::process_mnist_images(path, Xtrain, Ytrain, 100);

	cnn->train(Xtrain, Ytrain, 0.01, 30);
	return 0;
}

int main(){
	test1();	
}
