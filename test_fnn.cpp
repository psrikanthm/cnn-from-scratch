#include "linAlgebra.h"
#include "Util.h"
#include "fnn.h"

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
using namespace std;

int test1(){
	const char* path = "data/mnist_train.csv";
	std::vector<std::vector<double> > Xtrain, Ytrain; 
	pre_process::process_mnist_csv(path, Xtrain, Ytrain);
	int inputs = Xtrain[0].size();
	int outputs = Ytrain[0].size();
	std::vector<int> layers = {10};
	std::unique_ptr<NeuralNetwork> ann(new NeuralNetwork(inputs,layers,outputs));
	ann->train(Xtrain, Ytrain, 0.1, 50, 10);
	
	const char* test_path = "data/mnist_test.csv";
	std::vector<std::vector<double> > Xval, Yval; 
	pre_process::process_mnist_csv(test_path, Xval, Yval);
	ann->validate(Xval, Yval, 200);

	return 0;
}

int main(){
	test1();	
}
