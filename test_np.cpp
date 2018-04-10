#include "linAlgebra.h"
#include "Util.h"

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
using namespace std;

int test_applyFunction(){
	
	// declare a Matrix object
	int rows = 3; // set number of rows
	int columns = 2; // set number of columns

	std::unique_ptr<Matrix> A(new Matrix(rows, columns, true));
	std::unique_ptr<Matrix> B(new Matrix(rows, columns, true));
	std::vector<double> vec(A->getRows(), (double)1);

	std::vector<int> layers = {3};

	std::unique_ptr<Matrix> z = np::applyFunction(A, fns::softmax);
	z = np::normalize(z);
	std::unique_ptr<Matrix> C = np::applyFunction(z, log);
	double err = np::element_sum(z);
	cout << err << endl ;

	A->pretty_print();
	cout << endl ;
	z->pretty_print();
	cout << endl ;
	C->pretty_print();
	return 0;
}

int test_flatten(){

	int rows = 3; // set number of rows
	int columns = 2; // set number of columns
	std::unique_ptr<Matrix> A(new Matrix(rows, columns, true));
	std::unique_ptr<std::vector<double> > v = np::flatten(A);

	A->pretty_print();
	cout << endl;
	for(unsigned int i=0; i < v->size(); i++){
		cout << v->at(i) << " ";
	}
	cout << endl;
	return 0;
}

int test_multiply(){
	std::unique_ptr<Matrix> A(new Matrix(3, 3, true));
	std::unique_ptr<Matrix> K(new Matrix(2, 2, true));
	double C = np::multiply(K,A,1,1);
	A->pretty_print();
	cout << endl;
	K->pretty_print();
	cout << endl << C << endl;
	return 0;
}

int test_maximum(){
	std::unique_ptr<Matrix> A(new Matrix(4, 4, true));
	Shape window = {2,2};
	auto index = std::make_unique<Shape>(Shape{0,0});
	double C = np::maximum(A,0,0,window,index);
	A->pretty_print();
	cout << index->rows << "," << index->columns << " " << C << endl;
	C = np::maximum(A,0,2,window,index);
	cout << index->rows << "," << index->columns << " " << C << endl;
	C = np::maximum(A,2,0,window,index);
	cout << index->rows << "," << index->columns << " " << C << endl;
	C = np::maximum(A,2,2,window,index);
	cout << index->rows << "," << index->columns << " " << C << endl;
	return 0;
}

int test_dot(){

	int rows = 3; // set number of rows
	int columns = 2; // set number of columns
	std::unique_ptr<Matrix> A(new Matrix(rows, columns, true));
	std::unique_ptr<std::vector<double> > v = std::make_unique<std::vector<double> >();
	v->push_back(1);
	v->push_back(1);
	
	std::unique_ptr<std::vector<double> > v1 = np::dot(A,v);

	A->pretty_print();
	cout << endl;
	for(unsigned int i=0; i < v1->size(); i++){
		cout << v1->at(i) << " ";
	}
	cout << endl;
	return 0;
}

int main(){
	test_applyFunction();	
	test_flatten();
	test_multiply();
	test_maximum();
	test_dot();
}
