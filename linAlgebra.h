#ifndef LINALGEBRA_H
#define LINALGEBRA_H

#include "Matrix.h"
#include "Util.h"

#include <random>

namespace np{
	// scalar multiplication
	std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix> & m1, double value);
	// hadamard product of matrices
	std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2, 
								unsigned int slice=0);

	// hadamard product of vectors
	std::unique_ptr<std::vector<double> > multiply(std::unique_ptr<std::vector<double> > &v1,
						std::unique_ptr<std::vector<double> > &v2);

	// take dot product and sum all elements
	double multiply(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2,
					unsigned int xslice, unsigned int yslice);

	// dot product of 2 Matrices
	std::unique_ptr<Matrix> dot(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2,
								unsigned int slice=0);
	
	// dot product of Matrix and a vector
	std::unique_ptr<std::vector<double> > dot(std::unique_ptr<Matrix> & m1, 
								std::unique_ptr<std::vector<double> > & v, unsigned int v_slice=0);

	// dot product of 2 vectors
	std::unique_ptr<Matrix> dot(std::unique_ptr<std::vector<double> > & v1, 
								std::unique_ptr<std::vector<double> > & v2, unsigned int v2_slice=0);

	// addition
	std::unique_ptr<Matrix> add(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2);

	// subtraction (matrices)
	std::unique_ptr<Matrix> subtract(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2);
	
	// subtraction (vectors)
	std::unique_ptr<std::vector<double> > subtract(std::unique_ptr<std::vector<double> > & v1, 
							std::unique_ptr<std::vector<double> > & v2);

	// transpose
	std::unique_ptr<Matrix> transpose(std::unique_ptr<Matrix> & m1);

	// apply a function to every element of the matrix
	std::unique_ptr<Matrix> applyFunction(std::unique_ptr<Matrix> & m1, double (*function)(double));
	
	// apply a function to every element of the vector 
	std::unique_ptr<std::vector<double> > applyFunction(std::unique_ptr<std::vector<double> > &v, 
								double (*function)(double));

	// concatenate with another matrix
	std::unique_ptr<Matrix> concatenate(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2);

	// concatenate matrix with a vector
	std::unique_ptr<Matrix> concatenate(std::unique_ptr<Matrix> & m1, std::vector<double> & v);

	// normalize a matrix such that sum of each row is 1
	std::unique_ptr<Matrix> normalize(std::unique_ptr<Matrix> & m1);
	
	// normalize a vector such that sum of all elements is 1
	std::unique_ptr<std::vector<double> > normalize(std::unique_ptr<std::vector<double> > &v);

	// return sum of all elements in matrix
	double element_sum(std::unique_ptr<Matrix> & m1);
	
	// return sum of all elements in vector 
	double element_sum(std::unique_ptr<std::vector<double> > &v);

	// flatten the matrix. convert 2D matrix to 1D vector
	std::unique_ptr<std::vector<double> > flatten(std::unique_ptr<Matrix> & m1);
	
	// return the maximum of matrix within the boundaries specified by (xptr, yptr, window)
	// set the index of maximum element in index variable
	double maximum(std::unique_ptr<Matrix> & m1, unsigned int xptr, unsigned int yptr, 
					Shape window, std::unique_ptr<Shape> &index);

	// reshape a vector to matrix
	std::unique_ptr<Matrix> reshape(std::unique_ptr<std::vector<double> > &v, Shape shape);
}

#endif
