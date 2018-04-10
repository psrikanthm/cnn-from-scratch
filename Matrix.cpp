#include "Matrix.h"
#include "Util.h"

#include <iostream>
#include <assert.h>
#include <vector>
#include <random>

using namespace std;

Matrix::Matrix(){}

Matrix::Matrix(int rows, int columns, bool init){
	/*	If init=false, initialize with 0 matrix
	 *	If init=true, initialize with random matrix
	 */
	this -> rows = rows;
	this -> columns = columns;
	this -> array = vector<vector <double> >(rows, vector<double>(columns));
	
	int factor = 0;
	if (init) factor=1;
	// initialize Matrix using random numbers
	std::uniform_real_distribution<double> unif(-1,1); // uniform distribution lower and upper bound
	for(unsigned int i = 0; i < rows; i++){
		for(unsigned int j = 0; j < columns; j++){
			double a = unif(random_engine);
			this -> array[i][j] = (a * factor);	
		}
	}
}

Matrix::Matrix(vector<vector <double> > const &array){
	assert(array.size() != 0);
	this -> rows = array.size();
	this -> columns = array[0].size();
	this -> array = array;
}

void Matrix::set(int row, int column, double value){
	if(row < rows || column < columns)	this -> array[row][column] = value;
}

double Matrix::get(int row, int column){
	if(row >= rows || column >= columns) return (double)0;
	return this->array[row][column];
}

/* deprecated - since it is a redundant function
Shape Matrix::getShape(){
	Shape shape = {rows, columns};
	return shape;
}
*/

void Matrix::pretty_print(){
	for(int i = 0; i < array.size(); i++){
		for(int j = 0; j < array[i].size(); j++){
			cout << array[i][j] << " " ;
		}
		cout << endl;
	}
}
