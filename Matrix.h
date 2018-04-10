#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

class Matrix
{
public:
	Matrix();
	Matrix(int rows, int columns, bool init=false);
	Matrix(std::vector<std::vector<double> > const &array);
	void set(int row, int column, double value);
	double get(int row, int column);
	int getRows() const {return rows;};
	int getColumns() const {return columns;};
	void pretty_print();
private:
	std::vector<std::vector<double> > array;
	int rows;
	int columns;
};

#endif
