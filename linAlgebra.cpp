#include <vector>
#include <assert.h>
#include <cmath>
#include <float.h>

#include "Matrix.h"
#include "linAlgebra.h"

using namespace std;

namespace np{

	// scalar multiplication
	std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix> & m1, double value){
		std::unique_ptr<Matrix> m3(new Matrix(m1->getRows(), m1->getColumns(), false));
		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				m3->set(i,j,m1->get(i,j) * value);
			}
		}
		return m3;
	}

	// hadamard product of matrices
	std::unique_ptr<Matrix> multiply(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2,
										unsigned int slice){
		assert(m1->getRows() == m2->getRows() && m1->getColumns() == m2->getColumns() - slice);

		std::unique_ptr<Matrix> m3(new Matrix(m2->getRows(), m2->getColumns() - slice, true));
		for(unsigned int i=0; i < m3->getRows(); i++){
			for(unsigned int j=0; j < m3->getColumns(); j++){
				m3->set(i,j,m1->get(i,j) * m2->get(i,j));
			}
		}
		return m3;
	}
	
	// hadamard product of vectors
	std::unique_ptr<std::vector<double> > multiply(std::unique_ptr<std::vector<double> > &v1, 
				std::unique_ptr<std::vector<double> > &v2){
		assert(v1->size() == v2->size());
		std::unique_ptr<std::vector<double> > vr = std::make_unique<std::vector<double> >();
		for(unsigned int i = 0; i < v1->size(); i++){
			vr->push_back((v1->at(i) * v2->at(i)));
		}
		return vr;
	}

	// take dot product and sum all elements
	double multiply(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2,
					unsigned int xslice, unsigned int yslice){
		assert(m2->getRows() >= m1->getRows() && m2->getColumns() >= m1->getColumns());
		double accumulator = 0;
		for(unsigned int i = 0; i < m1->getRows(); i++){
			for(unsigned int j = 0; j < m1->getColumns(); j++){
				accumulator += (m1->get(i,j) * m2->get(xslice + i, yslice + j));
			}
		}
		return accumulator;
	}

	// dot product between two matrices
	std::unique_ptr<Matrix> dot(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2, 
								unsigned int slice){
		assert(m1->getColumns() == m2->getRows());

		std::unique_ptr<Matrix> m3(new Matrix(m1->getRows(), m2->getColumns() - slice, false));
		for(unsigned int i=0; i < m3->getRows(); i++){
			for(unsigned int j=0; j < m3->getColumns(); j++){
				double w = 0;
				for(unsigned int k=0; k < m1->getColumns(); k++){
					w += (m1->get(i,k) * m2->get(k,j));
				}
				m3->set(i,j,w);
			}
		}
		return m3;
	}

	// dot product with a vetor
	std::unique_ptr<std::vector<double> > dot(std::unique_ptr<Matrix> & m1,
					std::unique_ptr<std::vector<double> > & v, unsigned int v_slice){
		assert(m1->getColumns() == v->size() - v_slice);
		std::unique_ptr<std::vector<double> > vr = std::make_unique<std::vector<double> >();
		for(unsigned int i=0; i < m1->getRows(); i++){
			double w = 0;
			for(unsigned int j=0; j < m1->getColumns() - v_slice; j++){
				w += (m1->get(i,j) * v->at(j));
			}
			vr->push_back(w);
		}
		return vr;
	}
	
	// dot product of 2 vectors returning a Rank 1 Matrix
	std::unique_ptr<Matrix> dot(std::unique_ptr<std::vector<double> > & v1, 
								std::unique_ptr<std::vector<double> > & v2, unsigned int v2_slice){
		std::unique_ptr<Matrix> m3(new Matrix(v1->size(), v2->size() - v2_slice, true));
		for(unsigned int i=0; i < m3->getRows(); i++){
			for(unsigned int j=0; j< m3->getColumns(); j++){
				m3->set(i,j,(v1->at(i) * v2->at(j)));
			}
		}
		return m3;
	}

	// addition
	std::unique_ptr<Matrix> add(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2){
		assert(m1->getRows() == m2->getRows() && m1->getColumns() == m2->getColumns());

		std::unique_ptr<Matrix> m3(new Matrix(m1->getRows(), m1->getColumns(), true));
		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				m3->set(i,j,m1->get(i,j) + m2->get(i,j));
			}
		}
		return m3;
	}

	// subtraction (of matrices)
	std::unique_ptr<Matrix> subtract(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2){
		assert(m1->getRows() == m2->getRows() && m1->getColumns() == m2->getColumns());

		std::unique_ptr<Matrix> m3(new Matrix(m1->getRows(), m1->getColumns(), true));
		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				m3->set(i,j,m1->get(i,j) - m2->get(i,j));
			}
		}
		return m3;
	}

	// subtraction (of vectors)
	std::unique_ptr<std::vector<double> > subtract(std::unique_ptr<std::vector<double> > & v1, 
														std::unique_ptr<std::vector<double> > & v2){
		assert(v1->size() == v2->size());

		std::unique_ptr<std::vector<double> > vr = std::make_unique<std::vector<double> >();
		for(unsigned int i=0; i < v1->size(); i++){
			vr->push_back(v1->at(i) - v2->at(i));
		}
		return vr;
	}

	// transpose
	std::unique_ptr<Matrix> transpose(std::unique_ptr<Matrix> & m1){
		std::unique_ptr<Matrix> m3(new Matrix(m1->getColumns(), m1->getRows(), false));
		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				m3->set(j,i,m1->get(i,j));
			}
		}
		return m3;
	}

	// apply a function to every element of the matrix
	std::unique_ptr<Matrix> applyFunction(std::unique_ptr<Matrix> & m1, double (*active_fn)(double)){
		std::unique_ptr<Matrix> m3(new Matrix(m1->getRows(), m1->getColumns(), false));
		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				double ret = (*active_fn)(m1->get(i,j));
				if(isnan(ret))	m3->set(i,j,0);
				else	m3->set(i,j,ret);
			}
		}
		return m3;
	}
	
	// apply a function to every element of the vector 
	std::unique_ptr<std::vector<double> > applyFunction(std::unique_ptr<std::vector<double> > &v, 
								double (*active_fn)(double)){
		std::unique_ptr<std::vector<double> > vr = std::make_unique<std::vector<double> >();
		for(unsigned int i=0; i < v->size(); i++){
			if(!isnan(v->at(i))){ 
				double ret = (*active_fn)(v->at(i));
				if(isnan(ret))	vr->push_back(0);
				else	vr->push_back(ret);
			}else {
				vr->push_back(0);
			}
		}
		return vr;
	}

	// concatenate matrices 
	std::unique_ptr<Matrix> concatenate(std::unique_ptr<Matrix> & m1, std::unique_ptr<Matrix> & m2){
		assert(m1->getRows() == m2->getRows());

		std::unique_ptr<Matrix> m3(new Matrix(m1->getRows(), m1->getColumns() + m2->getColumns(), true));
		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				m3->set(i,j,m1->get(i,j));
			}
			for(unsigned int j=m1->getColumns(); j < m1->getColumns() + m2->getColumns(); j++){
				m3->set(i,j,m2->get(i,j - m1->getColumns()));
			}
		}
		return m3;
	}
	
	// concatenate matrix with a vector as additional column
	std::unique_ptr<Matrix> concatenate(std::unique_ptr<Matrix> & m1, std::vector<double> & v){
		assert(m1->getRows() == v.size());

		std::unique_ptr<Matrix> m3(new Matrix(m1->getRows(), m1->getColumns() + 1, true));
		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				m3->set(i,j,m1->get(i,j));
			}
			m3->set(i, m1->getColumns(), v[i]);
		}
		return m3;
	}
	
	// normalize a matrix such that sum of each row is 1
	std::unique_ptr<Matrix> normalize(std::unique_ptr<Matrix> & m1){
		std::unique_ptr<Matrix> m3(new Matrix(m1->getRows(), m1->getColumns(), true));

		for(unsigned int i=0; i < m1->getRows(); i++){
			double sum = 0;
			for(unsigned int j=0; j < m1->getColumns(); j++){
				sum += m1->get(i,j);
			}
			for(unsigned int j=0; j < m1->getColumns(); j++){
				m3->set(i,j,(m1->get(i,j)/sum));
			}
		}
		return m3;
	}
	
	// normalize a vector such that sum of all elements is 1
	std::unique_ptr<std::vector<double> > normalize(std::unique_ptr<std::vector<double> > &v){
		std::unique_ptr<std::vector<double> > vr = std::make_unique<std::vector<double> >();
		double sum = 0;
		for(unsigned int i=0; i < v->size(); i++){
			sum += v->at(i);
		}
		assert(sum != 0);
		for(unsigned int i=0; i < v->size(); i++){
			vr->push_back(v->at(i)/sum);
		}
		return vr;
	}
	
	// return sum of all elements in matrix
	double element_sum(std::unique_ptr<Matrix> & m1){
		double sum = 0;
		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				sum += m1->get(i,j);
			}
		}
		return sum;
	}

	//	return sum of all elements in a vector
	double element_sum(std::unique_ptr<std::vector<double> > &v){
		double sum = 0;
		for(unsigned int i=0; i < v->size(); i++){
			sum += v->at(i);
		}
		return sum;	
	}
	
	// flatten the matrix. convert 2D matrix to 1D vector
	std::unique_ptr<std::vector<double> > flatten(std::unique_ptr<Matrix> & m1){
		std::unique_ptr<std::vector<double> > v(new std::vector<double>(m1->getRows() * m1->getColumns()));

		for(unsigned int i=0; i < m1->getRows(); i++){
			for(unsigned int j=0; j < m1->getColumns(); j++){
				v->at((i*m1->getColumns()) + j) = m1->get(i,j);
			}
		}
		return v;
	}
	
	// return the maximum of matrix within the boundaries specified by (xptr, yptr, window)
	// set the index of maximum element in index variable
	double maximum(std::unique_ptr<Matrix> & m1, unsigned int xptr, unsigned int yptr, 
					Shape window, std::unique_ptr<Shape> &index){
		assert(xptr + window.rows <= m1->getRows() && yptr + window.columns <= m1->getColumns());

		double max=-DBL_MAX;
		unsigned int i=xptr;
		while(i-xptr < window.rows && i < m1->getRows()){
			unsigned int j=yptr;
			while(j-yptr < window.columns && j < m1->getColumns()){
				if(m1->get(i,j) > max){
					max = m1->get(i,j);
					index->rows = i;
					index->columns = j;
				}	
				j++;
			}
			i++;
		}
		return max;
	}

	// reshape a vector into a matrix
	std::unique_ptr<Matrix> reshape(std::unique_ptr<std::vector<double> > &v, Shape shape){
		assert((shape.rows * shape.columns) == v->size());
		std::unique_ptr<Matrix> m3(new Matrix(shape.rows, shape.columns, true));
		for(unsigned int i=0; i < shape.rows; i++){
			for(unsigned int j=0; j < shape.columns; j++){
				m3->set(i,j,v->at((i*shape.rows) + j));
			}
		}
		return m3;
	}
}
