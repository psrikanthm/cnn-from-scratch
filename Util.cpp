#include "Util.h"
#include "Matrix.h"

#include <random>
#include <time.h>
#include <cmath>
#include <fstream>
#include <float.h>

#include <opencv2/opencv.hpp>
#include <boost/tokenizer.hpp>


using namespace std;

int seed = time(0);
std::default_random_engine random_engine(seed); 

namespace fns{
	double relu(double x){
		if(x > 0) return x;
		else return (double) 0;
	}
	double sigmoid(double x){
		return (1.0/(1.0 + exp(-x)));
	}
	double tan(double x){
		return tanh(x);
	}
	double relu_gradient(double x){
		if(x > 0) return (double) 1;
		else return (double) 0.2;
	}
	double sigmoid_gradient(double x){
		return (x*(1-x));
	}
	double tan_gradient(double x){
		return (1-(x*x));
	}
	double softmax(double x){
		if(isnan(x)) return 0;
		return exp(x);
	}
}

namespace pre_process{
	int process_mnist_images(const char* path, std::vector<std::unique_ptr<Matrix> > &Xtrain, 
		std::vector<std::unique_ptr<std::vector<double> > > &Ytrain, unsigned int nr_images){
		std::string str(path);	// convert char* to string
		const int width = 28;
		const int height = 28;
		const int LABELS = 10;
	
		for(unsigned int i=0; i < LABELS; i++){
			std::vector<cv::String> files;	// vector of strings to store file names
			cv::glob(path + std::to_string(i), files, true);
				// true means recursively read from path
			for(unsigned int k=0; k < (nr_images/LABELS); k++){
				cv::Mat img = cv::imread(files[k]);
				if(img.empty()) continue;	//only proceed further if the file is not empty
				std::unique_ptr<Matrix> image = std::make_unique<Matrix>(width, height, true);
				for(unsigned int h=0; h<height; h++){
					for(unsigned int w=0; w<width; w++){
						image->set(h,w,(double)(img.at<uchar>(h,w)/255.0));
					}
				}
				Xtrain.emplace_back(std::move(image));
				std::unique_ptr<std::vector<double> > vr = std::make_unique<std::vector<double> >(LABELS, 0);				
				(*vr)[i] = 1.0;
				Ytrain.emplace_back(std::move(vr));
			}
		}
		return 0;
	}
	
	int process_mnist_csv(const char* filename, std::vector<std::vector<double> > &Xtrain, 
		std::vector<std::vector<double> > &Ytrain){
		std::string data(filename);
		ifstream in(data.c_str());

		if(!in.is_open()) return 1;

		typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
		std::vector<std::string> svec;
		std::string line;

		while(getline(in, line)){
			Tokenizer tok(line);
			auto it = tok.begin();
			int label = std::stoi(*it);
			std::vector<double> labels(10, 0.0);
			labels[label] = 1.0;


			svec.assign(std::next(it, 1), tok.end());

			std::vector<double> dvec(svec.size());
			std::transform(svec.begin(), svec.end(), dvec.begin(), [](const std::string& val)
			{
				return (std::stod(val)/255); // divide by 255 for normalization, since each pixel is 8 bit
			});

			Xtrain.push_back(dvec);
			Ytrain.push_back(labels);
		}
		cout << "processed the input file" << endl;
		return 0;
	}
	
	void process_image(const char* filename){
		std::vector<double> image;
		cv::Mat img = cv::imread(filename);
		if(img.empty()){
			std::cout << "No Image" << std::endl;
		}
		else{
			if(img.isContinuous()){
				image.assign(img.datastart, img.dataend);
				for(unsigned int j=0; j < image.size(); j++){
					cout << image[j] << " " ;
				}
				cout << endl << image.size();
			}
			else{
				std::cout << "Not Continous !" << std::endl;
			}
		}
	}
}
