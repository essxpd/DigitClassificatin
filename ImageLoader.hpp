#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "FloatMatrix.hpp"

using namespace std;

class ImageLoader
{
public:
	ImageLoader();
	
	//bool loadImages(vector<vector<float>>&, string, int& rows, int& cols);
	bool loadImages(FloatMatrix&, string);
	bool loadImages(vector<float>&, string);
	bool loadLabels(vector<float>&, string);
private:
};
