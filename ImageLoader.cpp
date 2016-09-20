#include "ImageLoader.hpp"

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

ImageLoader::ImageLoader()
{
}

//bool ImageLoader::loadImages(vector<vector<float>>& images, string filePath, int& rows, int& cols)
bool ImageLoader::loadImages(FloatMatrix& images, string filePath)
{
	ifstream file(filePath, ios::binary);
	
	if (!file.is_open())
	{
		cerr << "File failed to open." << endl;
		return false;
	}
	
	// Read magic number
	int magic_number = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = ReverseInt(magic_number);	
	if (magic_number != 2051)
	{
		cerr << "Magic number inncorrect" << endl;
		return false;
	}

	// Read number of images
	int numCases = 0;
	file.read((char*)&numCases, sizeof(numCases));
	numCases = ReverseInt(numCases);
	
	int rows, cols;

	// Read image dimensions
	file.read((char*)&rows, sizeof(rows));
	file.read((char*)&cols, sizeof(cols));
	rows = ReverseInt(rows);
	cols = ReverseInt(cols);
	
	//images = vector<vector<float>>(numCases, vector<float>(rows * cols));
	images = FloatMatrix(rows * cols, numCases);
	float* img_ptr = &(images.data()[0]);

	unsigned char temp = 0;

	for(int i = 0; i < numCases; ++i)
	{
		for(int x = 0; x < cols; ++x)
		{
			for(int y = 0; y < rows; ++y)
			{
				file.read((char*)&temp, sizeof(temp));
				images(y*cols + x, i) = temp;
			}
		}
	}

/*
	for(int i = 0, l = rows * cols; i < l; ++i)
	{
		for(int j = 0; j < numCases; ++j)
		{
			file.read((char*)&temp, sizeof(temp));
			images(i, j) = temp;	
		}
	}
*/
/*
	for(int j = 0; j < numCases; ++j)
	{
		for(int i = 0, l = rows * cols; i < l; ++i)
		{
			file.read((char*)&temp, sizeof(temp));
			images(i, j) = temp;	
		}
	}
*/
/*
	for(int i = 0, l = rows * cols; i < numCases * l; ++i)
	{
		file.read((char*)&temp, sizeof(temp));
		img_ptr[i] = temp;
	}
*/
/*
	for (int i = 0; i < numCases; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			for (int k = 0; k < rows; ++k)
			{
				file.read((char*)&temp, sizeof(temp));
				//images[i][j*cols + k] = (float)temp;
				//images(j*cols + k, i) = (float)temp;
				//images(k*cols + j, i) = (float)temp;
				//images(i,  j*cols + k) = (float)temp;
				//images(j * rows + k, i) = (float)temp;
			}
		}
		
		for(int j = 0, l = rows * cols; j < l; ++j)
		{
			file.read((char*)&temp, sizeof(temp));
			//img_ptr[numCases*l + j] = (float)temp;
			cout
			img_ptr[i * numCases + j] = (float)temp;
		}
	}
*/
/*
	for(int i = 0; i < images.size1() * images.size2(); ++i)
	{
		file.read((char*)&temp, sizeof(temp));
		img_ptr[i] = (float)temp;
	}
*/
	return true;
}


bool ImageLoader::loadImages(vector<float>& images, string filePath)
{
	ifstream file(filePath, ios::binary);
	
	if (!file.is_open())
	{
		cerr << "File failed to open." << endl;
		return false;
	}
	
	// Read magic number
	int magic_number = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = ReverseInt(magic_number);	
	if (magic_number != 2051)
	{
		cerr << "Magic number inncorrect" << endl;
		return false;
	}

	// Read number of images
	int numCases = 0;
	file.read((char*)&numCases, sizeof(numCases));
	numCases = ReverseInt(numCases);
	
	int rows, cols;

	// Read image dimensions
	file.read((char*)&rows, sizeof(rows));
	file.read((char*)&cols, sizeof(cols));
	rows = ReverseInt(rows);
	cols = ReverseInt(cols);
	
	//images = vector<vector<float>>(numCases, vector<float>(rows * cols));
	images = vector<float>(numCases*rows*cols);

	unsigned char temp = 0;

	for(int i = 0; i < images.size(); ++i)
	{
		file.read((char*)&temp, sizeof(temp));
		images[i] = (float)temp;
	}

	return true;
}


bool ImageLoader::loadLabels(vector<float>& labels, string filePath)
{
	ifstream file(filePath, ios::binary);
	
	if (!file.is_open())
	{
		cerr << "File failed to open." << endl;
		return false;
	}

	// Read magic number
	int magic_number = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = ReverseInt(magic_number);
	if (magic_number != 2049)
	{
		cerr << "Magic number inncorrect" << endl;
		return false;
	}

	// Read number of images
	int numCases = 0;
	file.read((char*)&numCases, sizeof(numCases));
	numCases = ReverseInt(numCases);
	
	labels = vector<float>(numCases);
	unsigned char temp = 0;

	for (int i = 0; i < numCases; ++i)
	{
		file.read((char*)&temp, sizeof(temp));
		labels[i] = (float)temp;
	}
	return true;
}

