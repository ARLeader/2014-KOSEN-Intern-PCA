/*
	Class	:	Tester
	Author	:	Lattasit Haritaworn
	Date	:	2014/06/27
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace cv;
using namespace std;

int displayImage(Mat image)
{
	if (!image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", image);
	return 0;

}

Mat readImage(string path, int format)
{
	Mat image;
	image = imread(path, format);

	return image;
}

void printMatrix(Mat matrix)
{
	  
	cout << matrix << endl;
}

Mat convertMatToVector(Mat matrix)
{
	Mat row = matrix.reshape(1, 1);
	return row;
}

Mat loadBatchImages(string filePath)
{
	ifstream fileStream;
	string line;
	Mat coAll, tmp;
	coAll = Mat::ones(224, 1, CV_8U); //Row vector of 224,1
	int imgNum = 0;

	fileStream.open(filePath, std::ifstream::in);
	if (fileStream.is_open()) 
	{
		while (getline(fileStream, line)){
			hconcat(coAll, convertMatToVector(readImage(line, 0)).t(), coAll);
		}
		coAll = coAll.colRange(1, coAll.cols);
	}
	else{
		cout << "ERROR OPENING FILES";
	}
	fileStream.close();
	//import from path
	return coAll;
}

Mat findMeanVector(Mat matrix)
{

	Mat mean;
	reduce(matrix, mean, 1, CV_REDUCE_AVG, -1); reduce(matrix, mean, 1, CV_REDUCE_AVG, -1);
	return mean;

}

int main(int argc, char** argv)
{	
	string dir = "C:\\faceimg\\01\\DIR.txt";
	Mat face01 = loadBatchImages(dir);
	
	
	ofstream fileStream;
	fileStream.open("C:\\faceimg\\01\\FACE01.txt");
	fileStream << face01;


	Mat covar01,mean01,eigen01;
	mean01 =(face01);
	calcCovarMatrix(face01, covar01, mean01, CV_COVAR_COLS, CV_64F);
	eigen(covar01, eigen01, -1, -1);
	waitKey(0);
	return(0);
}

