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

/*Create new windows displaying image of a given matrix.*/
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

/*Load an image from a given path and return with a matrix of the image.*/
Mat readImage(string path, int format)
{
	Mat image;
	try
	{
	image = imread(path, format);
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
	return image;
}

void printMatrix(Mat matrix)
{
	  
	cout << matrix << endl;
}


Mat convertMatToVector(Mat matrix)
{
	Mat row = matrix.t();
	row = row.reshape(1, 1);
	return row;
}

/*Load image(s) specified by given filePath in form of vectors.*/
Mat loadBatchImages(string filePath)
{
	string line;
	Mat coAll, tmp;
	try
	{
		ifstream fileStream;
		
		coAll = Mat::ones(224, 1, CV_8U); //Create a column of 1 for using with concat.
		int imgNum = 0;

		fileStream.open(filePath, std::ifstream::in);
		if (fileStream.is_open())
		{
			while (getline(fileStream, line)){
				hconcat(coAll, convertMatToVector(readImage(line, 0)).t(), coAll);
			}
			coAll = coAll.colRange(1, coAll.cols); //Remove first column vector.
		}
		else{
			cout << "ERROR OPENING FILES";
		}
		fileStream.close();
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
	return coAll;
	
}

/*OBSOLETE return mean vector column of given matrix*/
Mat findMeanVector(Mat matrix)
{

	Mat mean;
	reduce(matrix, mean, 1, CV_REDUCE_AVG, -1); reduce(matrix, mean, 1, CV_REDUCE_AVG, -1);
	return mean;

}

/*Write a given image to a file in the given path*/
void writeMatToFile(Mat matrix,string filePath)
{
	try
	{
		ofstream fileStream;
		fileStream.open(filePath);
		fileStream << matrix;
		fileStream.close();

	}
	catch (int e)
	{
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
}


int main(int argc, char** argv)
{	
	string dir = "C:\\faceimg\\01\\DIR.txt";
	Mat face01 = loadBatchImages(dir);
	
	writeMatToFile(face01, "C:\\faceimg\\01\\FACE01.txt");
	
	
	Mat a = readImage("C:\\faceimg\\01\\31.pgm",0);
	imwrite("C:\\faceimg\\01\\a.pgm", a);
	Mat av = convertMatToVector(a);
	imwrite("C:\\faceimg\\01\\av.pgm", av);

	Mat covar01,mean01,eigenvec01,eigenval01,nCovar01;
	calcCovarMatrix(face01, covar01, mean01, CV_COVAR_COLS, CV_64F);
	eigen(covar01, eigenval01,eigenvec01, -1, -1);
	normalize(covar01, nCovar01, 1, 0, NORM_MINMAX, -1, Mat());
	writeMatToFile(nCovar01, "C:\\faceimg\\01\\NCOVAR01.txt");
	writeMatToFile(covar01, "C:\\faceimg\\01\\COVAR01.txt");
	writeMatToFile(eigenval01, "C:\\faceimg\\01\\EIGENVAL01.txt");
	writeMatToFile(eigenvec01, "C:\\faceimg\\01\\EIGENVEC01.txt");
	
	int numOfComponent = 64;
	PCA pca(face01, noArray(), CV_PCA_DATA_AS_COL, numOfComponent);
	Mat pcaEigenval01 = pca.eigenvalues.clone();
	Mat pcaEigenvec01 = pca.eigenvectors.clone();
	writeMatToFile(pcaEigenval01, "C:\\faceimg\\01\\PCAEIGENVAL01.txt");
	writeMatToFile(pcaEigenvec01, "C:\\faceimg\\01\\PCAEIGENVEC01.txt");
	return(0);
}

