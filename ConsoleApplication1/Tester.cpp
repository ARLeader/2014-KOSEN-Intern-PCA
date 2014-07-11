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

int MAXIMUM_COMPONENTS_NUMBER = 3;

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

/*Load image(s) specified by given filePath in form of vectors.*/
Mat loadBatchImages(string filePath)
{
	string line, path, classlabel;
	Mat coAll;
	try
	{
		ifstream fileStream;
		
		coAll = Mat::ones(1, 224, CV_8U); //Create a row of 1 for using with concat.
		int imgNum = 0;

		fileStream.open(filePath, std::ifstream::in);
		if (fileStream.is_open())
		{
			while (getline(fileStream, line)){
				vconcat(coAll, readImage(line, 0).reshape(1,1), coAll);
			}
			coAll = coAll.rowRange(1, coAll.rows); //Remove first row vector.
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

/*Load image(s) specified by given filePath in form of vectors.*/
void loadBatchImages(const string& filePath, Mat& images, Mat& labels, char separator = ';')
{
	string line, path, classlabel;
	Mat coAll;
	try
	{
		ifstream fileStream;
		
		images = Mat::ones(1, 224, CV_8U); //Create a row of 1 for using with concat.
		int imgNum = 0;

		fileStream.open(filePath, std::ifstream::in);
		if (fileStream.is_open())
		{
			string line, path, classlabel;
			while (getline(fileStream, line)) {
				stringstream liness(line);
				getline(liness, path, separator);
				getline(liness, classlabel);
				if (!path.empty() && !classlabel.empty()) {
					vconcat(images, readImage(path, 0).reshape(1,1), images);
					labels.push_back(atoi(classlabel.c_str()));
				}
			}
			images = images.rowRange(1, images.rows); //Remove first row vector.
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

/*Initialize multiple PCA from given mat and labels*/
void initMassPCA(Mat faces, vector<int> label,vector<PCA>& pcas)
{
	if (faces.rows != label.size())
		cout << "Data and Label rows not equal.";
	else
	{
		int labelCount = 1, dataStart = 0, faceCount = 0;
		for (int i = 0; i < label.size(); i++)
		{
			if (label.at(i) != labelCount)
			{
				
				labelCount = label.at(i);
				faceCount++;
				pcas.push_back(PCA(faces.rowRange(dataStart, i), Mat(), CV_PCA_DATA_AS_ROW, MAXIMUM_COMPONENTS_NUMBER));
				dataStart = i;
			}
		}
		pcas.push_back(PCA(faces.rowRange(dataStart, label.size()), Mat(), CV_PCA_DATA_AS_ROW, MAXIMUM_COMPONENTS_NUMBER));
	}

}

int main(int argc, char** argv)
{	
	string dir = "C:\\faceimg\\01\\DIR.txt";
	string csvdir = "C:\\faceimg\\faceDIR.txt";
	string outdir = "C:\\faceout\\";
	//Mat face01 = loadBatchImages(dir);
	Mat label;
	Mat faces;
	loadBatchImages(csvdir, faces, label, ';');
	writeMatToFile(faces, outdir + "faces.txt");
	writeMatToFile(label, outdir + "label.txt");

	vector<PCA> pcaFaces;
	initMassPCA(faces,label,pcaFaces);

	PCA testPCA = pcaFaces.back();
	writeMatToFile(testPCA.eigenvectors, outdir + "PCAEIGEN31.TXT");

	/*Mat covar01,mean01,eigenvec01,eigenval01,nCovar01;
	calcCovarMatrix(face01, covar01, mean01, CV_COVAR_ROWSS | CV_COVAR_NORMAL | CV_COVAR_SCALE, CV_64F);
	eigen(covar01, eigenval01,eigenvec01, -1, -1);
	normalize(covar01, nCovar01, 1, 0, NORM_MINMAX, -1, Mat());
	writeMatToFile(nCovar01, "C:\\faceimg\\01\\NCOVAR01.txt");
	writeMatToFile(covar01, "C:\\faceimg\\01\\COVAR01.txt");
	writeMatToFile(eigenval01, "C:\\faceimg\\01\\EIGENVAL01.txt");
	writeMatToFile(eigenvec01, "C:\\faceimg\\01\\EIGENVEC01.txt");
	*/

	
	/*
	int numOfComponent = 0;
	PCA pca(face01, noArray(), CV_PCA_DATA_AS_COL, numOfComponent);
	Mat pcaEigenval01 = pca.eigenvalues.clone();
	Mat pcaEigenvec01 = pca.eigenvectors.clone();
	writeMatToFile(pcaEigenval01, "C:\\faceimg\\01\\PCAEIGENVAL01.txt");
	writeMatToFile(pcaEigenvec01, "C:\\faceimg\\01\\PCAEIGENVEC01.txt");
	*/
	
	return(0);
	
}

