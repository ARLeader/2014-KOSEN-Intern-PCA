/*
	Class	:	Tester
	Author	:	Lattasit Haritaworn
	Date	:	2014/06/27
*/
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#if defined WIN32 || defined _WIN32
#include <conio.h>		// For _kbhit() on Windows
#include <direct.h>		// For mkdir(path) on Windows
#include <Windows.h>
#define snprintf sprintf_s	// Visual Studio on Windows comes with sprintf_s() instead of snprintf()
#define mkdir _mkdir
#define getch _getch
#else
#include <stdio.h>		// For getchar() on Linux
#include <termios.h>	// For kbhit() on Linux
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>	// For mkdir(path, options) on Linux
#endif


#include <opencv2/core/core.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>


using namespace cv;
using namespace std;

int MAXIMUM_COMPONENTS_NUMBER = 6;

Size DEFAULT_IMAGE_SIZE(30, 30); // Dafault dimension for faces in the database
string csvdir = "C:\\imgface\\faceDIR.txt";
string outdir = "C:\\faceout\\";
vector<int> label;
Mat faces;
vector<Mat> faceVec;
int nFaces = 0;

// Prototype Declarations
int displayImage(Mat image);
Mat readImage(string path, int format);
Mat loadBatchImages(string filePath);
void loadBatchImages(string& filePath, char separator);
void loadSavedPCA(string& filePath, vector<PCA> pcaVector, char separator);
void initMassPCA(Mat faces, vector<int> label, vector<PCA>& pcas, string filePath);
void writeMatToFile(Mat matrix, string filePath);
void initMassPCA(Mat faces, vector<int> label, vector<PCA>& pcas);
int FaceRecog(Mat testImage, vector<PCA> PCAs);
void testFaceRecog();

String convertNum2String(int input);
String convertNum2String(double input);
Mat getProjVec(Mat eigenVecFace, Mat testFace);
int findLength(Mat dbFace, Mat testFace);
double getRange(PCA facePca, Mat testFace);
void recordFaces();
Mat readMatrix(string filePath);

int findLength(Mat testFace)
{
	int component = 0, count = 0;
	double sum = 00.00;
	int face = 0;
	for (int i = 0; i<nFaces; i++)
	{
		//cout << faceVec.size() << " " << i << " " << faceVec.at(i).rows << "," << faceVec.at(i).rows << endl;
		PCA facePca(faceVec.at(i), noArray(), CV_PCA_DATA_AS_COL, 0);
		double range = getRange(facePca, testFace);
		if (range > sum)
		{
			sum = range;
			face = i;
		}
	}
	
	return face;
}

double getRange(PCA facePca, Mat testFace)
{
	testFace.convertTo(testFace, 5);
	normalize(testFace, testFace);
	Mat coef = getProjVec(facePca.eigenvectors, testFace);
	Size projVecS = coef.size();
	double sumAll = 0.0, keepVal = 0.00;
	int z = 0;
	
	for (int i = 0; i<projVecS.height; i++){
		keepVal = coef.at<double>(i, 0);

		sumAll += (keepVal*keepVal);
	}
	return sqrt(sumAll);
}

Mat getProjVec(Mat eigenVecFace, Mat testFace)
{		
	Size eVFS = eigenVecFace.size();
	Size inputFaceS = testFace.size();
	Mat keepDotVal(eVFS.height, inputFaceS.width, CV_64F);
	Size a = keepDotVal.size();
	for (int i = 0; i < eVFS.height; i++)
	{
		double collect = 0.00;
		for (int j = 0; j < eVFS.width; j++)
		{
			if (i >= eigenVecFace.rows || j >= eigenVecFace.cols || j >= testFace.size().height)
				cout << endl << i << " " << j << " " << eigenVecFace.rows << " " << eigenVecFace.cols << " " << testFace.size().height;
			collect += eigenVecFace.at<double>(i, j)*testFace.at<double>(j, 0);
		}
		keepDotVal.col(0).row(i) = collect;
	}
	return keepDotVal;
}


int main(int argc, char** argv)
{	
	loadBatchImages(csvdir, ';');
	recordFaces();
	testFaceRecog();
	cout << "SUCCESS";
	
	return(0);
	
}

/*Return input number as string*/
String convertNum2String(int input){
	ostringstream ssc;
	ssc << input;
	return ssc.str();
}

/*Return input number as string*/
String convertNum2String(double input){
	ostringstream ssc;
	ssc << input;
	return ssc.str();
}


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
		resize(image, image, DEFAULT_IMAGE_SIZE);
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception Nr. " << e << endl;
	}
	return image;
}

/*OBSOLETE*/
/*Load image(s) specified by given filePath in form of vectors.*/
Mat loadBatchImages(string filePath)
{
	string line, path, classlabel;
	Mat coAll;
	try
	{
		ifstream fileStream;
		
		coAll = Mat::ones(1, DEFAULT_IMAGE_SIZE.height * DEFAULT_IMAGE_SIZE.width, CV_8U); // Create a row of 1 for using with concat.
		int imgNum = 0;
		fileStream.open(filePath, std::ifstream::in);
		if (fileStream.is_open())
		{
			while (getline(fileStream, line)){
				vconcat(coAll, readImage(line, CV_LOAD_IMAGE_GRAYSCALE).reshape(1, 1), coAll);
			}
			coAll = coAll.rowRange(1, coAll.rows); // Remove first row vector.
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
void loadBatchImages(string& filePath, char separator = ';')
{
	string line, path, classlabel;
	Mat coAll;
	try
	{
		ifstream fileStream;

		faces = Mat::ones(DEFAULT_IMAGE_SIZE.height * DEFAULT_IMAGE_SIZE.width, 1, CV_8U); // Create a row of 1 for using with concat.
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
					hconcat(faces, readImage(path, CV_LOAD_IMAGE_GRAYSCALE).reshape(1, 1).t(), faces);
					label.push_back(atoi(classlabel.c_str()));
				}
			}
			faces = faces.colRange(1, faces.cols); // Remove first row vector.
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

/*Initialize multiple PCA from given mat and labels*/
void recordFaces()
{
	int labelCount = 1, dataStart = 0, faceCount = 1;
	for (int i = 0; i < label.size(); i++)
	{
		if (label.at(i) != labelCount)
		{
			faceCount++;
			Mat tmpFace = faces.colRange(dataStart, i);
			faceVec.push_back(tmpFace);
			writeMatToFile(tmpFace, outdir + "facedata\\face" + convertNum2String(labelCount) + ".txt");
			dataStart = i;
			labelCount = label.at(i);
		}
	}
	Mat tmpFace = faces.colRange(dataStart, label.size());
	faceVec.push_back(tmpFace);
	writeMatToFile(tmpFace, outdir + "facedata\\face" + convertNum2String(nFaces) + ".txt");
	nFaces = faceCount;
}

/*UNDER DEVELOPMENT */
/*Load saved eigenvector, eigenvalue, and mean vector from file and assign to PCA.*/
void loadSavedPCA(string& filePath, vector<PCA> pcaVector, char separator = ';')
{
	string line, path;
	try
	{
		ifstream fileStream;
		fileStream.open(filePath, std::ifstream::in);
		if (fileStream.is_open())
		{
			string line, path, classlabel;
			while (getline(fileStream, line)) {
				stringstream liness(line);
				getline(liness, path, separator);
				getline(liness, classlabel);
				if (!path.empty() && !classlabel.empty()) {
					// 			vconcat(images, readImage(path, CV_LOAD_IMAGE_GRAYSCALE).reshape(1, 1), images);
					// 			labels.push_back(atoi(classlabel.c_str()));
				}
			}
			// 	images = images.rowRange(1, images.rows); // Remove first row vector.
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

/*UNDER DEVELOPMENT*/
/*Initialize multiple PCA from given mat and labels*/
void initMassPCA(Mat faces, vector<int> label, vector<PCA>& pcas, string filePath)
{
	if (faces.rows != label.size())
		cout << "Data and Label rows not equal.";
	else
	{
		ofstream fileStream;
		fileStream.open(filePath);

		int labelCount = 1, dataStart = 0, faceCount = 1;
		for (int i = 0; i < label.size(); i++)
		{

			if (label.at(i) != labelCount)
			{
				faceCount++;
				cout << "Initiate PCA of Person No. " << faceCount << endl;

				labelCount = label.at(i);

				pcas.push_back(PCA(faces.rowRange(dataStart, i), Mat(), CV_PCA_DATA_AS_ROW, MAXIMUM_COMPONENTS_NUMBER));
				dataStart = i;


			}
		}
		pcas.push_back(PCA(faces.rowRange(dataStart, label.size()), Mat(), CV_PCA_DATA_AS_ROW, MAXIMUM_COMPONENTS_NUMBER));
	}

}

/*Write a given image to a file in the given path*/
void writeMatToFile(Mat matrix, string filePath)
{
	try
	{
		FileStorage fs(filePath, FileStorage::WRITE);
		fs << "input" << matrix;
		fs.release();
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
}

/*Return matrix of given filePath*/
Mat readMatrix(string filePath){
	Mat face;
	FileStorage fs(filePath, FileStorage::READ);

	fs["input"] >> face;

	fs.release();
	return face;
}


/*Initialize multiple PCA from given mat and labels*/
void initMassPCA(Mat faces, vector<int> label, vector<PCA>& pcas)
{
	if (faces.rows != label.size())
		cout << "Data and Label rows not equal.";
	else
	{
		int labelCount = 1, dataStart = 0, faceCount = 1;
		for (int i = 0; i < label.size(); i++)
		{

			if (label.at(i) != labelCount)
			{
				faceCount++;
				cout << "Initiate PCA of Person No. " << faceCount << endl;

				labelCount = label.at(i);

				pcas.push_back(PCA(faces.rowRange(dataStart, i), Mat(), CV_PCA_DATA_AS_ROW, MAXIMUM_COMPONENTS_NUMBER));
				dataStart = i;

			}
		}
		pcas.push_back(PCA(faces.rowRange(dataStart, label.size()), Mat(), CV_PCA_DATA_AS_ROW, MAXIMUM_COMPONENTS_NUMBER));
	}

}

/*Return result of Face Recognition of a testImage*/
int FaceRecog(Mat testImage, vector<PCA> PCAs)
{
	Mat coeff = Mat::ones(1, MAXIMUM_COMPONENTS_NUMBER, CV_32F);
	Mat coeffSum;
	double coeffMax = 0;
	int PCAcount = 1, coeffMaxIndex, coeffMinIndex;
	for each (PCA pca in PCAs)
	{
		vconcat(coeff, pca.project(testImage), coeff);
		PCAcount++;
	}

	coeff = coeff.rowRange(1, coeff.rows);
	multiply(coeff, coeff, coeffSum);
	reduce(coeffSum, coeffSum, 1, CV_REDUCE_SUM, -1);

	for (int i = 0; i < coeffSum.rows; i++)
	{
		if (coeffSum.at<double>(i) > coeffMax)
		{
			coeffMax = coeffSum.at<double>(i);
			coeffMaxIndex = i;
		}
	}


	return coeffMaxIndex;
}

/*Report Result of Face Recognition on every faces provided*/
void testFaceRecog()
{
	int result;
	double correctCount = 0;
	Mat recognizerCountMat = Mat::zeros(label.size(), 1, CV_16S), recognizerCorrectMat = Mat::zeros(label.size(), 1, CV_16S);
	vector<int> recognizerCount = recognizerCountMat, recognizerCorrect = recognizerCorrectMat;
	for (int testFaceNum = 0; testFaceNum < faces.cols; testFaceNum++)
	{
		result = findLength(faces.col(testFaceNum));
		cout << "The face Number " << testFaceNum << " has been recognized as Face Number : " << result << endl;
		if (result == label.at(testFaceNum))
		{
			cout << "PONG! ";
			recognizerCorrect[label.at(testFaceNum)]++;
			correctCount++;
		}
		recognizerCount[result]++;
	}


	cout << endl << "The Result is Here" << endl;
	int count = 0;
	for (int i = 0; i < recognizerCount.size(); i++)
	{
		cout << "Face Number : " << count << "\tHas the Score of : " << recognizerCount[i] << "\tWith " << recognizerCorrect[i] << " Correct Rate." << endl;
		count++;
	}
	cout << "This program has correctly recognize " << correctCount << " faces" << endl << "Out of " << faces.cols << endl << "Which is calculated as " << (correctCount / faces.cols) * 100 << " Percents." << endl;

}
