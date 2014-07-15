/*
	Class	:	Tester
	Author	:	Lattasit Haritaworn
	Date	:	2014/06/27
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace cv;
using namespace std;

int MAXIMUM_COMPONENTS_NUMBER = 3;
Size DEFAULT_IMAGE_SIZE(100, 100);

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
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
	return image;
}

/*Load image(s) specified by given filePath in form of vectors.*/
Mat loadBatchImages(string filePath)
{
	string line, path, classlabel;
	Mat coAll;
	try
	{
		ifstream fileStream;
		
		coAll = Mat::ones(1, DEFAULT_IMAGE_SIZE.height * DEFAULT_IMAGE_SIZE.width, CV_8U); //Create a row of 1 for using with concat.
		int imgNum = 0;

		fileStream.open(filePath, std::ifstream::in);
		if (fileStream.is_open())
		{
			while (getline(fileStream, line)){
				vconcat(coAll, readImage(line, CV_LOAD_IMAGE_GRAYSCALE).reshape(1, 1), coAll);
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
		
		images = Mat::ones(1, DEFAULT_IMAGE_SIZE.height * DEFAULT_IMAGE_SIZE.width, CV_8U); //Create a row of 1 for using with concat.
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
					vconcat(images, readImage(path, CV_LOAD_IMAGE_GRAYSCALE).reshape(1,1), images);
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


			cout << "Initiate PCA of Person No. " << i << endl;

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

/*Return result of Face Recognition of a testImage*/
int FaceRecog(Mat testImage, vector<PCA> PCAs)
{
	Mat coeff = Mat::ones(1, MAXIMUM_COMPONENTS_NUMBER, CV_32F);
	Mat coeffSum;
	float coeffMax = 0;
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
		if (coeffSum.at<float>(i) > coeffMax)
		{
			coeffMax = coeffSum.at<float>(i);
			coeffMaxIndex = i;
		}
	}


	return coeffMaxIndex;
}

/*Report Result of Face Recognition on every faces provided*/
void testFaceRecog(Mat faces, vector<PCA> pcaFaces, vector<int> label)
{
	int result;
	double correctCount = 0;
	Mat recognizerCountMet = Mat::zeros(pcaFaces.size(), 1, CV_16S), recognizerCorrectMet = Mat::zeros(pcaFaces.size(), 1, CV_16S);
	vector<int> recognizerCount = recognizerCountMet, recognizerCorrect = recognizerCorrectMet;

	for (int testFaceNum = 0; testFaceNum < faces.rows; testFaceNum++)
	{
		result = FaceRecog(faces.row(testFaceNum), pcaFaces);
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
	cout << "This program has correctly recognize " << correctCount << " faces" << endl << "Out of " << faces.rows << endl << "Which is calculated as " << (correctCount / faces.rows) * 100 << " Percents." << endl;

}

int main(int argc, char** argv)
{	
	string csvdir = "C:\\camface\\faceDIR.txt";
	string outdir = "C:\\faceout\\";
	
	Mat label;
	Mat faces;
	loadBatchImages(csvdir, faces, label, ';');
	//writeMatToFile(faces, outdir + "faces.txt");
	//writeMatToFile(label, outdir + "label.txt");

	vector<PCA> pcaFaces;
	initMassPCA(faces,label,pcaFaces);

	testFaceRecog(faces, pcaFaces, label);
	
	return(0);
	
}

