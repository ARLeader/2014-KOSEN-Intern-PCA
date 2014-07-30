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
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <conio.h>		// For _kbhit() on Windows
#include <string>

#include <Python.h>
#include <pythonrun.h>

#define VK_ESCAPE 0x1B



using namespace cv;
using namespace std;

/*-----Global variable---------*/
Mat outputGray;						// result picture is here.
string name = "";
string result = "Unknown";

vector<int> label;
Mat faces;
vector<Mat> faceVec;
int nFaces = 0;

// Variable for debugging
int loopCount = 0;
bool doFaceRecog = true;

int MAXIMUM_COMPONENTS_NUMBER = 200; // The amount of PCA components

Size DEFAULT_IMAGE_SIZE(30, 30); // Dafault dimension for faces in the database
string indir = "C:\\camface\\";
string camdir = indir + "captures\\";
string csvdir = indir + "db\\faceDIR.txt";
string pythonScriptPath = indir + "db\\create_csv.py";

string outdir = "C:\\faceout\\";

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
int testFaceRecog(Mat greyImage);
void testFaceRecog();
void runPythonScript(string filePath, string argument);

String convertNum2String(int input);
String convertNum2String(double input);
void recordFaces();
Mat readMatrix(string filePath);
void showLabel(Rect_<int> faces, Mat img, string result);
int findLength(Mat testFace);
double getRange(PCA facePca, Mat testFace);
Mat getProjVec(Mat eigenVecFace, Mat testFace);
void startCamera();

int main(int argc, char** argv)
{	
	runPythonScript(pythonScriptPath, indir);
	loadBatchImages(csvdir, ';');
	recordFaces();
	startCamera();
	
	//testFaceRecog();
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

/*Load image(s) specified by given filePath in form of vectors.*/
void loadBatchImages(string& filePath, char separator = ';')
{
	string line, path, classlabel;
	Mat coAll;
	try
	{
		ifstream fileStream;

		faces = Mat::ones(DEFAULT_IMAGE_SIZE.height * DEFAULT_IMAGE_SIZE.width, 1, CV_8U); // Create a row vector for using with concat.
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
			faces = faces.colRange(1, faces.cols); // Remove the first row vector.
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

/*Write all images to a file in format of a column vector*/
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

/*Write a given Mat to a file in the given path*/
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

/*Predict the input face with the database*/
int testFaceRecog(Mat greyImage)
{
	int result;

	resize(greyImage, greyImage, DEFAULT_IMAGE_SIZE);
	result = findLength(greyImage.reshape(1,1).t());


	cout << "The face has been recognized as Face Number : " << result + 1 << endl;
	return result;
}

/*Report Result of Face Recognition on every faces provided*/
void testFaceRecog()
{
	int result;
	double correctCount = 0;
	Mat recognizerCountMat = Mat::zeros(nFaces, 1, CV_16S), recognizerCorrectMat = Mat::zeros(nFaces, 1, CV_16S);
	vector<int> recognizerCount = recognizerCountMat, recognizerCorrect = recognizerCorrectMat;


	for (int testFaceNum = 0; testFaceNum < faces.cols; testFaceNum++)
	{
		result = findLength(faces.col(testFaceNum));

		cout << "The face no. " << testFaceNum << " has been recognized as Face Number : " << result;
		if (result == label.at(testFaceNum))
		{
			cout << "PONG! ";
			recognizerCorrect[label.at(testFaceNum)]++;
			correctCount++;
		}
		cout << endl;
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

/*Display image label around the detected face*/
void showLabel(Rect_<int> faces, Mat img, string result)
{
	//string box_text = format("Prediction = %d", prediction);
	string box_text = result;
	//position of image
	double pos_x = std::max(faces.x - 10, 0);
	double pos_y = std::max(faces.y - 10, 0);
	// And now put it into the image:
	putText(img, box_text, Point(pos_x, pos_y), FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0, 255, 0), 1.5);
}

/*Predict testFace with faces in database*/
int findLength(Mat testFace)
{
	double sum = 00.00;
	int face = 0;
	for (int i = 0; i<nFaces; i++)
	{
		//cout << faceVec.size() << " " << i << " " << faceVec.at(i).rows << "," << faceVec.at(i).rows << endl;
		PCA facePca(faceVec.at(i), noArray(), CV_PCA_DATA_AS_COL, MAXIMUM_COMPONENTS_NUMBER);
		double range = getRange(facePca, testFace);
		cout << endl<< i + 1 << " " << range << endl;

		if (range > sum)
		{
			sum = range;
			face = i;
		}
	}
	cout << endl;
	return face;
}

/*Return the square sum of projected vectors*/
double getRange(PCA facePca, Mat testFace)
{
	testFace.convertTo(testFace, 5);
	normalize(testFace, testFace);
	loopCount++;
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

/*Return Mat multiplication of projected vector and eigenvecs*/
Mat getProjVec(Mat eigenVecFace, Mat testFace)
{
	eigenVecFace.convertTo(eigenVecFace, CV_64FC1);
	testFace.convertTo(testFace, CV_64FC1);
	return eigenVecFace * testFace;
}

/*Under Development*/
/*Execute python script on selected path*/
void runPythonScript(string filePath,string argument)
{
	Py_Initialize();

	//PyRun_SimpleString("import sys");
	//string app = filePath;
	//PyRun_SimpleString(app.c_str());

	Py_Finalize();
}

/*Initialize and Start video capture device*/
void startCamera()
{
	//setup video capture device and link it to the first capture device
	VideoCapture captureDevice;
	captureDevice.open(0);


	//create the cascade classifier object used for the face detection
	CascadeClassifier face_cas_alt2 = CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml");
	Mat captureFrame, tmp, grayscaleFrame;
	std::vector<Rect> faces;
	char keyPressed = 0;
	bool isFace, newperson = true;
	int picNum = 0;
	String result = "Unknown";
	//create a window to present the results
	namedWindow("outputCapture", WINDOW_AUTOSIZE);
	namedWindow("faceGray", WINDOW_AUTOSIZE);

	//create a loop to capture and find faces
	while (true)
	{
		try{
			//capture a new image frame
			captureDevice >> captureFrame;

			//convert captured image to gray scale and equalize it
			cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
			equalizeHist(grayscaleFrame, grayscaleFrame);

			//create a vector array to store the face found 
			//find faces and store them in the vector array
			face_cas_alt2.detectMultiScale(grayscaleFrame, faces, 1.1, 3, 0, Size(75, 75), Size(300, 300));

			//print the output
			for (int i = 0; i < faces.size(); i++)
			{
				//Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
				//Point pt2(faces[i].x, faces[i].y);
				double percentage = 0.15;
				double wMargin = faces[i].width * percentage, hMargin = faces[i].height * percentage;
				Point pt1((faces[i].x) + (faces[i].width - wMargin), (faces[i].y) + (faces[i].height - hMargin));
				Point pt2(faces[i].x + wMargin, faces[i].y + hMargin);

				// select only <Mat> face image from Vector<Rect>faces
				try{
					//Un-Cropped output
					//tmp = captureFrame;
					//tmp = tmp.colRange(faces[i].x, faces[i].x + faces[i].width);
					//tmp = tmp.rowRange(faces[i].y, faces[i].y + faces[i].height);


					tmp = captureFrame;
					tmp = tmp.colRange((faces[i].x + wMargin), (faces[i].x) + (faces[i].width - wMargin));
					tmp = tmp.rowRange((faces[i].y + hMargin), (faces[i].y) + (faces[i].height - hMargin));

					// tranform to grayScale
					cvtColor(tmp, outputGray, CV_BGR2GRAY);
					equalizeHist(outputGray, outputGray);




				}
				catch (exception ex){}

				//show image
				try {
					imshow("faceGray", outputGray);
					isFace = true;
					cout << "Detection Face: Found!!! \tat x: " << faces[i].x << " and y: " << faces[i].y << '\n';
				}
				catch (exception ex){
					cout << "file not match" << endl;
				}
				//draw a rectangle for all found faces in the vector array on the original image.
				rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
				//draw a label for all found on upper left of the original image.


				//Do face recognition on datected face
				if(doFaceRecog)result = convertNum2String(1 + testFaceRecog(outputGray));
				showLabel(faces[i], captureFrame, result);

			}
			imshow("outputCapture", captureFrame);
			//wait for key
			// ESCAPE key to quit
			// N key to capture a face with filename of the input name for 40 frames.
			// I key to capture a face with filename of the input name indefinitely until press T
			try{
				if (_kbhit())
				{
					keyPressed = _getch();
					if (keyPressed == VK_ESCAPE) {	// Check if the user hit the 'Escape' key
						break;	// Stop processing input.
					}

					cout << keyPressed;

					switch (keyPressed) {
					case 'n':	// Add a new person to the training set.
						if (newperson){
							cout << '\n' << "Enter your name: ";
							cin >> name;
							newperson = false;
						}
						if (name.length() > 1 & isFace == true){
							imwrite(camdir + name + to_string(picNum) + ".pgm", outputGray);
							cout << '\n' << "Writing:" << camdir << picNum << "_" << name << ".pgm" << '\n';
							picNum++;
							isFace = false; //for checking new face
							if (picNum >= 40)
							{
								keyPressed = 't';
							}
						}
						break;
					case 't':	// Start training or stop write picture
						cout << "Recognizing person in the camera ..." << '\n';
						picNum = 0;
						keyPressed = 0;
						newperson = true;
						name.clear();
						break;
					case 'i':	// get picture infinite until press any key
						if (newperson){
							cout << '\n' << "Enter your name: ";
							cin >> name;
							newperson = false;
						}
						if (name.length() > 1 & isFace == true){
							imwrite(camdir + name + to_string(picNum) + ".pgm", outputGray);
							cout << '\n' << "Writing:" << camdir << picNum << "_" << name << ".pgm" << '\n';
							picNum++;
							isFace = false; //for checking new face
						}
						break;
					}
				}

			}
			catch (exception ex){
			}
			// wait for 40 ms
			waitKey(40);

			//destroy result of detection window
			destroyWindow("faceGray");

		}
		catch (exception ex){
			cout << "exception No. UNKNOWN" << '\n';
		}
	}
}
