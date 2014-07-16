/*
Class	:	OpenCvProject
Author	:	Nat anunta
Edit Num:	12
Method	:	Face detection + keyPress call function + limit loop
Date	:	2014/07/16
Status	:	Extend feature(complete) ,edit filename
*/

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <conio.h>		// For _kbhit() on Windows
#include <string>

#include <iostream>
#include <stdio.h>
#define VK_ESCAPE 0x1B

using namespace std;
using namespace cv;

/*-----Global variable---------*/
Mat outputGray;						// result picture is here.
string name = "";
string result = "Unknown";

/*-----Show Label of face that detect----------*/
void label(Rect_<int> faces, Mat img, string result){
	//string box_text = format("Prediction = %d", prediction);
	string box_text = "Is Face :" + result;
	//position of image
	double pos_x = std::max(faces.x - 10, 0);
	double pos_y = std::max(faces.y - 10, 0);
	// And now put it into the image:
	putText(img, box_text, Point(pos_x, pos_y), FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0, 255, 0), 1.5);
}

int main(int argc, const char** argv)
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
	//create a window to present the results
	namedWindow("outputCapture", WINDOW_AUTOSIZE);
	namedWindow("faceGray", WINDOW_AUTOSIZE);

	//create a loop to capture and find faces
	while (true)
	{
		try{
			//capture a new image frame
			captureDevice >> captureFrame;

			//convert captured image to gray scale and equalize
			cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
			equalizeHist(grayscaleFrame, grayscaleFrame);

			//create a vector array to store the face found //find faces and store them in the vector array
			face_cas_alt2.detectMultiScale(grayscaleFrame, faces, 1.1, 3, 0, Size(75, 75), Size(300, 300));

			//print the output
			for (int i = 0; i < faces.size(); i++)
			{
				Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
				Point pt2(faces[i].x, faces[i].y);

				// select only <Mat> face image from Vector<Rect>faces
				try{
					tmp = captureFrame;
					tmp = tmp.colRange(faces[i].x, faces[i].x + faces[i].width);
					tmp = tmp.rowRange(faces[i].y, faces[i].y + faces[i].height);
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
				label(faces[i], captureFrame, result);
				result = "Unknown";
			}
			imshow("outputCapture", captureFrame);
			//wait for key
			try{
				if (_kbhit()){
					keyPressed = _getch();
					if (keyPressed == VK_ESCAPE) {	// Check if the user hit the 'Escape' key
						break;	// Stop processing input.
					}
				}
				switch (keyPressed) {
				case 'n':	// Add a new person to the training set.
					if (newperson){
						cout << '\n' << "Enter your name: ";
						cin >> name;
						newperson = false;
					}
					if (name.length() > 1 & isFace == true){
						imwrite("DB_new\\" + name + to_string(picNum) + ".pgm", outputGray);
						cout << '\n' << "Writing:" << "DB_new\\" << picNum << "_" << name << ".pgm" << '\n';
						picNum++;
						isFace = false; //for checking new face
						if (picNum >= 40)
						{
							keyPressed = 't';
						}
						/*-----------fuction to add picture here----------------*/
					}
					break;
				case 't':	// Start training or stop write picture
					cout << "Recognizing person in the camera ..." << '\n';
					picNum = 0;
					keyPressed = 0;
					newperson = true;
					name.clear();
					/*-----------fuction to Recognize here----------------*/
					/*-----------fuction to Recognize here----------------*/
					/*-----------fuction to Recognize here----------------*/
					break;
				case 'i':	// get picture infinite until press any key
					if (newperson){
						cout << '\n' << "Enter your name: ";
						cin >> name;
						newperson = false;
					}
					if (name.length() > 1 & isFace == true){
						imwrite("DB_new\\" + name + to_string(picNum) + ".pgm", outputGray);
						cout << '\n' << "Writing:" << "DB_new\\" << picNum << "_" << name << ".pgm" << '\n';
						picNum++;
						isFace = false; //for checking new face
						/*-----------fuction to add picture here----------------*/
					}
					break;
				}
			}
			catch (exception ex){
			}
			// wait for 40 ms
			waitKey(40);

			//destroy result of detection window
			destroyWindow("faceGray");

			//return outputGray; 
		}
		catch (exception ex){
			cout << "exception No. UNKNOWN" << '\n';
		}
	}
	return 0;
}
