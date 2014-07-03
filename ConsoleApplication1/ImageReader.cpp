/*
Class	:	Tester
Author	:	Lattasit Haritaworn
Date	:	2014/06/27
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat readImage(string path)
{
	Mat image;
	image = imread(path);
	return image;
}