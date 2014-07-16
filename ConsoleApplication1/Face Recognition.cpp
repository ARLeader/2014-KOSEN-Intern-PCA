#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <string>
#include <math.h>
using namespace std;
using namespace cv;


//--------------------------------------------------------
//--------- Find EigenVector and EigenValue---------------
//--------------------------------------------------------

Mat eigenVec(Mat covariance){
	Mat eigenVec, eigenVal;
	eigen(covariance, eigenVal, eigenVec, -1, -1);
	return eigenVec;
}

Mat eigenVal(Mat covariance){
	Mat eigenVal, eigenVec;
	eigen(covariance, eigenVal, eigenVec, -1, -1);
	return eigenVal;
}

//---------------------------------------------------
//---------------  Integer to String  ---------------
//---------------------------------------------------

String convertNum2String(int input){
	ostringstream ssc;
	ssc << input;
	return ssc.str();
}

//---------------------------------------------------
//--------------Write out a Cov to File--------------
//---------------------------------------------------

void writeMat2File(Mat cov, int i, string check){
	string str1;
	if(check == "Covariance"){
		if(i < 10)
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Covariance Matrix\\FaceCovariance0";
		else
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Covariance Matrix\\FaceCovariance";
	}
	else{
		if(i < 10)
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Mean\\FaceMean0";
		else
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Mean\\FaceMean";
	}
	string str2 = convertNum2String(i);
	string str3 = ".csv"; 
	string str4 = str1 + str2 + str3;
	FileStorage fs(str4, FileStorage::WRITE);
	fs << "input" << cov;
	fs.release();

}

//---------------------------------------------------
//--------------------Get matrix---------------------
//---------------------------------------------------

Mat readMatrix(string dirF){
	Mat face;
	FileStorage fs(dirF, FileStorage::READ);
	
	fs["input"] >> face;

	fs.release();
	//coV(face);
	return face;
}

//---------------------------------------------------
//---------------Write out Face Matrix---------------
//---------------------------------------------------

void writeMatrixForFace(Mat input, int face){
	string str1;
	if(face < 10)
		str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Keep Matrix of Face\\Face0";
	else
		str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Keep Matrix of Face\\Face";
	string str2 = convertNum2String(face);
	string str3 = ".csv"; 
	string str4 = str1 + str2 + str3;
	FileStorage fs(str4, FileStorage::WRITE);
	fs << "input" << input;
	fs.release();
}

//---------------------------------------------------
//---------------Write covariance Matr---------------
//---------------------------------------------------

void writeCovMat(){
	string dirIm, str1, str2 ,str3;
	Mat getFace, cov, mu;
	//Where i is number of face 
	for(int i=1;i<=31;i++){
		if(i <10)
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Keep Matrix of Face\\Face0";
		else
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Keep Matrix of Face\\Face";
		str2 = convertNum2String(i);
		str3 = ".csv";
		dirIm = str1 + str2 + str3;
		getFace = readMatrix(dirIm);

		//---------------------------------------------------
		//---------------Get Covariance Matrix---------------
		//---------------------------------------------------

		calcCovarMatrix(getFace, cov, mu, CV_COVAR_SCALE | CV_COVAR_COLS| CV_COVAR_NORMAL);
		Mat getCov, eigenValue, eigenVector;
		eigenValue = eigenVal(cov);
		eigenVector = eigenVec(cov);
		
		if(i < 10)
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\EigenVector\\FaceEigenVec0";
		else
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\EigenVector\\FaceEigenVec";
		str2 = convertNum2String(i);
		str3 = ".csv"; 
		string str4 = str1 + str2 + str3;
		FileStorage fs(str4, FileStorage::WRITE);
		fs << "input" << eigenVector;
		fs.release();

		
		if(i < 10)
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\EigenValue\\FaceEigenVal0";
		else
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\EigenValue\\FaceEigenVal";
		str2 = convertNum2String(i);
		str3 = ".csv"; 
		str4 = str1 + str2 + str3;
		FileStorage fss(str4, FileStorage::WRITE);
		fss << "input" << eigenValue;
		fss.release();

		writeMat2File(cov, i, "Covariance");
		writeMat2File(mu, i, "Mean");
	}
}

//---------------------------------------------------
//-------Get all data and save to CSV file-----------
//---------------------------------------------------

void getAllData(){
	Mat a;
	// Where y is person and x is type of type of that person

	for(int y=1;y<=31;y++){
		Mat keepData = Mat::zeros(224,64, CV_8UC1);
		for(int x = 1;x<=64;x++){
			string st1;
			if(y <10)
				st1 = "C:\\Users\\N_ny\\Desktop\\Intership\\16x14\\16x14\\";
			else
				st1 = "C:\\Users\\N_ny\\Desktop\\Intership\\16x14\\16x14\\";
			string str2 = convertNum2String(y);
			string str3;
			if(x < 10)
				str3 = "\\0";
			else 
				str3 = "\\";
			//Convert int to string
			
			string str4 = convertNum2String(x);
			string str5 = ".pgm";

			//Combine directory
			string dStr = st1 + str2 + str3 + str4 + str5;
			
			//Read Image
			Mat image = imread(dStr,0);

			//resize to 1x 16*14 and transpost
			image = image.reshape(1,1);
			image = image.t();
			int check = x-1;
			//Collect all image Data
			for(int k=0;k<224;k++){
				image.col(0).row(k).copyTo(keepData.col(check).row(k));
				//cout << image.col(0).row(k) << ">>>" << keepData.col(check).row(k);
			}
		}
		writeMatrixForFace(keepData,y);
	}
	
}

//---------------------------------------------------
//-------Get all data and save to CSV file-----------
//---------------------------------------------------

void getAllEigen(){
	Mat getCov, eigenValue, eigenVector;
	// where i is a person
	for(int i=1;i<31;i++){
		//cout << i;
		string str1;
		if(i < 10)
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Covariance Matrix\\FaceCovariance0";
		else
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Covariance Matrix\\FaceCovariance";
		string str2 = convertNum2String(i);
		string str3 = ".csv";
		string str4 = str1 + str2 + str3;
		getCov = readMatrix(str4);
		eigenValue = eigenVal(getCov);
		eigenVector = eigenVec(getCov);
		
		if(i < 10)
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\EigenVector\\FaceEigenVec0";
		else
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\EigenVector\\FaceEigenVec";
		str2 = convertNum2String(i);
		str3 = ".csv"; 
		str4 = str1 + str2 + str3;
		FileStorage fs(str4, FileStorage::WRITE);
		fs << "input" << eigenVector;
		fs.release();

		
		if(i < 10)
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\EigenValue\\FaceEigenVal0";
		else
			str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\EigenValue\\FaceEigenVal";
		str2 = convertNum2String(i);
		str3 = ".csv"; 
		str4 = str1 + str2 + str3;
		FileStorage fss(str4, FileStorage::WRITE);
		fss << "input" << eigenValue;
		fss.release();
	}

}

//---------------------------------------------------
//----------Input Image apply with PCA Func----------
//---------------------------------------------------

Mat readSampleImage(string dirImage){

	Mat image = imread(dirImage,0);	
	//resize to 1x 16*14 and transpost
	image = image.reshape(1,1);
	image = image.t();
	return image;
}

//---------------------------------------------------
//--------Get range of the Projection vector---------
//---------------------------------------------------

float getRangeProj(Mat projVec){
	Size projVecS = projVec.size();
	float sumAll = 0.0, keepVal = 0.00;
	int z=0;
	
	for(int i=0;i<projVecS.height;i++){
		keepVal = projVec.at<double>(i,0);
		
		sumAll += (keepVal*keepVal);
	}
	
	return sqrt(sumAll);
}


//---------------------------------------------------
//-----------Get Coefficient of vector---------------
//---------------------------------------------------
Mat getProjVec(Mat eigenVecFace, Mat inputFace){
	
	Size eVFS = eigenVecFace.size();
	Size inputFaceS = inputFace.size();
	Mat keepDotVal(eVFS.height,inputFaceS.width,CV_64F);
	Size a = keepDotVal.size();
	
	for(int i=0;i<eVFS.height;i++){
		float collect = 0.00;
		for(int j=0;j<eVFS.width;j++){
			collect += eigenVecFace.at<float>(i,j)*inputFace.at<float>(j,0);
		}
		keepDotVal.col(0).row(i) = collect;
	}
	return keepDotVal;
}


//---------------------------------------------------
//---------Using a PCA to find range of PJG----------
//---------------------------------------------------

void findLength(){
	Mat keepValue;
	int component=0, count = 0;
	// where j is number of face to test and i is a number of person of subspace !!!!
	for(int j=1;j<33;j++){
		float sum=00.00;
		int face = 0 ;
		for(int i=1;i<32;i++){
			//if(i == 1 || i == 3 || i == 2 || i == 8 || i ==24)continue;
			Mat keepFace(224,1,CV_32F), testFace,hoge(224,1,CV_64F),fuga(224,1,CV_64F);
			string str1,str2,str3,str4;
			if(i < 10)
				str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Keep Matrix of Face\\Face0";
			else
				str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Keep Matrix of Face\\Face";
			str2 = convertNum2String(i);
			str3 = ".csv";
			str4 = str1 + str2 + str3;
			keepFace = readMatrix(str4);
			//Size sss = keepFace.size();
			//cout << "Train Face :: " << "height:" <<sss.height << "width:@" << sss.width << endl;
			
			// Read Sample image
			if(j < 10)
				str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Test Image\\0";
			else
				str1 = "C:\\Users\\N_ny\\Desktop\\Intership\\Test Image\\";
			str2 = convertNum2String(j);
			str3 = ".pgm";
			str4 = str1 + str2 + str3;

			testFace = readSampleImage(str4);
			//cout << keepFace.size();
			PCA facePca(keepFace, noArray(), CV_PCA_DATA_AS_COL, 0);
			
			//Normallize the eigenvector !!!
			
			testFace.convertTo(fuga, 5);
			//cout << "Fuga >>>>> " << fuga;
			normalize(fuga,hoge);
			//cout << "Hoge >>>>> " << hoge;
			Mat coef = getProjVec(facePca.eigenvectors,hoge);
			//cout << coef << endl;
			//cout << "Project >> " << x;
			float range= getRangeProj(coef);
			//cout << endl << "Range of Projection vector ::  " << range << endl;
			if(sum < range){
				cout << "change " << sum << "->" << range << "(" << face << "->"<< i << ")" << endl;
				sum = range;
				//cout << endl << "Show sum : " << sum << endl;
				face = i;
			}
		}
		cout << endl << "Show Final recognize Face :: :: :: " << face << "  Face number :: " << j << " Max :::" << sum << endl;
		if(face == j)
			count += 1;
	}
	cout << endl << "Finally >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> "<< count;
}

//---------------------------------------------------
//-------          Main Function          -----------
//---------------------------------------------------

int main()
{
	int n;
	Mat imageF, cov, mu, gray, eigenVal, eigenVec;

	// Try to read and show image
	Mat image = imread("C:\\Users\\N_ny\\Desktop\\Intership\\Opencv Face Database\\att_faces\\s1\\1.pgm",0);
	if(! image.data )  // Check for invalid input
       {
              cout <<  "Could not open or find the image" << std::endl ;
              return -1;
       }
	//cout << image;
	//namedWindow( "window", CV_WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "window", image ); // Show our image inside it.

	//----------------------------------------------------- Receive all data back from function
	//getAllData();

	//----------------------------------------------------- Principle Component Analysis
	findLength();

	scanf("%d",&n);
	return 0;
}