#include "kernel_gpu.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <cv.h>
#include <highgui.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sys/timeb.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string.h>

using namespace cv;
using namespace std;

static void StereoCalib(const char *filename, int nx, int ny, int useUncalibrated, float _squareSize, 
		CvMat *mx1, CvMat *my1, CvMat *mx2, CvMat *my2) {
	int displayCorners = 0;
	bool isVerticalStereo = false;	//OpenCV can handle left-right
								//or up-down camera arrangements
	const int maxScale = 1;
	const float squareSize = _squareSize;	//Chessboard square size in cm
	int i, j, lr, nframes, n = nx * ny, N = 0;
	vector<string> imageNames[2];
	vector<CvPoint3D32f> objectPoints;
	vector<CvPoint2D32f> points[2];
	vector<int> npoints;
	vector<uchar> active[2];
	vector<CvPoint2D32f> temp(n);
	CvSize imageSize = {0, 0};
	
	//ARRAY AND VECTOR STORAGE:
	double M1[3][3], M2[3][3], D1[5], D2[5];
	double R[3][3], T[3], E[3][3], F[3][3];
	double Q[4][4];
    
	CvMat _M1 = cvMat(3, 3, CV_64F, M1);
	CvMat _M2 = cvMat(3, 3, CV_64F, M2);
	CvMat _D1 = cvMat(1, 5, CV_64F, D1);
	CvMat _D2 = cvMat(1, 5, CV_64F, D2);
	CvMat _R = cvMat(3, 3, CV_64F, R);
	CvMat _T = cvMat(3, 1, CV_64F, T);
	CvMat _E = cvMat(3, 3, CV_64F, E);
	CvMat _F = cvMat(3, 3, CV_64F, F);
	CvMat _Q = cvMat(4,4, CV_64F, Q);
    
	if (displayCorners)
		cvNamedWindow("corners", 1);
	
	FILE *f = fopen(filename, "rt");
	if (!f) {
		fprintf(stderr, "can not open file %s\n", filename);
		exit(1);
	}
	
	// READ IN THE LIST OF CHESSBOARDS:
	for (i = 0; ; i++) {
		char buf[1024];
		int count = 0, result = 0;
		lr = i % 2;
		vector<CvPoint2D32f>& pts = points[lr];
		
		if (!fgets(buf, sizeof(buf) - 3, f))
			break;
		
		size_t len = strlen(buf);
		while (len > 0 && isspace(buf[len - 1]))
			buf[--len] = '\0';
		
		if (buf[0] == '#')
			continue;
		
		IplImage *img = cvLoadImage(buf, 0);
		if (!img)
			break;
		
		imageSize = cvGetSize(img);
		imageNames[lr].push_back(buf);
		
		//FIND CHESSBOARDS AND CORNERS THEREIN:
		for (int s = 1; s <= maxScale; s++) {
			IplImage *timg = img;
			
			result = cvFindChessboardCorners(timg, cvSize(nx, ny),
					&temp[0], &count,
					CV_CALIB_CB_ADAPTIVE_THRESH |
					CV_CALIB_CB_NORMALIZE_IMAGE);
			
			if (timg != img)
				cvReleaseImage(&timg);
			
			if (result || s == maxScale)
				for (j = 0; j < count; j++) {
					temp[j].x /= s;
					temp[j].y /= s;
				}
			
			if (result)
				break;
		}
		
		if (displayCorners) {
			printf("%s\n", buf);
			IplImage *cimg = cvCreateImage(imageSize, 8, 3);
			
			cvCvtColor(img, cimg, CV_GRAY2BGR);
			cvDrawChessboardCorners(cimg, cvSize(nx, ny), &temp[0],
					count, result);
			cvShowImage("corners", cimg);
			cvReleaseImage(&cimg);
			
			if (cvWaitKey(0) == 27)	//Allow ESC to quit
				exit(-1);
		} else
			putchar('.');
		
		N = pts.size();
		pts.resize(N + n, cvPoint2D32f(0, 0));
		active[lr].push_back((uchar) result);
		//assert( result != 0 );
		
		if (result) {
			//Calibration will suffer without subpixel interpolation
			cvFindCornerSubPix(img, &temp[0], count,
					cvSize(11, 11), cvSize(-1, -1),
					cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,
					30, 0.01));
			copy(temp.begin(), temp.end(), pts.begin() + N);
		}
		
		cvReleaseImage(&img);
	}
	
	fclose(f);
	printf("\n");
	
	//HARVEST CHESSBOARD 3D OBJECT POINT LIST:
	nframes = active[0].size();	//Number of good chessboads found
	objectPoints.resize(nframes * n);
	
	for (i = 0; i < ny; i++)
		for (j = 0; j < nx; j++)
			objectPoints[i * nx + j] = cvPoint3D32f(i * squareSize, j * squareSize, 0);
	
	for (i = 1; i < nframes; i++)
		copy(objectPoints.begin(), objectPoints.begin() + n,
				objectPoints.begin() + i * n);
	
	npoints.resize(nframes, n);
	N = nframes * n;
	CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0]);
	CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0]);
	CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0]);
	CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0]);
	cvSetIdentity(&_M1);
	cvSetIdentity(&_M2);
	cvZero(&_D1);
	cvZero(&_D2);
	
	//CALIBRATE THE STEREO CAMERAS
	printf("Running stereo calibration ...");
	fflush(stdout);
	cvStereoCalibrate(&_objectPoints, &_imagePoints1,
			&_imagePoints2, &_npoints,
			&_M1, &_D1, &_M2, &_D2,
			imageSize, &_R, &_T, &_E, &_F,
			cvTermCriteria(CV_TERMCRIT_ITER+
			CV_TERMCRIT_EPS, 100, 1e-5),
			CV_CALIB_FIX_ASPECT_RATIO +
			CV_CALIB_ZERO_TANGENT_DIST +
			CV_CALIB_SAME_FOCAL_LENGTH);
	printf(" done\n");
	
	//CALIBRATION QUALITY CHECK
	//because the output fundamental matrix implicitly
	//includes all the output information,
	//we can check the quality of calibration using the
	//epipolar geometry constraint: m2^t*F*m1=0
	vector<CvPoint3D32f> lines[2];
	points[0].resize(N);
	points[1].resize(N);
	_imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0]);
	_imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0]);
	lines[0].resize(N);
	lines[1].resize(N);
	CvMat _L1 = cvMat(1, N, CV_32FC3, &lines[0][0]);
	CvMat _L2 = cvMat(1, N, CV_32FC3, &lines[1][0]);
	
	//Always work in undistorted space
	cvUndistortPoints(&_imagePoints1, &_imagePoints1,
			&_M1, &_D1, 0, &_M1);
	cvUndistortPoints(&_imagePoints2, &_imagePoints2,
			&_M2, &_D2, 0, &_M2);
	cvComputeCorrespondEpilines(&_imagePoints1, 1, &_F, &_L1);
	cvComputeCorrespondEpilines(&_imagePoints2, 2, &_F, &_L2);
	
	double avgErr = 0;
	for (i = 0; i < N; i++) {
		double err = fabs(points[0][i].x * lines[1][i].x
				+ points[0][i].y * lines[1][i].y + lines[1][i].z)
				+ fabs(points[1][i].x * lines[0][i].x
				+ points[1][i].y * lines[0][i].y + lines[0][i].z);
		avgErr += err;
	}
	printf("avg err = %g\n", avgErr/(nframes * n));
	
	//COMPUTE AND DISPLAY RECTIFICATION
	CvMat *pair;
	double R1[3][3], R2[3][3], P1[3][4], P2[3][4];
	CvMat _R1 = cvMat(3, 3, CV_64F, R1);
	CvMat _R2 = cvMat(3, 3, CV_64F, R2);
	
	// IF BY CALIBRATED (BOUGUET'S METHOD)
	if (useUncalibrated == 0) {
		CvMat _P1 = cvMat(3, 4, CV_64F, P1);
		CvMat _P2 = cvMat(3, 4, CV_64F, P2);
		cvStereoRectify(&_M1, &_M2, &_D1, &_D2, imageSize,
				&_R, &_T, &_R1, &_R2, &_P1, &_P2, &_Q,
				0/*CV_CALIB_ZERO_DISPARITY*/);
		
		isVerticalStereo = fabs(P2[1][3]) > fabs(P2[0][3]);
		
		//Precompute maps for cvRemap()
		cvInitUndistortRectifyMap(&_M1, &_D1, &_R1, &_P1, mx1, my1);
		cvInitUndistortRectifyMap(&_M2, &_D2, &_R2, &_P2, mx2, my2);
		
		//Save parameters
		cvSave("M1.xml", &_M1);
		cvSave("D1.xml", &_D1);
		cvSave("R1.xml", &_R1);
		cvSave("P1.xml", &_P1);
		cvSave("M2.xml", &_M2);
		cvSave("D2.xml", &_D2);
		cvSave("R2.xml", &_R2);
		cvSave("P2.xml", &_P2);
		cvSave("Q.xml", &_Q);
		cvSave("mx1.xml", mx1);
		cvSave("my1.xml", my1);
		cvSave("mx2.xml", mx2);
		cvSave("my2.xml", my2);
	} else if (useUncalibrated == 1 || useUncalibrated == 2) {	//OR ELSE HARTLEY'S METHOD
		//use intrinsic parameters of each camera, but
		//compute the rectification transformation directly
		//from the fundamental matrix
		double H1[3][3], H2[3][3], iM[3][3];
		CvMat _H1 = cvMat(3, 3, CV_64F, H1);
		CvMat _H2 = cvMat(3, 3, CV_64F, H2);
		CvMat _iM = cvMat(3, 3, CV_64F, iM);
		
		//Just to show you could have independently used F
		if (useUncalibrated == 2)
			cvFindFundamentalMat(&_imagePoints1,
					&_imagePoints2, &_F);
		
		cvStereoRectifyUncalibrated(&_imagePoints1, &_imagePoints2, &_F,
				imageSize, &_H1, &_H2, 3);
		cvInvert(&_M1, &_iM);
		cvMatMul(&_H1, &_M1, &_R1);
		cvMatMul(&_iM, &_R1, &_R1);
		cvInvert(&_M2, &_iM);
		cvMatMul(&_H2, &_M2, &_R2);
		cvMatMul(&_iM, &_R2, &_R2);
		
		//Precompute map for cvRemap()
		cvInitUndistortRectifyMap(&_M1, &_D1, &_R1, &_M1, mx1, my1);
		cvInitUndistortRectifyMap(&_M2, &_D1, &_R2, &_M2, mx2, my2);
	} else
		assert(0);
}

long long timeUSec() {
	boost::posix_time::ptime now =
			boost::posix_time::microsec_clock::local_time();
	return now.time_of_day().total_microseconds();
}

void loadImages(unsigned char **gDispMap, CvSize *imageSize, boost::posix_time::ptime *start, bool rectify, 
		Mat *leftImg, Mat *rightImg, unsigned char **gLeftRekt, unsigned char **gRightRekt) {
	string leftPath, rightPath;
	
	cout << "Left image: ";
	cin >> leftPath;
	cout << "Right image: ";
	cin >> rightPath;
	cout << endl;
	
	//(*start) = timeUSec();
	(*start) = boost::posix_time::microsec_clock::local_time();
	
	*leftImg = imread(leftPath, CV_LOAD_IMAGE_GRAYSCALE);
	*rightImg = imread(rightPath, CV_LOAD_IMAGE_GRAYSCALE);
	
	if ((*leftImg).cols != (*rightImg).cols || (*leftImg).rows != (*rightImg).rows) {
		cout << "The left and right picture doesn't have the same size." << endl;
		exit(-1);
	}
	
	if ((*leftImg).rows != (*imageSize).height || (*leftImg).cols != (*imageSize).width) {
		if (rectify && (*imageSize).height > 0 && (*imageSize).width > 0) {
			cout << "The pictures need to have the same proportions as the pictures " <<
					" that were used for calibration." << endl;
			exit(-1);
		}
		
		(*imageSize).height = (*leftImg).rows;
		(*imageSize).width = (*leftImg).cols;
		
		cudaFree(*gLeftRekt);
		cudaFree(*gRightRekt);
		
		if (cudaMalloc((void **) gDispMap, (*imageSize).height * (*imageSize).width  
				* sizeof(unsigned char)) != cudaSuccess) {
			cerr << "ERROR: Failed cudaMalloc of disparity map" << endl;
			exit(-1);
		}
		if (cudaMalloc((void **) gLeftRekt, (*imageSize).width * (*imageSize).height 
				* sizeof(unsigned char)) != cudaSuccess) {
			cerr << "ERROR: Failed cudaMalloc" << endl;
			exit(-1);
		}
		if (cudaMalloc((void **) gRightRekt, (*imageSize).width * (*imageSize).height 
				* sizeof(unsigned char)) != cudaSuccess) {
			cerr << "ERROR: Failed cudaMalloc" << endl;
			exit(-1);
		}
	}
}

int main(int argc, char** argv) {
	//long long start, end;
	boost::posix_time::ptime start, end;
	int algo = 0, algonew;
	string output, input;
	CvSize imageSize = {0, 0};
	unsigned char *gDispMap = 0, *dispMap, *gLeftRekt = 0, *gRightRekt = 0;
	int device;
	struct cudaDeviceProp prop;
	bool loadNewImages = true;
	int frame = 3;
	int delta_min = 5;
	int delta_max = 20;
	int steps = 10;
	float borderVal = 125000.0f;
	float medLeft, medRight, stdDevLeft, stdDevRight;
	CvMat *mx1, *my1, *mx2, *my2;
	Mat mapxl, mapyl, mapxr, mapyr, leftImg, rightImg, leftRektMat, rightRektMat;
	int nx, ny;
	float squareSize;
	int fail = 0;
	bool rectify = true, calibrated = false;
	
	//Check command line
	if (argc != 5 && argc != 2) {
		cout << "USAGE: " << argv[0] << " imageList nx ny squareSize" << endl;
		cout << "\t imageList: Filename of the image list (string). Example : list.txt" << endl;
		cout << "\t nx: Number of horizontal squares (int > 0). Example: 9" << endl;
		cout << "\t ny: Number of vertical squares (int > 0). Example: 6" << endl;
		cout << "\t squareSize: Size of a square (float > 0). Example: 2.5" << endl;
		
		return 1;
	} else if (argc == 2) {
		if (strcmp(argv[1], "-no_rect") == 0)
			rectify = false;
		else {
			fail = 1;
			cout << "ERROR: false arguments" << endl;
		}
	} else {
		nx = atoi(argv[2]);
		ny = atoi(argv[3]);
		squareSize = (float) atof(argv[4]);
		
		if (nx <= 0) {
			fail = 1;
			cout << "ERROR: nx value can not be <= 0" << endl;
		}
	
		if (ny <= 0) {
			fail = 1;
			cout << "ERROR: ny value can not be <= 0" << endl;
		}
	}
	
	if (fail != 0)
		return 1;
	
	//Set properties of device for kernel execution
	if (cudaSuccess != cudaGetDevice(&device)) {
		cerr << "ERROR: Failed cudaGetDevice" << endl;
		return -1;
	}
	
	if (cudaSuccess != cudaGetDeviceProperties(&prop, device)) {
		cerr << "ERROR: Failed cudaGetDeviceProperties" << endl;
		return -1;
	}
	
	namedWindow("Disparity Map", CV_WINDOW_AUTOSIZE);
	
	while (true) {
		if (loadNewImages) {
			loadImages(&gDispMap, &imageSize, &start, rectify, 
					&leftImg, &rightImg, &gLeftRekt, &gRightRekt);
			
			if (rectify) {
				if (!calibrated) {
					mx1 = cvCreateMat(imageSize.height, imageSize.width, CV_32F);
					my1 = cvCreateMat(imageSize.height, imageSize.width, CV_32F);
					mx2 = cvCreateMat(imageSize.height, imageSize.width, CV_32F);
					my2 = cvCreateMat(imageSize.height, imageSize.width, CV_32F);
					
					StereoCalib(argv[1], nx, ny, 0, squareSize, mx1, my1, mx2, my2);
					
					mapxl = Mat(mx1);
					mapyl = Mat(my1);
					mapxr = Mat(mx2);
					mapyr = Mat(my2);
					
					calibrated = true;
				}
				
				Scalar_<unsigned char> zero(0);
				leftRektMat = Mat(imageSize.height, imageSize.width, DataType<unsigned char>::type, zero);
				rightRektMat = Mat(imageSize.height, imageSize.width, DataType<unsigned char>::type, zero);
				
				cout << "Rectification ... ";
				remap(leftImg, leftRektMat, mapxl, mapyl, INTER_LINEAR);
				remap(rightImg, rightRektMat, mapxr, mapyr, INTER_LINEAR);
				cout << "done" << endl;
				
				imshow("Disparity Map", leftRektMat);
				waitKey(0);
				imwrite("leftrekt.jpg", leftRektMat);
				
				imshow("Disparity Map", rightRektMat);
				waitKey(0);
				imwrite("rightrekt.jpg", rightRektMat);
				
				if (cudaMemcpy(gLeftRekt, leftRektMat.data, imageSize.width * imageSize.height 
						* sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
					cout << "ERROR: Failed cudaMemcpy" << endl;
					return -1;
				}
				if (cudaMemcpy(gRightRekt, rightRektMat.data, imageSize.width * imageSize.height 
						* sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
					cout << "ERROR: Failed cudaMemcpy" << endl;
					return -1;
				}
			} else {
				if (cudaMemcpy(gLeftRekt, leftImg.data, imageSize.height * imageSize.width  
						* sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
					cout << "ERROR: Failed cudaMemcpy of left image" << endl;
					exit(-1);
				}
				if (cudaMemcpy(gRightRekt, rightImg.data, imageSize.height * imageSize.width  
						* sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
					cout << "ERROR: Failed cudaMemcpy of right image" << endl;
					exit(-1);
				}
			}
		} else
			start = boost::posix_time::microsec_clock::local_time();
			//start = timeUSec();
		
		dim3 blockDim(16, (int) (prop.maxThreadsPerBlock / 32), 1);
		dim3 gridDim((int) ceil(double(imageSize.width) / double(blockDim.x)), 
				(int) ceil(double(imageSize.height) / double(blockDim.y)));
		
		cout << "Block matching ... ";
		cuda_blockmatching((rectify) ? leftRektMat.data : leftImg.data, (rectify) ? rightRektMat.data : rightImg.data, gLeftRekt, 
				gRightRekt, gDispMap, imageSize.width, imageSize.height, gridDim, blockDim, frame, delta_min, delta_max, borderVal, steps, algo);
		cout << "done" << endl;
		
		dispMap = new unsigned char[imageSize.width * imageSize.height];
		memset(dispMap, 0, imageSize.width * imageSize.height * sizeof(unsigned char));
		cudaMemcpy(dispMap, gDispMap, imageSize.width * imageSize.height 
				* sizeof(unsigned char), cudaMemcpyDeviceToHost);
		
		end = boost::posix_time::microsec_clock::local_time();
		boost::posix_time::time_duration msdiff = end - start;
		cout << "Time: " << msdiff.total_microseconds() / 1000.0 << "ms" << endl;
		cout << "FPS: " << 1000000 / msdiff.total_microseconds() << endl << endl;
		
		Mat dispMat = Mat(imageSize.height, imageSize.width, DataType<unsigned char>::type/*CV_8UC1*/);
		dispMat.data = dispMap;
		
		imshow("Disparity Map", dispMat);
		waitKey(0);
		
		loadNewImages = false;
		
		while (true) {
			cout << "Command: ";
			cin >> input;
			
			if (input.compare("set_delta_max") == 0) {
				cout << "int delta_max = ";
				cin >> delta_max;
				
				break;
			} else if (input.compare("set_delta_min") == 0) {
				cout << "double delta_min = ";
				cin >> delta_min;
				
				break;
			} else if (input.compare("set_borderVal") == 0) {
				cout << "double borderVal = ";
				cin >> borderVal;
				
				break;
			} else if (input.compare("set_frame") == 0) {
				cout << "int frame = ";
				cin >> frame;
				
				break;
			} else if (input.compare("set_steps") == 0) {
				cout << "int steps = ";
				cin >> steps;
				
				break;
			} else if (input.compare("set_algorithm") == 0) {
				cout << "Sum of Squared Distances (SSD) = 0" 
						<< endl << "Normalized SSD = 1"
						<< endl << "Normalized Cross Correlation = 2"
						<< endl << "Zero-mean Normalized Cross Correlation (local) = 3" 
						<< endl << "Zero-mean Normalized Cross Correlation (global) = 4" 
						<< endl << endl << "Algorithm = ";
				cin >> algonew;
				
				if (algonew > -1 && algonew < 5) {
					if (algo != algonew) {
						if (algonew == 0) {
							borderVal = 125000.0f;
						} else if (algonew == 1) {
							borderVal = 100.0f;
						} else if (algonew == 2 || algonew == 3 || algonew == 4) {
							borderVal = 0.5f;
						}
					}
					
					algo = algonew;
					
					break;
				} else 
					cout << endl << "You haven't selected a proper number for an algorithm." << endl;
			} else if (input.compare("save") == 0) {
				cout << "Outfile name (*.jpg): ";
				cin >> output;
				
				while (output.rfind(".jpg") != output.length() - 4) {
					cout << "The outfile isn't a JPEG-File!" << endl
							<< "Outfile name (*.jpg): ";
					cin >> output;
				}
				
				vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
				compression_params.push_back(100);
				
				imwrite(output, dispMat, compression_params);
			} else if (input.compare("load_image") == 0) {
				loadNewImages = true;
				
				break;
			} else if (input.compare("get_params") == 0) {
				cout << endl << "delta_max = " << delta_max << endl <<
						"delta_min = " << delta_min << endl <<
						"borderVal = " << borderVal << endl <<
						"frame = " << frame << endl <<
						"steps = " << steps << endl << endl;
			} else if (input.compare("exit") == 0) {
				cudaFree(gLeftRekt);
				cudaFree(gRightRekt);
				cudaFree(gDispMap);
				delete[] dispMap;
				
				exit(0);
			} else {
				if (input.compare("help") != 0 && input.compare("h") != 0)
					cout << endl << "Incorrect usage!";
				
				cout << endl << "Possible commands:" << endl 
						<< "  * 'set_delta_min': Set minimum disparity" << endl
						<< "  * 'set_delta_max': Set maximum disparity" << endl 
						<< "  * 'set_borderVal': Set maximum error" << endl 
						<< "  * 'set_frame': Set size of block" << endl 
						<< "  * 'set_steps': Set steps between two pixels" << endl 
						<< "  * 'set_algorithm': Set the algorithm for disparity calculation" << endl
						<< "  * 'get_params': Show current parameters" << endl 
						<< "  * 'save': Save disparity map" << endl 
						<< "  * 'load_image': Load new pair of images" << endl 
						<< "  * 'help' or 'h': Show help" << endl 
						<< "  * 'exit': Exit program" << endl << endl;
			}
		}
		
		delete[] dispMap;
		cout << endl;
	}
	
	return 0;
}

