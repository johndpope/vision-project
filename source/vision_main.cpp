#include "vision_main.h"
#include "glut.h"

#include <cmath>
#include <cstdio>

#include <Windows.h>
#include <Ole2.h>

#include <Kinect.h>

#include <NuiKinectFusionCameraPoseFinder.h>
#include <iostream>
#include <fstream>
//cv includes
#include "opencv2/highgui.hpp"

#include "opencv2\core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2\aruco.hpp"
#include "opencv2\aruco\charuco.hpp"
#include "opencv2\aruco\dictionary.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2\xfeatures2d\nonfree.hpp"
#include "opencv2\xfeatures2d.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2\core\cuda_stream_accessor.hpp"
#include <opencv2/aruco/charuco.hpp>


//nui
#include "NuiKinectFusionDepthProcessor.h"
#include "NuiKinectFusionApi.h"
#include "NuiKinectFusionBase.h"
#include "NuiKinectFusionCameraPoseFinder.h"

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

using namespace aruco;

// We'll be using buffer objects to store the kinect point cloud
GLuint vboId;
GLuint cboId;

// Intermediate Buffers
unsigned char rgbimage[colorwidth*colorheight * 4];    // Stores RGB color image
unsigned char bgrimage[colorwidth*colorheight * 4];    //stores bgr color image
unsigned short bgrimage2[colorwidth*colorheight * 4];    //stores bgr color image
int is_aruco[colorwidth*colorheight];
ColorSpacePoint color_position[width*height];
int color_index;
ColorSpacePoint depth2rgb[width*height];             // Maps depth pixels to rgb pixels
ColorSpacePoint dummy2[colorwidth*colorheight];             // Maps depth pixels to rgb pixels
CameraSpacePoint depth2xyz[width*height];			 // Maps depth pixels to 3d coordinates
CameraSpacePoint depth2xyz_found[width*height];			 // Maps to the 3d coordinates of the pixels that are found
CameraSpacePoint depth2xyz_found_different_dim[width*height / 2];
DepthSpacePoint *depth_found;
// Kinect Variables
IKinectSensor* sensor;             // Kinect sensor
IMultiSourceFrameReader* reader;   // Kinect data source
CameraIntrinsics cameraIntrinsics_kinect[1];
ICoordinateMapper* mapper;         // Converts between depth, color, and 3d coordinates

//success for camera calibration
Mat intrinsic = Mat(3, 3, CV_32FC1);
Mat distCoeffs;
vector< Vec3d > rvecs, tvecs, tvecs_new;
vector<Vec3d> rvec_aruco, tvec_aruco;
Vec3d rvec_new, tvec_new;
Mat I = Mat(colorheight, colorwidth, CV_8UC4, &bgrimage);
Mat I_flipped;// = Mat(colorheight, colorwidth, CV_8UC4, &rgbimage);
Mat I_flipped_flipped;
Mat I_undistorted = Mat(colorheight, colorwidth, CV_8UC4);
Mat I_inrange;
Mat I_inrangegreen;
Mat I_inrangeblue;
Mat I_inrangered;
Mat I_inrangeyellow;
Mat hsv;
Mat I_gray;
Mat I_gray_resize;
vector<vector<Point3f>> object_points;
vector<vector<Point2f>> image_points;
int successes = 0;
int numBoards = 20;
int numCornersHor = 4;
int numCornersVer = 5;
bool calibrated = false; // assuming camera is not calibrationed
vector<float> pointerworld;
vector<float> pointerlocal;
unsigned short *buf_color_position;
int found_colour[height*width];
ofstream myfile;



//Threshold

int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
int write_counter = 0;
Mat src, src_gray, dst;
Mat dummy;
char* window_name = "Threshold Demo";
char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";
int run_counter = 0;

//Aruco
vector<Point2f> output_axis;
cv::Mat markerImage;
Ptr < Dictionary > dictionary;
vector< int > markerIds;
vector< vector<Point2f> > markerCorners, rejectedCandidates;
vector<vector<Point2f>> markerCorners_resize;
vector<Point2f> aruco_center;

float markerLength = 0.1;
Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
cv::Mat rvec(3, 1, cv::DataType<double>::type);
cv::Mat tvec(3, 1, cv::DataType<double>::type);
vector<Point3f> obj;
vector<Point2f> corners;
vector<Point3d> greenLower;
vector<Point3d> greenUpper;
//Calibration
cv::Mat distortion(4, 1, cv::DataType<double>::type);
cv::Mat instrinsics(3, 3, cv::DataType<double>::type);

//3d stuff
vector<Point3f> axis_3;
vector<Point3f> geo_deform;
vector<Point2f> output_deform;
vector<Point2f> tracking_colors;
vector<Point3f> mesh_geometry;
vector<Point2f> mesh_geometry_display;
Mat A;
bool cuda_init = false;
bool first_geo_init;
std::vector<DepthSpacePoint> depthSpace(colorwidth * colorheight);
DepthSpacePoint depthSpace2[colorwidth*colorheight];
//DepthSpacePoint depthSSSS[colorwidth*colorheight];
cuda::GpuMat input_gpu;
cuda::GpuMat output_gpu;
Mat output;
int display_counter;

//charuco
int squaresX = 4;
int squaresY = 5;
float squareLength = 0.04;
float markerLength_charuco = 0.02;
int dictionaryId = 1;
int margins = squareLength - markerLength_charuco;
int aruco_center_id = 0;
Size imageSize;
bool aruco_begin = false;// this means that we have not begun the tracking.
vector<Point2f> aruco_position;//we have t+1
vector<Point2f> meshnode_position;//we have t

vector<Point2f> diff;//the diff vector for 
//chessboard markers


//
//
////Ptr<aruco::CharucoBoard> board_charuco = aruco::CharucoBoard::create(squaresX, squaresY, (float)squareLength, (float)markerLength_charuco, dictionary);
//Ptr<aruco::CharucoBoard> board_charuco; //= aruco::CharucoBoard::create(squaresX, squaresY, (float)squareLength, (float)markerLength_charuco, dictionary);


void Threshold_Demo(int, void*)
{
	/* 0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
	*/

	threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);
	
	imshow(window_name, dst);
}
//-------------End of threshold function stuf------------------------//
Geometry *geo_ptr;

bool initKinect() {
	if (FAILED(GetDefaultKinectSensor(&sensor))) {
		return false;
	}
	if (sensor) {
		sensor->get_CoordinateMapper(&mapper);

		sensor->Open();
		sensor->OpenMultiSourceFrameReader(
			FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
			&reader);
		return reader;
	}
	else {
		return false;
	}

}

void getDepthData(IMultiSourceFrame* frame, GLubyte* dest) {
	IDepthFrame* depthframe;
	IDepthFrameReference* frameref = NULL;
	frame->get_DepthFrameReference(&frameref);
	frameref->AcquireFrame(&depthframe);
	if (frameref) frameref->Release();

	if (!depthframe) return;

	// Get data from frame
	unsigned int sz;
	unsigned short* buf;

	depthframe->AccessUnderlyingBuffer(&sz, &buf);

	// Write vertex coordinates
	mapper->MapDepthFrameToCameraSpace(width*height, buf, width*height, depth2xyz);
	//mapper->MapDepthFrameToCameraSpace(width*height, buf, width*height / 2, depth2xyz_found_different_dim);

	float* fdest = (float*)dest;
	for (int i = 0; i < sz; i++) {

		*fdest++ = depth2xyz[i].X;
		*fdest++ = depth2xyz[i].Y;
		*fdest++ = depth2xyz[i].Z;

	}
	//mapper->MapColorFrameToDepthSpace(width*height, buf_color_position, 2*width*height, depth_found);
	mapper->MapDepthFrameToColorSpace(sz, buf, width*height, depth2rgb);

	mapper->MapColorFrameToDepthSpace(sz, buf, colorwidth*colorheight, depthSpace2);
	//mapper->MapDepthPointsToColorSpace(1, &depthSpace2[0], sz, buf, 1, dummycolor);
	// Fill in depth2rgb map


	//
	if (depthframe) depthframe->Release();
}

void get_mesh(Geometry *p){
	int e = p->return_numElems();
	// global_geo.return_numElems()
	if (1){
		p->setSudoNode(147);
		p->setSudoForcex(diff[0].x/8.0);
		p->setSudoForcey(diff[0].y /8.0);
	}
	else {
		p->setSudoNode(20);
		p->setSudoForcex(0);
		p->setSudoForcey(0);
	}
	/*if (display_counter == 500){
	p->setSudoNode(20);
	p->setSudoForcex(-100);
	p->setSudoForcey(-100);
	}*/
	display_counter++;
	/*if (!cuda_init){
		p->initialize_CUDA();
		cuda_init = true;
	}*/
	p->make_K_matrix();
	p->find_b();

	p->update_vector();
	p->update_dynamic_vectors();
	p->update_dynamic_xyz();
	mesh_geometry.empty();
	for (int i = 0; i <p->return_numNodes(); i++){

		//double dx = (geo_deform[1].x-geo_deform[0].x );
		if (first_geo_init == true){
			mesh_geometry.push_back(Point3f(((p->return_x(i))) / 1.0, (p->return_y(i)) / 1.0, 0));
		}
		else {
			mesh_geometry[i] = (Point3f(((p->return_x(i))) / 1.0, (p->return_y(i)) / 1.0, 0));
		}
		
		
		
	}
}


void draw_mesh(Geometry *p, Mat I){
	int e = p->return_numElems();
	// global_geo.return_numElems()
	meshnode_position.clear();
	for (int i = 0; i < p->return_numElems(); i++){
		int node_considered4=0;

		int node_considered1 = p->node_number_inElem(i, 0);
		int node_considered2 = p->node_number_inElem(i, 1);
		int node_considered3 = p->node_number_inElem(i, 2);
		if (p->return_dim() == 3){
			int node_considered4 = p->node_number_inElem(i, 3);
		}




		int thickness = 1;
		int lineType = 8;

		//GpuMat image1(Size(1902, 1080), CV_8U);
		
		line(I, mesh_geometry_display[node_considered1], mesh_geometry_display[node_considered2], Scalar(100, 50, 255), thickness, lineType);

		line(I, mesh_geometry_display[node_considered3], mesh_geometry_display[node_considered1], Scalar(100, 50, 255), thickness, lineType);

		if (p->return_dim() == 3){
			line(I, mesh_geometry_display[node_considered2], mesh_geometry_display[node_considered4], Scalar(100, 50, 255), thickness, lineType);
			line(I, mesh_geometry_display[node_considered4], mesh_geometry_display[node_considered3], Scalar(100, 50, 255), thickness, lineType);

			line(I, mesh_geometry_display[node_considered1], mesh_geometry_display[node_considered4], Scalar(100, 50, 255), thickness, lineType);

			line(I, mesh_geometry_display[node_considered3], mesh_geometry_display[node_considered2], Scalar(100, 50, 255), thickness, lineType);
			
		}
		else {
			line(I, mesh_geometry_display[node_considered2], mesh_geometry_display[node_considered3], Scalar(100, 50, 255), thickness, lineType);
		}
		
		if (node_considered1 == 147){
			meshnode_position.push_back((mesh_geometry_display[node_considered1]));
			circle(I, mesh_geometry_display[node_considered1], 20, Scalar(0, 100, 255), 4);
			putText(I, to_string(mesh_geometry_display[node_considered1].x) +"   "+ to_string(mesh_geometry_display[node_considered1].y), mesh_geometry_display[node_considered1], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		}
		if (0){ // if draw zero u points
			bool yes = false;
			int node_yes;
			for (int m = 0; m < 9; m++){
				if ((node_considered1 == m)){
					yes = true;
					node_yes = node_considered1;
				}
				else if ((node_considered3 == m)){
					yes = true;
					node_yes = node_considered3;
				}
			}

			if (yes){
				//meshnode_position.push_back((mesh_geometry_display[node_considered1]));
				circle(I, mesh_geometry_display[node_yes], 5, Scalar(100, 58, 58), 5);
				//putText(I, to_string(mesh_geometry_display[node_considered1].x) + "   " + to_string(mesh_geometry_display[node_considered1].y), mesh_geometry_display[node_considered1], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
			}
		}
		
		//putText(I, to_string(node_considered2), mesh_geometry_display[node_considered2], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
/*
		circle(I, mesh_geometry_display[node_considered1], 100 / 32.0, Scalar(200, 100, 80), -1, 1);
		circle(I, mesh_geometry_display[node_considered2], 100 / 32.0, Scalar(200,100, 80), -1, 1);
		circle(I, mesh_geometry_display[node_considered3], 100 / 32.0, Scalar(200, 100, 80), -1, 1);
		
		double dx = (geo_deform[1].x - geo_deform[0].x);
		string ss = "dx : " + to_string(dx);
		putText(I, (ss), Point2f(50.0,50.0), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		if (p->return_dim() == 3){

			circle(I, mesh_geometry_display[node_considered4], 70 / 32.0, Scalar(200, 0, 80), -1, 1);
		}*/



	}

}

//----------------------get RGB DATA-----------------------//
void getRgbData(IMultiSourceFrame* frame, GLubyte* dest) {
	IColorFrame* colorframe;
	IColorFrameReference* frameref = NULL;
	frame->get_ColorFrameReference(&frameref);
	frameref->AcquireFrame(&colorframe);
	if (frameref) frameref->Release();

	if (!colorframe) return;
	unsigned int sz;
	unsigned char*  buffer;

	//colorframe->AccessRawUnderlyingBuffer(&sz, &buffer);
	// Get data from frame

	//mapper->MapColorFrameToDepthSpace(width*height, buf, width*height, dummy2);
	
	colorframe->CopyConvertedFrameDataToArray(colorwidth*colorheight * 4, rgbimage, ColorImageFormat_Rgba);
	colorframe->CopyConvertedFrameDataToArray(colorwidth*colorheight * 4, bgrimage, ColorImageFormat_Bgra);
	
	//colorframe->CopyConvertedFrameDataToArray(colorwidth*colorheight * 4, reinterpret_cast<BYTE*>(bgrimage2), ColorImageFormat_Bgra);
	
	std::clock_t start_K11;
	double duration_vision;
	

	


	if (0){

		undistort(I, I_undistorted, intrinsic, distCoeffs);
		imshow("undistorted", I_undistorted);
	}

	if (0){ // iF chess board
		
	}

	if (0){//if aruco
		detectMarkers(src_gray, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
		//detectMarkers(dummy, dictionary, markerCorners_resize, markerIds, parameters, rejectedCandidates);
		if (markerIds.size() > 3){
			for (unsigned int i = 0; i < markerIds.size(); i++){

				//cout << "markerId: " << markerIds[i] << endl;

				for (int j = 0; j < 4; j++){

					markerCorners[i][j].x = markerCorners[i][j].x*(colorwidth / (colorwidth / 2.0));
					markerCorners[i][j].y = markerCorners[i][j].y*(colorheight / (colorheight / 2.0));

				}
			}


			//cv::aruco::drawDetectedMarkers(src_gray, markerCorners, markerIds);
			aruco::estimatePoseSingleMarkers(markerCorners, markerLength, instrinsics, distortion, rvecs, tvecs);
			for (int j = 0; j < markerIds.size(); j++){
				if (markerIds[j] == 0){
					rvec_new = rvecs[j];
					tvec_new = tvecs[j];
				}
			}




			for (unsigned int i = 0; i < markerIds.size(); i++){
				//int index = (int)(((markerCorners[i][0].x)*((double)(width*1.0) / colorwidth)) + width*(markerCorners[i][0].y)*((double)(1.0*height) / colorheight));
				/*int idx2 = ((int)markerCorners_resize[i][0].x) + width*((int)markerCorners_resize[i][0].y);
				CameraSpacePoint q = depth2xyz[idx2];t
				*/

				if (markerIds[i] == 0){
					corners[0].x = markerCorners[i][0].x;
					corners[0].y = markerCorners[i][0].y;
					/*geo_deform[0].x = q.X;
					  geo_deform[0].y = q.Y;
					  geo_deform[0].z = q.Z;*/
					circle(I_flipped, Point(corners[0].x, corners[0].y), 150 / 32.0, Scalar(200, 0, 50), -1, 1);
				}
				else if (markerIds[i] == 6){
					corners[1].x = markerCorners[i][0].x;
					corners[1].y = markerCorners[i][0].y;
					/*		geo_deform[1].x = q.X;
							geo_deform[1].y = q.Y;
							geo_deform[1].z = q.Z;*/
					circle(I_flipped, Point(corners[1].x, corners[1].y), 150 / 32.0, Scalar(200, 0, 60), -1, 1);
				}
				else if (markerIds[i] == 4){
					corners[2].x = markerCorners[i][0].x;
					corners[2].y = markerCorners[i][0].y;
					/*	geo_deform[2].x = q.X;
						geo_deform[2].y = q.Y;
						geo_deform[2].z = q.Z;*/
					circle(I_flipped, Point(corners[2].x, corners[2].y), 150 / 32.0, Scalar(200, 0, 70), -1, 1);
				}
				else if (markerIds[i] == 1){
					corners[3].x = markerCorners[i][0].x;
					corners[3].y = markerCorners[i][0].y;
					/*	geo_deform[3].x = q.X;
						geo_deform[3].y = q.Y;
						geo_deform[3].z = q.Z;*/
					circle(I_flipped, Point(corners[3].x, corners[3].y), 150 / 32.0, Scalar(200, 0, 80), -1, 1);

				}
				if (0){

				}
				solvePnP(Mat(geo_deform), Mat(corners), instrinsics, distortion, rvec_new, tvec_new, false);
				//markerCorners[i][0].x = markerCorners[i][0].x*((int)colorwidth / 960);
				//markerCorners[i][0].y = markerCorners[i][0].y*((int)colorheight / 540);

				//ellipse(I_flipped, Point(markerCorners[i][0].x, markerCorners[i][0].y), Size(10 / 4.0, 10 / 16.0), 0, 0, 360, Scalar(255, 0, 255), 10, 1);

				//projectPoints(axis_3, rvecs[i], tvecs[i], instrinsics, distortion, output_axis);
				projectPoints(geo_deform, rvec_new, tvec_new, instrinsics, distortion, output_deform);
				int thickness = 0.5;
				int lineType = 8;

				line(I_flipped, output_deform[0], output_deform[1], Scalar(255, 0, 255), thickness, lineType);
				line(I_flipped, output_deform[1], output_deform[3], Scalar(255, 0, 255), thickness, lineType);
				line(I_flipped, output_deform[3], output_deform[2], Scalar(255, 0, 255), thickness, lineType);
				line(I_flipped, output_deform[2], output_deform[0], Scalar(255, 0, 255), thickness, lineType);
				
			}


		}
	}
	
	//-----------------------------tracking circles------------------------------//
	if (1){
		
		

		
		//------------------image processing to find the contour---------------//
	
		input_gpu.upload(I);
		
		cuda::cvtColor(input_gpu, output_gpu, COLOR_BGR2HSV);
		
		output_gpu.download(hsv);

		//inRange(I, blcklow, blckhigh, I_inrangeyellow);//////////////////
		//imshow("I_gray_resize", I_inrangeyellow);

		cvtColor(I, I_gray, CV_BGR2GRAY);
		double resize_num = 4.0;
		resize(I_gray, I_gray_resize, Size(colorwidth / resize_num, colorheight / resize_num));
		//resize(hsv, hsv, Size(colorwidth / 4, colorheight / 4));

		flip(I_gray_resize, I_gray_resize, 1);
		
		flip(I, I_flipped,1);
		//imshow("I_gray_resize", I_gray_resize);
		/*imwrite(to_string(write_counter) + ".png", I_flipped);
		write_counter++;*/
		int numSquares = numCornersHor * numCornersVer;
		Size board_sz = Size(numCornersHor, numCornersVer);
		bool found = findChessboardCorners(I_gray_resize, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found)
		{
			//cornerSubPix(I, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			

			for (unsigned int i = 0; i < numSquares; i++){
				
				corners[i].x = colorwidth - corners[i].x*resize_num;//
				corners[i].y = corners[i].y*resize_num;
					
			
				

			}
			drawChessboardCorners(I, board_sz, corners, found);
		}
		//imshow("ff", I_gray_resize);
		//-------------------ARUCO-------------------------
		if (1){
			start_K11 = std::clock();

			detectMarkers(I_gray_resize, dictionary, markerCorners, markerIds);
			aruco_center.clear();
			aruco_position.clear();
			//vector< Point2f > charucoCorners; vector< int > markerIds, charucoIds;
			if (markerIds.size() > 0){
				//	//cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);


				for (unsigned int i = 0; i < markerIds.size(); i++){
					double x_ave = 0;
					double y_ave = 0;
					for (int j = 0; j < 4; j++){
						markerCorners[i][j].x = colorwidth - markerCorners[i][j].x*resize_num;//colorwidth -
						markerCorners[i][j].y = markerCorners[i][j].y*resize_num;
						/*Point2f center((markerCorners[i][j].x), (markerCorners[i][j].y));
						
						circle(I, center, 2, Scalar(0, 0, 255), 3, 8, 0);*/
						x_ave = x_ave + markerCorners[i][j].x;
						y_ave = y_ave + markerCorners[i][j].y;
					
					}
					aruco_center.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));
					aruco_position.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));
					//putText(I, ".", Point((int)aruco_center[i].x, (int)aruco_center[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
					circle(I, aruco_center[i], 10, cv::Scalar(255, 0, 0), 3);
					/*if (markerIds[i] == 12){
						
						aruco_center_id = i;
					}
					*/

				}
				//aruco::estimatePoseSingleMarkers(markerCorners, markerLength, instrinsics, distortion, rvec_aruco, tvec_aruco);
				////IF WE ARE WRITING TO FILE THE CENTERS OF THE ARUCO MARKERS
				if (0){
					double c_x = instrinsics.at<double>(2);
					double f_x = instrinsics.at<double>(0);
					double c_y = instrinsics.at<double>(5);
					double f_y = instrinsics.at<double>(4);
					if (0){
						cout << "cx: " << c_x << endl;
						cout << "fx: " << f_x << endl;
						cout << "cy: " << c_y << endl;
						cout << "fy: " << f_y << endl;
					}
					string outputmesg;
					string outputcoord;
					//std::ofstream in_disp(to_string(write_counter) + "aruco_center.txt");
					for (unsigned int i = 0; i < markerIds.size(); i++){
						int index3 = ((int)aruco_center[i].y)*colorwidth + (int)aruco_center[i].x;
						ColorSpacePoint dummycolor;

						int _X = (int)depthSpace2[index3].X;
						int _Y = (int)depthSpace2[index3].Y;
						// _X = (int)(aruco_center[i].x*static_cast<double>(width)/colorwidth);
						// _Y = (int)(aruco_center[i].y*static_cast<double>(height) / colorheight);
						double actualx;
						double actualy;
						double actualz;
						if ((_X >= 0) && (_X < width) && (_Y >= 0) && (_Y < height)){
							int depth_index = (_Y*width) + _X;

							/*CameraSpacePoint q = depth2xyz[depth_index]*/
							ColorSpacePoint p = depth2rgb[depth_index];
							CameraSpacePoint world_point_camera = depth2xyz[depth_index];
							int idx = ((int)p.X) + colorwidth*((int)p.Y);
							actualx = (p.X - c_x)*world_point_camera.Z / f_x;
							actualy = (p.Y - c_y)*world_point_camera.Z / f_y;
							actualz = world_point_camera.Z;
							//if (actualz >= INFINITE)break;



							//in_disp << markerIds[i] << " " << actualx << " " << actualy << " " << actualz << endl;
							outputmesg = to_string(markerIds[i]);// +" Pos: " + to_string(actualx) + " " + to_string(actualy) + " " + to_string(actualz);
							outputcoord = "id:" + to_string(markerIds[i]) + " Pos: " + to_string(actualx) + " " + to_string(actualy) + " " + to_string(actualz);
							putText(I, outputmesg, Point((int)aruco_center[i].x, (int)aruco_center[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
							putText(I, outputcoord, Point((int)50, 10 * markerIds[i]), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);

							

						}

					}
					//write_counter++;
					//in_disp.close();
				}
				

				/*for (unsigned int i = 0; i < markerIds.size(); i++){
					projectPoints(axis_3, rvec_aruco[i], tvec_aruco[i], instrinsics, distortion, output_deform);
					int thickness = 2;
					int lineType = 8;
					line(I, output_deform[0], output_deform[1], Scalar(100, 0, 0), thickness, lineType);
					line(I, output_deform[0], output_deform[2], Scalar(100, 200, 0), thickness, lineType);
					line(I, output_deform[0], output_deform[3], Scalar(50, 200, 100), thickness, lineType);
					line(I, output_deform[1], output_deform[5], Scalar(100, 200, 300), thickness, lineType);
					line(I, output_deform[1], output_deform[4], Scalar(100, 200, 300), thickness, lineType);
					line(I, output_deform[2], output_deform[4], Scalar(100, 200, 300), thickness, lineType);
					line(I, output_deform[2], output_deform[6], Scalar(100, 200, 300), thickness, lineType);
					line(I, output_deform[6], output_deform[7], Scalar(100, 200, 300), thickness, lineType);
					line(I, output_deform[3], output_deform[6], Scalar(100, 200, 300), thickness, lineType);
					line(I, output_deform[5], output_deform[7], Scalar(100, 200, 300), thickness, lineType);
					line(I, output_deform[3], output_deform[5], Scalar(100, 200, 300), thickness, lineType);
					line(I, output_deform[4], output_deform[7], Scalar(100, 200, 300), thickness, lineType);

				}*/
			}

			//solvePnP(Mat(geo_deform), Mat(markerCorners), instrinsics, distortion, rvec_new, tvec_new, false);
			
			//charuco marker detection
			//aruco::refineDetectedMarkers(I_gray_resize, board_charuco, markerCorners, markerIds, rejectedCandidates, instrinsics, distortion);
			/*if (0) {
				std::vector<cv::Point2f> charucoCorners;
				std::vector<int> charucoIds;
				cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, I_gray_resize, board_charuco, charucoCorners, charucoIds, instrinsics, distortion);
				aruco::drawDetectedCornersCharuco(I_gray_resize, charucoCorners, charucoIds, cv::Scalar(100, 100, 100));
			}*/
			
		}
		//int interpolatedCorners = 0;
		//if (markerIds.size() > 0)
		//	interpolatedCorners =aruco::interpolateCornersCharuco(markerCorners, markerIds, I_gray_resize, board_charuco,markerCorners, markerIds, instrinsics, distortion);
		//
		//bool validPose = false;
		//if (instrinsics.total() != 0)
		//	validPose = aruco::estimatePoseCharucoBoard(markerCorners, markerIds, board_charuco,
		//	instrinsics, distortion, rvec_aruco, tvec_aruco);
		//if (markerIds.size() > 0) {
		//	///aruco::drawDetectedMarkers(I, markerCorners);
		//}
		//imshow("good image", I_flipped);
//---------------ARUCO END---------------------
		if (0){
			

			Scalar greenlow = Scalar(50, 50, 50);
			Scalar greenhigh = Scalar(75, 220, 220);

			Scalar bluelow = Scalar(80, 50, 50);
			Scalar bluehigh = Scalar(140, 255, 255);


			Scalar redlow = Scalar(0, 100, 0);
			Scalar redhigh = Scalar(10, 255, 255);


			Scalar yellowlow = Scalar(20, 124, 123);
			Scalar yellowhigh = Scalar(30, 256, 256);


			inRange(hsv, yellowlow, yellowhigh, I_inrangeyellow);
			inRange(hsv, bluelow, bluehigh, I_inrangeblue);
			inRange(hsv, redlow, redhigh, I_inrangered);
			inRange(hsv, greenlow, greenhigh, I_inrangegreen);
			/*
			imshow("yellow", I_inrangeyellow);
			imshow("green", I_inrangegreen);
			imshow("red", I_inrangered);
*/
			
			bitwise_or(I_inrangeyellow, I_inrangeblue, I_inrangeblue);
			bitwise_or(I_inrangered, I_inrangegreen, I_inrangegreen);
			bitwise_or(I_inrangeblue, I_inrangegreen, I_inrange);
			imshow("blue", I_inrange);
			Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(0, 0));
			erode(I_inrange, I_inrange, element);

			dilate(I_inrange, I_inrange, element);
			imshow("blue", I_inrange);
			//imshow("i-range", I_inrange);
			vector<Vec4i> hierarchy;
			vector<vector<Point> > contours;


			findContours(I_inrange, contours, RETR_LIST, CHAIN_APPROX_NONE);
			vector<Point2f>center(contours.size());
			vector<int>color_bool(contours.size());
			vector<float>radius(contours.size());
			vector<string>world_string(contours.size());
			vector<Point2f>world_3d(contours.size());
			vector<vector<Point> > contours_poly(contours.size());

			//////////////////////////


			for (int i = 0; i < contours.size(); i++)
			{

				approxPolyDP(Mat(contours[i]), contours_poly[i], 4, true);
				minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);


			}

			//////////////////////
			//std::ofstream in_disp(to_string(write_counter)+"mesh.txt");
			int x_pos = 0, y_pos = 0;
			int dummpy_used;
			string s;
			Mat colorImage = Mat::zeros(colorheight, colorwidth, CV_8UC4);
			double c_x = instrinsics.at<double>(2);
			double f_x = instrinsics.at<double>(0);
			double c_y = instrinsics.at<double>(5);
			double f_y = instrinsics.at<double>(4);
			
			


			for (int h = 0; h < colorheight; h++){//colorheight
				for (int w = 0; w < colorwidth; w++){//colorwidth
					int index3 = h*colorwidth + w;
					ColorSpacePoint dummycolor;

					int _X = (int)depthSpace2[index3].X;
					int _Y = (int)depthSpace2[index3].Y;
					if ((_X >= 0) && (_X < width) && (_Y >= 0) && (_Y < height)){
						int depth_index = (_Y*width) + _X;
						int index3_color = index3 * 4;
						/*CameraSpacePoint q = depth2xyz[depth_index]*/
						ColorSpacePoint p = depth2rgb[depth_index];
						CameraSpacePoint world_point_camera = depth2xyz[depth_index];
						int idx = ((int)p.X) + colorwidth*((int)p.Y);
						double actualx = (p.X - c_x)*world_point_camera.Z/f_x;
						double actualy = (p.Y - c_y)*world_point_camera.Z / f_y;
						
						//if ((rgbimage[4 * idx + 0] < 120) && (rgbimage[4 * idx + 1]>120) && (rgbimage[4 * idx + 2] < 90	)	){
							//in_disp << actualx << " " << actualy << " " << world_point_camera.Z << endl;
							/*colorImage.data[index3_color + 0] = rgbimage[4 * idx + 0];
							colorImage.data[index3_color + 1] = rgbimage[4 * idx + 1];*/
						colorImage.data[index3_color + 0] = 0;
						colorImage.data[index3_color + 1] = world_point_camera.Z * 50;

							colorImage.data[index3_color + 2] = 0;
							colorImage.data[index3_color + 3] = rgbimage[4 * idx + 3];
						//}
						
						/*colorImage.data[index3_color + 0] = rgbimage[4 * idx + 0];
						colorImage.data[index3_color + 1] = rgbimage[4 * idx + 1];
						colorImage.data[index3_color + 2] = rgbimage[4 * idx + 2];*/
						/*colorImage.data[index3_color + 0] = 0;
						colorImage.data[index3_color + 1] = aaaa.Z*100;
						colorImage.data[index3_color + 2] = 100;
						colorImage.data[index3_color + 3] = rgbimage[4 * idx + 3];*/
					}


				}
			}
			imshow("fd", colorImage);
			imwrite("calibrate"+to_string(write_counter) + ".png", colorImage);
			write_counter++;

			//in_disp.close();
			
			//cvtColor(colorImage, colorImage, CV_RGB2HSV);
			//inRange(colorImage, greenlow, greenhigh, I_inrangeyellow);
			//if ((rgbimage[4 * idx + 0] < 120) && (rgbimage[4 * idx + 1]>50) && (rgbimage[4 * idx + 2] < 90)){
			//	in_disp << aaaa.X << " " << aaaa.Y << " " << aaaa.Z << endl;

			//	//colorImage.data[index3_color + 3] = rgbimage[4 * idx + 3];

			//}
			/*imshow("color image", colorImage);
			imwrite(to_string(write_counter) + ".png", I_inrangeyellow);
			write_counter++;*/

			
			for (int i = 0; i < contours.size(); i++)
			{
				int w = center[i].x;
				int h = center[i].y;
				int index3 = h*colorwidth + w;
				int g = 0;
				if ((w >= 0) && (w < colorwidth) && (h >= 0) && (h < colorheight)){

				}
				else{
					index3 = 0;
				}
				if (radius[i] >INFINITY){//change!!
					if ((int)index3 < colorwidth*colorheight){

						int _X = (int)((depthSpace2[index3].X));
						int _Y = (int)((depthSpace2[index3].Y));
						if ((_X >= 0) && (_X < width) && (_Y >= 0) && (_Y < height)){
							int depth_index = (_Y*width) + _X;
							int index3_color = index3 * 4;
							CameraSpacePoint q = depth2xyz[depth_index];
							ColorSpacePoint p = depth2rgb[depth_index];
							int idx = ((int)p.X) + colorwidth*((int)p.Y);
							double c_x = instrinsics.at<double>(2);
							double f_x = instrinsics.at<double>(0);
							double c_y = instrinsics.at<double>(5);
							double f_y = instrinsics.at<double>(4);
							

							if ((rgbimage[4 * idx + 0] < 90) && (rgbimage[4 * idx + 1]>75) && (rgbimage[4 * idx + 2] < 90)){
								g = 200;
								//if (((int)w == (int)(center[i].x)) && ((int)h == (int)(center[i].y))){
								world_string[i] = "Green: " + to_string((p.X - c_x)*q.Z / f_x) + " " + to_string((p.Y - c_y)*q.Z / f_y) + " " + to_string(q.Z);
								world_3d[i] = Point2f(p.X, p.Y);

								tracking_colors[2].x = p.X;
								tracking_colors[2].y = p.Y;
								/*	mesh_geometry[1].x = geo_deform[2].x = q.X;
									mesh_geometry[1].y = geo_deform[2].y = q.Y;
									mesh_geometry[1].z = geo_deform[2].z = q.Z;*/

								geo_deform[2].x = (p.X - c_x)*q.Z / f_x;
								geo_deform[2].y = (p.Y - c_y)*q.Z / f_y;
								geo_deform[2].z = q.Z;

								//circle(colorImage, Point((int)center[i].x, (int)center[i].y), radius[i], Scalar(200, g, 20), -1, 1);
								if (world_string[i].size() != 0){
									//putText(I, world_string[i], Point((int)world_3d[i].x + 5, (int)world_3d[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
									circle(I, Point((int)world_3d[i].x, (int)world_3d[i].y), radius[i], Scalar(0, 255, 0), -1, 1);
								}
							}
							else if ((100 < rgbimage[4 * idx + 0]) && (rgbimage[4 * idx + 1] < 100) && (rgbimage[4 * idx + 2] < 100)){

								//if (((int)w == (int)(center[i].x)) && ((int)h == (int)(center[i].y))){
								world_string[i] = "Red: " + to_string((p.X - c_x)*q.Z / f_x) + " " + to_string((p.Y - c_y)*q.Z / f_y) + " " + to_string(q.Z);
								world_3d[i] = Point2f(p.X, p.Y);

								tracking_colors[3].x = p.X;
								tracking_colors[3].y = p.Y;
								/*mesh_geometry[76].x = geo_deform[3].x = q.X;
								mesh_geometry[76].y = geo_deform[3].y = q.Y;
								mesh_geometry[76].z = geo_deform[3].z = q.Z;*/

								geo_deform[3].x = (p.X - c_x)*q.Z / f_x;
								geo_deform[3].y = (p.Y - c_y)*q.Z / f_y;
								geo_deform[3].z = q.Z;
								//circle(colorImage, Point((int)center[i].x, (int)center[i].y), radius[i], Scalar(200, g, 20), -1, 1);
								if (world_string[i].size() != 0){
									//putText(I, world_string[i], Point((int)world_3d[i].x + 5, (int)world_3d[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
									circle(I, Point((int)world_3d[i].x, (int)world_3d[i].y), radius[i], Scalar(0, 0, 255), -1, 1);
								}
							}
							else if ((200 > rgbimage[4 * idx + 0]) && (200>rgbimage[4 * idx + 1] ) && (0<rgbimage[4 * idx + 2] )){

								//if (((int)w == (int)(center[i].x)) && ((int)h == (int)(center[i].y))){
								world_string[i] = "Blue: " + to_string((p.X - c_x)*q.Z / f_x) + " " + to_string((p.Y - c_y)*q.Z / f_y) + " " + to_string(q.Z);
								world_3d[i] = Point2f(p.X, p.Y);

								tracking_colors[0].x = p.X;
								tracking_colors[0].y = p.Y;
								/*mesh_geometry[0].x = geo_deform[0].x = q.X;
								mesh_geometry[0].y= geo_deform[0].y = q.Y;
								mesh_geometry[0].z = geo_deform[0].z = q.Z;*/
								geo_deform[0].x = (p.X - c_x)*q.Z / f_x;
								geo_deform[0].y = (p.Y - c_y)*q.Z / f_y;
								geo_deform[0].z = q.Z;

								//circle(colorImage, Point((int)center[i].x, (int)center[i].y), radius[i], Scalar(200, g, 20), -1, 1);
								if (world_string[i].size() != 0){
									putText(I, world_string[i], Point((int)world_3d[i].x + 5, (int)world_3d[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 0, 255), 2.0);
									circle(I, Point((int)world_3d[i].x, (int)world_3d[i].y), radius[i], Scalar(255, 0, 0), -1, 1);
								}
							}
							else if ((100 < rgbimage[4 * idx + 0]) && (rgbimage[4 * idx + 1] > 100) && (rgbimage[4 * idx + 2] < 80)){

								//if (((int)w == (int)(center[i].x)) && ((int)h == (int)(center[i].y))){
								world_string[i] = "Yellow: " + to_string((p.X - c_x)*q.Z / f_x) + " " + to_string((p.Y - c_y)*q.Z / f_y) + " " + to_string(q.Z);
								world_3d[i] = Point2f(p.X, p.Y);

								tracking_colors[1].x = p.X;
								tracking_colors[1].y = p.Y;
								/*mesh_geometry[75].x = geo_deform[1].x = q.X;
								mesh_geometry[75].y = geo_deform[1].y = q.Y;
								mesh_geometry[75].z = geo_deform[1].z = q.Z;*/

								geo_deform[1].x = (p.X - c_x)*q.Z / f_x;
								geo_deform[1].y = (p.Y - c_y)*q.Z / f_y;
								geo_deform[1].z = q.Z;
								//circle(colorImage, Point((int)center[i].x, (int)center[i].y), radius[i], Scalar(200, g, 20), -1, 1);
								if (world_string[i].size() != 0){
									putText(I, world_string[i], Point((int)world_3d[i].x + 5, (int)world_3d[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 20), 2.0);
									circle(I, Point((int)world_3d[i].x, (int)world_3d[i].y), radius[i], CV_RGB(255, 255, 20), -1, 1);
								}
							}


							

						}
					}
				}
			}
			double dx = (geo_deform[1].x - geo_deform[0].x);
			double dy = (geo_deform[1].y - geo_deform[3].y);
			string ssx = "dx : " + to_string(dx);
			string ssy = "dy : " + to_string(dy);
			putText(I, (ssx), Point2f(50.0, 50.0), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
			putText(I, (ssy), Point2f(50.0, 30.0), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
			int _DUMMY_ = 0;
			/*for (int i = 0; i < contours.size(); i++){
				if (!world_string[i].empty()){
					putText(I, world_string[i], Point2f(50.0, 50.0 + i*_DUMMY_), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 20), 2.0);
					_DUMMY_++;
				}
			}*/
		}
		
		
		//imshow("original", I);
		if (1)//DRAW VERY IMPORTANT!!!!!!!!!!!
		{
			
			//solvePnP(Mat(geo_deform), Mat(tracking_colors), instrinsics, distortion, rvec_new, tvec_new, false);
			if (found){
				
				solvePnP(Mat(obj), Mat(corners), instrinsics, distortion, rvec_new, tvec_new, false);
				projectPoints(mesh_geometry, rvec_new, tvec_new, instrinsics, distortion, mesh_geometry_display);
				draw_mesh(geo_ptr, I);
				
			}
			if (markerIds.size() > 0){ //if there is an aruco marker
				diff.clear();
				diff.push_back(Point2f((aruco_position[0].x - meshnode_position[0].x), (aruco_position[0].y - meshnode_position[0].y)));
				if (cv::norm(diff) > 100){
					diff.clear();
					diff.push_back(Point2f(0.0, 0.0));
				}
				putText(I, "Force : " + to_string(diff[0].x) + "  " + to_string(diff[0].y), Point2f(50.0, 50.0), 1, 1, Scalar(100, 100, 20));
			}
			else{ // if there isn't then put sudo force to zero
				diff.clear();
				diff.push_back(Point2f(0.0, 0.0));
				
			}
			//projectPoints(geo_deform, rvec_new, tvec_new, instrinsics, distortion, output_deform);
			if (0){
				projectPoints(mesh_geometry, rvec_aruco[aruco_center_id], tvec_aruco[aruco_center_id], instrinsics, distortion, mesh_geometry_display);
				draw_mesh(geo_ptr, I);
			}
			int thickness = 2;
			int lineType = 8;


			/*line(I, output_deform[0], output_deform[1], Scalar(255, 0, 255), thickness, lineType);
			line(I, output_deform[1], output_deform[3], Scalar(255, 0, 255), thickness, lineType);
			line(I, output_deform[3], output_deform[2], Scalar(255, 0, 255), thickness, lineType);
			line(I, output_deform[2], output_deform[0], Scalar(255, 0, 255), thickness, lineType);*/
			//line(I, Point2f(0, 0), Point2f(100, 100), Scalar(100, 10, 255), thickness, lineType);
			
			imshow("original", I);

			//imshow("colorimage", colorImage);
			//imshow("In range", I_inrange);
			first_geo_init = false;
			

		
			get_mesh(geo_ptr);
			
			//double dt = abs(std::clock() - start_K11);
			//cout << " dt : " << dt << endl;
			//duration_vision = (std::clock() - start_K11) / (double)CLOCKS_PER_SEC;
			//cout << "Duration vision: " << duration_vision << endl;
		}
	}

	//drawContours(I_inrange, contours, 0, Scalar(255,100,100), 2, 8);


	
	int found_index = 0;
	float* fdest = (float*)dest;
	for (int i = 0; i < width*height; i++) {

		ColorSpacePoint p = depth2rgb[i];
		CameraSpacePoint q = depth2xyz[i];
	
		//A.at<unsigned char>(row, col) = 244;

		//cout << i<<" "<<row << " " << col << endl;

		// Check if color pixel coordinates are in bounds
		if (p.X < 0 || p.Y < 0 || p.X > colorwidth || p.Y > colorheight) {
			*fdest++ = 0;
			*fdest++ = 0;
			*fdest++ = 0;
		}
		else {

		
			int idx = ((int)p.X) + colorwidth*((int)p.Y);
			
			*fdest++ = rgbimage[4 * idx + 0] / 255.;

			*fdest++ = rgbimage[4 * idx + 1] / 255.;
			*fdest++ = rgbimage[4 * idx + 2] / 255.;

		}
		// Don't copy alpha channel
	}


	
	
	

	if (colorframe) colorframe->Release();
}


//-----------------CAMERA CALIBRATION--------------------------//
void rgbcamera_calibration(IMultiSourceFrame* frame) {
	IColorFrame* colorframe;
	IColorFrameReference* frameref = NULL;
	frame->get_ColorFrameReference(&frameref);
	frameref->AcquireFrame(&colorframe);
	if (frameref) frameref->Release();

	if (!colorframe) return;

	int numSquares = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);

	int key = waitKey(1);








	

	
	for (int j = 0; j < numSquares; j++)
		obj.push_back(Point3f(j / numCornersHor, j%numCornersHor, 0.0f));



	// ------------------------------Get data from frame--------------------------------


	//frameref->AcquireFrame(&colorframe);

	//colorframe->CopyConvertedFrameDataToArray(colorwidth*colorheight * 4, rgbimage, ColorImageFormat_Bgra);
	//ScanImageAndReduceC(I, rgbimage);
	I = Mat(colorheight, colorwidth, CV_8UC4, &rgbimage);
	bool found = findChessboardCorners(I, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

	if (found)
	{
		//cornerSubPix(I, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
		drawChessboardCorners(I, board_sz, corners, found);


	}
	if (found != 0)
	{
		image_points.push_back(corners);
		object_points.push_back(obj);

		printf("Snap stored!");

		successes++;


	}
	imshow("hi", I);

	/*if (!calibrated){
		calibrateCamera(object_points, image_points, I.size(), intrinsic, distCoeffs, rvecs, tvecs);

		}*/


	if (colorframe) colorframe->Release();
	


}
void getKinectData() {
	color_index = 0;
	IMultiSourceFrame* frame = NULL;
	if (SUCCEEDED(reader->AcquireLatestFrame(&frame))) {
		GLubyte* ptr;
		glBindBuffer(GL_ARRAY_BUFFER, vboId);
		ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		if (ptr) {
			getDepthData(frame, ptr);
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, cboId);
		ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		if (ptr) {
			getRgbData(frame, ptr);
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		/*if (successes < numBoards){
			rgbcamera_calibration(frame);
			}
			else{
			if (!calibrated)
			calibrateCamera(object_points, image_points, I.size(), intrinsic, distCoeffs, rvecs, tvecs);
			calibrated = true;
			}*/

	}
	if (frame) frame->Release();
}

void rotateCamera() {
	static double angle = 0.;
	static double radius = 3.;
	double x = radius*sin(angle);
	double z = radius*(1 - cos(angle)) - radius / 2;
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity(); 
	gluLookAt(x, 0, z, 0, 0, radius / 2, 0, 1, 0);
	angle += 0;
}

void drawKinectData() {
	getKinectData();
	rotateCamera();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glVertexPointer(3, GL_FLOAT, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, cboId);
	glColorPointer(3, GL_FLOAT, 0, NULL);

	glPointSize(1.0f);
	glDrawArrays(GL_POINTS, 0, width*height);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}

int kinect_main(int argc, char* argv[], Geometry *p) {
	cout << "settping up kinect and camera" << endl;
	//Reading in the matrix and distortion_coefficients
	string filename = "camera.yml";
	//global_geo = *p;
	geo_ptr = p;

	FileStorage fs(filename, FileStorage::READ);

	fs["camera_matrix"] >> instrinsics;
	fs["distortion_coefficients"] >> distortion;

	//the 3d box
	axis_3.push_back(Point3f(0.0, 0.0, 0.0));
	axis_3.push_back(Point3f(0.1, 0, 0));
	axis_3.push_back(Point3f(0, 0.1, 0));
	axis_3.push_back(Point3f(0, 0, 0.1));
	axis_3.push_back(Point3f(0.1, 0.1, 0));
	axis_3.push_back(Point3f(0.1, 0, 0.1));
	axis_3.push_back(Point3f(0, 0.1, 0.1));
	axis_3.push_back(Point3f(0.1, 0.1, 0.1));
	
	//Our 2d object
	geo_deform.push_back(Point3f(0.0, -1.0, 0.0));
	geo_deform.push_back(Point3f(1.0, -1.0, 0));
	geo_deform.push_back(Point3f(0, 0.0, 0));
	geo_deform.push_back(Point3f(1.0, 0.0, 0.0));

	corners.push_back(Point2f(0.0, 0.0));
	corners.push_back(Point2f(0.0, 0.0));
	corners.push_back(Point2f(0.0, 0.0));

	corners.push_back(Point2f(0.0, 0.0));

	mesh_geometry_display.push_back(Point2f(0.0, 0.0));
	mesh_geometry_display.push_back(Point2f(0.0, 0.0));
	mesh_geometry_display.push_back(Point2f(0.0, 0.0));
	mesh_geometry_display.push_back(Point2f(0.0, 0.0));
	greenLower.push_back(Point3d(29, 86, 6));
	greenUpper.push_back(Point3d(64, 255, 255));

	//A = Mat::zeros(height, width, CV_8U);
	dictionary = cv::aruco::getPredefinedDictionary(cv::aruco:: DICT_4X4_50);
	ostringstream convert;
	for (int i = 0; i < 10; i++){
		cv::aruco::drawMarker(dictionary, i, 200, markerImage, 1);
		convert << i;
		imshow(convert.str(), markerImage);
	}
	//Initializing everything
	if (!init(argc, argv)) return 1;
	if (!initKinect()) return 1;
	

	// OpenGL setup
	glClearColor(0, 0, 0, 0);
	glClearDepth(1.0f);
	myfile.open("results.txt");
	//getting intrinsic depth camera values
	mapper->GetDepthCameraIntrinsics(cameraIntrinsics_kinect);
	buf_color_position = new unsigned short[2 * width*height];
	/*cameraIntrinsics_kinect->FocalLengthX = 5.9421434211923247e+02;
	cameraIntrinsics_kinect->FocalLengthY = 5.9104053696870778e+02;
	cameraIntrinsics_kinect->PrincipalPointX = 3.3930780975300314e+02;
	cameraIntrinsics_kinect->PrincipalPointY = 2.4273913761751615e+02;
	*/
	cout << "Focal Point X: " << cameraIntrinsics_kinect->FocalLengthX << endl;
	cout << "Focal Point Y: " << cameraIntrinsics_kinect->FocalLengthY << endl;

	cout << "Principal Point x:" << cameraIntrinsics_kinect->PrincipalPointX << endl;
	cout << "Principal Point y:" << cameraIntrinsics_kinect->PrincipalPointY << endl;

	// Set up array buffers
	const int dataSize = width*height * 3 * 4;
	glGenBuffers(1, &vboId);
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, dataSize, 0, GL_DYNAMIC_DRAW);
	glGenBuffers(1, &cboId);
	glBindBuffer(GL_ARRAY_BUFFER, cboId);
	glBufferData(GL_ARRAY_BUFFER, dataSize, 0, GL_DYNAMIC_DRAW);

	// Camera setup
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, width / (GLdouble)height, 0.1, 1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 0, 0, 0, 1, 0, 1, 0);
	const char **a;
	
	/*Geometry testing_geo;
	testing_geo.set_dim(3);
	testing_geo.read_nodes();
	testing_geo.read_elem();
	testing_geo.read_force();
	testing_geo.set_YoungPoisson(20000, 0.45);
	testing_geo.set_thickness(5);*/
		
		tracking_colors.push_back(Point2f(0.0, 0.0));
	tracking_colors.push_back(Point2f(0.0, 0.0));
	tracking_colors.push_back(Point2f(0.0, 0.0));

	tracking_colors.push_back(Point2f(0.0, 0.0));

	if (geo_ptr->get_dynamic() == true){
		geo_ptr->initialize_dynamic();
	
		geo_ptr->set_beta1(0.9); // if beta_2 >= beta1 and beta > 1/2 then the time stepping scheme is unconditionally stable.
		geo_ptr->set_beta2(0.9);
		geo_ptr->set_dt(1.5);
		geo_ptr->set_dynamic_alpha(0.056);//damping
		geo_ptr->set_dynamic_xi(0.016);//damping
		
	}
	diff.push_back(Point2f(0.0, 0.0));
	//global_geo.initialize_CUDA
	display_counter = 0;
	first_geo_init = true;
	get_mesh(geo_ptr);

	
	//intiliazing the number of points that will not move

	p->initialize_zerovector(9);
	//next we set what nodes we want to make stable
	int points[9];
	for (int i = 0; i < 9; i++){
		points[i] = i;
		
	}
	
	p->set_zero_nodes(points);

	////charuco
	imageSize.width = squaresX * squareLength + 2 * margins;
	imageSize.height = squaresY * squareLength + 2 * margins;
	//board_charuco = aruco::CharucoBoard::create(squaresX, squaresY, (float)squareLength, (float)markerLength_charuco, dictionary);
	//mesh(1,a);
	// Main loop
	rvec_aruco.push_back(Vec3d(0.0, 0.0, 0.0));
	tvec_aruco.push_back(Vec3d(0.0, 0.0, 0.0));
	int numSquares = numCornersHor * numCornersVer;
	//creating the reference chessboard
	for (int j = 0; j < numSquares; j++)
		obj.push_back(Point3f((j / numCornersHor)-5, j%numCornersHor, 0.0f));


	
	execute();
	return 0;
}
