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
int numCornersHor = 5;
int numCornersVer = 4;
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

float markerLength = 1;
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
Geometry global_geo;

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
	int e = global_geo.return_numElems();
	// global_geo.return_numElems()
	p->setSudoNode(20);
	p->setSudoForcex(1 / 6.0);
	p->setSudoForcey(1 / 6.0);
	if (!cuda_init){
		p->initialize_CUDA();
		cuda_init = true;
	}
	p->make_K_matrix();
	p->tt();
	mesh_geometry.empty();
	for (int i = 0; i <global_geo.return_numNodes(); i++){

		double dx = (geo_deform[1].x-geo_deform[0].x );
		if (first_geo_init == true){
			mesh_geometry.push_back(Point3f(((p->return_x(i) - 5.0f)), (p->return_y(i) - 5.0f), 0));
		}
		else {
			mesh_geometry[i] = (Point3f(((p->return_x(i) - 5.0f)),(p->return_y(i) - 5.0f), 0));
		}
		
		
		
	}
}


void draw_mesh(Geometry *p){
	int e = global_geo.return_numElems();
	// global_geo.return_numElems()

	for (int i = 0; i < global_geo.return_numElems(); i++){
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



		circle(I, mesh_geometry_display[node_considered1], 150 / 32.0, Scalar(200, 0, 80), -1, 1);
		circle(I, mesh_geometry_display[node_considered2], 150 / 32.0, Scalar(200, 0, 80), -1, 1);
		circle(I, mesh_geometry_display[node_considered3], 150 / 32.0, Scalar(200, 0, 80), -1, 1);
		putText(I, to_string(node_considered3), mesh_geometry_display[node_considered3], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		double dx = (geo_deform[1].x - geo_deform[0].x);
		string ss = "dx : " + to_string(dx);
		putText(I, (ss), Point2f(50.0,50.0), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		if (p->return_dim() == 3){

			circle(I, mesh_geometry_display[node_considered4], 70 / 32.0, Scalar(200, 0, 80), -1, 1);
		}



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
		cvtColor(I, I_gray, CV_BGR2GRAY);
		
		resize(I_gray, I_gray_resize, Size(colorwidth / 4, colorheight / 4));
		resize(hsv, hsv, Size(colorwidth / 4, colorheight / 4));

		cuda::flip(I_gray_resize, I_gray_resize, 1);
			
		
		//-------------------ARUCO-------------------------


		
		detectMarkers(I_gray_resize, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
	
		if (markerIds.size() > 0){
		//	//cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);

			
			for (unsigned int i = 0; i < markerIds.size(); i++){
				
				for (int j = 0; j < 4; j++){
					markerCorners[i][j].x = colorwidth - markerCorners[i][j].x*4.0;
					markerCorners[i][j].y = markerCorners[i][j].y*4.0;
					/*Point2f center((markerCorners[i][j].x), (markerCorners[i][j].y));

					circle(I, center, 2, Scalar(0, 0, 255), 3, 8, 0);*/
				}
				
			}
			aruco::estimatePoseSingleMarkers(markerCorners, markerLength, instrinsics, distortion, rvec_aruco, tvec_aruco);

		}
		
		
		//---------------ARUCO END---------------------
		
		Scalar greenlow = Scalar(40, 50, 50);
		Scalar greenhigh = Scalar(85, 220, 220);

		Scalar bluelow = Scalar(105, 50, 50);
		Scalar bluehigh = Scalar(125, 255, 255);

		
		Scalar redlow = Scalar(0, 100, 0);
		Scalar redhigh = Scalar(10, 255, 255);


		Scalar yellowlow = Scalar(20, 124, 123);
		Scalar yellowhigh = Scalar(30, 256, 256);
		
		
		inRange(hsv, yellowlow, yellowhigh, I_inrangeyellow);
		inRange(hsv, bluelow,bluehigh, I_inrangeblue);
		inRange(hsv, redlow, redhigh, I_inrangered);
		inRange(hsv, greenlow, greenhigh, I_inrangegreen);
		
		bitwise_or(I_inrangeyellow, I_inrangeblue, I_inrangeblue);
		bitwise_or(I_inrangered, I_inrangegreen, I_inrangegreen);
		bitwise_or(I_inrangeblue, I_inrangegreen, I_inrange);
		
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(0, 0));
		erode(I_inrange, I_inrange, element);
		
		dilate(I_inrange, I_inrange, element);
		imshow("i-range", I_inrange);
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
		clock_t startstart = clock();
		int x_pos=0, y_pos=0;
		int dummpy_used;
		string s;
		Mat colorImage = Mat::zeros(colorheight, colorwidth, CV_8UC4);
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
					int idx = ((int)p.X) + colorwidth*((int)p.Y);
					colorImage.data[index3_color + 0] = rgbimage[4 * idx + 0];
					colorImage.data[index3_color + 1] = rgbimage[4 * idx + 1];
					colorImage.data[index3_color + 2] = rgbimage[4 * idx + 2];
				}


			}
		}

		
		
		

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
			if (radius[i] > 5){
				if ((int)index3 < colorwidth*colorheight){

					int _X = (int)((depthSpace2[index3].X));
					int _Y = (int)((depthSpace2[index3].Y));
					if ((_X >= 0) && (_X < width) && (_Y >= 0) && (_Y < height)){
						int depth_index = (_Y*width) + _X;
						int index3_color = index3 * 4;
						CameraSpacePoint q = depth2xyz[depth_index];
						ColorSpacePoint p = depth2rgb[depth_index];
						int idx = ((int)p.X) + colorwidth*((int)p.Y);


						if ((rgbimage[4 * idx + 0] < 90) && (rgbimage[4 * idx + 1]>75) && (rgbimage[4 * idx + 2] < 90)){
							g = 200;
							//if (((int)w == (int)(center[i].x)) && ((int)h == (int)(center[i].y))){
							world_string[i] = "Green: " + to_string(q.X) + " " + to_string(q.Y) + " " + to_string(q.Z);
							world_3d[i] = Point2f(p.X, p.Y);
						
							tracking_colors[2].x = p.X;
							tracking_colors[2].y = p.Y;
						/*	mesh_geometry[1].x = geo_deform[2].x = q.X;
							mesh_geometry[1].y = geo_deform[2].y = q.Y;
							mesh_geometry[1].z = geo_deform[2].z = q.Z;*/

							geo_deform[2].x = q.X;
							 geo_deform[2].y = q.Y;
							geo_deform[2].z = q.Z;

							//circle(colorImage, Point((int)center[i].x, (int)center[i].y), radius[i], Scalar(200, g, 20), -1, 1);
							if (world_string[i].size() != 0){
								putText(I, world_string[i], Point((int)world_3d[i].x + 5, (int)world_3d[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
								circle(I, Point((int)world_3d[i].x, (int)world_3d[i].y), radius[i], Scalar(0, 255, 0), -1, 1);
							}
						}
						else if ((100 < rgbimage[4 * idx + 0]) && (rgbimage[4 * idx + 1] < 100) && (rgbimage[4 * idx + 2] < 100)){

							//if (((int)w == (int)(center[i].x)) && ((int)h == (int)(center[i].y))){
							world_string[i] = "Red: " + to_string(q.X) + " " + to_string(q.Y) + " " + to_string(q.Z);
							world_3d[i] = Point2f(p.X, p.Y);
							
							tracking_colors[3].x = p.X;
							tracking_colors[3].y = p.Y;
							/*mesh_geometry[76].x = geo_deform[3].x = q.X;
							mesh_geometry[76].y = geo_deform[3].y = q.Y;
							mesh_geometry[76].z = geo_deform[3].z = q.Z;*/

							geo_deform[3].x = q.X;
							geo_deform[3].y = q.Y;
							geo_deform[3].z = q.Z;
							//circle(colorImage, Point((int)center[i].x, (int)center[i].y), radius[i], Scalar(200, g, 20), -1, 1);
							if (world_string[i].size() != 0){
								putText(I, world_string[i], Point((int)world_3d[i].x + 5, (int)world_3d[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
								circle(I, Point((int)world_3d[i].x, (int)world_3d[i].y), radius[i], Scalar(0, 0, 255), -1, 1);
							}
						}
						else if ((80 > rgbimage[4 * idx + 0]) && (rgbimage[4 * idx + 1] < 80) && (rgbimage[4 * idx + 2] >60)){

							//if (((int)w == (int)(center[i].x)) && ((int)h == (int)(center[i].y))){
							world_string[i] = "Blue: " + to_string(q.X) + " " + to_string(q.Y) + " " + to_string(q.Z);
							world_3d[i] = Point2f(p.X, p.Y);
							
							tracking_colors[0].x = p.X;
							tracking_colors[0].y = p.Y;
							/*mesh_geometry[0].x = geo_deform[0].x = q.X;
							mesh_geometry[0].y= geo_deform[0].y = q.Y;
							mesh_geometry[0].z = geo_deform[0].z = q.Z;*/
							geo_deform[0].x = q.X;
							geo_deform[0].y = q.Y;
							 geo_deform[0].z = q.Z;

							//circle(colorImage, Point((int)center[i].x, (int)center[i].y), radius[i], Scalar(200, g, 20), -1, 1);
							if (world_string[i].size() != 0){
								putText(I, world_string[i], Point((int)world_3d[i].x + 5, (int)world_3d[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 0, 255), 2.0);
								circle(I, Point((int)world_3d[i].x, (int)world_3d[i].y), radius[i], Scalar(255, 0, 0), -1, 1);
							}
						}
						else if ((100 < rgbimage[4 * idx + 0]) && (rgbimage[4 * idx + 1] > 100) && (rgbimage[4 * idx + 2] < 80)){

							//if (((int)w == (int)(center[i].x)) && ((int)h == (int)(center[i].y))){
							world_string[i] = "Yellow: " + to_string(q.X) + " " + to_string(q.Y) + " " + to_string(q.Z);
							world_3d[i] = Point2f(p.X, p.Y);
						
							tracking_colors[1].x = p.X;
							tracking_colors[1].y = p.Y;
							/*mesh_geometry[75].x = geo_deform[1].x = q.X;
							mesh_geometry[75].y = geo_deform[1].y = q.Y;
							mesh_geometry[75].z = geo_deform[1].z = q.Z;*/

							 geo_deform[1].x = q.X;
							geo_deform[1].y = q.Y;
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

		start_K11 = std::clock();
		double dt = abs(startstart - start_K11);
		cout << " dt : " << dt << endl;

		
		solvePnP(Mat(geo_deform), Mat(tracking_colors), instrinsics, distortion, rvec_new, tvec_new, false);

		projectPoints(geo_deform, rvec_new, tvec_new, instrinsics, distortion, output_deform);
		if (markerIds.size() > 0){
			projectPoints(mesh_geometry, rvec_aruco[0], tvec_aruco[0], instrinsics, distortion, mesh_geometry_display);
			draw_mesh(&global_geo);
		}
		int thickness = 2;
		int lineType = 8;

		
		line(I, output_deform[0], output_deform[1], Scalar(255, 0, 255), thickness, lineType);
		line(I, output_deform[1], output_deform[3], Scalar(255, 0, 255), thickness, lineType);
		line(I, output_deform[3], output_deform[2], Scalar(255, 0, 255), thickness, lineType);
		line(I, output_deform[2], output_deform[0], Scalar(255, 0, 255), thickness, lineType);
		line(I, Point2f(0,0),Point2f(100,100), Scalar(100, 10, 255), thickness, lineType);
		
		imshow("original", I);
		//imshow("colorimage", colorImage);
		//imshow("In range", I_inrange);
		first_geo_init = false;
		get_mesh(&global_geo);
		duration_vision = (std::clock() - start_K11) / (double)CLOCKS_PER_SEC;
		cout << "Duration vision: " << duration_vision << endl;

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








	vector<Point2f> corners;

	vector<Point3f> obj;
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
	//Reading in the matrix and distortion_coefficients
	string filename = "camera.yml";
	global_geo = *p;

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

	A = Mat::zeros(height, width, CV_8U);
	dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
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
	
	Geometry testing_geo;
	testing_geo.set_dim(3);
	testing_geo.read_nodes();
	testing_geo.read_elem();
	testing_geo.read_force();
	testing_geo.set_YoungPoisson(20000, 0.45);
	testing_geo.set_thickness(5);
		
		tracking_colors.push_back(Point2f(0.0, 0.0));
	tracking_colors.push_back(Point2f(0.0, 0.0));
	tracking_colors.push_back(Point2f(0.0, 0.0));

	tracking_colors.push_back(Point2f(0.0, 0.0));
	
	first_geo_init = true;
	get_mesh(&global_geo);

	


	//mesh(1,a);
	// Main loop
	execute();
	return 0;
}
