
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
#include <math.h>
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


////for PCL and ICP

#if 1
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
//#include <pcl/features/organized_edge_detection.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/io/io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>  
#include <pcl/common/transforms.h>
#endif // 0


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

using namespace aruco;

// We'll be using buffer objects to store the kinect point cloud
GLuint vboId;
GLuint cboId;

const int widthd = 512;
const int heightd = 424;
// Intermediate Buffers
unsigned char rgbimage[colorwidth*colorheight * 4];    // Stores RGB color image
unsigned char bgrimage[colorwidth*colorheight * 4];    //stores bgr color image
//unsigned short bgrimage2[colorwidth*colorheight * 4];    //stores bgr color image
//unsigned short infrared_vec[widthd*heightd * 4];    //stores bgr color image
int is_aruco[colorwidth*colorheight];
ColorSpacePoint color_position[widthd*heightd];
int color_index;
ColorSpacePoint depth2rgb[widthd*heightd];             // Maps depth pixels to rgb pixels
ColorSpacePoint dummy2[colorwidth*colorheight];             // Maps depth pixels to rgb pixels
CameraSpacePoint depth2xyz[widthd*heightd];			 // Maps depth pixels to 3d coordinates
//CameraSpacePoint depth2xyz_found[widthd*heightd];			 // Maps to the 3d coordinates of the pixels that are found
//CameraSpacePoint depth2xyz_found_different_dim[widthd*heightd / 2];
DepthSpacePoint *depth_found;
// Kinect Variables
IKinectSensor* sensor;             // Kinect sensor
IMultiSourceFrameReader* reader;   // Kinect data source
CameraIntrinsics cameraIntrinsics_kinect[1];
ICoordinateMapper* mapper;         // Converts between depth, color, and 3d coordinates
int num_station_nodes = 0;
Mat colorImage = Mat::zeros(colorheight, colorwidth, CV_8UC4);
//int station_nodes[13] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
int station_nodes[1] = {0};
//int station_nodes[16] = { 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60 }; for "TOP"
//int station_nodes[16] = { 3    , 7  ,  11  ,  15,    19   , 23 ,   27 ,   31 ,   35  ,  39  ,  43  ,  47 ,   51   , 55 ,   59    ,63 };
//success for camera calibration
Mat intrinsic = Mat(3, 3, CV_32FC1);
Mat distCoeffs;
vector< Vec3d > rvecs, tvecs, tvecs_new;
vector<Vec3d> rvec_aruco, tvec_aruco;
Vec3d rvec_new, tvec_new;
Mat I = Mat(colorheight, colorwidth, CV_8UC4, &bgrimage);
Mat I_infrared = Mat(heightd, widthd, CV_16UC1);
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
Mat I_gray2;
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
int found_colour[heightd*widthd];
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
vector<Point2f> aruco_center(4);
vector<Point3f> circles_detected;
int width_depth, height_depth;
float markerLength = 0.1;
Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
cv::Mat rvec(3, 1, cv::DataType<double>::type);
cv::Mat tvec(3, 1, cv::DataType<double>::type);
vector<Point3f> obj;
vector<Point3f> partial_cube;
vector<Point2f> corners;
vector<Point3d> greenLower;
vector<Point3d> greenUpper;

bool initial_pnp = true;
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

//Defining studo force information
int num_nodes_interseted = 2;
vector<Point2f> aruco_position;//we have t+1
vector<Point2f> aruco_postion2;
vector<Point2f> meshnode_position;//we have t
vector<Point2f> meshnode_position2;
vector<Point2f> meshnode_position3;
vector<Point2f> meshnode_position4;
int node_interested = 15 - 1; //edge1 from matlab
int node_interested2 = 138 - 1; //edge3 from matlab
int node_interested3 = 14- 1; //edge4 from matlab
int node_interested4 = 137 - 1; //edge2 from matlab
vector<Point2f> diff;//the diff vector for sudoforce 1
vector<Point2f> diff2;// diff vector for sudoforce2
vector<Point2f> diff3;// diff vector for sudoforce2 
vector<Point2f> diff4;// diff vector for sudoforce2
vector<Point3f> diff3D;// diff vector for sudoforce2f
vector<Point3f> diff3D2;// diff vector for sudoforce2
int num_green_dots=0;
//int green_nodes[1] = { 108, 109, 110, 111, 112, 113, 208, 209, 210, 211, 212, 213, 308, 309, 310, 311, 312, 313 };
int green_nodes[1] = { 0 };
//chessboard markers
#define FEM_USE false
int counter = 0;
int markerCount_global = 0;
//
//
////Ptr<aruco::CharucoBoard> board_charuco = aruco::CharucoBoard::create(squaresX, squaresY, (float)squareLength, (float)markerLength_charuco, dictionary);
//Ptr<aruco::CharucoBoard> board_charuco; //= aruco::CharucoBoard::create(squaresX, squaresY, (float)squareLength, (float)markerLength_charuco, dictionary);

//PCL DECLARATIONS
#if 1
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_new(new pcl::PointCloud<pcl::PointXYZRGB>());

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_KINECT(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZ>::Ptr xcloud_box(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_generated_var(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr xcloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr xcloud_new(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr xcloud_box_mesh(new pcl::PointCloud<pcl::PointXYZ>());
pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Matrix4 transformation2;
Eigen::Affine3f *global_initial_ptr;
#if 1
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
#else // 0
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
#endif
pcl::PolygonMesh *global_mesh_ptr;
#endif // 0




#define WRITE_ARUCO_POS 1

#if WRITE_ARUCO_POS 1

#endif
///ICP
pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
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
#if 0

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
#if 1
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(cloud2);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud2, rgb, "sample cloud2");

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud2");
	viewer->removePointCloud("sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
#endif // 0

	return (viewer);
}
#endif // 0



bool initKinect() {
	if (FAILED(GetDefaultKinectSensor(&sensor))) {
		return false;
	}
	if (sensor) {
		sensor->get_CoordinateMapper(&mapper);

		sensor->Open();
		sensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color, &reader);
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

	if (!depthframe)
		return;

	// Get data from frame
	unsigned int sz;
	unsigned short* buf;

	depthframe->AccessUnderlyingBuffer(&sz, &buf);

	// Write vertex coordinates
	mapper->MapDepthFrameToCameraSpace(widthd*heightd, buf, widthd*heightd, depth2xyz);
	//mapper->MapDepthFrameToCameraSpace(widthd*heightd, buf, widthd*heightd / 2, depth2xyz_found_different_dim);

	float* fdest = (float*)dest;
#if 0
	if (cloud_KINECT->size() > 0)
		cloud_KINECT->clear();
	pcl::PointXYZRGB dummyVar;
#endif // 0

	//viewer->removeAllPointClouds();
	for (unsigned int i = 0; i < sz; i++) {
#if 1
		/*cloud->at(i).x = cloud->at(i).x / 100.0;
		cloud->at(i).y = cloud->at(i).y / 100.0;
		cloud->at(i).z = cloud->at(i).z / 100.0; */
		/** fdest++ = depth2xyz[i].X;
		*fdest++ = depth2xyz[i].Y;
		*fdest++ = depth2xyz[i].Z;*/
#else

		/*if ((depth2xyz[i].X + depth2xyz[i].Y + depth2xyz[i].Z)<100.0)*/
		dummyVar.x = depth2xyz[i].X;
		dummyVar.y = depth2xyz[i].Y;
		dummyVar.z = depth2xyz[i].Z;
		dummyVar.r = 203;
		dummyVar.g = 230;
		dummyVar.b = 100;
		//cloud_KINECT->points.push_back(pcl::PointXYZ(depth2xyz[i].X, depth2xyz[i].Y, depth2xyz[i].Z));

		cloud_KINECT->points.push_back(dummyVar);
#endif // 0


#if 0

#endif // 0


	}
#if 0

	viewer->updatePointCloud(cloud_KINECT, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "sample cloud");
	viewer->spinOnce();
	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
#endif // 0

	//viewer->spinOnce();
	//viewer->addPointCloud<pcl::PointXYZ>(cloud_KINECT, "sample cloud");
	//viewer->addPointCloud<pcl::PointXYZRGB>(cloud_, rgb, "sample cloud2");
	//mapper->MapColorFrameToDepthSpace(widthd*heightd, buf_color_position, 2*widthd*heightd, depth_found);
	mapper->MapDepthFrameToColorSpace(sz, buf, widthd*heightd, depth2rgb);

	mapper->MapColorFrameToDepthSpace(sz, buf, colorwidth*colorheight, depthSpace2);
	//mapper->MapDepthPointsToColorSpace(1, &depthSpace2[0], sz, buf, 1, dummycolor);
	// Fill in depth2rgb map


	//
	if (depthframe) depthframe->Release();
	}

void get_mesh(Geometry *p){
	int e = p->return_numElems();
	double total_FEM_time = std::clock();

	// global_geo.return_numElems()
	if (0){

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
		if (1){
			double divisor = 20.0;
#if 0
			//p->set_force_rest(true);
			p->setSudoNode(node_interested);
			p->setSudoForcex(-diff[0].x / divisor);
			p->setSudoForcey(-diff[0].y / divisor);
			//p->call_sudo_force_func();
			p->setSudoNode(node_interested2);
			p->setSudoForcex(-diff2[0].x / divisor);
			p->setSudoForcey(-diff2[0].y / divisor);
#endif // 0
#if 0

			//p->sudo_force_value.clear();
			p->sudo_force_index[0] = node_interested;
			p->sudo_force_index[1] = node_interested2;
			double divisor2 = 200.0;
			if (norm(diff[0]) < 3){
				p->sudo_force_value1[0] = -(diff[0].x / divisor2);
				p->sudo_force_value1[1] = -(diff[0].y / divisor2);
			}
			else{
				p->sudo_force_value1[0] = -1.0/(diff[0].x / divisor);
				p->sudo_force_value1[1] = -1.0/(diff[0].y / divisor);
			}

			if (norm(diff2[0]) < 3){
				p->sudo_force_value2[0] = -(diff2[0].x / divisor2);
				p->sudo_force_value2[1] = -(diff2[0].y / divisor2);
			}
			else{
				p->sudo_force_value2[0] = -1.0/(diff2[0].x / divisor);
				p->sudo_force_value2[1] = -1.0 / (diff2[0].y / divisor);
			}

			p->sudo_force_value1[0] = p->sudo_force_value1[0] / divisor2;
			p->sudo_force_value1[1] = p->sudo_force_value1[1] / divisor2;

			p->sudo_force_value2[0] = p->sudo_force_value2[0] / divisor2;
			p->sudo_force_value2[1] = p->sudo_force_value2[1] / divisor2;
			//p->sudo_force_value.push_back(-diff[0] / divisor);
			//p->sudo_force_value.push_back(-diff2[0] / divisor);

#endif // 0

#if 1

			//p->sudo_force_value.clear();
			p->sudo_force_index[0] = node_interested;
			p->sudo_force_index[1] = node_interested2;
			p->sudo_force_index[2] = node_interested3;
			p->sudo_force_index[3] = node_interested4;
			double d_est = 1.0;
			p->sudo_force_value1[0] = -(diff.at(0).x / divisor);
			p->sudo_force_value1[1] = -(diff.at(0).y / divisor);
			p->sudo_force_value2[0] = -(diff2.at(0).x / divisor);
			p->sudo_force_value2[1] = -(diff2.at(0).y / divisor);
			p->sudo_force_value3[0] = -(diff3.at(0).x / divisor);
			p->sudo_force_value3[1] = -(diff3.at(0).y / divisor);
			p->sudo_force_value4[0] = -(diff4.at(0).x / divisor);
			p->sudo_force_value4[1] = -(diff4.at(0).y / divisor);
#if 0
			p->sudo_force_value1[0] = -(diff[0].x / divisor);
			p->sudo_force_value1[1] = -(diff[0].y / divisor);
			p->sudo_force_value2[0] = -(diff2[0].x / divisor);
			p->sudo_force_value2[1] = -(diff2[0].y / divisor);
#endif // 0

			//p->sudo_force_value1[0] = -(0.0 / divisor);
			//p->sudo_force_value1[1] = -(0.0 / divisor);
			//p->sudo_force_value2[0] = -(0.0 / divisor);
			//p->sudo_force_value2[1] = -(0.0 / divisor); 

			//p->sudo_force_value.push_back(-diff[0] / divisor);
			//p->sudo_force_value.push_back(-diff2[0] / divisor);

#endif // 0



			/*	p->set_force_rest(false);
			p->call_sudo_force_func();*/
			/*int sign_x;
			int sign_y;
			if (diff[0].x < 0){
			sign_x = -1;
			}
			else if (diff[0].x >= 0){
			sign_x = 1;
			}
			if (diff[0].y < 0){
			sign_y = -1;
			}
			else if (diff[0].y >= 0){
			sign_y = 1;
			}

			p->setSudoForcex(-sign_x*log(sign_x*diff[0].x/1500.0+1));
			p->setSudoForcey(-sign_y*log(sign_y*diff[0].y /1500.0 + 1));*/
			/*if (norm(diff[0]) > 30.0){
			p->initialize_dynamic();
			}*/

			/*else{
			p->setSudoForcex(0.0);
			p->setSudoForcey(0.0);
			}*/
			/*		p->setSudoForcex(1.0);
			p->setSudoForcey(1.0);*/
		}
		else {
			p->setSudoNode(node_interested);
			p->setSudoForcex(0);
			p->setSudoForcey(0);
		}
		p->make_K_matrix();



		if (p->get_dynamic()){//p->get_dynamic()
			p->find_b();

			p->update_vector();
			p->update_dynamic_vectors();
			p->update_dynamic_xyz();
		}
		else {
			p->tt();
		}
	}
	else {

	}
	cout << "TOTAL FEM TIME : " << (total_FEM_time - std::clock()) / ((double)CLOCKS_PER_SEC) << endl;

	mesh_geometry.empty();
	mesh_generated_var->points.clear();
	pcl::PointXYZ dummy_var;
	for (int i = 0; i < p->return_numNodes(); i++){
		if (p->return_dim() == 2){
			//double dx = (geo_deform[1].x-geo_deform[0].x );
			if (first_geo_init == true){
				mesh_geometry.push_back(Point3f(((p->return_x(i))), (p->return_y(i)), 0.0));
				dummy_var.x = p->return_x(i);
				dummy_var.y = p->return_y(i);
				dummy_var.z = p->return_z(i);
				mesh_generated_var->points.push_back(dummy_var);
			}
			else {
				mesh_geometry[i] = (Point3f(((p->return_x(i))), (p->return_y(i)), 0.0));
				dummy_var.x = p->return_x(i);
				dummy_var.y = p->return_y(i);
				dummy_var.z = p->return_z(i);
				mesh_generated_var->points.push_back(dummy_var);
			}
		}
		else if (p->return_dim() == 3){
			if (first_geo_init == true){
				mesh_geometry.push_back(Point3f(((p->return_x(i))), (p->return_y(i)), p->return_z(i)));
				dummy_var.x = p->return_x(i);
				dummy_var.y = p->return_y(i);
				dummy_var.z = p->return_z(i);
				mesh_generated_var->points.push_back(dummy_var);
			}
			else {
				mesh_geometry[i] = (Point3f(((p->return_x(i))), (p->return_y(i)), p->return_z(i)));
				dummy_var.x = p->return_x(i);
				dummy_var.y = p->return_y(i);
				dummy_var.z = p->return_z(i);
				mesh_generated_var->points.push_back(dummy_var);
			}
		}



	}
	
	
	pcl::transformPointCloud(*mesh_generated_var, *mesh_generated_var, *global_initial_ptr);
	diff3D2.clear();
	diff3D2.push_back(Point3f(mesh_generated_var->points.at(1 - 1).x, mesh_generated_var->points.at(1 - 1).y, (mesh_generated_var->points.at(1 - 1).z)));

	viewer->updatePointCloud(mesh_generated_var, "mesh generated");

	viewer->spinOnce();
}


void draw_mesh(Geometry *p, Mat I){
	int e = p->return_numElems();
	// global_geo.return_numElems()
	meshnode_position.clear();
	meshnode_position2.clear();
	meshnode_position3.clear();
	meshnode_position4.clear();

	for (int j = 0; j < num_green_dots; j++){
		
		circle(I, mesh_geometry_display[green_nodes[j]], 7, Scalar(50, 10, 100), 2);
	}
	for (int i = 0; i < p->return_numElems(); i++){
		int node_considered4;

		int node_considered1 = p->node_number_inElem(i, 0);
		int node_considered2 = p->node_number_inElem(i, 1);
		int node_considered3 = p->node_number_inElem(i, 2);
		if (p->return_dim() == 3){
			node_considered4 = p->node_number_inElem(i, 3);
		}




		int thickness = 1.9;
		LineTypes lineType = LINE_AA;

		//GpuMat image1(Size(1902, 1080), CV_8U);
		Scalar color_line = Scalar(0, 0,0);
		line(I, mesh_geometry_display[node_considered1], mesh_geometry_display[node_considered2], color_line, thickness, lineType);

		line(I, mesh_geometry_display[node_considered3], mesh_geometry_display[node_considered1], color_line, thickness, lineType);

		if (p->return_dim() == 3){
			line(I, mesh_geometry_display[node_considered2], mesh_geometry_display[node_considered4], Scalar(100, 50, 255), thickness, lineType);
			line(I, mesh_geometry_display[node_considered4], mesh_geometry_display[node_considered3], Scalar(100, 50, 255), thickness, lineType);

			line(I, mesh_geometry_display[node_considered1], mesh_geometry_display[node_considered4], Scalar(100, 50, 255), thickness, lineType);

			line(I, mesh_geometry_display[node_considered3], mesh_geometry_display[node_considered2], Scalar(100, 50, 255), thickness, lineType);
			line(I, mesh_geometry_display[node_considered2], mesh_geometry_display[node_considered3], color_line, thickness, lineType);
		}
		else {
			line(I, mesh_geometry_display[node_considered2], mesh_geometry_display[node_considered3], color_line, thickness, lineType);
		}
		//not good programming, node_intersted1
		//-----------------Getting the mesh nodes in camera space--------------//
		if (node_considered1 == node_interested){
			meshnode_position.push_back((mesh_geometry_display[node_considered1]));
			circle(I, mesh_geometry_display[node_considered1], 20, Scalar(0, 100, 255), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered1].x) + "   " + to_string(mesh_geometry_display[node_considered1].y), mesh_geometry_display[node_considered1], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		}
		else if (node_considered2 == node_interested){
			meshnode_position.push_back((mesh_geometry_display[node_considered2]));
			circle(I, mesh_geometry_display[node_considered2], 20, Scalar(0, 100, 255), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered2].x) + "   " + to_string(mesh_geometry_display[node_considered2].y), mesh_geometry_display[node_considered2], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

		}
		else if (node_considered3 == node_interested){
			meshnode_position.push_back((mesh_geometry_display[node_considered3]));
			circle(I, mesh_geometry_display[node_considered3], 20, Scalar(0, 100, 255), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered3].x) + "   " + to_string(mesh_geometry_display[node_considered3].y), mesh_geometry_display[node_considered3], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

		}
		//node_interested2
		if (node_considered1 == node_interested2){
			meshnode_position2.push_back((mesh_geometry_display[node_considered1]));
			circle(I, mesh_geometry_display[node_considered1], 20, Scalar(255, 100, 255), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered1].x) + "   " + to_string(mesh_geometry_display[node_considered1].y), mesh_geometry_display[node_considered1], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		}
		else if (node_considered2 == node_interested2){
			meshnode_position2.push_back((mesh_geometry_display[node_considered2]));
			circle(I, mesh_geometry_display[node_considered2], 20, Scalar(255, 100, 255), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered2].x) + "   " + to_string(mesh_geometry_display[node_considered2].y), mesh_geometry_display[node_considered2], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

		}
		else if (node_considered3 == node_interested2){
			meshnode_position2.push_back((mesh_geometry_display[node_considered3]));
			circle(I, mesh_geometry_display[node_considered3], 20, Scalar(255, 100, 255), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered3].x) + "   " + to_string(mesh_geometry_display[node_considered3].y), mesh_geometry_display[node_considered3], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

		}
		//node_interested3

		if (node_considered1 == node_interested3){
			meshnode_position3.push_back((mesh_geometry_display[node_considered1]));
			circle(I, mesh_geometry_display[node_considered1], 20, Scalar(255, 100, 100), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered1].x) + "   " + to_string(mesh_geometry_display[node_considered1].y), mesh_geometry_display[node_considered1], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		}
		else if (node_considered2 == node_interested3){
			meshnode_position3.push_back((mesh_geometry_display[node_considered2]));
			circle(I, mesh_geometry_display[node_considered2], 20, Scalar(255, 100, 100), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered2].x) + "   " + to_string(mesh_geometry_display[node_considered2].y), mesh_geometry_display[node_considered2], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

		}
		else if (node_considered3 == node_interested3){
			meshnode_position3.push_back((mesh_geometry_display[node_considered3]));
			circle(I, mesh_geometry_display[node_considered3], 20, Scalar(255, 100, 100), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered3].x) + "   " + to_string(mesh_geometry_display[node_considered3].y), mesh_geometry_display[node_considered3], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

		}

		if (node_considered1 == node_interested4){
			meshnode_position4.push_back((mesh_geometry_display[node_considered1]));
			circle(I, mesh_geometry_display[node_considered1], 20, Scalar(255, 100, 100), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered1].x) + "   " + to_string(mesh_geometry_display[node_considered1].y), mesh_geometry_display[node_considered1], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		}
		else if (node_considered2 == node_interested4){
			meshnode_position4.push_back((mesh_geometry_display[node_considered2]));
			circle(I, mesh_geometry_display[node_considered2], 20, Scalar(255, 100, 100), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered2].x) + "   " + to_string(mesh_geometry_display[node_considered2].y), mesh_geometry_display[node_considered2], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

		}
		else if (node_considered3 == node_interested4){
			meshnode_position4.push_back((mesh_geometry_display[node_considered3]));
			circle(I, mesh_geometry_display[node_considered3], 20, Scalar(255, 100, 100), 4);
			//putText(I, to_string(mesh_geometry_display[node_considered3].x) + "   " + to_string(mesh_geometry_display[node_considered3].y), mesh_geometry_display[node_considered3], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

		}

	
		if (1){ // if draw zero u points
			bool yes = false;
			int node_yes;
			int dummy_node;
			for (int m = 0; m < num_station_nodes; m++){
				dummy_node = station_nodes[m];
				if ((node_considered1 == dummy_node)){
					yes = true;
					node_yes = node_considered1;
				}
				else if ((node_considered3 == dummy_node)){
					yes = true;
					node_yes = node_considered3;
				}
				else if (node_considered2 == dummy_node){
					yes = true;
					node_yes = node_considered2;
				}
				if (p->return_dim() == 3){
					if (node_considered4 == dummy_node){
						yes = true;
						node_yes = node_considered4;

					}
				}
			}

			if (yes){
				//meshnode_position.push_back((mesh_geometry_display[node_considered1]));
				circle(I, mesh_geometry_display[node_yes], 5, Scalar(100, 58, 58), 5);
				//putText(I, to_string(mesh_geometry_display[node_considered1].x) + "   " + to_string(mesh_geometry_display[node_considered1].y), mesh_geometry_display[node_considered1], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
			}
		}

#if 0
		putText(I, to_string(node_considered1), mesh_geometry_display[node_considered1], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		putText(I, to_string(node_considered2), mesh_geometry_display[node_considered2], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
		putText(I, to_string(node_considered3), mesh_geometry_display[node_considered3], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);

#endif // 0
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

		//circle(I, mesh_geometry_display[node_considered1], 100 / 80.0, Scalar(10, 200, 255), -1, 1);
		//circle(I, mesh_geometry_display[node_considered2], 100 / 80.0, Scalar(10, 200, 255), -1, 1);
		//circle(I, mesh_geometry_display[node_considered3], 100 / 80.0, Scalar(10, 200, 255), -1, 1);

	}

}

//----------------------get RGB DATA-----------------------//
void getRgbData(IMultiSourceFrame* frame, GLubyte* dest) {
	IColorFrame* colorframe;
	//	IInfraredFrame* infraredFrame;
	//	IInfraredFrameReference* infraredref = NULL;
	IColorFrameReference* frameref = NULL;
	frame->get_ColorFrameReference(&frameref);
	//	frame->get_InfraredFrameReference(&infraredref);
	frameref->AcquireFrame(&colorframe);

	//	infraredref->AcquireFrame(&infraredFrame);
	if (frameref) frameref->Release();
	//if (infraredref) infraredref->Release();

	if (!colorframe) return;
	//if (!infraredFrame) return;
	unsigned int sz;
	unsigned char*  buffer;

	//colorframe->AccessRawUnderlyingBuffer(&sz, &buffer);
	// Get data from frame

	//mapper->MapColorFrameToDepthSpace(widthd*heightd, buf, widthd*heightd, dummy2);

	colorframe->CopyConvertedFrameDataToArray(colorwidth*colorheight * 4, rgbimage, ColorImageFormat_Rgba);
	colorframe->CopyConvertedFrameDataToArray(colorwidth*colorheight * 4, bgrimage, ColorImageFormat_Bgra);
	//infraredFrame->CopyFrameDataToArray(512 * 424*4,infrared_vec);
	//infraredFrame->CopyFrameDataToArray(widthd*heightd*4, reinterpret_cast<UINT16*>(I_infrared.data) );
	//colorframe->CopyConvertedFrameDataToArray(colorwidth*colorheight * 4, reinterpret_cast<BYTE*>(bgrimage2), ColorImageFormat_Bgra);

	std::clock_t aruco_time;
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


		std::ofstream in_disp_aruco("aruco" + to_string(counter) + ".txt");
		counter++;
		//------------------image processing to find the contour---------------//
		const float dp = 2.0f;
		const float minDist = 0.0;
		const int minRadius = 16;
		const int maxRadius = 30;
		const int cannyThreshold = 00;
		const int votesThreshold = 0;
		cv::Ptr<cv::cuda::HoughCirclesDetector> houghCircles = cv::cuda::createHoughCirclesDetector(dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);
		cv::cuda::GpuMat d_circles;
		//imwrite("test" + to_string(display_counter) + ".jpg", I);

		input_gpu.upload(I);

		cuda::cvtColor(input_gpu, output_gpu, COLOR_BGR2HSV);
		
		//d_circles.download(circles_detected);
		output_gpu.download(hsv);

		//inRange(I, blcklow, blckhigh, I_inrangeyellow);//////////////////
		//imshow("I_gray_resize", I_inrangeyellow);

		cvtColor(I, I_gray, CV_BGR2GRAY);




		double resize_num = 2.0;

		resize(I_gray, I_gray_resize, Size(colorwidth / resize_num, colorheight / resize_num));
		//resize(hsv, hsv, Size(colorwidth / 4, colorheight / 4));

		flip(I_gray_resize, I_gray_resize, 1);

		flip(I, I_flipped, 1);
		//imshow("I_gray_resize", I_gray_resize);
		/*imwrite(to_string(write_counter) + ".png", I_flipped);
		write_counter++;*/
		unsigned int numSquares = numCornersHor * numCornersVer;
		Size board_sz = Size(numCornersHor, numCornersVer);
		//bool found = findChessboardCorners(I_gray_resize, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		bool found = false;
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

			aruco_time = std::clock();

			//detectMarkers(I_gray_resize, dictionary, markerCorners, markerIds);
			/*aruco_center.clear();
			aruco_center.reserve(6);*/

			aruco_position.clear();
			//vector< Point2f > charucoCorners; vector< int > markerIds, charucoIds;
			if (markerIds.size() > 0){
				//	//cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
				int dummy_size = markerIds.size();
				if (dummy_size>4) {
					markerCount_global = 4;
				}
				else{
					markerCount_global = dummy_size;
					
				}

				for (unsigned int i = 0; i < markerCount_global; i++){
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
					if (markerIds[i] == 4){
						aruco_center.at(0) = (Point2f(x_ave / 4.0, y_ave / 4.0));
						circle(I, aruco_center[0], 10, cv::Scalar(255, 0, 0), 3);
						aruco_position.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));
					}
					else if (markerIds[i] == 9){
						aruco_center.at(1) = (Point2f(x_ave / 4.0, y_ave / 4.0));
						circle(I, aruco_center[1], 10, cv::Scalar(255, 0, 0), 3);
						aruco_position.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));
					}
					else if (markerIds[i] == 1){
						aruco_center.at(2) = (Point2f(x_ave / 4.0, y_ave / 4.0));
						circle(I, aruco_center[2], 10, cv::Scalar(255, 0, 0), 3);
						aruco_position.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));
					}
					else if (markerIds[i] == 2){
						aruco_center.at(3) = (Point2f(x_ave / 4.0, y_ave / 4.0));
						circle(I, aruco_center[3], 10, cv::Scalar(255, 0, 0), 3);
						aruco_position.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));
					}
					/*else if (markerIds[i] == 3){
						aruco_center.at(4) = (Point2f(x_ave / 4.0, y_ave / 4.0));
						circle(I, aruco_center[i], 10, cv::Scalar(255, 0, 0), 3);
						aruco_position.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));
					}
					else if (markerIds[i] == 0){
						aruco_center.at(5) = (Point2f(x_ave / 4.0, y_ave / 4.0));
						circle(I, aruco_center[i], 10, cv::Scalar(255, 0, 0), 3);
						aruco_position.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));
					}*/
					//aruco_center.push_back(Point2f(x_ave / 4.0, y_ave / 4.0));

					//putText(I, ".", Point((int)aruco_center[i].x, (int)aruco_center[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);

					/*if (markerIds[i] == 12){

					aruco_center_id = i;
					}
					*/

				}

				//aruco::estimatePoseSingleMarkers(markerCorners, markerLength, instrinsics, distortion, rvec_aruco, tvec_aruco);
				////IF WE ARE WRITING TO FILE THE CENTERS OF THE ARUCO MARKERS
				cloud->points.clear();
				cloud_new->points.clear();
				pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_aruco(new pcl::PointCloud<pcl::PointXYZ>());
				pcl::PointXYZRGB dummyVar;
				pcl::PointXYZ dummyVarPos;
				if (1){ // detection of aruco marker positions and plotting them on the PCL visulizer
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
					for (unsigned int i = 0; i < markerCount_global; i++){
						int index3 = ((int)aruco_center[i].y)*colorwidth + (int)aruco_center[i].x;
						ColorSpacePoint dummycolor;

						int _X = (int)depthSpace2[index3].X;
						int _Y = (int)depthSpace2[index3].Y;
						// _X = (int)(aruco_center[i].x*static_cast<double>(width)/colorwidth);
						// _Y = (int)(aruco_center[i].y*static_cast<double>(height) / colorheight);
						double actualx;
						double actualy;
						double actualz;
						if ((_X >= 0) && (_X < widthd) && (_Y >= 0) && (_Y < heightd)){
							int depth_index = (_Y*widthd) + _X;

							/*CameraSpacePoint q = depth2xyz[depth_index]*/
							ColorSpacePoint p = depth2rgb[depth_index];
							CameraSpacePoint world_point_camera = depth2xyz[depth_index];
							int idx = ((int)p.X) + colorwidth*((int)p.Y);
							actualx = (p.X - c_x)*world_point_camera.Z / f_x;
							actualy = (p.Y - c_y)*world_point_camera.Z / f_y;
							actualz = world_point_camera.Z;
							//if (actualz >= INFINITE)break;


							in_disp_aruco << markerIds[i] << endl;
							in_disp_aruco << actualx << " " << actualy << " " << actualz << endl;
							//in_disp << markerIds[i] << " " << actualx << " " << actualy << " " << actualz << endl;
							outputmesg = to_string(markerIds[i]);// +" Pos: " + to_string(actualx) + " " + to_string(actualy) + " " + to_string(actualz);
							outputcoord = "id:" + to_string(markerIds[i]) + " Pos: " + to_string(actualx) + " " + to_string(actualy) + " " + to_string(actualz);
							putText(I, outputmesg, Point((int)aruco_center[i].x, (int)aruco_center[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
							putText(I, outputcoord, Point((int)50, 10 * markerIds[i]), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);



							colorImage.data[index3 + 0] = rgbimage[4 * idx + 0]; // i assume here that colorimage is bgr?
							colorImage.data[index3 + 1] = rgbimage[4 * idx + 1];

							colorImage.data[index3 + 2] = rgbimage[4 * idx + 2];
							colorImage.data[index3 + 3] = rgbimage[4 * idx + 3];
							dummyVar.x = dummyVarPos.x = actualx;
							dummyVar.y = dummyVarPos.y = actualy;
							dummyVar.z = dummyVarPos.z = world_point_camera.Z;
							dummyVar.r = 255;
							dummyVar.g = 255;
							dummyVar.b = 50;
							//cloud_KINECT->points.push_back(pcl::PointXYZ(depth2xyz[i].X, depth2xyz[i].Y, depth2xyz[i].Z));

							cloud_new->points.push_back(dummyVar);
							dummy_aruco->points.push_back(dummyVarPos);

						}

					}


					//write_counter++;
					//in_disp.close();
				}

				cout << "ARUCO TRACKING TIME : " << (aruco_time - std::clock()) / ((double)CLOCKS_PER_SEC) << endl;
				if (dummy_aruco->points.size() == 4){
					icp.setInputTarget(dummy_aruco);
					icp.align(*xcloud_box);
					
					/*pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> TESVD;
					TESVD.estimateRigidTransformation(*xcloud_box, *dummy_aruco, transformation2);*/
					icp.setInputSource(xcloud_box);
					
#if 1
					pcl::transformPointCloud(*xcloud_box_mesh, *xcloud_box_mesh, icp.getFinalTransformation());
					Eigen::Matrix4f f_4;
					f_4 = icp.getFinalTransformation();
					
					Eigen::Affine3f dummy_trans;
					dummy_trans = f_4;
					*global_initial_ptr = dummy_trans* (*global_initial_ptr);

					cout << " transformation matrix rotation: " << endl;
					
					cout << global_initial_ptr->rotation() << endl;

					cout << " transformation matrix translation: " << endl;

					cout << global_initial_ptr->translation() << endl;
					pcl::toPCLPointCloud2(*xcloud_box_mesh, global_mesh_ptr->cloud);
					viewer->updatePolygonMesh(*global_mesh_ptr, "meshes");



#endif // 0

				}
				//FOR 3D FORCES BUT NOT IMPLEMENTED YET
				/*diff3D.clear();
				if (dummy_aruco->points.size() == 4){
					for (int mn = 0; mn < 4; mn++){
						diff3D.push_back(Point3f(abs(xcloud_box->points[mn].x - dummy_aruco->points[mn].x), abs(xcloud_box->points[mn].y - dummy_aruco->points[mn].y), abs(xcloud_box->points[mn].z - dummy_aruco->points[mn].z)));
					}
					diff3D2.at(0).x = (diff3D2.at(0).x - aruco_center[2].x);
					diff3D2.at(0).y = diff3D2.at(0).y - aruco_center[2].y;
					
					diff3D2.at(0).z = diff3D2.at(0).z - dummy_aruco->points[2].z;
					cout << "diff vectors: " << endl;
					cout << diff3D2 << endl;
				}*/


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


		double c_x = instrinsics.at<double>(2);
		double f_x = instrinsics.at<double>(0);
		double c_y = instrinsics.at<double>(5);
		double f_y = instrinsics.at<double>(4);
		/*Mat dummy_image = Mat::zeros(colorheight, colorwidth, CV_8UC4);*/
		colorImage.release();
		colorImage = Mat::zeros(colorheight, colorwidth, CV_8UC4);
		cloud->points.clear();
		pcl::PointXYZRGB dummyVar;
		//pcl::PointXYZRGB point_cloud_considered;
#if 1
		for (int h = 0; h < colorheight; h +=3 ){//colorheight
			for (int w = 0; w < colorwidth; w += 3){//colorwidth
				int index3 = h*colorwidth + w;
				ColorSpacePoint dummycolor;

				int _X = (int)depthSpace2[index3].X;
				int _Y = (int)depthSpace2[index3].Y;
				if ((_X >= 0) && (_X < widthd) && (_Y >= 0) && (_Y < heightd)){
					int depth_index = (_Y*widthd) + _X;
					int index3_color = index3 * 4;
					/*CameraSpacePoint q = depth2xyz[depth_index]*/
					ColorSpacePoint p = depth2rgb[depth_index];
					CameraSpacePoint world_point_camera = depth2xyz[depth_index];
					//int idx = ((int)p.X) + colorwidth*((int)p.Y);
					int idx = ((int)p.X) + colorwidth*((int)p.Y);
					double actualx = (p.X - c_x)*world_point_camera.Z / f_x;
					double actualy = (p.Y - c_y)*world_point_camera.Z / f_y;

					//if ((rgbimage[4 * idx + 0] < 120) && (rgbimage[4 * idx + 1]>120) && (rgbimage[4 * idx + 2] < 90	)	){
					//in_disp << actualx << " " << actualy << " " << world_point_camera.Z << endl;
					/*colorImage.data[index3_color + 0] = rgbimage[4 * idx + 0];
					colorImage.data[index3_color + 1] = rgbimage[4 * idx + 1];*/
					//current
					//if ((world_point_camera.Z < 2.0)){
						if ((0.0 < world_point_camera.Z)){
							//if ((rgbimage[4 * idx + 0] <150) && (rgbimage[4 * idx + 1] > 100) && (rgbimage[4 * idx + 2] < 150)){
							colorImage.data[index3_color + 0] = rgbimage[4 * idx + 0]; // i assume here that colorimage is bgr?
							colorImage.data[index3_color + 1] = rgbimage[4 * idx + 1];

							colorImage.data[index3_color + 2] = rgbimage[4 * idx + 2];
							colorImage.data[index3_color + 3] = rgbimage[4 * idx + 3];
							dummyVar.x = actualx;
							dummyVar.y = actualy;
							/*dummyVar.x = world_point_camera.X;
							dummyVar.y = world_point_camera.Y;*/
							dummyVar.z = world_point_camera.Z;
							dummyVar.r = rgbimage[4 * idx + 0];
							dummyVar.g = rgbimage[4 * idx + 1];
							dummyVar.b = rgbimage[4 * idx + 2];
							//cloud_KINECT->points.push_back(pcl::PointXYZ(depth2xyz[i].X, depth2xyz[i].Y, depth2xyz[i].Z));

							cloud->points.push_back(dummyVar);

							//}
						}
						//}
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
#endif // 0a




		//pcl::Organi
		//viewer->update("cloudcloud");
		viewer->updatePointCloud(cloud, "sample cloud");
		viewer->updatePointCloud(cloud_new, "sample cloud2");
		viewer->updatePointCloud(xcloud_box, "box");
		viewer->spinOnce();
		in_disp_aruco.close();

		//input_gpu.upload(colorImage);

		//cuda::cvtColor(input_gpu, output_gpu, COLOR_RGBA2GRAY);

		//output_gpu.download(I_gray2);d
		//detectMarkers(I_gray2, dictionary, markerCorners, markerIds);
		imshow("fd", colorImage);


		//circle detection
#if 0
		Scalar greenlow = Scalar(60-20, 100, 100);
		Scalar greenhigh = Scalar(60+20, 255, 255);
		inRange(hsv, greenlow, greenhigh, I_inrangegreen);
		GaussianBlur(I_inrangegreen, I_inrangegreen, Size(9, 9), 2, 2);
		
		HoughCircles(I_inrangegreen, circles_detected, CV_HOUGH_GRADIENT, 1, I_inrangegreen.rows / 8, 100, 20, 0, 20);



		imshow("GREEN", I_inrangegreen);
#endif // 0

		//dummy_image.release();
		if (1){


			Scalar greenlow = Scalar(60 - 30, 100, 100);
			Scalar greenhigh = Scalar(60 + 30, 255, 255);

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

			///bitwise_or(I_inrangeyellow, I_inrangeblue, I_inrangeblue);
			//bitwise_or(I_inrangered, I_inrangegreen, I_inrangegreen);
			//bitwise_or(I_inrangeblue, I_inrangegreen, I_inrange);
			bitwise_or(I_inrangegreen, I_inrangegreen, I_inrange);
			
			Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(0, 0));
			GaussianBlur(I_inrange, I_inrange, Size(9, 9), 2, 2);
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

			for (int i = 0; i < contours.size(); i++)
			{

				

				circle(I, center[i], radius[i], 1);
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
					if ((_X >= 0) && (_X < widthd) && (_Y >= 0) && (_Y < heightd)){
						int depth_index = (_Y*widthd) + _X;
						int index3_color = index3 * 4;
						/*CameraSpacePoint q = depth2xyz[depth_index]*/
						ColorSpacePoint p = depth2rgb[depth_index];
						CameraSpacePoint world_point_camera = depth2xyz[depth_index];
						int idx = ((int)p.X) + colorwidth*((int)p.Y);
						double actualx = (p.X - c_x)*world_point_camera.Z / f_x;
						double actualy = (p.Y - c_y)*world_point_camera.Z / f_y;

						//if ((rgbimage[4 * idx + 0] < 120) && (rgbimage[4 * idx + 1]>120) && (rgbimage[4 * idx + 2] < 90	)	){
						//in_disp << actualx << " " << actualy << " " << world_point_camera.Z << endl;
						/*colorImage.data[index3_color + 0] = rgbimage[4 * idx + 0];
						colorImage.data[index3_color + 1] = rgbimage[4 * idx + 1];*/
						if (rgbimage[4 * idx + 0] > 100){
							colorImage.data[index3_color + 0] = rgbimage[4 * idx + 0];
							colorImage.data[index3_color + 1] = rgbimage[4 * idx + 1];

							colorImage.data[index3_color + 2] = rgbimage[4 * idx + 2];
							colorImage.data[index3_color + 3] = rgbimage[4 * idx + 3];
						}

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
			//imwrite("calibrate" + to_string(write_counter) + ".png", colorImage);
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
				if (radius[i] > INFINITY){//change!!
					if ((int)index3 < colorwidth*colorheight){

						int _X = (int)((depthSpace2[index3].X));
						int _Y = (int)((depthSpace2[index3].Y));
						if ((_X >= 0) && (_X < widthd) && (_Y >= 0) && (_Y < heightd)){
							int depth_index = (_Y*widthd) + _X;
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
							else if ((200 > rgbimage[4 * idx + 0]) && (200 > rgbimage[4 * idx + 1]) && (0 < rgbimage[4 * idx + 2])){

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
		if (0)//DRAW VERY IMPORTANT!!!!!!!!!!!
		{/*
		 diff.clear();
		 diff2.clear();
		 diff.push_back(Point2f(0.0, 0.0));
		 diff2.push_back(Point2f(0.0, 0.0));*/
			//solvePnP(Mat(geo_deform), Mat(tracking_colors), instrinsics, distortion, rvec_new, tvec_new, false);
			double time_augment = std::clock();
			if (1){
				if (((norm(diff) + norm(diff2) + norm(diff3) + norm(diff4))>45.00) || initial_pnp){//
					if (markerCount_global == 4){
						solvePnP(Mat(partial_cube), Mat(aruco_center), instrinsics, distortion, rvec_new, tvec_new, false);
						
						initial_pnp = false;
					}

				}
				projectPoints(mesh_geometry, rvec_new, tvec_new, instrinsics, distortion, mesh_geometry_display);
#if 0
				for (int j_dummy = 0; j_dummy < mesh_geometry_display.size(); j_dummy++){
					circle(I, mesh_geometry_display[j_dummy], 20, Scalar(0, 100, 255), 4);

			}
#endif // 0


				draw_mesh(geo_ptr, I);

				cout << "TOTAL SOLVEPNP AND PROJECTION TIME : " << (time_augment - std::clock()) / ((double)CLOCKS_PER_SEC) << endl;
		}
			if (markerCount_global==4){ //if there is an aruco marker
				for (int dummy_i = 0; dummy_i < markerCount_global; dummy_i++){ // changed limit from .size()
					if (markerIds[dummy_i] == 1){
						diff.clear();
						diff.push_back(Point2f((aruco_position[dummy_i].x - meshnode_position[0].x), (aruco_position[dummy_i].y - meshnode_position[0].y)));
						if (cv::norm(diff) > 100){
							diff.clear();
							diff.push_back(Point2f(0.0, 0.0));
						}
						putText(I, "Hello 1: " + to_string(diff[0].x) + "  " + to_string(diff[0].y), Point2f(50.0, 50.0), 2, 1.5, Scalar(100, 100, 255), 2, -1);

					}
					else if (markerIds[dummy_i] == 2){
						diff2.clear();
						diff2.push_back(Point2f((aruco_position[dummy_i].x - meshnode_position2[0].x), (aruco_position[dummy_i].y - meshnode_position2[0].y)));
						if (cv::norm(diff2) > 100){
							diff2.clear();
							diff2.push_back(Point2f(0.0, 0.0));
						}
						putText(I, "David 2: " + to_string(diff2[0].x) + "  " + to_string(diff2[0].y), Point2f(50.0, 150.0), 2, 1.5, Scalar(100, 100, 255), 2, -1);

					}
					else if (markerIds[dummy_i] == 9){
						diff3.clear();
						diff3.push_back(Point2f((aruco_position[dummy_i].x - meshnode_position3[0].x), (aruco_position[dummy_i].y - meshnode_position3[0].y)));
						if (cv::norm(diff3) > 100){
							diff3.clear();
							diff3.push_back(Point2f(0.0, 0.0));
						}
						putText(I, "You 3: " + to_string(diff3[0].x) + "  " + to_string(diff3[0].y), Point2f(50.0, 250.0), 2, 1.5, Scalar(100, 100, 255), 2, -1);

					}
					else if (markerIds[dummy_i] == 4){
						diff4.clear();
						diff4.push_back(Point2f((aruco_position[dummy_i].x - meshnode_position4[0].x), (aruco_position[dummy_i].y - meshnode_position4[0].y)));
						if (cv::norm(diff4) > 100){
							diff4.clear();
							diff4.push_back(Point2f(0.0, 0.0));
						}
						putText(I, "Suck 4: " + to_string(diff4[0].x) + "  " + to_string(diff4[0].y), Point2f(50.0, 350.0), 2, 1.5, Scalar(100, 100, 255), 2, -1);

					}

				}
				//diff.clear();
			}
			else{ // if there isn't then put sudo force to zero
				diff.clear();
				diff.push_back(Point2f(0.0, 0.0));
				diff2.clear();
				diff2.push_back(Point2f(0.0, 0.0));
				diff3.clear();
				diff3.push_back(Point2f(0.0, 0.0));
				diff4.clear();
				diff4.push_back(Point2f(0.0, 0.0));

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
			for (int jk = 0; jk < circles_detected.size(); jk++){
				circle(I, Point2f(circles_detected[jk].x, circles_detected[jk].y), circles_detected[jk].z, Scalar(0, 100, 255), 4);
			}

			//imshow("original", I);
		

			//imshow("colorimage", colorImage);
			//imshow("In range", I_inrange);
			first_geo_init = false;

			
			get_mesh(geo_ptr);
			
#if 0

#endif // FEM_USE


			//double dt = abs(std::clock() - start_K11);
			//cout << " dt : " << dt << endl;
			//duration_vision = (std::clock() - start_K11) / (double)CLOCKS_PER_SEC;
			//cout << "Duration vision: " << duration_vision << endl;
	}

		imshow("original", I);
		//imwrite("with_mesh" + to_string(display_counter) + ".jpg", I);
		
		waitKey(1);

}

	//drawContours(I_inrange, contours, 0, Scalar(255,100,100), 2, 8);



	int found_index = 0;
	float* fdest = (float*)dest;
	for (int i = 0; i < widthd*heightd; i++) {

		ColorSpacePoint p = depth2rgb[i];
		CameraSpacePoint q = depth2xyz[i];

		//A.at<unsigned char>(row, col) = 244;

		//cout << i<<" "<<row << " " << col << endl;

		// Check if color pixel coordinates are in bounds
#if 0
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
#endif // 0

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
#if 1
		GLubyte* ptr;
		glBindBuffer(GL_ARRAY_BUFFER, vboId);
		ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		if (ptr) {
			getDepthData(frame, ptr);
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, cboId);
		ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		if (1) {
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
#endif // 0


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
	//rotateCamera();

	/*glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glVertexPointer(3, GL_FLOAT, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, cboId);
	glColorPointer(3, GL_FLOAT, 0, NULL);

	glPointSize(1.0f);
	glDrawArrays(GL_POINTS, 0, widthd*heightd);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);*/
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
	dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
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
	buf_color_position = new unsigned short[2 * widthd*heightd];
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
	const int dataSize = widthd*heightd * 3 * 4;
	glGenBuffers(1, &vboId);
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, dataSize, 0, GL_DYNAMIC_DRAW);
	glGenBuffers(1, &cboId);
	glBindBuffer(GL_ARRAY_BUFFER, cboId);
	glBufferData(GL_ARRAY_BUFFER, dataSize, 0, GL_DYNAMIC_DRAW);


	diff3D2.push_back(Point3f(0, 0, 0));
	//// Camera setup
	//glViewport(0, 0, width, height);
	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//gluPerspective(45, width / (GLdouble)height, 0.1, 1000);
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();
	//gluLookAt(0, 0, 0, 0, 0, 1, 0, 1, 0);
	//const char **a;

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

#if 0
		geo_ptr->set_beta1(0.9); // if beta_2 >= beta1 and beta > 1/2 then the time stepping scheme is unconditionally stable.
		geo_ptr->set_beta2(0.9);
		geo_ptr->set_dt(1.0);
#endif // 0
		p->set_beta1(0.85); // if beta_2 >= beta1 and beta > 1/2 then the time stepping scheme is unconditionally stable.
		p->set_beta2(0.95);
		p->set_dt(0.12);

#if 0
		geo_ptr->set_dynamic_alpha(0.2);
		geo_ptr->set_dynamic_xi(1.90);
#endif // 0
		p->set_dynamic_alpha(1.2);
		p->set_dynamic_xi(2.8);


	}


	p->set_beta1(0.9); // if beta_2 >= beta1 and beta > 1/2 then the time stepping scheme is unconditionally stable.
	p->set_beta2(0.9);
	p->set_dt(0.08);
	p->set_dynamic_alpha(0.2);
	p->set_dynamic_xi(0.23);
	diff.push_back(Point2f(0.0, 0.0));
	diff2.push_back(Point2f(0.0, 0.0));
	diff3.push_back(Point2f(0.0, 0.0));
	diff4.push_back(Point2f(0.0, 0.0));
	//global_geo.initialize_CUDA
	display_counter = 0;
	first_geo_init = true;
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	// Define a translation of 2.5 meters on the x axis.
	
	Eigen::Matrix4f transform_initial = Eigen::Matrix4f::Identity();

	transform_initial(0, 0) = 0.99999;
	transform_initial(1, 1) = 0.99999;
	transform_initial(2, 2) = 0.99999;

	transform_initial(0, 1) = 0.00255439;
	transform_initial(0, 2) = 0.00594739;
	transform_initial(0, 3) = -0.00488;


	transform_initial(1, 0) = -0.00255439;
	transform_initial(1, 2) = -0.00932893;
	transform_initial(1, 3) = 0.00725061;

	transform_initial(2, 0) = -0.00597098;
	transform_initial(2, 1) = 0.00931379;
	transform_initial(2, 3) = -0.0018118;
	

	transform_2.translation() << 0.0538385, 0.266005, 0.948146;
	
	//transform_2.rotate(Eigen::AngleAxisf(1.7724, Eigen::Vector3f(0.1891  ,  0.3916  ,  0.6310)));
	global_initial_ptr = &transform_2;
	get_mesh(geo_ptr);


	//intiliazing the number of points that will not move

	p->initialize_zerovector(num_station_nodes);
	//next we set what nodes we want to make stable
	int points[8];
	/*for (int i = 0; i < num_station_nodes; i++){

	points[i] = i;

	}*/

	p->set_zero_nodes(station_nodes);

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
		obj.push_back(Point3f((j / numCornersHor) - 5, j%numCornersHor, 0.0f));




	//Initialising PCL POINTS, the declaration is GLOBAL
	pcl::PolygonMesh triangles;

	std::string knot_name = "texturedknot.stl";
	std::string duck_name = "duck_triangulate.stl";
	std::string laura_name = "Laurana50k.stl";
	pcl::io::loadPolygonFileSTL(knot_name, triangles);

#if 1 //////////////////////////////LARGE IF
#if 1

	pcl::fromPCLPointCloud2(triangles.cloud, *cloud);
	pcl::fromPCLPointCloud2(triangles.cloud, *cloud_new);
	for (int i = 0; i < cloud->size(); i++){
		cloud->at(i).x = cloud->at(i).x / 100.0;
		cloud->at(i).y = cloud->at(i).y / 100.0;
		cloud->at(i).z = cloud->at(i).z / 100.0;
		cloud->at(i).r = 255.0;
		cloud->at(i).g = 150.0;
		cloud->at(i).b = 150.0;
		xcloud->points.push_back(pcl::PointXYZ(cloud->at(i).x, cloud->at(i).y, cloud->at(i).z));



		cloud_new->at(i).x = (cloud_new->at(i).x / 100.0) + 1.0;
		cloud_new->at(i).y = (cloud_new->at(i).y / 100.0) + 10.5;
		cloud_new->at(i).z = (cloud_new->at(i).z / 100.0) + 6.0;

		float theta = 3.1415 / 5.0;
		double dummyx;
		double dummyy;

		dummyx = cloud_new->at(i).x*cosf(theta) + cloud_new->at(i).y*sinf(theta);
		dummyy = -cloud_new->at(i).x*sinf(theta) + cloud_new->at(i).y*cosf(theta);
		cloud_new->at(i).x = dummyx;

		cloud_new->at(i).y = dummyy;
		cloud_new->at(i).rgb = (cloud->at(i).x);
		xcloud_new->points.push_back(pcl::PointXYZ(cloud_new->at(i).x, cloud_new->at(i).y, cloud_new->at(i).z));


	}

	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(cloud_new);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_new, rgb, "sample cloud2");

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "sample cloud2");

	viewer->addCoordinateSystem(0.10);
	viewer->initCameraParameters();
	//viewer->removePointCloud("sample cloud");
	//viewer->spinOnce();
#endif // 0


	//viewer->removePointCloud("sample cloud");

	while (0)
	{
		IMultiSourceFrame* frame = NULL;
		viewer->updatePointCloud(cloud, "sample cloud");
		//viewer->*/
		viewer->spinOnce();
		if (SUCCEEDED(reader->AcquireLatestFrame(&frame))) {
			//cout << "s" << endl;
			IDepthFrame* depthframe;
			IDepthFrameReference* frameref = NULL;
			frame->get_DepthFrameReference(&frameref);
			frameref->AcquireFrame(&depthframe);
			if (frameref) frameref->Release();

			if (!depthframe)
				cout << "error" << endl;;

			// Get data from frame
			unsigned int sz;
			unsigned short* buf;

			depthframe->AccessUnderlyingBuffer(&sz, &buf);

			// Write vertex coordinates
			mapper->MapDepthFrameToCameraSpace(widthd*heightd, buf, widthd*heightd, depth2xyz);
			//mapper->MapDepthFrameToCameraSpace(widthd*heightd, buf, widthd*heightd / 2, depth2xyz_found_different_dim);


			if (cloud_KINECT->size() > 0)
				cloud_KINECT->clear();
			pcl::PointXYZRGB dummyVar;
			//viewer->removeAllPointClouds();
			for (unsigned int i = 0; i < sz; i = i + 20) {
#if 0
				cloud->at(i).x = cloud->at(i).x / 100.0;
				cloud->at(i).y = cloud->at(i).y / 100.0;
				cloud->at(i).z = cloud->at(i).z / 100.0;
#endif // 0

				/*if ((depth2xyz[i].X + depth2xyz[i].Y + depth2xyz[i].Z)<100.0)*/
				dummyVar.x = depth2xyz[i].X;
				dummyVar.y = depth2xyz[i].Y;
				dummyVar.z = depth2xyz[i].Z;
				dummyVar.r = 203;
				dummyVar.g = 230;
				dummyVar.b = 100;
				//cloud_KINECT->points.push_back(pcl::PointXYZ(depth2xyz[i].X, depth2xyz[i].Y, depth2xyz[i].Z));

				cloud_KINECT->points.push_back(dummyVar);

#if 0
				* fdest++ = depth2xyz[i].X;
				*fdest++ = depth2xyz[i].Y;
				*fdest++ = depth2xyz[i].Z;
#endif // 0


		}

			viewer->updatePointCloud(cloud_KINECT, "sample cloud");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "sample cloud2");
			viewer->spinOnce();
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			//viewer->spinOnce();
			//viewer->addPointCloud<pcl::PointXYZ>(cloud_KINECT, "sample cloud");
			//viewer->addPointCloud<pcl::PointXYZRGB>(cloud_, rgb, "sample cloud2");
			//mapper->MapColorFrameToDepthSpace(widthd*heightd, buf_color_position, 2*widthd*heightd, depth_found);
			mapper->MapDepthFrameToColorSpace(sz, buf, widthd*heightd, depth2rgb);

			mapper->MapColorFrameToDepthSpace(sz, buf, colorwidth*colorheight, depthSpace2);
			//mapper->MapDepthPointsToColorSpace(1, &depthSpace2[0], sz, buf, 1, dummycolor);
			// Fill in depth2rgb map


			//
			if (depthframe) depthframe->Release();
	}
		if (frame) frame->Release();
		/*getKinectData();
		rotateCamera();
		;*/
		/*getKinectData();
		rotateCamera();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, vboId);
		glVertexPointer(3, GL_FLOAT, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, cboId);
		glColorPointer(3, GL_FLOAT, 0, NULL);

		glPointSize(1.0f);
		glDrawArrays(GL_POINTS, 0, widthd*heightd);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);*/
		/*viewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));*/
}

	//making a box to fit
	/*double h = 0.06332;
	double d = 0.1224;
	double w = 0.14364;*/
double d = 0.0945825;
double h = 0.1253975;


	double w = 0.14364;
	xcloud_box->points.push_back(pcl::PointXYZ(d, 0, 0.0));
	xcloud_box->points.push_back(pcl::PointXYZ(0.0, 0.0, 0.0));


	xcloud_box->points.push_back(pcl::PointXYZ(0.0, h, 0.0));
	//xcloud_box->points.push_back(pcl::PointXYZ(0.0, h/2.0, 0.0));
#if 0
	xcloud_box->points.push_back(pcl::PointXYZ(0.0, 0.0, w));
	xcloud_box->points.push_back(pcl::PointXYZ(d, 0.0, w));
#endif // 0

	xcloud_box->points.push_back(pcl::PointXYZ(d, h, 0.0));






	partial_cube.push_back(cv::Point3f(d, 0, 0.0));

	partial_cube.push_back(cv::Point3f(0.0, 0.0, 0.0));

	

	partial_cube.push_back(cv::Point3f(0.0, h, 0.0));
	//xcloud_box->points.push_back(pcl::PointXYZ(0.0, h/2.0, 0.0));
#if 0
	partial_cube.push_back(cv::Point3f(0.0, 0.0, w));
	partial_cube.push_back(cv::Point3f(d, 0.0, w));
#endif // 0

	partial_cube.push_back(cv::Point3f(d, h, 0.0));






	
	/*xcloud_box->points.push_back(pcl::PointXYZ(0.0, h, w));
	xcloud_box->points.push_back(pcl::PointXYZ(d, h, w));*/
	//for (int i = 0; i < xcloud_box->points.size(); i++){
	//	xcloud_box->points[i].x = xcloud_box->points[i].x;
	//	xcloud_box->points[i].y = xcloud_box->points[i].y;
	//	xcloud_box->points[i].z = xcloud_box->points[i].z;

	//}

	//pcl::transformPointCloud(*xcloud_box, *xcloud_box, transform_initial);

	viewer->addPointCloud(xcloud_box, "box");
	mesh_generated_var->points.push_back(pcl::PointXYZ(0.0, 0.0, 0.0));
	viewer->addPointCloud(mesh_generated_var, "mesh generated");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 20, "box");


	icp.setInputSource(xcloud_box);

	icp.setMaximumIterations(1);
#if 1

	cloud_KINECT->points.push_back(pcl::PointXYZRGB(1.0, 1.0, 1.0));
	pcl::PolygonMesh mesh;
	std::string cube_name = "boxpcl.obj";
	pcl::io::loadPolygonFileOBJ(cube_name, mesh);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "mesh generated");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5, "mesh generated");
	viewer->addPolygonMesh(mesh, "meshes", 0);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.5, 0.2, 0.7, "meshes");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "meshes");
#endif // 0

	mesh.polygons[0].vertices[0] = 1;
	
	// The same rotation matrix as before; theta radians arround Z axis

	pcl::fromPCLPointCloud2(mesh.cloud, *xcloud_box_mesh);
	pcl::transformPointCloud(*xcloud_box_mesh, *xcloud_box_mesh, *global_initial_ptr);
	pcl::transformPointCloud(*xcloud_box, *xcloud_box, *global_initial_ptr);
	pcl::transformPointCloud(*mesh_generated_var, *mesh_generated_var, *global_initial_ptr);
	pcl::toPCLPointCloud2(*xcloud_box_mesh, mesh.cloud);
	viewer->updatePolygonMesh(mesh, "meshes");
	global_mesh_ptr = &mesh;
#endif // 0 //////////////////////////////LARGE IF
	
	//pcl::transformPointCloud()
	execute();
	return 0;
}



