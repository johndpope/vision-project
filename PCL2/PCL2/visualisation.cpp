/* \author Geoffrey Biggs */


#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/io/io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include "Windows.h"
#include <cstdlib>
#include <Kinect.h>
#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>

#include <vector>

#include "aruco_functions.h"
#define VIEWER
CameraSpacePoint depth2xyz1[512 * 424];
CameraSpacePoint depth2xyz2[512 * 424];
CameraSpacePoint depth2xyz3[512 * 424];
CameraSpacePoint depth2xyz4[512 * 424];
CameraSpacePoint depth2xyz5[512 * 424];
//std::vector<std::array<CameraSpacePoint, 512 * 424>> average_depth;
CameraSpacePoint depth2xyz[512 * 424];
double dx = 0;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
// --------------
// -----Help-----
// --------------
void
printUsage(const char* progName)
{
	std::cout << "\n\nUsage: " << progName << " [options]\n\n"
		<< "Options:\n"
		<< "-------------------------------------------\n"
		<< "-h           this help\n"
		<< "-s           Simple visualisation example\n"
		<< "-r           RGB colour visualisation example\n"
		<< "-c           Custom colour visualisation example\n"
		<< "-n           Normals visualisation example\n"
		<< "-a           Shapes visualisation example\n"
		<< "-v           Viewports example\n"
		<< "-i           Interaction Customization example\n"
		<< "\n\n";
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 100, 0);
	//viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem(0.1);
	viewer->initCameraParameters();
	return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(cloud2);
	//viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	//viewer->addPointCloud<pcl::PointXYZRGB>(cloud2, rgb, "sample cloud2");

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud2");
	//viewer->removePointCloud("sample cloud");
	viewer->addCoordinateSystem(0.1);

	viewer->initCameraParameters();
	return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> customColourVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis(
	pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
	// --------------------------------------------------------
	// -----Open 3D viewer and add point cloud and normals-----
	// --------------------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 0.05, "normals");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> shapesVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	//------------------------------------
	//-----Add shapes at cloud points-----
	//------------------------------------
	viewer->addLine<pcl::PointXYZRGB>(cloud->points[0],
		cloud->points[cloud->size() - 1], "line");
	viewer->addSphere(cloud->points[0], 0.2, 0.5, 0.5, 0.0, "sphere");

	//---------------------------------------
	//-----Add shapes at other locations-----
	//---------------------------------------
	pcl::ModelCoefficients coeffs;
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(1.0);
	coeffs.values.push_back(0.0);
	viewer->addPlane(coeffs, "plane");
	coeffs.values.clear();
	coeffs.values.push_back(0.3);
	coeffs.values.push_back(0.3);
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(1.0);
	coeffs.values.push_back(0.0);
	coeffs.values.push_back(5.0);
	viewer->addCone(coeffs, "cone");

	return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> viewportsVis(
	pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals1, pcl::PointCloud<pcl::Normal>::ConstPtr normals2)
{
	// --------------------------------------------------------
	// -----Open 3D viewer and add point cloud and normals-----
	// --------------------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->initCameraParameters();

	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->addText("Radius: 0.01", 10, 10, "v1 text", v1);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud1", v1);

	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
	viewer->addText("Radius: 0.1", 10, 10, "v2 text", v2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, single_color, "sample cloud2", v2);

	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
	viewer->addCoordinateSystem(1.0);

	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals1, 10, 0.05, "normals1", v1);
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals2, 10, 0.05, "normals2", v2);

	return (viewer);
}


unsigned int text_id = 0;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
	void* viewer_void)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
	if (event.getKeySym() == "r" && event.keyDown())
	{
		std::cout << "r was pressed => removing all text" << std::endl;

		char str[512];
		for (unsigned int i = 0; i < text_id; ++i)
		{
			sprintf(str, "text#%03d", i);
			viewer->removeShape(str);
		}
		text_id = 0;
	}
}

void mouseEventOccurred(const pcl::visualization::MouseEvent &event,
	void* viewer_void)
{
	/*boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
	if (event.getButton() == pcl::visualization::MouseEvent::LeftButton &&
	event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease)
	{
	std::cout << "Left mouse button released at position (" << event.getX() << ", " << event.getY() << ")" << std::endl;

	char str[512];
	sprintf(str, "text#%03d", text_id++);
	viewer->addText("clicked here", event.getX(), event.getY(), str);
	}*/
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis()
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);

	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);
	viewer->registerMouseCallback(mouseEventOccurred, (void*)&viewer);

	return (viewer);
}

template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL){
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}
// --------------
// -----Main-----
// --------------
int
main(int argc, char** argv)
{
	// --------------------------------------
	// -----Parse Command Line Arguments-----
	// --------------------------------------
#if 0
	if (pcl::console::find_argument(argc, argv, "-h") >= 0)
	{
		printUsage(argv[0]);
		return 0;
	}
	bool simple(false), rgb(false), custom_c(false), normals(false),
		shapes(false), viewports(false), interaction_customization(false);
	if (pcl::console::find_argument(argc, argv, "-s") >= 0)
	{
		simple = true;
		std::cout << "Simple visualisation example\n";
	}
	else if (pcl::console::find_argument(argc, argv, "-c") >= 0)
	{
		custom_c = true;
		std::cout << "Custom colour visualisation example\n";
	}
	else if (pcl::console::find_argument(argc, argv, "-r") >= 0)
	{
		rgb = true;
		std::cout << "RGB colour visualisation example\n";
	}
	else if (pcl::console::find_argument(argc, argv, "-n") >= 0)
	{
		normals = true;
		std::cout << "Normals visualisation example\n";
	}
	else if (pcl::console::find_argument(argc, argv, "-a") >= 0)
	{
		shapes = true;
		std::cout << "Shapes visualisation example\n";
	}
	else if (pcl::console::find_argument(argc, argv, "-v") >= 0)
	{
		viewports = true;
		std::cout << "Viewports example\n";
	}
	else if (pcl::console::find_argument(argc, argv, "-i") >= 0)
	{
		interaction_customization = true;
		std::cout << "Interaction Customization example\n";
	}
	else
	{
		printUsage(argv[0]);
		return 0;
	}
#endif // 0
	bool simple(false), rgb(false), custom_c(false), normals(false),
		shapes(false), viewports(false), interaction_customization(false);
	rgb = true;

	// ------------------------------------
	// -----Create example point cloud-----
	// ------------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr green_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr green_points1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr green_points2(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr green_points3(new pcl::PointCloud<pcl::PointXYZ>);


	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);


	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr2(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr2(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::cout << "Genarating example point clouds.\n\n";
	// We're going to make an ellipse extruded along the z-axis. The colour for
	// the XYZRGB cloud will gradually go from red to green to blue.
	uint8_t r(255), g(15), b(15);
#if 1
	for (float z(-1.0); z <= 1.0; z += 0.05)
	{
		for (float y(-1.0); y <= 1.0; y += 0.05){
			//for (float angle(0.0); angle <= 360.0; angle += 5.0)
			//{
			pcl::PointXYZ basic_point;
			basic_point.x = cosf((z*2.0));
			basic_point.y = sinf(y*5.0);//sinf(pcl::deg2rad(angle))
			basic_point.z = cosf(2.0*z) + 4;
			basic_cloud_ptr->points.push_back(basic_point);

			pcl::PointXYZRGB point;
			point.x = basic_point.x;
			point.y = basic_point.y;
			point.z = basic_point.z;
			uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
				static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			point.rgb = *reinterpret_cast<float*>(&rgb);
			point_cloud_ptr->points.push_back(point);
			//}
			if (z < 0.0)
			{
				r -= 12;
				g += 12;
			}
			else
			{
				g -= 12;
				b += 12;
			}
		}
	}
#endif // 0

#if 0
	basic_cloud_ptr2->points.push_back(pcl::PointXYZ(0.0, 0.0, 0.0));
	basic_cloud_ptr2->points.push_back(pcl::PointXYZ(1.0, 0.0, 0.0));
	basic_cloud_ptr2->points.push_back(pcl::PointXYZ(0.0, 1.0, 0.0));
	basic_cloud_ptr2->points.push_back(pcl::PointXYZ(0.0, 0.0, 1.0));

	basic_cloud_ptr->points.push_back(pcl::PointXYZ(0.0, 5.0, 0.0));
	basic_cloud_ptr->points.push_back(pcl::PointXYZ(1.0, 5.0, 0.0));
	basic_cloud_ptr->points.push_back(pcl::PointXYZ(0.0, 1.0, 0.0));
	basic_cloud_ptr->points.push_back(pcl::PointXYZ(0.0, 0.0, 1.0));  
#endif // 0

#if 1
	for (float z(-1.0); z <= 1.0; z += 0.05)
	{
		for (float y(-1.0); y <= 1.0; y += 0.05){
			//for (float angle(0.0); angle <= 360.0; angle += 5.0)
			//{
			pcl::PointXYZ basic_point;
			basic_point.x = cosf((z*2.0));
			basic_point.y = sinf(y*5.0);// sinf(pcl::deg2rad(angle));
			basic_point.z = cosf(z*2.0);
			basic_cloud_ptr2->points.push_back(basic_point);

			pcl::PointXYZRGB point;
			point.x = basic_point.x;
			point.y = basic_point.y;
			point.z = basic_point.z;
			uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
				static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			point.rgb = *reinterpret_cast<float*>(&rgb);
			point_cloud_ptr2->points.push_back(point);
			//	}
			if (z < 0.0)
			{
				r -= 12;
				g += 12;
			}
			else
			{
				g -= 12;
				b += 12;
			}
		}
	}
#endif // 0

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_markers(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_new(new pcl::PointCloud<pcl::PointXYZRGB>());

	pcl::PointCloud<pcl::PointXYZ>::Ptr xcloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr xcloud_new(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PolygonMesh triangles;

	std::string knot_name = "texturedknot.stl";
	std::string duck_name = "duck_triangulate.stl";
	std::string laura_name = "Laurana50k.stl";
	std::string stick_name = "test_stick.stl";
	pcl::io::loadPolygonFileSTL(stick_name, triangles);


	pcl::fromPCLPointCloud2(triangles.cloud, *cloud);
	pcl::fromPCLPointCloud2(triangles.cloud, *cloud_new);
	for (int i = 0; i < 6; i++){
		cloud->at(i).x = cloud->at(i).x *10;
		cloud->at(i).y = cloud->at(i).y*10;
		cloud->at(i).z = cloud->at(i).z*10;
		cloud->at(i).r = 255.0;
		cloud->at(i).g = 150.0;
		cloud->at(i).b = 150.0;
		xcloud->points.push_back(pcl::PointXYZ(cloud->at(i).x, cloud->at(i).y, cloud->at(i).z));



		cloud_new->at(i).x = (cloud_new->at(i).x) + 1.0;
		cloud_new->at(i).y = (cloud_new->at(i).y ) + 10.5;
		cloud_new->at(i).z = (cloud_new->at(i).z ) + 6.0;

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

	/*xcloud->width = (int)xcloud->points.size();
	xcloud_new->width = (int)xcloud_new->points.size();
	basic_cloud_ptr->width = (int)basic_cloud_ptr->points.size();
	basic_cloud_ptr->height = 1;
	point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
	point_cloud_ptr->height = 1;


	basic_cloud_ptr2->width = (int)basic_cloud_ptr2->points.size();
	basic_cloud_ptr2->height = 1;
	point_cloud_ptr2->width = (int)point_cloud_ptr2->points.size();
	point_cloud_ptr2->height = 1;
	*/
	// ----------------------------------------------------------------
	// -----Calculate surface normals with a search radius of 0.05-----
	// ----------------------------------------------------------------
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud(point_cloud_ptr);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	ne.setSearchMethod(tree);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch(0.05);
	ne.compute(*cloud_normals1);

	// ---------------------------------------------------------------
	// -----Calculate surface normals with a search radius of 0.1-----
	// ---------------------------------------------------------------
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch(0.1);
	ne.compute(*cloud_normals2);


	if (simple)
	{
		viewer = simpleVis(basic_cloud_ptr);
	}
	else if (rgb)
	{

#ifdef VIEWER
		viewer = rgbVis(cloud, cloud_new);
#endif // VIEWER

	}
	else if (custom_c)
	{
		viewer = customColourVis(basic_cloud_ptr);
	}
	else if (normals)
	{
		viewer = normalsVis(point_cloud_ptr, cloud_normals2);
	}
	else if (shapes)
	{
		viewer = shapesVis(point_cloud_ptr);
	}
	else if (viewports)
	{
		viewer = viewportsVis(point_cloud_ptr, cloud_normals1, cloud_normals2);
	}
	else if (interaction_customization)
	{
		viewer = interactionCustomizationVis();
	}

	//--------------------
	// -----Main loop-----
	//--------------------
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(xcloud);
	icp.setInputTarget(xcloud_new);
	pcl::PointCloud<pcl::PointXYZ>::Ptr Final_ptr(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>);
	//icp.setTransformationEpsilon(1e-8);
	//icp.setEuclideanFitnessEpsilon(0.00000001);
	//icp.setMaximumIterations(10000);
	icp.setMaximumIterations(1);
	icp.align(*Final);


	viewer->addPolygonMesh(triangles, "stickmesh");


	//pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr22;

	std::cout << "has converged:" << icp.hasConverged() << " score: " <<
		icp.getFitnessScore() << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;

	//icp.setTransformationEpsilon (1e-8);
#ifdef VIEWER
	viewer->addPointCloud<pcl::PointXYZ>(Final, "final");
	viewer->removePointCloud("final");
	viewer->removePointCloud("final");
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloudcloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloudcloud");
#endif // VIEWER


	IKinectSensor* pSensor;
	HRESULT hResult = S_OK;
	hResult = GetDefaultKinectSensor(&pSensor);

	if (FAILED(hResult)){
		std::cerr << "Error : GetDefaultKinectSensor" << std::endl;
		return -1;
	}

	hResult = pSensor->Open();
	if (FAILED(hResult)){
		std::cerr << "Error : IKinectSensor::Open()" << std::endl;
		return -1;
	}

	// Source
	IColorFrameSource* pColorSource;
	hResult = pSensor->get_ColorFrameSource(&pColorSource);
	if (FAILED(hResult)){
		std::cerr << "Error : IKinectSensor::get_ColorFrameSource()" << std::endl;
		return -1;
	}

	IDepthFrameSource* pDepthSource;
	hResult = pSensor->get_DepthFrameSource(&pDepthSource);
	if (FAILED(hResult)){
		std::cerr << "Error : IKinectSensor::get_DepthFrameSource()" << std::endl;
		return -1;
	}

	IInfraredFrameSource* pInfraredSource;
	hResult = pSensor->get_InfraredFrameSource(&pInfraredSource);
	if (FAILED(hResult)){
		std::cerr << "Error : IKinectSensor::pInfraredSource()" << std::endl;
		return -1;
	}

	// Reader
	IColorFrameReader* pColorReader;
	hResult = pColorSource->OpenReader(&pColorReader);
	if (FAILED(hResult)){
		std::cerr << "Error : IColorFrameSource::OpenReader()" << std::endl;
		return -1;
	}

	IDepthFrameReader* pDepthReader;
	hResult = pDepthSource->OpenReader(&pDepthReader);
	if (FAILED(hResult)){
		std::cerr << "Error : IDepthFrameSource::OpenReader()" << std::endl;
		return -1;
	}

	IInfraredFrameReader* pInfraredReader;
	hResult = pInfraredSource->OpenReader(&pInfraredReader);
	if (FAILED(hResult)){
		std::cerr << "Error : IInfraredFrameReader::OpenReader()" << std::endl;
		return -1;
	}

	// Description
	IFrameDescription* pColorDescription;
	hResult = pColorSource->get_FrameDescription(&pColorDescription);
	if (FAILED(hResult)){
		std::cerr << "Error : IColorFrameSource::get_FrameDescription()" << std::endl;
		return -1;
	}
	IFrameDescription* pInfraredDescription;
	hResult = pInfraredSource->get_FrameDescription(&pInfraredDescription);
	if (FAILED(hResult)){
		std::cerr << "Error : pInfraredDescription::get_FrameDescription()" << std::endl;
		return -1;
	}

	pcl::PointXYZ p[5];
	p[0] = pcl::PointXYZ(-0.128187299, -0.312860161, 1.20415008	);
	p[1] = pcl::PointXYZ(-0.0660647824, -0.287979186, 1.19415009);
	p[2]=pcl::PointXYZ(-0.133042291, -0.247751921, 1.21715009);
	p[3] = pcl::PointXYZ(-0.0526939221, -0.252760977, 1.25715005);
	p[4] = pcl::PointXYZ(-0.114646740, -0.287160486, 1.30615008);

	p[0] = pcl::PointXYZ(0.0,0.0,0.0);
	p[1] = pcl::PointXYZ(0.0,0.0,0.0);
	p[2] = pcl::PointXYZ(0.0,0.0,0.0);
	p[3] = pcl::PointXYZ(0.0,0.0,0.0);
	p[4] = pcl::PointXYZ(0.0,0.0,0.0);

	for (int i = 0; i < 5; i++){
	
		green_points->push_back(p[i]);
		green_points1->push_back(p[i]);
		green_points2->push_back(p[i]);
		green_points3->push_back(p[i]);

	
	}

	pcl::PointXYZ running_average[5];
	running_average[0] = pcl::PointXYZ(0.0, 0.0, 0.0);
	running_average[1] = pcl::PointXYZ(0.0, 0.0, 0.0);
	running_average[2] = pcl::PointXYZ(0.0, 0.0, 0.0);
	running_average[3] = pcl::PointXYZ(0.0, 0.0, 0.0);
	running_average[4] = pcl::PointXYZ(0.0, 0.0, 0.0);


	//

	

	viewer->addPointCloud<pcl::PointXYZ>(green_points, "greenpoints");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 20, "greenpoints");
	int colorWidth = 0;
	int colorHeight = 0;
	pColorDescription->get_Width(&colorWidth); // 1920
	pColorDescription->get_Height(&colorHeight); // 1080
	unsigned int colorBufferSize = colorWidth * colorHeight * 4 * sizeof(unsigned char);

	int infraredWidth = 0;
	int infraredHeight = 0;
	pInfraredDescription->get_Width(&infraredWidth);
	pInfraredDescription->get_Height(&infraredHeight);

	viewer->addPointCloud<pcl::PointXYZ>(Final, "final");
	cv::Mat colorBufferMat(colorHeight, colorWidth, CV_8UC4);
	cv::Mat colorMat(colorHeight / 2, colorWidth / 2, CV_8UC4);
	cv::namedWindow("Color");

	cv::Mat infraredBufferMat(infraredHeight, infraredWidth, CV_16UC1);
	cv::Mat infraredMat(infraredHeight, infraredHeight, CV_8UC1);

	unsigned int infraredBufferSize = infraredHeight * infraredHeight * sizeof(unsigned int);
	IFrameDescription* pDepthDescription;
	hResult = pDepthSource->get_FrameDescription(&pDepthDescription);
	if (FAILED(hResult)){
		std::cerr << "Error : IDepthFrameSource::get_FrameDescription()" << std::endl;
		return -1;
	}

	int depthWidth = 0;
	int depthHeight = 0;
	pDepthDescription->get_Width(&depthWidth); // 512
	pDepthDescription->get_Height(&depthHeight); // 424
	unsigned int depthBufferSize = depthWidth * depthHeight * sizeof(unsigned short);

	cv::Mat depthBufferMat(depthHeight, depthWidth, CV_16UC1);
	cv::Mat depthMat(depthHeight, depthWidth, CV_8UC1);
	cv::namedWindow("Depth");

	// Coordinate Mapper
	ICoordinateMapper* pCoordinateMapper;
	hResult = pSensor->get_CoordinateMapper(&pCoordinateMapper);
	if (FAILED(hResult)){
		std::cerr << "Error : IKinectSensor::get_CoordinateMapper()" << std::endl;
		return -1;
	}

	cv::Mat coordinateMapperMat(depthHeight, depthWidth, CV_8UC4);
	cv::namedWindow("CoordinateMapper");

	unsigned short minDepth, maxDepth;
	pDepthSource->get_DepthMinReliableDistance(&minDepth);
	pDepthSource->get_DepthMaxReliableDistance(&maxDepth);
	std::vector<DepthSpacePoint> depthSpacePoints(colorWidth*colorHeight);
	std::vector<ColorSpacePoint> colorSpacePoints(depthWidth * depthHeight);
	bool tracker_position_set = false;
	

	for (int i = 0; i < 512 * 424; i++){
		CameraSpacePoint zero;
		zero.X = zero.Y = zero.Z = 0;
		depth2xyz1[i] = depth2xyz2[i] = depth2xyz3[i] = zero;

	}
	while (1){
		// Color Frame
		std::clock_t start = std::clock();
		IColorFrame* pColorFrame = nullptr;
		hResult = pColorReader->AcquireLatestFrame(&pColorFrame);
		if (SUCCEEDED(hResult)){
			hResult = pColorFrame->CopyConvertedFrameDataToArray(colorBufferSize, reinterpret_cast<BYTE*>(colorBufferMat.data), ColorImageFormat::ColorImageFormat_Bgra);
			if (SUCCEEDED(hResult)){
				cv::resize(colorBufferMat, colorMat, cv::Size(), 0.5, 0.5);
			}
		}
		//SafeRelease( pColorFrame );

		// Depth Frame
		IDepthFrame* pDepthFrame = nullptr;
		hResult = pDepthReader->AcquireLatestFrame(&pDepthFrame);
		if (SUCCEEDED(hResult)){
			hResult = pDepthFrame->AccessUnderlyingBuffer(&depthBufferSize, reinterpret_cast<UINT16**>(&depthBufferMat.data));
			if (SUCCEEDED(hResult)){
				depthBufferMat.convertTo(depthMat, CV_8U, -255.0f / 8000.0f, 255.0f);
			}
		}

		//infrared frame

		IInfraredFrame* pInfraredFrame = nullptr;

		hResult = pInfraredReader->AcquireLatestFrame(&pInfraredFrame);
		if (SUCCEEDED(hResult)){

			hResult = pInfraredFrame->AccessUnderlyingBuffer(&infraredBufferSize, reinterpret_cast<UINT16**>(&infraredBufferMat.data));
			if (SUCCEEDED(hResult)){
				infraredBufferMat.convertTo(infraredMat, CV_8UC1, 0.02, 0.2);
			}
		}
		//SafeRelease( pDepthFrame );
		int x_shift = 0;
		
		// Mapping (Depth to Color)
		if (SUCCEEDED(hResult)){
			
			

			hResult = pCoordinateMapper->MapDepthFrameToColorSpace(depthWidth * depthHeight, reinterpret_cast<UINT16*>(depthBufferMat.data), depthWidth * depthHeight, &colorSpacePoints[0]);
			hResult = pCoordinateMapper->MapColorFrameToDepthSpace(depthWidth * depthHeight, reinterpret_cast<UINT16*>(depthBufferMat.data), colorWidth*colorHeight, &depthSpacePoints[0]);
			hResult = pCoordinateMapper->MapDepthFrameToCameraSpace(depthWidth * depthHeight, reinterpret_cast<UINT16*>(depthBufferMat.data), depthWidth * depthHeight, depth2xyz);
			
			/*if ( 5){
				for (unsigned k = 0; k < depthWidth*depthHeight; k++){
					
					depth2xyz1[k] = depth2xyz2[k];
					depth2xyz2[k] = depth2xyz3[k];
					depth2xyz3[k] = depth2xyz4[k];
					depth2xyz4[k] = depth2xyz5[k];
					depth2xyz5[k] = depth2xyz[k];


				}
				

				for (unsigned k = 0; k < depthWidth*depthHeight; k++){

					depth2xyz[k].X = (depth2xyz1[k].X + depth2xyz2[k].X + depth2xyz3[k].X + depth2xyz4[k].X+depth2xyz5[k].X) / 3.0;
					depth2xyz[k].Y = (depth2xyz1[k].Y + depth2xyz2[k].Y + depth2xyz3[k].Y + depth2xyz4[k].Y + depth2xyz5[k].Y) / 3.0;
					depth2xyz[k].Z = (depth2xyz1[k].Z + depth2xyz2[k].Z + depth2xyz3[k].Z + depth2xyz4[k].Z + depth2xyz5[k].Z) / 3.0;


				}

			}*/
			
			
			
			pcl::PointXYZRGB dummyVar;
			cloud->clear();
			cloud_markers->clear();

#if 0
			for (unsigned int i = 0; i < depthBufferSize; i++) {
#if 0
				cloud->at(i).x = cloud->at(i).x / 100.0;
				cloud->at(i).y = cloud->at(i).y / 100.0;
				cloud->at(i).z = cloud->at(i).z / 100.0;
#endif // 0
				ColorSpacePoint point = colorSpacePoints[i];
				int colorX = static_cast<int>((point.X)) + x_shift;
				int colorY = static_cast<int>((point.Y));
				if ((colorX >= 0) && (colorX < colorWidth) && (colorY >= 0) && (colorY < colorHeight)/* && ( depth >= minDepth ) && ( depth <= maxDepth )*/){
					if (depth2xyz[i].Z*depth2xyz[i].Z < 4){
						/*if ((depth2xyz[i].X + depth2xyz[i].Y + depth2xyz[i].Z)<100.0)*/
						dummyVar.x = depth2xyz[i].X;
						dummyVar.y = depth2xyz[i].Y;
						dummyVar.z = depth2xyz[i].Z;

						cv::Vec4b dummycolor = colorBufferMat.at<cv::Vec4b>(colorY, colorX);
						dummyVar.r = dummycolor[2];
						dummyVar.g = dummycolor[1];
						dummyVar.b = dummycolor[0];
					}

					/*dummyVar.r = color_dummy(0);
					dummyVar.g = color_dummy(1);
					dummyVar.b = color_dummy(2);*/
					//cloud_KINECT->points.push_back(pcl::PointXYZ(depth2xyz[i].X, depth2xyz[i].Y, depth2xyz[i].Z));

					cloud->points.push_back(dummyVar);
				}

#if 0
				* fdest++ = depth2xyz[i].X;
				*fdest++ = depth2xyz[i].Y;
				*fdest++ = depth2xyz[i].Z;
#endif // 0


			}
#else
			for (unsigned int i = 0; i < colorWidth; i=i+50) {
				for (unsigned int j = 0; j < colorHeight; j=j+50){
#if 0
					cloud->at(i).x = cloud->at(i).x / 100.0;
					cloud->at(i).y = cloud->at(i).y / 100.0;
					cloud->at(i).z = cloud->at(i).z / 100.0;
#endif // 0
					int color_index = i + j*colorWidth;
					DepthSpacePoint colorToDepth = depthSpacePoints[color_index];
					cv::Vec4b dummycolor2 = colorBufferMat.at<cv::Vec4b>(j,i); //Note that the images is row by column
					int depthX = (int)colorToDepth.X;
					int depthY = (int)colorToDepth.Y;
					int depthIndex = depthX + depthY * depthWidth;
					if ((depthX >= 0) && (depthX < depthWidth) && (depthY >= 0) && (depthY < depthHeight)) {
						CameraSpacePoint depthToCamera = depth2xyz[depthIndex];



						//ColorSpacePoint point = colorSpacePoints[i];
						//int colorX = static_cast<int>((point.X)) + x_shift;
						//int colorY = static_cast<int>((point.Y));
						if (depthToCamera.Z*depthToCamera.Z < 10)/* && ( depth >= minDepth ) && ( depth <= maxDepth )*/{

							/*if ((depth2xyz[i].X + depth2xyz[i].Y + depth2xyz[i].Z)<100.0)*/
							dummyVar.x = depthToCamera.X;
							dummyVar.y = depthToCamera.Y;
							dummyVar.z = depthToCamera.Z;


							dummyVar.r = dummycolor2[2];
							dummyVar.g = dummycolor2[1];
							dummyVar.b = dummycolor2[0];


							/*dummyVar.r = color_dummy(0);
							dummyVar.g = color_dummy(1);
							dummyVar.b = color_dummy(2);*/
							//cloud_KINECT->points.push_back(pcl::PointXYZ(depth2xyz[i].X, depth2xyz[i].Y, depth2xyz[i].Z));

							cloud->points.push_back(dummyVar);
						}
					}
				}
			}
#endif // 0


			if (SUCCEEDED(hResult)){
				coordinateMapperMat = cv::Scalar(0, 0, 0, 0);
				for (int y = 0; y < depthHeight; y=y+1){
					for (int x = 0; x < depthWidth; x=x+1){
						unsigned int index = y * depthWidth + x;
						ColorSpacePoint point = colorSpacePoints[index];
						CameraSpacePoint D3 = depth2xyz[index];

						//int colorX = ((int)(point.X ));
						//int colorY = ((int)(point.Y ));
						int colorX = static_cast<int>(std::floor(point.X + 0.5)) + x_shift;
						int colorY = static_cast<int>(std::floor(point.Y + 0.5));
						unsigned short depth = depthBufferMat.at<unsigned short>(y, x);
						if ((colorX >= 0) && (colorX < colorWidth) && (colorY >= 0) && (colorY < colorHeight) && (D3.Z < 2)/* && ( depth >= minDepth ) && ( depth <= maxDepth )*/){
							coordinateMapperMat.at<cv::Vec4b>(y, x) = colorBufferMat.at<cv::Vec4b>(colorY, colorX);
						}
					}
				}
			}
		}

		SafeRelease(pColorFrame);
		SafeRelease(pDepthFrame);
		SafeRelease(pInfraredFrame);
		
		
		//CIRCLE Detection
		
		std::vector<cv::Point3f> circles_detected;


		cv::Mat infraredMatColor(infraredHeight, infraredWidth, CV_8UC4);
		infraredMat = 255 - infraredMat;

		cv::Mat inverted_image(infraredHeight, infraredWidth, CV_64F);
		
		
		//cv::threshold(infraredMat, infraredMat, 0, 254, CV_THRESH_BINARY);
		cv::SimpleBlobDetector::Params params;

		// Change thresholds


		// Filter by Area.
		params.filterByArea = true;
		params.minArea = 45;

		// Filter by Circularity
		params.filterByCircularity = true;
		params.minCircularity = 0.4;

		// Filter by Convexity
		params.filterByConvexity = true;
		params.minConvexity = 0.5;

		// Filter by Inertia
		params.filterByInertia = true;
		params.minInertiaRatio = 0.4;

		cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

		// Detect blobs.
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat gray;
		cv::Mat hsv;
		cv::Mat color_HSV;
		//cv::cvtColor(colorMat, gray, cv::COLOR_BGR2GRAY);
		
		cv::cvtColor(colorMat, color_HSV, cv::COLOR_BGR2HSV);
		
		cv::inRange(color_HSV, cv::Scalar(60 - 20, 100, 50), cv::Scalar(60 + 20, 255, 255), gray);
		gray = 255 - gray;
		
		cv::GaussianBlur(gray, gray, cv::Size(5, 5), 2, 2);

		

		detector->detect(gray, keypoints);
#if 0 //HUOUGHCIRCLES
		HoughCircles(infraredMat, circles_detected, CV_HOUGH_GRADIENT, 1, infraredMat.rows / 8, 50, 20, 0, 50);
		infraredMat.convertTo(infraredMatColor, CV_8UC4);
		for (auto cc = circles_detected.begin(); cc != circles_detected.end(); ++cc){
			circle(infraredMat, cv::Point2f(cc->x, cc->y), cc->z, cv::Scalar(100, 100, 200), 2);
		}
#endif // 0 //HUOUGHCIRCLES

		cv::Mat im_with_keypoints;
		drawKeypoints(gray, keypoints, gray, cv::Scalar(0, 0, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("keypoints", gray);
		

		pcl::PointXYZRGB dummyVar;
		pcl::PointXYZ target_points;
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_points_ptr(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto cc = keypoints.begin(); cc != keypoints.end(); ++cc){
			

			int color_x = (int)cc->pt.x;
			int color_y = (int)cc->pt.y;
			circle(colorMat, cv::Point2f(color_x, color_y), 5, cv::Scalar(100, 100, 200), 2);
			color_x = color_x * 2;
			color_y = color_y * 2;
			DepthSpacePoint depth_dummy = depthSpacePoints[color_x + color_y *colorWidth];
			int depth_x = (int)depth_dummy.X;
			int depth_y =(int)depth_dummy.Y;

			if ((depth_x >= 0) && (depth_x < depthWidth) && (depth_y >= 0) && (depth_y < depthHeight)){

				CameraSpacePoint camera_d = depth2xyz[depth_x + depth_y*depthWidth];
				if (camera_d.Z*camera_d.Z < 25){
					pcl::PointXYZ direction_norm;

					dummyVar.x = target_points.x= camera_d.X;
					dummyVar.y = target_points.y=camera_d.Y;
					dummyVar.z = target_points.z=camera_d.Z + 0.01215;
					//double length = sqrt(dummyVar.x*dummyVar.x + dummyVar.y*dummyVar.y + dummyVar.z*dummyVar.z);

				/*	dummyVar.x = dummyVar.x + dummyVar.x *0.01215 / length;
					dummyVar.y = dummyVar.y + dummyVar.y *0.01215 / length;*/
					//dummyVar.z = dummyVar.z + dummyVar.z *0.01215 / length;
			
					dummyVar.r = 000;
					dummyVar.g = 0;
					dummyVar.b = 200;

					cloud_markers->push_back(dummyVar);
					target_points_ptr->push_back(target_points);
				}



			}
			if (tracker_position_set == false){
				if (target_points_ptr->size() == 6){
					green_points = target_points_ptr;
					
					cout << target_points_ptr->points[0] << endl;
					tracker_position_set = true;
				}
			}




		}


		
		icp.setInputSource(green_points);
		icp.setInputTarget(target_points_ptr);
		icp.align(*green_points);

#if 0
		std::vector<cv::Point2f> aruco_centers = find_aruco_center(cv::Mat(colorMat), colorWidth);
		if (!aruco_centers.empty())
			cv::circle(colorMat, aruco_centers[0], 5, cv::Scalar(100, 200, 0));
#endif // 0


		//green_points1 = green_points2;
		//green_points2 = green_points3;
		//green_points3 = green_points;

		//*green_points = (*green_points1) + (*green_points2) ;
		//*green_points = *green_points + (*green_points3);
		//for (auto counter_i = green_points->begin(); counter_i != green_points->end(); ++counter_i){

		//	counter_i->x = counter_i->x / 3.0;
		//	counter_i->y = counter_i->y / 3.0;
		//	counter_i->z = counter_i->z / 3.0;
		//}
		std::cout << "has converged:" << icp.hasConverged() << " score: " <<
			icp.getFitnessScore() << std::endl;
		

		cv::imshow("Color", colorMat);
		cv::imshow("Depth", depthMat);
		cv::imshow("Infrared", infraredMat);
		cv::imshow("CoordinateMapper", coordinateMapperMat);

		if (cv::waitKey(30) == VK_ESCAPE){
			break;
		}


#ifdef VIEWER
		/*viewer->removePointCloud("cloudcloud");
		viewer->addPointCloud(cloud, "cloudcloud");*/
		viewer->addText("Data and information:", 100, 500,123,123,23, "initialized_status");
		viewer->addText("Not initialized the probe", 100, 490,32,33,32, "initialized_status_1");
		viewer->updatePointCloud(cloud, "cloudcloud");
		viewer->updatePointCloud(cloud_markers, "final");
		viewer->updatePointCloud(green_points, "greenpoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloudcloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 25, "final");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 1, "cloudcloud");
		viewer->spinOnce();
#endif // VIEWER

		//boost::this_thread::sleep(boost::posix_time::microseconds(100000));

		std::clock_t end_time = std::clock();
		std::cout << "Time for algorithm: " << (start - end_time) / (double)(CLOCKS_PER_SEC) << std::endl;
	}

#ifdef VIEWER
	while (!viewer->wasStopped())
	{

		//viewer->removePointCloud("sample cloud2");
		//for (int i = 0; i < basic_cloud_ptr2->size(); i++){
		//	basic_cloud_ptr2->points[i].x += dx;
		//	basic_cloud_ptr2->points[i].y += dx;
#if 1
		icp.setInputSource(Final);
		icp.setInputTarget(basic_cloud_ptr2);
		icp.align(*Final);
		std::cout << "has converged:" << icp.hasConverged() << " score: " <<
			icp.getFitnessScore() << std::endl;
		viewer->removePointCloud("final");
		viewer->addPointCloud<pcl::PointXYZ>(Final, "final");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "final");

#endif 



		//}
		//dx += 0.0001;

		//viewer->addPointCloud<pcl::PointXYZ>(basic_cloud_ptr2, "sample cloud2");

		//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
		//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud2");

		viewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
#endif // VIEWER


	SafeRelease(pDepthSource);
	SafeRelease(pDepthReader);

	if (pSensor){
		pSensor->Close();
	}
	SafeRelease(pSensor);
	cv::destroyAllWindows();
}