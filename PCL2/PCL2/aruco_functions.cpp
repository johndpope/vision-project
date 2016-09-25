#include "opencv2\aruco\charuco.hpp"
#include "opencv2\aruco\dictionary.hpp"
#include "opencv2\aruco.hpp"
#include <opencv2\opencv.hpp>
#include "aruco_functions.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <boost/thread/thread.hpp>
#include <opencv2/highgui.hpp>
std::vector<cv::Point2f> find_aruco_center(cv::Mat input_image,int x_diff){
	cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
	cv::Ptr < cv::aruco::Dictionary > dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

	//Variables
	std::vector< std::vector<cv::Point2f> > markerCorners, rejectedCandidates;
	std::vector< int > markerIds;


	cv::Mat gray;

	cv::cvtColor(input_image, gray, cv::COLOR_RGBA2GRAY);
	cv::flip(gray, gray, 1);


	cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

	std::vector<cv::Point2f> markerCenter;
	double x_ave, y_ave;
	for (auto i = markerCorners.begin(); i != markerCorners.end(); ++i ){
		x_ave = y_ave = 0;
		for (auto j = i->begin(); j != i->end(); ++j){
			x_ave += j->x;
			y_ave += j->y;
		}


		markerCenter.push_back(cv::Point2f((x_diff / 2)-x_ave / 4.0  , y_ave / 4.0));
	}

#if 0  //DRAW
	if (!markerCenter.empty())
		cv::circle(gray, markerCenter[0], 10, cv::Scalar(100, 200, 200));
	cv::aruco::drawDetectedMarkers(gray, markerCorners, markerIds);
	cv::imshow("ARUCO", gray);
#endif // 0  //DRAW

	return markerCenter;
}