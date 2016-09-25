#ifndef ARUCO_CENTER_H
#define ARUCO_CENTER_H
#include <opencv2\opencv.hpp>
#include "opencv2\aruco\charuco.hpp"
#include "opencv2\aruco\dictionary.hpp"
#include "opencv2\aruco.hpp"
std::vector<cv::Point2f> find_aruco_center(cv::Mat,int);
#endif // !ARUCO_CENTER_H
