//#pragma once
#include "cudaFEM_read.cuh"
 //int width = 512;
 //int height = 424;
const int colorwidth = 1920;
const int colorheight = 1080;

void drawKinectData();
int kinect_main(int argc, char* argv[], Geometry *p);
