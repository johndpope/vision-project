/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA)
* associated with this source code for terms and conditions that govern
* your use of this NVIDIA software.
*
*/
#include <iostream>
#include "string"
#include "fstream"
#include "cudaFEM_read.cuh"

#include <stdio.h> 
#include <stdlib.h>
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include <iostream>
#include "cuda_runtime.h"
#include <cuda.h>


#include <cufft.h>
#include "fstream"

#include <cuda.h>
#include <cusolverSp.h>
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include <cusparse.h>
#include <vector>
#include <cassert>
#include "Utilities.cuh"
#include <ctime>

#include "cuda_functions.cuh"
#define nodesinelemX(node,el,nodesPerElem) (node + nodesPerElem*el)
#define threeD21D(row_d,col_d,el_d,width_d,depth_d) (row_d+width_d*(col_d+depth_d*el_d))
#define nodesDisplacementX(dof,node,dimension) (dof + node*dimension)
#define IDX2C(i,j,ld) (((j)*(ld))+( i )) 

//This is for the local K matrix
//NOTE:::: nu and E are not initilized
__device__ inline float atomicAdda(float* address, double value)

{

	float ret = atomicExch(address, 0.0f);

	float old = ret + (float) value;

	while ((old = atomicExch(address, old)) != 0.0f)

	{

		old = atomicExch(address, 0.0f) + old;

	}

	return ret;

};
__global__ void make_K_cuda(double *E_vector, int *nodesInElem, double *x_vector, double *y_vector, double *z_vector, int *displaceInElem_device, float *d_A_dense,int numnodes) {
	//int x = threadIdx.x + blockIdx.x*blockDim.x; //if we have a 3D problem then this will go from 0 to 11
	int row;
	int dummy_node;
	int loop_node;
	int dummy_row;
	int dummy_col;
	int DOF[12];
	int counter;
	int offset = threadIdx.x + blockIdx.x*blockDim.x; // offset will essentaillay be the element counter
	int max_limit = 12 * 12 * 4374;
	double E = 200000;
	double nu = 0.45;
	double x14 = x_vector[nodesInElem[nodesinelemX(0, offset, 4)]] - x_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double x24 = x_vector[nodesInElem[nodesinelemX(1, offset, 4)]] - x_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double x34 = x_vector[nodesInElem[nodesinelemX(2, offset, 4)]] - x_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double y14 = y_vector[nodesInElem[nodesinelemX(0, offset, 4)]] - y_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double y24 = y_vector[nodesInElem[nodesinelemX(1, offset, 4)]] - y_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double y34 = y_vector[nodesInElem[nodesinelemX(2, offset, 4)]] - y_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double z14 = z_vector[nodesInElem[nodesinelemX(0, offset, 4)]] - z_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double z24 = z_vector[nodesInElem[nodesinelemX(1, offset, 4)]] - z_vector[nodesInElem[nodesinelemX(3, offset, 4)]];
	double z34 = z_vector[nodesInElem[nodesinelemX(2, offset, 4)]] - z_vector[nodesInElem[nodesinelemX(3, offset, 4)]];

	//std::cout << x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * 34) + z14*(x24*y34 - y24*x34) << std::endl;
	double det_J = (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34));

	double J_bar11 = (y24*z34 - z24*y34) / det_J;
	double J_bar12 = (z14*y34 - y14*z34) / det_J;
	double J_bar13 = (y14*z24 - z14*y24) / det_J;
	double J_bar21 = (z24*x34 - x24*z34) / det_J;
	double J_bar22 = (x14*z34 - z14*x34) / det_J;
	double J_bar23 = (z14*x24 - x14*z24) / det_J;
	double J_bar31 = (x24*y34 - y24*x34) / det_J;
	double J_bar32 = (y14*x34 - x14*y34) / det_J;
	double J_bar33 = (x14*y24 - y14*x24) / det_J;

	double J_star1 = -(J_bar11 + J_bar12 + J_bar13);
	double J_star2 = -(J_bar21 + J_bar22 + J_bar23);
	double J_star3 = -(J_bar31 + J_bar32 + J_bar33);

	
	
		E_vector[offset*144 + 0] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 1] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 2] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 3] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 4] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 5] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 6] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 7] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 8] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 9] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 10] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 11] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 12] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 13] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 14] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 15] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 16] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 17] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 18] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 19] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 20] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 21] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 22] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 23] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 24] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 25] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 26] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 27] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 28] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 29] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 30] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 31] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 32] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 33] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 34] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 35] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 36] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 37] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 38] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 39] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 40] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 41] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 42] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 43] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 44] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 45] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 46] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 47] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 48] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 49] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 50] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 51] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 52] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 53] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 54] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 55] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 56] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 57] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 58] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 59] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 60] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 61] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 62] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 63] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 64] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 65] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 66] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 67] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 68] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 69] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 70] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 71] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 72] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 73] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 74] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 75] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 76] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 77] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 78] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 79] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 80] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 81] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 82] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 83] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 84] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 85] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 86] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 87] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 88] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 89] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 90] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 91] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 92] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 93] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 94] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 95] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 96] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 97] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 98] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 99] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 100] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 101] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 102] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 103] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 104] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 105] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 106] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 107] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 108] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 109] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 110] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 111] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 112] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 113] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 114] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 115] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 116] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 117] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 118] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 119] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 120] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 121] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 122] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 123] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 124] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 125] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 126] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 127] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 128] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 129] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 130] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 131] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 132] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 133] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 134] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 135] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 136] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 137] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 138] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 139] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 140] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 141] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 142] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1);
		E_vector[offset*144 + 143] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1);
		

		counter = 0;
		for (int npe = 0; npe < 4; npe++){
			dummy_node = nodesInElem[nodesinelemX(npe, offset, 4)]; // The row of the matrix we looking at will be k_th element and npe (nodes per element) 	
			for (int dof = 0; dof < 3; dof++){

				DOF[counter] = displaceInElem_device[nodesDisplacementX(dof, dummy_node, 3)];
				counter++;
			}
		}

		//we will use atomic add because we will be writting to a single location multiple times (perhaps) 
		for (int c = 0; c < 12; c++){
			for (int r = 0; r < 12; r++){

				//d_A_dense[IDX2C(DOF[c], DOF[r], 3000)] = d_A_dense[IDX2C(DOF[c], DOF[r], 3000)] + E_vector[offset * 144 + c*12+r];
				atomicAdda(&(d_A_dense[IDX2C(DOF[c], DOF[r], 3 * numnodes)]), E_vector[offset * 144 + c * 12 + r]);
				//IDX2C(DOF[c], DOF[r], 3000)
				//K[IDX2C(DOF[r], DOF[c], numP*dim)] = K[IDX2C(DOF[r], DOF[c], numP*dim)] + E[k][r][c];
			}
		}


}

//This is for the global K matrix
__global__ void make_global_K(){

}

//working version??
//__global__ void make_K_cuda(double *E_vector, int *nodesInElem, double *x_vector, double *y_vector, double *z_vector) {
//	int x = threadIdx.x + blockIdx.x*blockDim.x; //if we have a 3D problem then this will go from 0 to 11
//	int y = threadIdx.y + blockIdx.y*blockDim.y; //the blockdim for x and y should be the same, which is 12
//	int z = threadIdx.z + blockIdx.z*blockDim.z; //This will control 
//	int offset = x + 12 * (y + z * 12);
//	int max_limit = 12 * 12*4374;
//	double E = 200000;
//	double nu = 0.45;
//	double x14 = x_vector[nodesInElem[nodesinelemX(0, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double x24 = x_vector[nodesInElem[nodesinelemX(1, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double x34 = x_vector[nodesInElem[nodesinelemX(2, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double y14 = y_vector[nodesInElem[nodesinelemX(0, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double y24 = y_vector[nodesInElem[nodesinelemX(1, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double y34 = y_vector[nodesInElem[nodesinelemX(2, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double z14 = z_vector[nodesInElem[nodesinelemX(0, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double z24 = z_vector[nodesInElem[nodesinelemX(1, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double z34 = z_vector[nodesInElem[nodesinelemX(2, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//
//	//std::cout << x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * 34) + z14*(x24*y34 - y24*x34) << std::endl;
//	double det_J = (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34));
//
//	double J_bar11 = (y24*z34 - z24*y34) / det_J;
//	double J_bar12 = (z14*y34 - y14*z34) / det_J;
//	double J_bar13 = (y14*z24 - z14*y24) / det_J;
//	double J_bar21 = (z24*x34 - x24*z34) / det_J;
//	double J_bar22 = (x14*z34 - z14*x34) / det_J;
//	double J_bar23 = (z14*x24 - x14*z24) / det_J;
//	double J_bar31 = (x24*y34 - y24*x34) / det_J;
//	double J_bar32 = (y14*x34 - x14*y34) / det_J;
//	double J_bar33 = (x14*y24 - y14*x24) / det_J;
//
//	double J_star1 = -(J_bar11 + J_bar12 + J_bar13);
//	double J_star2 = -(J_bar21 + J_bar22 + J_bar23);
//	double J_star3 = -(J_bar31 + J_bar32 + J_bar33);
//	//__syncthreads();
//	//B_Matrix testing
//	//if ((x == 0) && (y == 0)){ E_vector[offset] = J_bar11; }
//	//if ((x == 0) && (y == 1)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 2)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 3)){ E_vector[offset] = J_bar12; }
//	//	if ((x == 0) && (y == 4)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 5)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 6)){ E_vector[offset] = J_bar13; }
//	//	if ((x == 0) && (y == 7)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 8)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 9)){ E_vector[offset] = J_star1; }
//	//	if ((x == 0) && (y == 10)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 11)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 0)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 1)){ E_vector[offset] = J_bar21; }
//	//	if ((x == 1) && (y == 2)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 3)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 4)){ E_vector[offset] = J_bar22; }
//	//	if ((x == 1) && (y == 5)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 6)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 7)){ E_vector[offset] = J_bar23; }
//	//	if ((x == 1) && (y == 8)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 9)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 10)){ E_vector[offset] = J_star2; }
//	//	if ((x == 1) && (y == 11)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 0)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 1)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 2)){ E_vector[offset] = J_bar31; }
//	//	if ((x == 2) && (y == 3)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 4)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 5)){ E_vector[offset] = J_bar32; }
//	//	if ((x == 2) && (y == 6)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 7)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 8)){ E_vector[offset] = J_bar33; }
//	//	if ((x == 2) && (y == 9)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 10)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 11)){ E_vector[offset] = J_star3; }
//	//	if ((x == 3) && (y == 0)){ E_vector[offset] = J_bar21; }
//	//	if ((x == 3) && (y == 1)){ E_vector[offset] = J_bar11; }
//	//	if ((x == 3) && (y == 2)){ E_vector[offset] = 0; }
//	//	if ((x == 3) && (y == 3)){ E_vector[offset] = J_bar22; }
//	//	if ((x == 3) && (y == 4)){ E_vector[offset] = J_bar12; }
//	//	if ((x == 3) && (y == 5)){ E_vector[offset] = 0; }
//	//	if ((x == 3) && (y == 6)){ E_vector[offset] = J_bar23; }
//	//	if ((x == 3) && (y == 7)){ E_vector[offset] = J_bar13; }
//	//	if ((x == 3) && (y == 8)){ E_vector[offset] = 0; }
//	//	if ((x == 3) && (y == 9)){ E_vector[offset] = J_star2; }
//	//	if ((x == 3) && (y == 10)){ E_vector[offset] = J_star1; }
//	//	if ((x == 3) && (y == 11)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 0)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 1)){ E_vector[offset] = J_bar31; }
//	//	if ((x == 4) && (y == 2)){ E_vector[offset] = J_bar21; }
//	//	if ((x == 4) && (y == 3)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 4)){ E_vector[offset] = J_bar32; }
//	//	if ((x == 4) && (y == 5)){ E_vector[offset] = J_bar22; }
//	//	if ((x == 4) && (y == 6)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 7)){ E_vector[offset] = J_bar33; }
//	//	if ((x == 4) && (y == 8)){ E_vector[offset] = J_bar23; }
//	//	if ((x == 4) && (y == 9)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 10)){ E_vector[offset] = J_star3; }
//	//	if ((x == 4) && (y == 11)){ E_vector[offset] = J_star2; }
//	//	if ((x == 5) && (y == 0)){ E_vector[offset] = J_bar31; }
//	//	if ((x == 5) && (y == 1)){ E_vector[offset] = 0; }
//	//	if ((x == 5) && (y == 2)){ E_vector[offset] = J_bar11; }
//	//	if ((x == 5) && (y == 3)){ E_vector[offset] = J_bar32; }
//	//	if ((x == 5) && (y == 4)){ E_vector[offset] = 0; }
//	//	if ((x == 5) && (y == 5)){ E_vector[offset] = J_bar12; }
//	//	if ((x == 5) && (y == 6)){ E_vector[offset] = J_bar33; }
//	//	if ((x == 5) && (y == 7)){ E_vector[offset] = 0; }
//	//	if ((x == 5) && (y == 8)){ E_vector[offset] = J_bar13; }
//	//	if ((x == 5) && (y == 9)){ E_vector[offset] = J_star3; }
//	//	if ((x == 5) && (y == 10)){ E_vector[offset] = 0; }
//	//	if ((x == 5) && (y == 11)){ E_vector[offset] = J_star1; }
//	
//		if ((x == 0) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11* det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11* det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11* det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12* det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12* det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12* det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		//__syncthreads();
//	
//}
//

//
//__global__ void make_K_cuda(double *E_vector, int *nodesInElem, double *x_vector, double *y_vector, double *z_vector) {
//	int x = threadIdx.x + blockIdx.x*blockDim.x; //if we have a 3D problem then this will go from 0 to 11
//	int y = threadIdx.y + blockIdx.y*blockDim.y; //the blockdim for x and y should be the same, which is 12
//	int z = threadIdx.z + blockIdx.z*blockDim.z; //This will control 
//	int offset = x + 12 * (y + z * 12);
//	int max_limit = 12 * 12*4374;
//	double E = 200000;
//	double nu = 0.45;
//	double x14 = x_vector[nodesInElem[nodesinelemX(0, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double x24 = x_vector[nodesInElem[nodesinelemX(1, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double x34 = x_vector[nodesInElem[nodesinelemX(2, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double y14 = y_vector[nodesInElem[nodesinelemX(0, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double y24 = y_vector[nodesInElem[nodesinelemX(1, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double y34 = y_vector[nodesInElem[nodesinelemX(2, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double z14 = z_vector[nodesInElem[nodesinelemX(0, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double z24 = z_vector[nodesInElem[nodesinelemX(1, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//	double z34 = z_vector[nodesInElem[nodesinelemX(2, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];
//
//	//std::cout << x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * 34) + z14*(x24*y34 - y24*x34) << std::endl;
//	double det_J = (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34));
//
//	double J_bar11 = (y24*z34 - z24*y34) / det_J;
//	double J_bar12 = (z14*y34 - y14*z34) / det_J;
//	double J_bar13 = (y14*z24 - z14*y24) / det_J;
//	double J_bar21 = (z24*x34 - x24*z34) / det_J;
//	double J_bar22 = (x14*z34 - z14*x34) / det_J;
//	double J_bar23 = (z14*x24 - x14*z24) / det_J;
//	double J_bar31 = (x24*y34 - y24*x34) / det_J;
//	double J_bar32 = (y14*x34 - x14*y34) / det_J;
//	double J_bar33 = (x14*y24 - y14*x24) / det_J;
//
//	double J_star1 = -(J_bar11 + J_bar12 + J_bar13);
//	double J_star2 = -(J_bar21 + J_bar22 + J_bar23);
//	double J_star3 = -(J_bar31 + J_bar32 + J_bar33);
//	//__syncthreads();
//	//B_Matrix testing
//	//if ((x == 0) && (y == 0)){ E_vector[offset] = J_bar11; }
//	//if ((x == 0) && (y == 1)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 2)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 3)){ E_vector[offset] = J_bar12; }
//	//	if ((x == 0) && (y == 4)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 5)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 6)){ E_vector[offset] = J_bar13; }
//	//	if ((x == 0) && (y == 7)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 8)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 9)){ E_vector[offset] = J_star1; }
//	//	if ((x == 0) && (y == 10)){ E_vector[offset] = 0; }
//	//	if ((x == 0) && (y == 11)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 0)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 1)){ E_vector[offset] = J_bar21; }
//	//	if ((x == 1) && (y == 2)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 3)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 4)){ E_vector[offset] = J_bar22; }
//	//	if ((x == 1) && (y == 5)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 6)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 7)){ E_vector[offset] = J_bar23; }
//	//	if ((x == 1) && (y == 8)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 9)){ E_vector[offset] = 0; }
//	//	if ((x == 1) && (y == 10)){ E_vector[offset] = J_star2; }
//	//	if ((x == 1) && (y == 11)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 0)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 1)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 2)){ E_vector[offset] = J_bar31; }
//	//	if ((x == 2) && (y == 3)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 4)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 5)){ E_vector[offset] = J_bar32; }
//	//	if ((x == 2) && (y == 6)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 7)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 8)){ E_vector[offset] = J_bar33; }
//	//	if ((x == 2) && (y == 9)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 10)){ E_vector[offset] = 0; }
//	//	if ((x == 2) && (y == 11)){ E_vector[offset] = J_star3; }
//	//	if ((x == 3) && (y == 0)){ E_vector[offset] = J_bar21; }
//	//	if ((x == 3) && (y == 1)){ E_vector[offset] = J_bar11; }
//	//	if ((x == 3) && (y == 2)){ E_vector[offset] = 0; }
//	//	if ((x == 3) && (y == 3)){ E_vector[offset] = J_bar22; }
//	//	if ((x == 3) && (y == 4)){ E_vector[offset] = J_bar12; }
//	//	if ((x == 3) && (y == 5)){ E_vector[offset] = 0; }
//	//	if ((x == 3) && (y == 6)){ E_vector[offset] = J_bar23; }
//	//	if ((x == 3) && (y == 7)){ E_vector[offset] = J_bar13; }
//	//	if ((x == 3) && (y == 8)){ E_vector[offset] = 0; }
//	//	if ((x == 3) && (y == 9)){ E_vector[offset] = J_star2; }
//	//	if ((x == 3) && (y == 10)){ E_vector[offset] = J_star1; }
//	//	if ((x == 3) && (y == 11)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 0)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 1)){ E_vector[offset] = J_bar31; }
//	//	if ((x == 4) && (y == 2)){ E_vector[offset] = J_bar21; }
//	//	if ((x == 4) && (y == 3)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 4)){ E_vector[offset] = J_bar32; }
//	//	if ((x == 4) && (y == 5)){ E_vector[offset] = J_bar22; }
//	//	if ((x == 4) && (y == 6)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 7)){ E_vector[offset] = J_bar33; }
//	//	if ((x == 4) && (y == 8)){ E_vector[offset] = J_bar23; }
//	//	if ((x == 4) && (y == 9)){ E_vector[offset] = 0; }
//	//	if ((x == 4) && (y == 10)){ E_vector[offset] = J_star3; }
//	//	if ((x == 4) && (y == 11)){ E_vector[offset] = J_star2; }
//	//	if ((x == 5) && (y == 0)){ E_vector[offset] = J_bar31; }
//	//	if ((x == 5) && (y == 1)){ E_vector[offset] = 0; }
//	//	if ((x == 5) && (y == 2)){ E_vector[offset] = J_bar11; }
//	//	if ((x == 5) && (y == 3)){ E_vector[offset] = J_bar32; }
//	//	if ((x == 5) && (y == 4)){ E_vector[offset] = 0; }
//	//	if ((x == 5) && (y == 5)){ E_vector[offset] = J_bar12; }
//	//	if ((x == 5) && (y == 6)){ E_vector[offset] = J_bar33; }
//	//	if ((x == 5) && (y == 7)){ E_vector[offset] = 0; }
//	//	if ((x == 5) && (y == 8)){ E_vector[offset] = J_bar13; }
//	//	if ((x == 5) && (y == 9)){ E_vector[offset] = J_star3; }
//	//	if ((x == 5) && (y == 10)){ E_vector[offset] = 0; }
//	//	if ((x == 5) && (y == 11)){ E_vector[offset] = J_star1; }
//	
//		if ((x == 0) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11* det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 0) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11* det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 1) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11* det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 2) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12* det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 3) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12* det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 4) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12* det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 5) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 6) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 7) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 8) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 9) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 10) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 0)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 9)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 10)){ E_vector[offset] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
//		if ((x == 11) && (y == 11)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
//		//__syncthreads();
//	
//}
////


//working 2d cuda
/*
__global__ void make_K_cuda(double *E_vector, int *nodesInElem, double *x_vector, double *y_vector, double *z_vector) {
	 __shared__ double det_J_shared[72];
	 double det_J;
	 double J_bar11;
	double J_bar12;
	double J_bar13;
	double J_bar21;
	double J_bar22;
	double J_bar23;
	double J_bar31;
	double J_bar32;
	double J_bar33 ;
	 double x14;
	 double x24;
	 double x34;
	 double y14;
	 double y24;
	 double y34;
	 double z14;
	 double z24;
	 double z34;
	double E = 200000;
	double nu = 0.45;
	int x = threadIdx.x + blockIdx.x*blockDim.x; //if we have a 3D problem then this will go from 0 to 11
	int z = threadIdx.y + blockIdx.y*blockDim.y; //the blockdim for x and y should be the same, which is 12

	int offset = x + 12 * ( z * 12);
	int max_limit = 12 * 12 * 4374;
	int cacheIndex = threadIdx.x;
	
	 x14 = x_vector[nodesInElem[nodesinelemX(0, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
	 x24 = x_vector[nodesInElem[nodesinelemX(1, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
	 x34 = x_vector[nodesInElem[nodesinelemX(2, z, 4)]] - x_vector[nodesInElem[nodesinelemX(3, z, 4)]];
	 y14 = y_vector[nodesInElem[nodesinelemX(0, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
	 y24 = y_vector[nodesInElem[nodesinelemX(1, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
	 y34 = y_vector[nodesInElem[nodesinelemX(2, z, 4)]] - y_vector[nodesInElem[nodesinelemX(3, z, 4)]];
	 z14 = z_vector[nodesInElem[nodesinelemX(0, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];
	 z24 = z_vector[nodesInElem[nodesinelemX(1, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];
	 z34 = z_vector[nodesInElem[nodesinelemX(2, z, 4)]] - z_vector[nodesInElem[nodesinelemX(3, z, 4)]];

	//std::cout << x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * 34) + z14*(x24*y34 - y24*x34) << std::endl;

	//these lines take up 0.02 ms -begin

		 det_J = (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34));
		
	

	 
	 //det_J = det_J_shared[cacheIndex];
	J_bar11 = (y24*z34 - z24*y34) / det_J;
	 J_bar12 = (z14*y34 - y14*z34) / det_J;
	 J_bar13 = (y14*z24 - z14*y24) / det_J;
	 J_bar21 = (z24*x34 - x24*z34) / det_J;
	 J_bar22 = (x14*z34 - z14*x34) / det_J;
	 J_bar23 = (z14*x24 - x14*z24) / det_J;
	 J_bar31 = (x24*y34 - y24*x34) / det_J;
	 J_bar32 = (y14*x34 - x14*y34) / det_J;
	 J_bar33 = (x14*y24 - y14*x24) / det_J;

	double J_star1 = -(J_bar11 + J_bar12 + J_bar13);
	double J_star2 = -(J_bar21 + J_bar22 + J_bar23);
	double J_star3 = -(J_bar31 + J_bar32 + J_bar33);
	//-endd
	//__syncthreads();
	//B_Matrix testing
	//if ((x == 0) && (y == 0)){ E_vector[offset] = J_bar11; }
	//if ((x == 0) && (y == 1)){ E_vector[offset] = 0; }
	//	if ((x == 0) && (y == 2)){ E_vector[offset] = 0; }
	//	if ((x == 0) && (y == 3)){ E_vector[offset] = J_bar12; }
	//	if ((x == 0) && (y == 4)){ E_vector[offset] = 0; }
	//	if ((x == 0) && (y == 5)){ E_vector[offset] = 0; }
	//	if ((x == 0) && (y == 6)){ E_vector[offset] = J_bar13; }
	//	if ((x == 0) && (y == 7)){ E_vector[offset] = 0; }
	//	if ((x == 0) && (y == 8)){ E_vector[offset] = 0; }
	//	if ((x == 0) && (y == 9)){ E_vector[offset] = J_star1; }
	//	if ((x == 0) && (y == 10)){ E_vector[offset] = 0; }
	//	if ((x == 0) && (y == 11)){ E_vector[offset] = 0; }
	//	if ((x == 1) && (y == 0)){ E_vector[offset] = 0; }
	//	if ((x == 1) && (y == 1)){ E_vector[offset] = J_bar21; }
	//	if ((x == 1) && (y == 2)){ E_vector[offset] = 0; }
	//	if ((x == 1) && (y == 3)){ E_vector[offset] = 0; }
	//	if ((x == 1) && (y == 4)){ E_vector[offset] = J_bar22; }
	//	if ((x == 1) && (y == 5)){ E_vector[offset] = 0; }
	//	if ((x == 1) && (y == 6)){ E_vector[offset] = 0; }
	//	if ((x == 1) && (y == 7)){ E_vector[offset] = J_bar23; }
	//	if ((x == 1) && (y == 8)){ E_vector[offset] = 0; }
	//	if ((x == 1) && (y == 9)){ E_vector[offset] = 0; }
	//	if ((x == 1) && (y == 10)){ E_vector[offset] = J_star2; }
	//	if ((x == 1) && (y == 11)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 0)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 1)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 2)){ E_vector[offset] = J_bar31; }
	//	if ((x == 2) && (y == 3)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 4)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 5)){ E_vector[offset] = J_bar32; }
	//	if ((x == 2) && (y == 6)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 7)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 8)){ E_vector[offset] = J_bar33; }
	//	if ((x == 2) && (y == 9)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 10)){ E_vector[offset] = 0; }
	//	if ((x == 2) && (y == 11)){ E_vector[offset] = J_star3; }
	//	if ((x == 3) && (y == 0)){ E_vector[offset] = J_bar21; }
	//	if ((x == 3) && (y == 1)){ E_vector[offset] = J_bar11; }
	//	if ((x == 3) && (y == 2)){ E_vector[offset] = 0; }
	//	if ((x == 3) && (y == 3)){ E_vector[offset] = J_bar22; }
	//	if ((x == 3) && (y == 4)){ E_vector[offset] = J_bar12; }
	//	if ((x == 3) && (y == 5)){ E_vector[offset] = 0; }
	//	if ((x == 3) && (y == 6)){ E_vector[offset] = J_bar23; }
	//	if ((x == 3) && (y == 7)){ E_vector[offset] = J_bar13; }
	//	if ((x == 3) && (y == 8)){ E_vector[offset] = 0; }
	//	if ((x == 3) && (y == 9)){ E_vector[offset] = J_star2; }
	//	if ((x == 3) && (y == 10)){ E_vector[offset] = J_star1; }
	//	if ((x == 3) && (y == 11)){ E_vector[offset] = 0; }
	//	if ((x == 4) && (y == 0)){ E_vector[offset] = 0; }
	//	if ((x == 4) && (y == 1)){ E_vector[offset] = J_bar31; }
	//	if ((x == 4) && (y == 2)){ E_vector[offset] = J_bar21; }
	//	if ((x == 4) && (y == 3)){ E_vector[offset] = 0; }
	//	if ((x == 4) && (y == 4)){ E_vector[offset] = J_bar32; }
	//	if ((x == 4) && (y == 5)){ E_vector[offset] = J_bar22; }
	//	if ((x == 4) && (y == 6)){ E_vector[offset] = 0; }
	//	if ((x == 4) && (y == 7)){ E_vector[offset] = J_bar33; }
	//	if ((x == 4) && (y == 8)){ E_vector[offset] = J_bar23; }
	//	if ((x == 4) && (y == 9)){ E_vector[offset] = 0; }
	//	if ((x == 4) && (y == 10)){ E_vector[offset] = J_star3; }
	//	if ((x == 4) && (y == 11)){ E_vector[offset] = J_star2; }
	//	if ((x == 5) && (y == 0)){ E_vector[offset] = J_bar31; }
	//	if ((x == 5) && (y == 1)){ E_vector[offset] = 0; }
	//	if ((x == 5) && (y == 2)){ E_vector[offset] = J_bar11; }
	//	if ((x == 5) && (y == 3)){ E_vector[offset] = J_bar32; }
	//	if ((x == 5) && (y == 4)){ E_vector[offset] = 0; }
	//	if ((x == 5) && (y == 5)){ E_vector[offset] = J_bar12; }
	//	if ((x == 5) && (y == 6)){ E_vector[offset] = J_bar33; }
	//	if ((x == 5) && (y == 7)){ E_vector[offset] = 0; }
	//	if ((x == 5) && (y == 8)){ E_vector[offset] = J_bar13; }
	//	if ((x == 5) && (y == 9)){ E_vector[offset] = J_star3; }
	//	if ((x == 5) && (y == 10)){ E_vector[offset] = 0; }
	//	if ((x == 5) && (y == 11)){ E_vector[offset] = J_star1; }
	x = x +1;

	if ((x == 1)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 2)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 3)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 4)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 5)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 6)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 7)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 8)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 9)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 10)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 11)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 12)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 13)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 14)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 15)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 16)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 17)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 18)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 19)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 20)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 21)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 22)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 23)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 24)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 25)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar11*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 26)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 27)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar11 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar21 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar31 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 28)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 29)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 30)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 31)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 32)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 33)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 34)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 35)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 36)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 37)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 38)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 39)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 40)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 41)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 42)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 43)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 44)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 45)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 46)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 47)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 48)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 49)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 50)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 51)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 52)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 53)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 54)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 55)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 56)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 57)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 58)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 59)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 60)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 61)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 62)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 63)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar12*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar32*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 64)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar12*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 65)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 66)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar12 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar22 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar32 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 67)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 68)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 69)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 70)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 71)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 72)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 73)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 74)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 75)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 76)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 77)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 78)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 79)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 80)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 81)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 82)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 83)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 84)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 85)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar21*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 86)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 87)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 88)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar22*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 89)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 90)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 91)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar23*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 92)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 93)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 94)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 95)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 96)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 97)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 98)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar31*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 99)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 100)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 101)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar32*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 102)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_bar13*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_bar23*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_bar33*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 103)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar13*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 104)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_bar33*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar33*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 105)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_bar13 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_bar23 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_bar33 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 106)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 107)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 108)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 109)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 110)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 111)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 112)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 113)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 114)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 115)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 116)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 117)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 118)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 119)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 120)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 121)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 122)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 123)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 124)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 125)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 126)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 127)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 128)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 129)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*nu / (-2 * nu*nu - nu + 1); }
	if ((x == 130)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star2*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 131)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 132)){ E_vector[offset] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 133)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 134)){ E_vector[offset] = 0.166666666666667*E*J_bar21*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 135)){ E_vector[offset] = 0.166666666666667*E*J_bar11*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar21*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar31*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 136)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 137)){ E_vector[offset] = 0.166666666666667*E*J_bar22*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 138)){ E_vector[offset] = 0.166666666666667*E*J_bar12*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar22*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar32*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 139)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 140)){ E_vector[offset] = 0.166666666666667*E*J_bar23*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 141)){ E_vector[offset] = 0.166666666666667*E*J_bar13*J_star1*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar23*J_star2*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_bar33*J_star3*det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	if ((x == 142)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star1*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 143)){ E_vector[offset] = 0.166666666666667*E*J_star2*J_star3*det_J*nu / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star3*det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1); }
	if ((x == 144)){ E_vector[offset] = 0.166666666666667*E*J_star1*J_star1 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star2*J_star2 * det_J*(-0.5*nu + 0.5) / (-2 * nu*nu - nu + 1) + 0.166666666666667*E*J_star3*J_star3 * det_J*(-nu + 1.0) / (-2 * nu*nu - nu + 1); }
	
}
//


*/