#ifndef CUDA_FUNCIONTS_FEM
#define CUDA_FUNCIONTS_FEM

#include <stdio.h>


#include <math.h>
#include <GLFW/glfw3.h>
#include "cudaFEM_read.cuh"
#include <linmath.h>
__global__ void make_K_cuda3d(double *E_vector, int *nodesInElem_device, double *x_vector, double *y_vector, double *z_vector, int *displaceInElem_device, float *d_A_dense, int *numnodes);//3D
__global__ void make_K_cuda2d(double *K, int *nodesInElem, double *x_vector, double *y_vector, int *displaceInElem_device, float *d_A_dense, int numnodes, double thickness,double young_E,double nu,double alpha,double beta1,double beta2, double rho, double dt,double c_xi,int numE);//2D
__global__ void make_global_K(void); 
#endif //CUDA_DRAW_FEM
