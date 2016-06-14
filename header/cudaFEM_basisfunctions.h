#ifndef FEM_BASISFUNCTIONS

#include <stdlib.h>
#include <iostream>


using namespace std;

// Create a general function pointer variable type 


// -----------------------------------2D linear-barycentric elements --------------------------
double Linear2DJacobianDet_Barycentric(int *nodes, double *x, double *y);
void Linear2DBarycentric_B(int *nodes, double *x, double *y, double **term);
#define FEM_BASISFUNCTIONS
#endif
