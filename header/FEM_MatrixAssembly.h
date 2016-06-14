#ifndef FEM_MATRIXASSEMBLY

#include <stdlib.h>
#include <iostream>

using namespace std;

#include "FEM_BasisFunctions.h"
#include "FEM_Misc.h"
//------------------------------Barycentric coordinates----------------------------------------------
void Linear2DBarycentric_D(double **term, double nu, double youngE);
void AssembleLocalElementMatrixBarycentric2D(int *nodes, double *x, double *y, double *c, int elementType, int dimension, double **E,double nu,double youngE,double thickness);

void AssembleGlobalElementMatrixBarycentric(int numP, int numE, int nodesPerElem, int **elem, double ***E, double **K, int **displaceInElem);

void ApplyEssentialBoundaryConditionsBarycentric(int numP, int numBC, int *localcoord, int *elemForce, double *forceVec_x,double *forceVec_y, double *f, double **K, int **nodesInElem, double thickness, double *x, double *y,int **displaceInElem);	// outputs

void LU_factorize(double **A, double **P, double **L, double **U, int N, double *b);



#define FEM_MATRIXASSEMBLY
#endif