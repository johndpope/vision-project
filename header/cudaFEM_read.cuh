
#ifndef CUDA_READ_FEM
#define CUDA_READ_FEM

#include <iostream>
#include "string"
#include "fstream"
#include "cudaFEM_read.cuh"

#include <stdio.h> 
#include <stdlib.h>
#include <cuda_runtime.h> 
#include "cublas_v2.h" 
#include <iostream>


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
void read_nodes(int *numNodes);

class Geometry{
	//Dimensions
	int dim;


	//Material properties
	double Young, Poisson;
	double thickness;

	//Nodes
	int numNodes;
	double *x=NULL, *y=NULL, *z = NULL;
	double *d_x = NULL, *d_y = NULL, *d_z = NULL;

	//Elements
	int numE;
	int numNodesPerElem;
	int **nodesInElem=NULL;
	int *nodesInElem_host = NULL;
	int *nodesInElem_device = NULL;
	double ***E = NULL;					// Array of local element stiffness matrices
	double *E_vector_host = NULL;// local elements in array form
	double *E_vector_device = NULL;
	int **displaceInElem=NULL;
	int *displaceInElem_host=NULL;
	int *displaceInElem_device=NULL;


	//Force
	int numForceBC; // This is the number of forces acting on the surface
	int *elemForce=NULL; // Which element the force is acting on
	int *localcoordForce=NULL; //Which side of the element, it will be the opposite of the local coord
	double *forceVec_x=NULL; //What is the force/unit
	double *forceVec_y=NULL;

	//K_matrix
	double **K = NULL;
	double *K_vector_form = NULL;
	double *u = NULL;
	double *f = NULL;

	
	//For mouse movement's sudo force
	double sudo_node_force;
	double sudo_force_x;
	double sudo_force_y;

	//cuda allocations
	int Nrows;                        // --- Number of rows
	int Ncols;                        // --- Number of columns
	int N;
	double duration_K;
	cusparseHandle_t handle;
	cusparseMatDescr_t descrA;
	cusparseMatDescr_t      descr_L = 0;
	float *h_A_dense;
	
	float *d_A_dense;
	double *d_A_dense_double; 
	double *h_A_dense_double;

                  // --- Leading dimension of dense matrix
	int *d_nnzPerVector; 
	int *h_nnzPerVector;
	int nnz;
	int lda;


	//device side dense matrix
	float *d_A;
	int *d_A_RowIndices;
	int *d_A_ColIndices;


	//Memory used in cholesky factorization
	csric02Info_t info_A = 0; 
	csrsv2Info_t  info_L = 0;  
	csrsv2Info_t  info_Lt = 0; 
	
public:
	Geometry();
	~Geometry();
	void read_nodes(void);
	void read_elem(void);
	void read_force(void);
	
	void set_dim(int dim_input){ dim = dim_input; std::cout << "Dimension is: " << dim; };
	void set_YoungPoisson(double Young_input, double Poisson_input){ Young = Young_input; Poisson = Poisson_input; std::cout << "Young: " <<Young<<"  Poisson:" << Poisson<<std::endl; };
	void set_thickness(double thickness_input){ thickness = thickness_input; std::cout << "Thickness is: " << thickness << std::endl; };
	
	int return_numNodes(){ return numNodes; };
	int return_numElems(){ return numE; };
	int return_dim(){ return dim; };
	double return_x(int pos){ return x[pos]; };
	double return_y(int pos){ return y[pos]; };
	double return_z(int pos){ return z[pos]; };
	double node_number_inElem(int El_num, int node_num){ return nodesInElem[El_num][node_num]; };

	//Initilize matrices
	void initilizeMatrices(void);
	//Function to setup K matrix
	//********************2D**************
	void Linear2DBarycentric_B(int *nodes, double *x, double *y, double **term);
	double Linear2DJacobianDet_Barycentric(int *nodes, double *x, double *y);
	void Linear2DBarycentric_D(double **term, double nu, double youngE);
	void AssembleLocalElementMatrixBarycentric2D(int *nodes, double *x, double *y, int dimension, double **E, double nu, double youngE, double thickness);
	void AssembleGlobalElementMatrixBarycentric(int numP, int numE, int nodesPerElem, int **elem, double ***E, float *K, int **displaceInElem);
	
	//*******************3D***************
	
	void Linear3DBarycentric_D(double **term, double nu, double youngE);
	void Linear3DBarycentric_B(int *nodes, double *x, double *y, double *z, double **term);
	double Linear3DJacobianDet_Barycentric(int *nodes, double *x, double *y, double *z);
	void AssembleLocalElementMatrixBarycentric3D(int *nodes, double *x, double *y, double *z, int dimension, double **E, double nu, double youngE, double thickness);

	//*******************cuda************
	void Linear3DBarycentric_B_CUDA_host(void);

	void Linear3DBarycentric_globalK_host(void);

	void make_K_matrix(void);

	//Functions to setup f
	void ApplySudoForcesBarycentric(int numP, int numBC, int *localcoord, int *elemForce, double force_x, double force_y, double *f, int **nodesInElem, double thickness, double *x, double *y, int **displaceInElem);
	void ApplyEssentialBoundaryConditionsBarycentric(int numP, int numBC, int *localcoord, int *elemForce, double *forceVec_x, double *forceVec_y, double *f, double **K, int **nodesInElem, double thickness, double *x, double *y, int **displaceInElem);
	void make_surface_f(void);


	//set sudo forces
	void setSudoNode(int node){ sudo_node_force = node; };
	void setSudoForcex(double x_force){ sudo_force_x = x_force; };
	void setSudoForcey(double y_force){ sudo_force_y = y_force; };


	//Solver
	void initialize_CUDA(void);
	int tt(void);

	
};


#endif //CUDA_SOLVER_FEM