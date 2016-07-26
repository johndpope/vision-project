
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
	double density;

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
	double ***M = NULL;
	double *E_vector_host = NULL;// local elements in array form
	double *E_vector_device = NULL;	
	// This 2D array will have information regarding the d.o.f of each element, i.e. displaceInElem[1][0] will give us the
	// index for displacement of the first node and its 0th d.o.f, the second entry can range from 0-dim.
	int **displaceInElem=NULL;  
	int *displaceInElem_host=NULL;
	int *displaceInElem_device=NULL;


	//Force
	int numForceBC; // This is the number of forces acting on the surface
	int *elemForce=NULL; // Which element the force is acting on
	int *localcoordForce=NULL; //Which side of the element, it will be the opposite of the local coord
	double *forceVec_x=NULL; //What is the force/unit
	double *forceVec_y=NULL;

	//K_matrix nad M_matrix
	double **K = NULL;
	double *K_vector_form = NULL;
	double *u = NULL;
	double *f = NULL;
	//double **M = NULL;
	
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
	double *h_M_dense;
	
	float *d_A_dense;
	double *d_A_dense_double; 
	double *h_A_dense_double;

                  // --- Leading dimension of dense matrix
	int *d_nnzPerVector; 
	int *h_nnzPerVector;
	int nnz;
	int lda;

	//float *h_x = NULL;


	//device side dense matrix
	float *d_A;
	int *d_A_RowIndices;
	int *d_A_ColIndices;

	//------Implicit variables
	bool dynamic;
	double beta_1;
	double beta_2;
	double dt;
	float *L = NULL;
	float *device_L = NULL;
	float *b_rhs = NULL;
	//float *device_b_rhs = NULL;
	//double *u = NULL;

	double *u_dot = NULL;
	double *u_doubledot = NULL;
	double *u_doubledot_old = NULL;

	//for capacitance matrix parameters
	double c_alpha = 0;
	double c_xi = 0;
	

	//Memory used in cholesky factorization
	csric02Info_t info_A = 0; 
	csrsv2Info_t  info_L = 0;  
	csrsv2Info_t  info_Lt = 0; 
	
	//set dirichlet conditions
	int *vector_zero_nodes = NULL;
	int numNodesZero;

	//CUDA USE BOOL
	bool cuda_use = false;

	
public:
	Geometry();
	~Geometry();
	//VON MISES STRESS
	double *global_stress_mises = NULL;
	void read_nodes(void);
	void read_elem(void);
	void read_force(void);
	
	void set_dim(int dim_input){ dim = dim_input; std::cout << "Dimension is: " << dim; };
	void set_YoungPoisson(double Young_input, double Poisson_input){ Young = Young_input; Poisson = Poisson_input; std::cout << "Young: " <<Young<<"  Poisson:" << Poisson<<std::endl; };
	void set_thickness(double thickness_input){ thickness = thickness_input; std::cout << "Thickness is: " << thickness << std::endl; };
	void set_density(double density_input){ density = density_input; };


	int return_numNodes(){ return numNodes; }
	int return_numElems(){ return numE; }
	int return_dim(){ return dim; }

	double return_x(int pos){ return x[pos]; }
	double return_y(int pos){ return y[pos]; }
	double return_z(int pos){ return z[pos]; }

	int node_number_inElem(int El_num, int node_num){ return nodesInElem[El_num][node_num]; }

	//Initilize matrices
	void initilizeMatrices(void);
	//Function to setup K matrix
	//********************2D**************
	void Linear2DBarycentric_B(int *nodes, double *x, double *y, double **term);
	double Linear2DJacobianDet_Barycentric(int *nodes, double *x, double *y);
	void Linear2DBarycentric_D(double **term, double nu, double youngE);
	void AssembleLocalElementMatrixBarycentric2D(int,int *nodes,int **, double *x, double *y, int dimension, double **E,double **M, double nu, double youngE, double thickness);
	void AssembleGlobalElementMatrixBarycentric(int numP, int numE, int nodesPerElem, int **elem, double ***E,double ***M, float *K,double *global_M, int **displaceInElem);
	


	//*******************3D***************
	
	void Linear3DBarycentric_D(double **term, double nu, double youngE);
	void Linear3DBarycentric_B(int *nodes, double *x, double *y, double *z, double **term);
	double Linear3DJacobianDet_Barycentric(int *nodes, double *x, double *y, double *z);
	void AssembleLocalElementMatrixBarycentric3D(int *nodes, double *x, double *y, double *z, int dimension, double **E, double nu, double youngE, double thickness);

	//*******************cuda************
	void Linear2DBarycentric_B_CUDA_host(void);//Solves the LHS of the equation for the dynamic elasticity equation
	void Linear3DBarycentric_B_CUDA_host(void);

	void Linear3DBarycentric_globalK_host(void);

	void make_K_matrix(void);

	//Functions to setup f
	void ApplySudoForcesBarycentric(int numP, int numBC, int *localcoord, int *elemForce, double force_x, double force_y, double *f, int **nodesInElem, double thickness, double *x, double *y, int **displaceInElem);
	void ApplyEssentialBoundaryConditionsBarycentric(int numP, int numBC, int *localcoord, int *elemForce, double forceVec_x, double forceVec_y, double *f, double **K, int **nodesInElem, double thickness, double *x, double *y, int **displaceInElem);
	void make_surface_f(void);


	//set sudo forces
	void setSudoNode(int node){ sudo_node_force = node; };
	void setSudoForcex(double x_force){ sudo_force_x = x_force; };
	void setSudoForcey(double y_force){ sudo_force_y = y_force; };

	//------------Dynamic problem ---------//
	void set_dt(double DT){ dt = DT; };
	void set_beta1(double b1){ beta_1 = b1; };
	void set_beta2(double b2){ beta_2 = b2; };
	
	void find_b(void);
	void initialize_dynamic(void);
	void update_vector(void);
	void set_dynamic(bool tf){ dynamic = tf; };
	bool get_dynamic(){ return dynamic; }; //Bool variable for determining if we are using dynamic FEM
	void update_dynamic_vectors(void);
	void update_dynamic_xyz(void);
	void set_dynamic_alpha(double alpha){ c_alpha = alpha; };
	void set_dynamic_xi(double xi){ c_xi = xi; };
	//Solver
	void initialize_CUDA(void);
	int tt(void);

	//setting boundary condition
	void initialize_zerovector(int numberofelements); // Initializing an array with all of the non moving nodes
	void set_zero_nodes(int *); // Set the nodes that will not move, this will be done to the LHS and RHIS of the system of equations Ax = b;
	void set_zero_AxB(void);
	

	//cuda use set and read
	void set_cuda_use(bool t_f){ cuda_use = t_f; };
	bool get_cuda_use(void){ return cuda_use; };
	
};


#endif //CUDA_SOLVER_FEM