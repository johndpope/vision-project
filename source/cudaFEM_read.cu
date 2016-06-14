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
#include "cuda_functions.cuh"
#define max(a,b) ((a) > (b) ? (a) : (b))
#define IDX2C(i,j,ld) (((j)*(ld))+( i )) 
#define threeD21D(row_d,col_d,el_d,width_d,depth_d) (row_d+width_d*(col_d+depth_d*el_d))
#define nodesinelemX(node,el,nodesPerElem) (node + nodesPerElem*el)
#define nodesDisplacementX(dof,node,dimension) (dof + node*dimension)
Geometry::Geometry(){
	std::cout << "Geometry Object created" << std::endl;
}

Geometry::~Geometry(){
	std::cout << "Geometry Object deleted" << std::endl;

	//deleteing dynamic arrays
	delete[] x;
	delete[] y;


	for (int e = 0; e < numE; e++){

		for (int i = 0; i < numNodesPerElem*dim; i++){
			delete E[e][i];
		}
		delete E[e];
		delete nodesInElem[e];
	}

	for (int i = 0; i < numNodes; i++){
		delete displaceInElem[i];

	}






	for (int i = 0; i < numNodes*dim; i++) {
		delete K[i];

	}
	delete[] K;
	delete[] u;
	delete[] f;
	delete[] displaceInElem;
	delete[] E;
	delete[] nodesInElem;
	delete[] E_vector_host;

	//
	cudaFree(d_A_dense);
	cudaFree(d_nnzPerVector);
	cudaFree(d_A);
	cudaFree(d_A_RowIndices);
	cudaFree(d_A_ColIndices);
	cudaFree(nodesInElem_device);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	cudaFree(E_vector_device);
	free(h_nnzPerVector);

	//free(h_A_dense);


}

void Geometry::read_nodes(){

	std::ifstream in_matrix("FEM_Nodes.txt");

	if (!in_matrix){
		std::cout << "cannot open Nodes \n";
	}

	in_matrix >> numNodes;

	x = new double[numNodes];
	y = new double[numNodes];
	z = new double[numNodes];

	if (dim == 3){

		for (int i = 0; i < numNodes; i++){
			in_matrix >> x[i] >> y[i] >> z[i];
		}
	}
	else if(dim ==2){
		for (int i = 0; i < numNodes; i++){
			in_matrix >> x[i] >> y[i];
			z[i] = 0;
		}
	}

	in_matrix.close();



}


void Geometry::read_elem(){

	std::ifstream in_elem("FEM_Elem.txt");

	if (!in_elem){
		std::cout << "cannot open Element file \n";
	}

	in_elem >> numE >> numNodesPerElem;

	//Allocating E matrix 3x3x3 matrix
	E = new double**[numE];
	nodesInElem = new int*[numE];
	nodesInElem_host = new int[numE*numNodesPerElem];
	nodesInElem_device = new int[numE*numNodesPerElem];




	cudaMalloc((void**)&nodesInElem_device, numE*numNodesPerElem*sizeof(int));

	for (int e = 0; e < numE; e++){
		E[e] = new double*[numNodesPerElem*dim];
		nodesInElem[e] = new int[numNodesPerElem];
		for (int i = 0; i < numNodesPerElem*dim; i++){
			E[e][i] = new double[numNodesPerElem*dim];
		}
	}
	E_vector_host = new double[numE*numNodesPerElem*dim*numNodesPerElem*dim];
	cudaMalloc((void**)&E_vector_device, numE*numNodesPerElem*dim*numNodesPerElem*dim*sizeof(double));
	//Populating the nodesinelem matrix
	for (int e = 0; e < numE; e++) {
		for (int i = 0; i < numNodesPerElem; i++)
			in_elem >> nodesInElem[e][i];
	}


	in_elem.close();


	for (int e = 0; e < numE; e++) {
		for (int i = 0; i < numNodesPerElem; i++){
			nodesInElem_host[nodesinelemX(i, e, numNodesPerElem)] = nodesInElem[e][i];
			//std::cout << nodesInElem_host[nodesinelemX(i, e, numNodesPerElem)] << std::endl;
		}
		//std::cout << std::endl;

	}

	cudaMemcpy(nodesInElem_device, nodesInElem_host, numE*numNodesPerElem*sizeof(int), cudaMemcpyHostToDevice);


	std::ifstream in_disp("FEM_displacement.txt");

	if (!in_disp){
		std::cout << "cannot open displacement file \n";
	}
	displaceInElem = new int*[numNodes];
	displaceInElem_host = new int[numNodes*dim];
	displaceInElem_device = new int[numNodes*dim];
	for (int i = 0; i < numNodes; i++){
		displaceInElem[i] = new int[3];

	}
	cudaMalloc((void**)&displaceInElem_device, numNodes*dim*sizeof(int));

	for (int i = 0; i < numNodes; i++){
		for (int j = 0; j < dim; j++){
			in_disp >> displaceInElem[i][j];
			
		}

	}

	for (int i = 0; i < numNodes; i++){
		for (int j = 0; j < dim; j++){
			
			displaceInElem_host[nodesDisplacementX(j, i, dim)] = displaceInElem[i][j];
		}

	}

	cudaMemcpy(displaceInElem_device, displaceInElem_host, numNodes*dim*sizeof(int), cudaMemcpyHostToDevice);
	in_disp.close();



}




void Geometry::read_force(){

	std::ifstream in_matrix("FEM_force.txt");

	if (!in_matrix){
		std::cout << "cannot open force file \n";
	}
	in_matrix >> numForceBC;
	elemForce = new int[numForceBC];
	localcoordForce = new int[numForceBC];
	forceVec_x = new double[numForceBC];
	forceVec_y = new double[numForceBC];
	for (int i = 0; i < numForceBC; i++){
		in_matrix >> elemForce[i] >> localcoordForce[i] >> forceVec_x[i] >> forceVec_y[i];
	}

	in_matrix.close();



}


void Geometry::initilizeMatrices(){

	cudaMalloc((void**)&d_x, numNodes*sizeof(double));
	cudaMalloc((void**)&d_y, numNodes*sizeof(double));
	cudaMalloc((void**)&d_z, numNodes*sizeof(double));
	K = new double*[numNodes*dim];
	h_A_dense = new float[numNodes*dim*numNodes*dim*sizeof(*h_A_dense)];
	d_A_dense_double = new double[numNodes*dim*numNodes*dim*sizeof(*d_A_dense_double)];
	h_A_dense_double = new double[numNodes*dim*numNodes*dim*sizeof(*h_A_dense_double)];

	
	gpuErrchk(cudaMalloc(&d_A_dense, numNodes*dim*numNodes*dim* sizeof(*d_A_dense)));
	gpuErrchk(cudaMalloc(&d_A_dense_double, numNodes*dim*numNodes*dim* sizeof(*d_A_dense_double)));

	//B = new double*[3];
	for (int i = 0; i < 6; i++){
		//B[i] = new double[3];
	}

	for (int i = 0; i < numNodes*dim; i++) {
		K[i] = new double[numNodes*dim];

	}

	u = new double[numNodes*dim];
	f = new double[numNodes*dim];

	for (int i = 0; i < numNodes*dim; i++){
		f[i] = 0;
	}
}
void Geometry::make_K_matrix(){
	std::clock_t start_K_local;
	std::clock_t start_K_global;
	start_K_local = std::clock();
	bool cuda_use = false;
	if (cuda_use){
		Linear3DBarycentric_B_CUDA_host();

	}
	else{
		for (int e = 0; e < numE; e++) {
			//cout << Linear2DJacobianDet_Barycentric(nodesInElem[e], x, y) << endl;

			if (dim == 2){
				AssembleLocalElementMatrixBarycentric2D(nodesInElem[e], x, y, dim, E[e], Poisson, Young, thickness);
			}
			else if (dim == 3){
				AssembleLocalElementMatrixBarycentric3D(nodesInElem[e], x, y, z, dim, E[e], Poisson, Young, thickness);

			}
		}
	}
	start_K_global = std::clock();
	double duration_K_local = (std::clock() - start_K_local) / (double)CLOCKS_PER_SEC;
	
	if (!cuda_use)
		AssembleGlobalElementMatrixBarycentric(numNodes*dim, numE, numNodesPerElem, nodesInElem, E, h_A_dense, displaceInElem);
	double duration_K_global = (std::clock() - start_K_global) / (double)CLOCKS_PER_SEC;
	//ApplyEssentialBoundaryConditionsBarycentric(numNodes*dim, numForceBC, localcoordForce, elemForce, forceVec_x, forceVec_y, f, K, nodesInElem, thickness, x, y, displaceInElem);
	ApplySudoForcesBarycentric(numNodes*dim, sudo_node_force, localcoordForce, elemForce, sudo_force_x, sudo_force_y, f, nodesInElem, thickness, x, y, displaceInElem);
	std::cout << "FPS time local K matrix: " << duration_K_local << std::endl;
	std::cout << "FPS time global K matrix: " << duration_K_global << std::endl;
	//std::cout << "sudo force x: " << sudo_force_x << " sudo_force y: " << sudo_force_y << std::endl;
}

void Geometry::AssembleGlobalElementMatrixBarycentric(int numP, int numE, int nodesPerElem, int **elem, double ***E, float *K, int **displaceInElem){
	//cout << numP << endl << endl << endl << endl;


	//Initialising several variables
	int i;
	int j;
	int row;
	int col;

	//Make a numPxnumP matrix all equal to zero
	for (j = 0; j < numP; j++){
		for (i = 0; i < numP; i++){
			K[IDX2C(j, i, numP)] = 0;
		}
	}
	int dummy_node;
	int loop_node;
	int dummy_row;
	int dummy_col;
	int *DOF = new int[numNodes*dim];
	int counter;

	for (int k = 0; k < numE; k++){
		counter = 0;
		for (int npe = 0; npe < numNodesPerElem; npe++){
			dummy_node = elem[k][npe]; // The row of the matrix we looking at will be k_th element and npe (nodes per element) 	
			for (int dof = 0; dof < dim; dof++){
				row = displaceInElem[dummy_node][dof];
				DOF[counter] = row;
				//cout << DOF[counter] << endl;
				counter++;
			}
		}
		for (int c = 0; c < numNodesPerElem*dim; c++){
			for (int r = 0; r < numNodesPerElem*dim; r++){

				K[IDX2C(DOF[c], DOF[r], numP)] = K[IDX2C(DOF[c], DOF[r], numP)] + E[k][r][c];
				//K[IDX2C(DOF[r], DOF[c], numP*dim)] = K[IDX2C(DOF[r], DOF[c], numP*dim)] + E[k][r][c];
			}
		}
	}


	//for (i = 0; i < numP; i++){
	//	for (j = 0; j< numP; j++){
	//	std::cout << K[IDX2C(j, i, numP)] << "   ";
	//	}
	//	std::cout<< std::endl;
	//	}

}

void Geometry::Linear2DBarycentric_B(int *nodes, double *x, double *y, double **term){
	//
	double J = Linear2DJacobianDet_Barycentric(nodes, x, y);
	
		double y23 = y[nodes[1]] - y[nodes[2]];//y23
		double y31 = y[nodes[2]] - y[nodes[0]];//y31
		double y12 = y[nodes[0]] - y[nodes[1]];//y12
		double x32 = x[nodes[2]] - x[nodes[1]];//x32
		double x13 = x[nodes[0]] - x[nodes[2]];//x13
		double x21 = x[nodes[1]] - x[nodes[0]];//x21
		for (int row = 0; row < 3; row++){
			for (int col = 0; col < 6; col++){
				term[row][col] = 0;
			}
		}

		term[0][0] = term[2][1] = y23 / (J);
		term[0][2] = term[2][3] = y31 / (J);
		term[0][4] = term[2][5] = y12 / (J);
		term[1][1] = term[2][0] = x32 / (J);
		term[1][3] = term[2][2] = x13 / (J);
		term[1][5] = term[2][4] = x21 / (J);
	
	/*else {
		double **A = new double*[4];
		double **T = new double*[3];
		double **result = new double*[3];
		for (int i = 0; i < 4; i++){
		A[i] = new double[6];
		}
		for (int i = 0; i < 3; i++){
		T[i] = new double[4];
		result[i] = new double[6];
		}

		for (int row = 0; row < 3; row++){
		for (int col = 0; col < 4; col++){
		T[row][col] = 0;
		}
		}

		T[0][0] = T[1][3] = T[2][1] = T[2][2] = 1;
		A[0][1] = A[0][3] = A[0][5] = 0;
		A[1][1] = A[1][3] = A[1][5] = 0;
		A[2][0] = A[2][2] = A[2][4] = 0;
		A[3][0] = A[3][2] = A[3][4] = 0;

		A[0][0] = A[2][1] = y23/J;
		A[0][2] = A[2][3] = y31/J;
		A[0][4] = A[2][5] = y12 / J;
		A[1][0] = A[3][1] = x32 / J;
		A[1][2] = A[3][3] = x13 / J;
		A[1][4] = A[3][5] = x21 / J;
	}
	*/
	

	
	//MatrixTimes(term, T, A, 3, 4, 4, 6);

}


void Geometry::Linear3DBarycentric_B(int *nodes, double *x, double *y, double *z, double **term){
	//
	double x14 = x[nodes[0]] - x[nodes[3]];
	double x24 = x[nodes[1]] - x[nodes[3]];
	double x34 = x[nodes[2]] - x[nodes[3]];
	double y14 = y[nodes[0]] - y[nodes[3]];
	double y24 = y[nodes[1]] - y[nodes[3]];
	double y34 = y[nodes[2]] - y[nodes[3]];
	double z14 = z[nodes[0]] - z[nodes[3]];
	double z24 = z[nodes[1]] - z[nodes[3]];
	double z34 = z[nodes[2]] - z[nodes[3]];

	double J = x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * x34) + z14*(x24*y34 - y24*x34);
	double J_bar11 = (y24*z34 - z24*y34) / J;
	double J_bar12 = (z14*y34 - y14*z34) / J;
	double J_bar13 = (y14*z24 - z14*y24) / J;
	double J_bar21 = (z24*x34 - x24*z34) / J;
	double J_bar22 = (x14*z34 - z14*x34) / J;
	double J_bar23 = (z14*x24 - x14*z24) / J;
	double J_bar31 = (x24*y34 - y24*x34) / J;
	double J_bar32 = (y14*x34 - x14*y34) / J;
	double J_bar33 = (x14*y24 - y14*x24) / J;

	/* term[0][0]  = (y24*z34 - z24*y34) / J;
	 term[0][1]= (z14*y34 - y14*z34) / J;
	 term[0][2] =(y14*z24 - z14*y24) / J;
	 term[1][0]= (z24*x34 - x24*z24) / J;
	 term[1][1] = (x14*z34 - z14*x34) / J;
	 term[1][2] = (z14*x24 - x14*z24) / J;
	 term[2][0]= (x24*y34 - y24*x34) / J;
	 term[2][1]= (y14*x34 - x14*y34) / J;
	 term[2][2] = (x14*y24 - y14*x24) / J;*/

	double J_star1 = -(J_bar11 + J_bar12 + J_bar13);
	double J_star2 = -(J_bar21 + J_bar22 + J_bar23);
	double J_star3 = -(J_bar31 + J_bar32 + J_bar33);


	/*double **A = new double*[4];
	double **T = new double*[3];
	double **result = new double*[3];
	for (int i = 0; i < 4; i++){
	A[i] = new double[6];
	}
	for (int i = 0; i < 3; i++){
	T[i] = new double[4];
	result[i] = new double[6];
	}

	for (int row = 0; row < 3; row++){
	for (int col = 0; col < 4; col++){
	T[row][col] = 0;
	}
	}

	T[0][0] = T[1][3] = T[2][1] = T[2][2] = 1;
	A[0][1] = A[0][3] = A[0][5] = 0;
	A[1][1] = A[1][3] = A[1][5] = 0;
	A[2][0] = A[2][2] = A[2][4] = 0;
	A[3][0] = A[3][2] = A[3][4] = 0;

	A[0][0] = A[2][1] = y23/J;
	A[0][2] = A[2][3] = y31/J;
	A[0][4] = A[2][5] = y12 / J;
	A[1][0] = A[3][1] = x32 / J;
	A[1][2] = A[3][3] = x13 / J;
	A[1][4] = A[3][5] = x21 / J;*/

	for (int row = 0; row < 6; row++){
		for (int col = 0; col < 12; col++){
			term[row][col] = 0;
		}

	}
	term[0][0] = term[3][1] = term[5][2] = J_bar11;
	term[1][1] = term[3][0] = term[4][2] = J_bar21;
	term[2][2] = term[5][0] = term[4][1] = J_bar31;

	term[0][3] = term[3][4] = term[5][5] = J_bar12;
	term[1][4] = term[3][3] = term[4][5] = J_bar22;
	term[2][5] = term[4][4] = term[5][3] = J_bar32;

	term[0][6] = term[3][7] = term[5][8] = J_bar13;
	term[1][7] = term[3][6] = term[4][8] = J_bar23;

	term[2][8] = term[4][7] = term[5][6] = J_bar33;

	term[0][9] = term[3][10] = term[5][11] = J_star1;
	term[1][10] = term[3][9] = term[4][11] = J_star2;
	term[2][11] = term[4][10] = term[5][9] = J_star3;

	//for (int row = 0; row < 6; row++){
	//	for (int col = 0; col < 12; col++){
	//		std::cout << term[row][col] << "   ";
	//	}
	//	std::cout << std::endl;
	//}
	//MatrixTimes(term, T, A, 3, 4, 4, 6);

}
double Geometry::Linear2DJacobianDet_Barycentric(int *nodes, double *x, double *y){
	double x13 = x[nodes[0]] - x[nodes[2]];
	double x23 = x[nodes[1]] - x[nodes[2]];
	double y13 = y[nodes[0]] - y[nodes[2]];
	double y23 = y[nodes[1]] - y[nodes[2]];

	return (x13*y23 - y13*x23);

}
double Geometry::Linear3DJacobianDet_Barycentric(int *nodes, double *x, double *y, double *z){
	double x14 = x[nodes[0]] - x[nodes[3]];
	double x24 = x[nodes[1]] - x[nodes[3]];
	double x34 = x[nodes[2]] - x[nodes[3]];
	double y14 = y[nodes[0]] - y[nodes[3]];
	double y24 = y[nodes[1]] - y[nodes[3]];
	double y34 = y[nodes[2]] - y[nodes[3]];
	double z14 = z[nodes[0]] - z[nodes[3]];
	double z24 = z[nodes[1]] - z[nodes[3]];
	double z34 = z[nodes[2]] - z[nodes[3]];

	//std::cout << x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 * 34) + z14*(x24*y34 - y24*x34) << std::endl;
	return (x14*(y24*z34 - y34*z24) - y14*(x24*z34 - z24 *x34) + z14*(x24*y34 - y24*x34));



}


void Geometry::Linear2DBarycentric_D(double **term, double nu, double youngE){
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			term[i][j] = 0;
		}
	}

	/*term[0][0] = term[1][1] = 1;

	term[0][1] = term[1][0] = nu;
	term[2][2] = (1 - nu) / 2;*/

	term[0][0] = term[1][1]=  1;
	term[0][1] = term[1][0] = nu;
	term[2][2] = (1 - nu) / 2;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			//term[i][j] = (youngE / (1 - nu*nu))*term[i][j];
			term[i][j] = (youngE / ((1 - nu*nu)))*term[i][j];
		}
	}

}


void Geometry::Linear3DBarycentric_D(double **term, double nu, double youngE){
	int multi = 2;
	for (int i = 0; i < 3 * multi; i++){
		for (int j = 0; j < 3 * multi; j++){
			term[i][j] = 0;
		}
	}

	/*term[0][0] = term[1][1] = 1;

	term[0][1] = term[1][0] = nu;
	term[2][2] = (1 - nu) / 2;*/

	term[0][0] = term[1][1] = term[2][2] = (1.0 - nu);
	term[0][1] = term[1][0] = term[0][2] = term[2][0] = term[1][2] = term[2][1] = nu;
	term[3][3] = term[4][4] = term[5][5] = (1.0 - nu) / 2.0;

	for (int i = 0; i < 3 * multi; i++){
		for (int j = 0; j < 3 * multi; j++){
			//term[i][j] = (youngE / (1 - nu*nu))*term[i][j];
			term[i][j] = (youngE / ((1 - 2 * nu)*(1 + nu)))*term[i][j];
		}
	}

	//for (int i = 0; i < 3 * multi; i++){
	//	for (int j = 0; j < 3 * multi; j++){
	//		std::cout << term[i][j] << "   ";
	//	}
	//	std::cout << std::endl;
	//}

}

void Geometry::AssembleLocalElementMatrixBarycentric2D(int *nodes, double *x, double *y, int dimension, double **E, double nu, double youngE, double thickness)
{
	// thte dimension for B is 3x6
	int n = 3;
	double **B = new double*[n];
	double **D = new double*[n];
	double **B_TXD = new double*[n * 2];
	double **integrand = new double*[n * 2];


	for (int i = 0; i < n; i++){

		B[i] = new double[n * 2];
		D[i] = new double[n];
	}
	for (int i = 0; i <n * 2; i++){

		B_TXD[i] = new double[n];
		integrand[i] = new double[n * 2];
	}


	double J = Linear2DJacobianDet_Barycentric(nodes, x, y);


	for (int row = 0; row < n * 2; row++){
		for (int col = 0; col < n; col++){
			B_TXD[row][col] = 0;
		}
	}

	for (int row = 0; row < n * 2; row++){
		for (int col = 0; col < n * 2; col++){
			integrand[row][col] = 0;
		}
	}

	//Allocating the B and D matrices
	Linear2DBarycentric_B(nodes, x, y, B);
	Linear2DBarycentric_D(D, nu, youngE);

	//std::cout << "B:MATRIX: " << std::endl;
	//for (int row = 0; row < 3; row++){
	//	for (int col = 0; col < 6; col++){
	//		std::cout << B[row][col]<< "    ";
	//	}
	//	std::cout << std::endl;
	//}

	//Finding B^T*D
	for (int row = 0; row < n * 2; row++){
		for (int col = 0; col < n; col++){
			for (int k = 0; k < n; k++){
				B_TXD[row][col] = B_TXD[row][col] + B[k][row] * D[k][col];
			}
		}
	}
	//Finding B^T*D*B
	for (int row = 0; row < n * 2; row++){
		for (int col = 0; col < n * 2; col++){
			for (int k = 0; k < n; k++){
				integrand[row][col] = integrand[row][col] + B_TXD[row][k] * B[k][col];
			}
		}
	}

	//std::cout << "B_T x D : " << std::endl;
	//for (int row = 0; row < 6; row++){
	//	for (int col = 0; col < 3; col++){
	//		std::cout << B_TXD[row][col] << "    ";
	//	}
	//	std::cout<<std::endl;
	//}



	//for (int row = 0; row < 6; row++){
	//	for (int col = 0; col < 3; col++){
	//		B_T[row][col] = B[col][row];
	//	}
	//}


	for (int row = 0; row < n * 2; row++){
		for (int col = 0; col < n * 2; col++){

			E[row][col] = thickness*(J / 2) * integrand[row][col];

		}

	}


	//std::cout << "K_elem : " << std::endl;
	//for (int row = 0; row < 6; row++){
	//	for (int col = 0; col < 6; col++){
	//		std::cout << E[row][col] << "    ";
	//	}
	//	std::cout << std::endl;
	//}





	for (int i = 0; i < n; i++){

		delete B[i];
		delete D[i];
	}
	for (int i = 0; i < n * 2; i++){

		delete B_TXD[i];
		delete integrand[i];
	}

	delete[] B;
	delete[] D;
	delete[] B_TXD;
	delete[] integrand;

}

//**************************3D************************************//
//3333333333333333333333333333333333333333333333333333333333333333//
//DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD//
//****************************************************************//
void Geometry::AssembleLocalElementMatrixBarycentric3D(int *nodes, double *x, double *y, double *z, int dimension, double **E, double nu, double youngE, double thickness)
{
	int multi = 2;
	double **B = new double*[3 * multi];
	double **D = new double*[3 * multi];
	double **B_TXD = new double*[6 * multi];
	double **integrand = new double*[6 * multi];


	for (int i = 0; i < 3 * multi; i++){

		B[i] = new double[6 * multi];
		D[i] = new double[3 * multi];
	}
	for (int i = 0; i < 6 * multi; i++){

		B_TXD[i] = new double[3 * multi];
		integrand[i] = new double[6 * multi];
	}


	double J = Linear3DJacobianDet_Barycentric(nodes, x, y, z);


	for (int row = 0; row < 6 * multi; row++){
		for (int col = 0; col < 3 * multi; col++){
			B_TXD[row][col] = 0;
		}
	}

	for (int row = 0; row < 6 * multi; row++){
		for (int col = 0; col < 6 * multi; col++){
			integrand[row][col] = 0;
		}
	}

	//Allocating the B and D matrices
	Linear3DBarycentric_B(nodes, x, y, z, B);
	Linear3DBarycentric_D(D, nu, youngE);

	//std::cout << "B:MATRIX: " << std::endl;
	//for (int row = 0; row < 3*multi; row++){
	//	for (int col = 0; col < 6*multi; col++){
	//		std::cout << B[row][col]<< "    ";
	//	}
	//	std::cout << std::endl;
	//}

	//Finding B^T*D
	for (int row = 0; row < 6 * multi; row++){
		for (int col = 0; col < 3 * multi; col++){
			for (int k = 0; k < 3 * multi; k++){
				B_TXD[row][col] = B_TXD[row][col] + B[k][row] * D[k][col];
			}
		}
	}
	//Finding B^T*D*B
	for (int row = 0; row < 6 * multi; row++){
		for (int col = 0; col < 6 * multi; col++){
			for (int k = 0; k < 3 * multi; k++){
				integrand[row][col] = integrand[row][col] + B_TXD[row][k] * B[k][col];
			}
		}
	}

	//std::cout << "B_T x D : " << std::endl;
	//for (int row = 0; row < 6; row++){
	//	for (int col = 0; col < 3; col++){
	//		std::cout << B_TXD[row][col] << "    ";
	//	}
	//	std::cout<<std::endl;
	//}



	//for (int row = 0; row < 6; row++){
	//	for (int col = 0; col < 3; col++){
	//		B_T[row][col] = B[col][row];
	//	}
	//}





	for (int row = 0; row < 6 * multi; row++){
		for (int col = 0; col < 6 * multi; col++){

			E[row][col] = integrand[row][col] * J / 6.0;

		}

	}


	//std::cout << "K_elem : " << std::endl;
	//for (int row = 0; row < 6*multi; row++){
	//	for (int col = 0; col < 6*multi; col++){
	//		std::cout << E[row][col] << " ";
	//	}
	//	std::cout << std::endl;
	//}





	for (int i = 0; i < 3 * multi; i++){

		delete B[i];
		delete D[i];
	}
	for (int i = 0; i < 6 * multi; i++){

		delete B_TXD[i];
		delete integrand[i];
	}

	delete[] B;
	delete[] D;
	delete[] B_TXD;
	delete[] integrand;

}
void Geometry::Linear3DBarycentric_B_CUDA_host(){

	int dummy_var;
	//dim3 blocks(1, 1, numE/5);//numE / (dim)
	//dim3 threads(numNodesPerElem*dim, numNodesPerElem*dim, 5);
	//dim3 blocks(144, (int)numE /( 32*15));//numE / (dim)
	//dim3 threads(1, (int)(32 * 15));

	//working 2d cuda
	dim3 blocks(72, 30);//numE / (dim)
	dim3 threads(2, numE / 30);
	/*for (int j = 0; j < numE;j++){
		for (int i = 0; i < numNodesPerElem; i++){
		nodesInElem_device[j][i] = nodesInElem[j][i];
		}
		}
		*/

	cudaMemcpy(d_x, x, numNodes*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, numNodes*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, z, numNodes*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(nodesInElem_device, nodesInElem, numE*numNodesPerElem*sizeof(int), cudaMemcpyHostToDevice);

	int max_limit = (numNodesPerElem*dim*numNodesPerElem*dim*numE);
	int threadsPerBlock = 256;
	int blocksPerGrid = (max_limit + threadsPerBlock - 1) / threadsPerBlock;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaMemset(d_A_dense, 0, numNodes*dim*numNodes*dim*sizeof(*d_A_dense));
	make_K_cuda << <numE / 144, 144 >> >(E_vector_device, nodesInElem_device, d_x, d_y, d_z, displaceInElem_device, d_A_dense);

	cudaMemcpy(h_A_dense, d_A_dense, numNodes*dim*numNodes*dim*sizeof(*d_A_dense), cudaMemcpyDeviceToHost);

	//cudaMemcpy(E_vector_host, E_vector_device, numNodesPerElem*dim*numNodesPerElem*dim*numE*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(x, d_x, numNodes*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(nodesInElem_host, nodesInElem_device, numE*numNodesPerElem*sizeof(int), cudaMemcpyDeviceToHost);

	//std::cout << " K _ CUDA " << std::endl;
	////for (int j = 0; j < 2; j++){
	////	for (int i = 0; i < numNodesPerElem; i++){
	////		std::cout << nodesInElem_host[nodesinelemX(i, j, numNodesPerElem)] << "  ";
	////	}
	////	std::cout << std::endl;
	////}
	//for (int j = 0; j < 10; j++){
	//	for (int i = 0; i < 10; i++){
	//		std::cout << h_A_dense[IDX2C(i, j, 3000)] << "  ";
	//	}
	//	std::cout << std::endl;
	//}



	////Print local K matrix
	//for (int e = 0; e < numE; e++){

	//	//std::cout << "element : " << e << std::endl;
	//	for (int i = 0; i < numNodesPerElem*dim; i++){
	//		for (int j = 0; j < numNodesPerElem*dim; j++){
	//			
	//			//E[e][i][j] = E_vector_host[threeD21D(i, j, e, numNodesPerElem*dim, numNodesPerElem*dim)];
	//			 //std::cout << E[e][i][j] << " ";
	//		}
	//		//std::cout << std::endl;
	//	}
	//	//std::cout << std::endl;
	//}

	//std::cout << std::endl << " the x value : " << x[0] << std::endl;
	/*(cudaMemcpy(&c, dev_c, sizeof(int),
		cudaMemcpyDeviceToHost));
		printf("2 + 7 = %d\n", c);
		(cudaFree(dev_c));*/

}


void Geometry::make_surface_f(){


}
void Geometry::ApplyEssentialBoundaryConditionsBarycentric(int numP, int numBC, int *localcoord, int *elemForce, double *forceVec_x, double *forceVec_y, double *f, double **K, int **nodesInElem, double thickness, double *x, double *y, int **displaceInElem){
	int local; // used to store local coord info
	int node_interest[2];// use two ints to tell us which 2 of the nodes in the element would be useful
	int row, col;
	int element;
	int node;
	double length;
	double x_1, y_1, x_2, y_2;
	for (int i = 0; i < numBC; i++){

		local = localcoord[i];

		if (local == 0){//Opposite to xi_1 direction
			node_interest[0] = 1;
			node_interest[1] = 2;
			x_1 = x[nodesInElem[elemForce[i]][node_interest[0]]];
			y_1 = y[nodesInElem[elemForce[i]][node_interest[0]]];
			x_2 = x[nodesInElem[elemForce[i]][node_interest[1]]];
			y_2 = y[nodesInElem[elemForce[i]][node_interest[1]]];
			length = sqrt(pow(x_1 - x_2, 2.0) + pow(y_1 - y_2, 2.0));
		}
		else if (local == 1){//Opposite to xi_2 direction
			node_interest[0] = 0;
			node_interest[1] = 2;
			x_1 = x[nodesInElem[elemForce[i]][node_interest[0]]];
			y_1 = y[nodesInElem[elemForce[i]][node_interest[0]]];
			x_2 = x[nodesInElem[elemForce[i]][node_interest[1]]];
			y_2 = y[nodesInElem[elemForce[i]][node_interest[1]]];
			length = sqrt(pow(x_1 - x_2, 2.0) + pow(y_1 - y_2, 2.0));
		}
		else if (local == 2){ // Opposite to xi_3 direction
			node_interest[0] = 0;
			node_interest[1] = 1;
			x_1 = x[nodesInElem[elemForce[i]][node_interest[0]]];
			y_1 = y[nodesInElem[elemForce[i]][node_interest[0]]];
			x_2 = x[nodesInElem[elemForce[i]][node_interest[1]]];
			y_2 = y[nodesInElem[elemForce[i]][node_interest[1]]];
			length = sqrt(pow(x_1 - x_2, 2.0) + pow(y_1 - y_2, 2.0));
		}
		//cout << endl << "length: " << length << endl;
		element = elemForce[i];
		for (int node_c = 0; node_c < 2; node_c++){
			node = nodesInElem[element][node_interest[node_c]];
			for (int dof = 0; dof < 2; dof++){
				row = displaceInElem[node][dof];

				for (int dummy_V = 0; dummy_V < numP; dummy_V++){
					//K[row][dummy_V] = 0;
				}
				//K[row][row] = 1;
				if (dof == 0){
					f[row] = f[row] + (length*thickness / 2)*forceVec_x[i];
				}
				else if (dof == 1){
					f[row] = f[row] + (length*thickness / 2)*forceVec_y[i];
				}
			}
		}



	}
}


void Geometry::ApplySudoForcesBarycentric(int numP, int numBC, int *localcoord, int *elemForce, double forceVec_x, double forceVec_y, double *f, int **nodesInElem, double thickness, double *x, double *y, int **displaceInElem){
	int local; // used to store local coord info
	int node_interest[2];// use two ints to tell us which 2 of the nodes in the element would be useful
	int row, col;
	int element;
	int node;
	double length;
	double x_1, y_1, x_2, y_2;


	for (int dummy_V = 0; dummy_V < numP; dummy_V++){
		f[dummy_V] = 0;
	}




	//cout << endl << "length: " << length << endl;

	for (int node_c = 5 ; node_c <  7; node_c++){

		for (int dof = 0; dof < dim; dof++){
			row = displaceInElem[node_c][dof];

			for (int dummy_V = 0; dummy_V < numP; dummy_V++){
				//K[row][dummy_V] = 0;
			}
			//K[row][row] = 1;
			if (dof == 0){
				f[row] += 70;
			}
			else if (dof == 1){
				f[row] +=70;
			}
			else if (dof == 2){
				f[row] += node_c/1728.0;
			}
		}
	}




}
void Geometry::initialize_CUDA(void){
	Nrows = numNodes*dim;                        // --- Number of rows
	Ncols = numNodes*dim;                        // --- Number of columns
	N = Nrows;
	cusparseSafeCall(cusparseCreate(&handle));
	
	//h_A_dense = (float*)malloc(Nrows*Ncols*sizeof(*h_A_dense));
	cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
	nnz = 0;                                // --- Number of nonzero elements in dense matrix
	lda = Nrows;                      // --- Leading dimension of dense matrix
	gpuErrchk(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));
	h_nnzPerVector = (int *)malloc(Nrows * sizeof(*h_nnzPerVector));


	//device side dense matrix
	gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));


	cusparseSafeCall(cusparseCreateMatDescr(&descr_L));
	cusparseSafeCall(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE));
	cusparseSafeCall(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
	cusparseSafeCall(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));

	//emeory in cholesky
	cusparseSafeCall(cusparseCreateCsric02Info(&info_A));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_L));
	cusparseSafeCall(cusparseCreateCsrsv2Info(&info_Lt));
}
int Geometry::tt()
{
	// --- Initialize cuSPARSE


	// --- Host side dense matrix

	double duration_K;


	// --- Column-major ordering
	/*h_A_dense[0] = 0.4612f;  h_A_dense[4] = -0.0006f;   h_A_dense[8] = 0.3566f; h_A_dense[12] = 0.0f;
	h_A_dense[1] = -0.0006f; h_A_dense[5] = 0.4640f;    h_A_dense[9] = -1000.0723f; h_A_dense[13] = 0.0f;
	h_A_dense[2] = 0.3566f;  h_A_dense[6] = 0.0723f;    h_A_dense[10] = 100.7543f; h_A_dense[14] = 0.0f;
	h_A_dense[3] = 0.f;      h_A_dense[7] = 0.0f;       h_A_dense[11] = 0.0f;    h_A_dense[15] = 0.1f;
	*/

	//for (int col = 0; col < Ncols; col++){
	//	for (int row = 0; row < Nrows; row++){

	//		h_A_dense[IDX2C(col, row, N)] = (float) h_A_dense_double[IDX2C(col, row, N)];
	//		//a[IDX2C(col, row, n)] = (float)ind++;
	//		//h_A_dense[IDX2C(col, row, N)] = 0;


	//	}

	//}
	for (int col = 0; col < Ncols; col++){

		h_A_dense[IDX2C(col, 0, N)] = 0;
		h_A_dense[IDX2C(col, 1, N)] = 0;
		if (dim == 3){
			h_A_dense[IDX2C(col, 2, N)] = 0;
		}
	}
	h_A_dense[IDX2C(0, 0, N)] = 1.0;
	h_A_dense[IDX2C(1, 1, N)] = 1.0;
	if (dim == 3){
		h_A_dense[IDX2C(2, 2, N)] = 1.0;
	}



	/*std::ofstream writenodes("global_K.txt");

	for (int j = 0; j < N; j++){
	for (int i = 0; i < N; i++){
	writenodes << h_A_dense[IDX2C(i, j, N)] << " ";
	}
	writenodes << std::endl;
	}

	writenodes.close();*/

	// --- Create device array and copy host array to it

	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, Nrows * Ncols * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

	// --- Descriptor for sparse matrix A




	// --- Device side number of nonzero elements per row

	cusparseSafeCall(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
	// --- Host side number of nonzero elements per row

	gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost));

	/*printf("Number of nonzero elements in dense matrix = %i\n\n", nnz);
	for (int i = 0; i < 10; ++i) printf("Number of nonzero elements in row %i = %i \n", i, h_nnzPerVector[i]);
	printf("\n");*/

	// --- Device side dense matrix
	gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
	gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));


	cusparseSafeCall(cusparseSdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));
	std::clock_t start_K;
	start_K = std::clock();
	// --- Host side dense matrix
	float *h_A = (float *)malloc(nnz * sizeof(*h_A));
	int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
	int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnz*sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));
	std::cout << nnz << std::endl;
	/*printf("\nOriginal matrix in CSR format\n\n");
	for (int i = 0; i < 10; ++i) printf("A[%i] = %.0f ", i, h_A[i]); printf("\n");

	printf("\n");
	for (int i = 0; i < (10 + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");

	for (int i = 0; i < 10; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);
	*/
	// --- Allocating and defining dense host and device data vectors

	float *h_x = (float *)malloc(Nrows * sizeof(float));
	/*h_x[0] = 100.0;  h_x[1] = 200.0; h_x[2] = 400.0; h_x[3] = 500.0;*/
	for (int i = 0; i < N; i++){
		h_x[i] = f[i];
	}
	if (dim == 3){
		h_x[0] = h_x[1] = h_x[2] = 0;
	}
	else {
		h_x[0] = h_x[1] =  0;
	}

	float *d_x;        gpuErrchk(cudaMalloc(&d_x, Nrows * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_x, h_x, Nrows * sizeof(float), cudaMemcpyHostToDevice));




	/******************************************/
	/* STEP 1: CREATE DESCRIPTORS FOR L AND U */
	/******************************************/




	/********************************************************************************************************/
	/* STEP 2: QUERY HOW MUCH MEMORY USED IN CHOLESKY FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/********************************************************************************************************/


	int pBufferSize_M, pBufferSize_L, pBufferSize_Lt;
	cusparseSafeCall(cusparseScsric02_bufferSize(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, &pBufferSize_M));
	cusparseSafeCall(cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, &pBufferSize_L));
	cusparseSafeCall(cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, &pBufferSize_Lt));

	int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));
	void *pBuffer = 0;  gpuErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));


	/******************************************************************************************************/
	/* STEP 3: ANALYZE THE THREE PROBLEMS: CHOLESKY FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
	/******************************************************************************************************/
	int structural_zero;

	cusparseSafeCall(cusparseScsric02_analysis(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));

	cusparseStatus_t status = cusparseXcsric02_zeroPivot(handle, info_A, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("A(%d,%d) is missing\n", structural_zero, structural_zero); }

	cusparseSafeCall(cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
	cusparseSafeCall(cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

	/*************************************/
	/* STEP 4: FACTORIZATION: A = L * L' */
	/*************************************/
	int numerical_zero;

	cusparseSafeCall(cusparseScsric02(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
	status = cusparseXcsric02_zeroPivot(handle, info_A, &numerical_zero);
	/*if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero); }
	*/

	gpuErrchk(cudaMemcpy(h_A, d_A, nnz * sizeof(float), cudaMemcpyDeviceToHost));
	/*printf("\nNon-zero elements in Cholesky matrix\n\n");
	for (int k = 0; k<10; k++) printf("%f\n", h_A[k]);*/

	cusparseSafeCall(cusparseScsr2dense(handle, Nrows, Ncols, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_A_dense, Nrows));

	/*printf("\nCholesky matrix\n\n");
	for (int i = 0; i < 10; i++) {
	std::cout << "[ ";
	for (int j = 0; j < 10; j++)
	std::cout << h_A_dense[i * Ncols + j] << " ";
	std::cout << "]\n";
	}*/

	/*********************/
	/* STEP 5: L * z = x */
	/*********************/
	// --- Allocating the intermediate result vector
	float *d_z;        gpuErrchk(cudaMalloc(&d_z, N * sizeof(float)));

	const float alpha = 1.;
	cusparseSafeCall(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, d_x, d_z, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));

	/**********************/
	/* STEP 5: L' * y = z */
	/**********************/
	// --- Allocating the host and device side result vector
	float *h_y = (float *)malloc(Ncols * sizeof(float));
	float *d_y;        gpuErrchk(cudaMalloc(&d_y, Ncols * sizeof(float)));

	cusparseSafeCall(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, d_z, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));

	cudaMemcpy(h_x, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
	printf("\n\nFinal result\n");
	/*for (int k = 0; k<20; k++) printf("dx[%i] = %f\n", k, h_x[k]);
	for (int k = 0; k<20; k++) printf("xs[%i] = %f\n", k, x[k]);*/


	for (int i = 0; i < numNodes; i++) {
		x[i] = x[i] + h_x[i * dim];
		y[i] = y[i] + h_x[i * dim + 1];
		if (dim == 3){
			z[i] = z[i] + h_x[i * dim + 2];
		}
		
	}

	free(h_A);
	free(h_A_RowIndices);
	free(h_A_ColIndices);
	//free(h_x);
	free(h_y);
	cudaFree(d_x);
	cudaFree(pBuffer);
	cudaFree(d_z);
	cudaFree(d_y);
	duration_K = (std::clock() - start_K) / (double)CLOCKS_PER_SEC;
	//std::cout << " change status : " << changeNode << std::endl;

	//std::cout << "FPS time: " <<1/duration_K << std::endl;

	//std::cout << "Duration: " << duration_K << std::endl;
	return 0;
}


//int Geometry::tt()
//{
//	// --- Initialize cuSPARSE
//	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));
//
//	const int Nrows = numNodes*dim;                        // --- Number of rows
//	const int Ncols = numNodes*dim;                        // --- Number of columns
//	const int N = Nrows;
//
//	// --- Host side dense matrix
//	double *h_A_dense = (double*)malloc(Nrows*Ncols*sizeof(*h_A_dense));
//
//	// --- Column-major ordering
//	/*h_A_dense[0] = 0.4612f;  h_A_dense[4] = -0.0006f;   h_A_dense[8] = 0.3566f; h_A_dense[12] = 0.0f;
//	h_A_dense[1] = -0.0006f; h_A_dense[5] = 0.4640f;    h_A_dense[9] = -1000.0723f; h_A_dense[13] = 0.0f;
//	h_A_dense[2] = 0.3566f;  h_A_dense[6] = 0.0723f;    h_A_dense[10] = 100.7543f; h_A_dense[14] = 0.0f;
//	h_A_dense[3] = 0.f;      h_A_dense[7] = 0.0f;       h_A_dense[11] = 0.0f;    h_A_dense[15] = 0.1f;
//	*/
//	for (int col = 0; col < Ncols; col++){
//		for (int row = 0; row < Nrows; row++){
//
//			h_A_dense[IDX2C(col, row, N)] = K[col][row];
//			//a[IDX2C(col, row, n)] = (double)ind++;
//			//h_A_dense[IDX2C(col, row, N)] = 0;
//
//
//		}
//
//	}
//	for (int col = 0; col < Ncols; col++){
//
//		h_A_dense[IDX2C(col, 0, N)] = 0;
//		h_A_dense[IDX2C(col, 1, N)] = 0;
//	}
//	h_A_dense[IDX2C(0, 0, N)] = 1;
//	h_A_dense[IDX2C(1, 1, N)] = 1;
//
//
//	// --- Create device array and copy host array to it
//	double *d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, Nrows * Ncols * sizeof(*d_A_dense)));
//	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, Nrows * Ncols * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
//
//	// --- Descriptor for sparse matrix A
//	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
//	cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
//	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
//
//	int nnz = 0;                                // --- Number of nonzero elements in dense matrix
//	const int lda = Nrows;                      // --- Leading dimension of dense matrix
//	// --- Device side number of nonzero elements per row
//	int *d_nnzPerVector;    gpuErrchk(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));
//	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, &nnz));
//	// --- Host side number of nonzero elements per row
//	int *h_nnzPerVector = (int *)malloc(Nrows * sizeof(*h_nnzPerVector));
//	gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost));
//
//	/*printf("Number of nonzero elements in dense matrix = %i\n\n", nnz);
//	for (int i = 0; i < 10; ++i) printf("Number of nonzero elements in row %i = %i \n", i, h_nnzPerVector[i]);
//	printf("\n");*/
//
//	// --- Device side dense matrix
//	double *d_A;            gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(*d_A)));
//	int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
//	int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(*d_A_ColIndices)));
//
//	cusparseSafeCall(cusparseDdense2csr(handle, Nrows, Ncols, descrA, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));
//
//	// --- Host side dense matrix
//	double *h_A = (double *)malloc(nnz * sizeof(*h_A));
//	int *h_A_RowIndices = (int *)malloc((Nrows + 1) * sizeof(*h_A_RowIndices));
//	int *h_A_ColIndices = (int *)malloc(nnz * sizeof(*h_A_ColIndices));
//	gpuErrchk(cudaMemcpy(h_A, d_A, nnz*sizeof(*h_A), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));
//
//	/*printf("\nOriginal matrix in CSR format\n\n");
//	for (int i = 0; i < 10; ++i) printf("A[%i] = %.0f ", i, h_A[i]); printf("\n");
//
//	printf("\n");
//	for (int i = 0; i < (10 + 1); ++i) printf("h_A_RowIndices[%i] = %i \n", i, h_A_RowIndices[i]); printf("\n");
//
//	for (int i = 0; i < 10; ++i) printf("h_A_ColIndices[%i] = %i \n", i, h_A_ColIndices[i]);
//	*/
//	// --- Allocating and defining dense host and device data vectors
//	double *h_x = (double *)malloc(Nrows * sizeof(double));
//	/*h_x[0] = 100.0;  h_x[1] = 200.0; h_x[2] = 400.0; h_x[3] = 500.0;*/
//	for (int i = 0; i < N; i++){
//		h_x[i] = f[i];
//	}
//	h_x[0] = h_x[1] = 0;
//
//	double *d_x;        gpuErrchk(cudaMalloc(&d_x, Nrows * sizeof(double)));
//	gpuErrchk(cudaMemcpy(d_x, h_x, Nrows * sizeof(double), cudaMemcpyHostToDevice));
//
//	/******************************************/
//	/* STEP 1: CREATE DESCRIPTORS FOR L AND U */
//	/******************************************/
//	cusparseMatDescr_t      descr_L = 0;
//	cusparseSafeCall(cusparseCreateMatDescr(&descr_L));
//	cusparseSafeCall(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE));
//	cusparseSafeCall(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
//	cusparseSafeCall(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
//	cusparseSafeCall(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));
//
//	/********************************************************************************************************/
//	/* STEP 2: QUERY HOW MUCH MEMORY USED IN CHOLESKY FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
//	/********************************************************************************************************/
//	csric02Info_t info_A = 0;  cusparseSafeCall(cusparseCreateCsric02Info(&info_A));
//	csrsv2Info_t  info_L = 0;  cusparseSafeCall(cusparseCreateCsrsv2Info(&info_L));
//	csrsv2Info_t  info_Lt = 0;  cusparseSafeCall(cusparseCreateCsrsv2Info(&info_Lt));
//
//	int pBufferSize_M, pBufferSize_L, pBufferSize_Lt;
//	cusparseSafeCall(cusparseDcsric02_bufferSize(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, &pBufferSize_M));
//	cusparseSafeCall(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, &pBufferSize_L));
//	cusparseSafeCall(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, &pBufferSize_Lt));
//
//	int pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));
//	void *pBuffer = 0;  gpuErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));
//
//	/******************************************************************************************************/
//	/* STEP 3: ANALYZE THE THREE PROBLEMS: CHOLESKY FACTORIZATION AND THE TWO FOLLOWING SYSTEM INVERSIONS */
//	/******************************************************************************************************/
//	int structural_zero;
//
//	cusparseSafeCall(cusparseDcsric02_analysis(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
//
//	cusparseStatus_t status = cusparseXcsric02_zeroPivot(handle, info_A, &structural_zero);
//	if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("A(%d,%d) is missing\n", structural_zero, structural_zero); }
//
//	cusparseSafeCall(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
//	cusparseSafeCall(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
//
//	/*************************************/
//	/* STEP 4: FACTORIZATION: A = L * L' */
//	/*************************************/
//	int numerical_zero;
//
//	cusparseSafeCall(cusparseDcsric02(handle, N, nnz, descrA, d_A, d_A_RowIndices, d_A_ColIndices, info_A, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
//	status = cusparseXcsric02_zeroPivot(handle, info_A, &numerical_zero);
//	/*if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero); }
//	*/
//
//	gpuErrchk(cudaMemcpy(h_A, d_A, nnz * sizeof(double), cudaMemcpyDeviceToHost));
//	/*printf("\nNon-zero elements in Cholesky matrix\n\n");
//	for (int k = 0; k<10; k++) printf("%f\n", h_A[k]);*/
//
//	cusparseSafeCall(cusparseDcsr2dense(handle, Nrows, Ncols, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_A_dense, Nrows));
//
//	/*printf("\nCholesky matrix\n\n");
//	for (int i = 0; i < 10; i++) {
//	std::cout << "[ ";
//	for (int j = 0; j < 10; j++)
//	std::cout << h_A_dense[i * Ncols + j] << " ";
//	std::cout << "]\n";
//	}*/
//
//	/*********************/
//	/* STEP 5: L * z = x */
//	/*********************/
//	// --- Allocating the intermediate result vector
//	double *d_z;        gpuErrchk(cudaMalloc(&d_z, N * sizeof(double)));
//
//	const double alpha = 1.;
//	cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_L, d_x, d_z, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer));
//
//	/**********************/
//	/* STEP 5: L' * y = z */
//	/**********************/
//	// --- Allocating the host and device side result vector
//	double *h_y = (double *)malloc(Ncols * sizeof(double));
//	double *d_y;        gpuErrchk(cudaMalloc(&d_y, Ncols * sizeof(double)));
//
//	cusparseSafeCall(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, N, nnz, &alpha, descr_L, d_A, d_A_RowIndices, d_A_ColIndices, info_Lt, d_z, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer));
//
//	cudaMemcpy(h_x, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);
//	/*printf("\n\nFinal result\n");
//	for (int k = 0; k<10; k++) printf("x[%i] = %f\n", k, h_x[k]);
//	*/
//	for (int i = 0; i < numNodes; i++) {
//		x[i] = x[i] + h_x[i * 2];
//		y[i] = y[i] + h_x[i * 2 + 1];
//	}
//	cudaFree(d_A_dense);
//	cudaFree(d_nnzPerVector);
//	cudaFree(d_A);
//	cudaFree(d_A_RowIndices);
//	cudaFree(d_A_ColIndices);
//	cudaFree(d_x);
//	cudaFree(pBuffer);
//	cudaFree(d_z);
//	cudaFree(d_y);
//
//	free(h_nnzPerVector);
//
//	free(h_A_dense);
//
//	free(h_A);
//	free(h_A_RowIndices);
//	free(h_A_ColIndices);
//	free(h_x);
//	free(h_y);
//
//	return 0;
//}