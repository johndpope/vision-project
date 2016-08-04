


#include "cudaFEM_read.cuh"
#include <fstream> 
#include <iostream>
#include <cstdio>
#include <ctime>
//#include "kinectotherfunctions.h"
//#include "kinect_vision.h"
#include "vision_main.h"

#include "FEM_draw.cuh"

using namespace std;

int main(void){


	int a;
	Geometry testing_geo;

	testing_geo.set_dim(3);
	testing_geo.read_nodes();
	testing_geo.read_elem();
	testing_geo.read_force();
	testing_geo.set_YoungPoisson(30000.0, 0.49);
	testing_geo.set_thickness(0.005);
	testing_geo.initilizeMatrices();
	testing_geo.initialize_CUDA();
	testing_geo.set_dynamic(false);
	testing_geo.set_cuda_use(true);
    //kinect_main(0, NULL,&testing_geo);
	
	 //maain(0, NULL);
	

	
	//cout << "x_value is" << testing_geo.return_y(1) << endl;
	//cout << "numNodes:" << testing_geo.return_numNodes() << endl;
	//
	//
	
	if (0){
		testing_geo.initilizeMatrices();
		std::ofstream writenodes("FEM_position_result.txt");
		double duration_K;
		double duration_solver;
		for (int i = 0; i < 20; i++){
			cout << "Iteration:" << i << endl;
			std::clock_t start_K;
			std::clock_t start_solver;

			//Assemble K
			start_K = std::clock();

			testing_geo.make_K_matrix();
			testing_geo.make_surface_f();
			

			duration_K = (std::clock() - start_K) / (double)CLOCKS_PER_SEC;

			//Solve Ax = b
			start_solver = std::clock();

			testing_geo.tt();
			duration_solver = (std::clock() - start_solver) / (double)CLOCKS_PER_SEC;
			for (int i = 0; i < testing_geo.return_numNodes(); i++) {
				writenodes << testing_geo.return_x(i) << "   " << testing_geo.return_y(i) << endl;
			}

			//cout << "K time: " << duration_K << endl;
			//cout << "Solver time: " << duration_solver << endl;
		}

		writenodes.close();
	}
 	//cuda_solver();
	//magma_solver();
	draw_things(&testing_geo);
	cin >> a; 

	return 0;
}