#ifndef CUDA_DRAW_FEM
#define CUDA_DRAW_FEM

#include <stdio.h>


#include <math.h>
#include <GLFW/glfw3.h>
#include "cudaFEM_read.cuh"
#include <linmath.h>
void init(void);
void display(void);
void reshape(GLFWwindow* window, int w, int h);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double x, double y);
void DrawBoingBall(void);
void BounceBall(double dt);
void DrawBoingBallBand(GLfloat long_lo, GLfloat long_hi);
void DrawGrid(void);
int draw_things(Geometry *p);
void drawMesh(Geometry *p);
#endif //CUDA_DRAW_FEM