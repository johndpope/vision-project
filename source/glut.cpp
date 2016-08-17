#include "glut.h"
#include "vision_main.h"

#include <iostream>
using namespace std;
void draw() {
   drawKinectData();
 // glutSwapBuffers();
  /* for (int i = 0; i < 100; i += 10){
	   cout << i << endl;
	   cout << "X[1] : " << depth2xyz[i].X << endl;
	   cout << "Y[1] : " << depth2xyz[i].Y << endl;
	   cout << "Z[1] : " << depth2xyz[i].Z << endl;
   }*/
}

void execute() {
	
    glutMainLoop();
}

bool init(int argc, char* argv[]) {
    glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(width,height);
    glutCreateWindow("Kinect SDK Tutorial");
    glutDisplayFunc(draw);
    glutIdleFunc(draw);
	glewInit();
    return true;
}
