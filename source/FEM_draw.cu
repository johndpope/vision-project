
#include <stdio.h>
#define _USE_MATH_DEFINES

#include <math.h>
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cstdio>
#include <linmath.h>
#include "FEM_draw.cuh"
#include "cudaFEM_read.cuh"

//Include for CMIZ
#include "zinc/context.hpp"
#include "zinc/element.hpp"
#include "zinc/field.hpp"
#include "zinc/fieldcache.hpp"
#include "zinc/fieldmodule.hpp"
#include <zinc/fieldvectoroperators.hpp>
#include "zinc/region.hpp"
#include "zinc/sceneviewer.hpp"
#include "zinc/scene.hpp"
#include <iostream>
using namespace OpenCMISS::Zinc;

#define RADIUS          150.0f
#define STEP_LONGITUDE   22.5f                   /* 22.5 makes 8 bands like original Boing */
#define STEP_LATITUDE    22.5f

#define DIST_BALL       (RADIUS * 2.f + RADIUS * 0.1f)

#define VIEW_SCENE_DIST (DIST_BALL * 3.f + 200.f)/* distance from viewer to middle of boing area */
#define GRID_SIZE       (RADIUS * 4.5f)          /* length (width) of grid */
#define BOUNCE_HEIGHT   (RADIUS * 2.1f)
#define BOUNCE_WIDTH    (RADIUS * 2.1f)

#define SHADOW_OFFSET_X -20.f
#define SHADOW_OFFSET_Y  10.f
#define SHADOW_OFFSET_Z   0.f

#define WALL_L_OFFSET   0.f
#define WALL_R_OFFSET   5.f
float3 rotate = make_float3(0.0, 0.0, 0.0);
/* Animation speed (50.0 mimics the original GLUT demo speed) */
#define ANIMATION_SPEED 50.f

/* Maximum allowed delta time per physics iteration */
#define MAX_DELTA_T 0.02f

/* Draw ball, or its shadow */
typedef enum { DRAW_BALL, DRAW_BALL_SHADOW } DRAW_BALL_ENUM;

/* Vertex type */
typedef struct { float x; float y; float z; } vertex_t;

/* Global vars */
int width, height;
GLfloat deg_rot_y = 0.f;
GLfloat deg_rot_y_inc = 2.f;
GLboolean override_pos = GL_FALSE;
GLfloat cursor_x = 0.f;
GLfloat cursor_y = 0.f;
GLfloat ball_x = -RADIUS;
GLfloat ball_y = -RADIUS;
GLfloat ball_x_inc = 1.f;
GLfloat ball_y_inc = 2.f;
DRAW_BALL_ENUM drawBallHow;
double dx, dy;
double  t;
double  t_old = 0.f;
double  dt;
double x_win_min, y_win_min;
int closest_Node = 0;
int closest_Node_new;
int node_selected;
bool changeNode = false;
float mouse_old_x = 0;
float mouse_old_y = 0;
float distance_change = 1.0f;
float3 translation = make_float3(0.0, 0.0, 0.0);
/* Random number generator */
#ifndef RAND_MAX
#define RAND_MAX 4095
#endif


/*****************************************************************************
* Truncate a degree.
*****************************************************************************/
GLfloat TruncateDeg(GLfloat deg)
{
	if (deg >= 360.f)
		return (deg - 360.f);
	else
		return deg;
}

/*****************************************************************************
* Convert a degree (360-based) into a radian.
* 360' = 2 * PI
*****************************************************************************/
double deg2rad(double deg)
{
	return deg / 360 * (2 * M_PI);
}

/*****************************************************************************
* 360' sin().
*****************************************************************************/
double sin_deg(double deg)
{
	return sin(deg2rad(deg));
}

/*****************************************************************************
* 360' cos().
*****************************************************************************/
double cos_deg(double deg)
{
	return cos(deg2rad(deg));
}

/*****************************************************************************
* Compute a cross product (for a normal vector).
*
* c = a x b
*****************************************************************************/
void CrossProduct(vertex_t a, vertex_t b, vertex_t c, vertex_t *n)
{
	GLfloat u1, u2, u3;
	GLfloat v1, v2, v3;

	u1 = b.x - a.x;
	u2 = b.y - a.y;
	u3 = b.y - a.z;

	v1 = c.x - a.x;
	v2 = c.y - a.y;
	v3 = c.z - a.z;

	n->x = u2 * v3 - v2 * v3;
	n->y = u3 * v1 - v3 * u1;
	n->z = u1 * v2 - v1 * u2;
}


#define BOING_DEBUG 0


/*****************************************************************************
* init()
*****************************************************************************/
void init(void)
{
	/*
	* Clear background.
	*/
	glClearColor(0.55f, 0.55f, 0.55f, 0.f);

	glShadeModel(GL_FLAT);
}


/*****************************************************************************
* display()
*****************************************************************************/
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();

	drawBallHow = DRAW_BALL_SHADOW;
	DrawBoingBall();

	DrawGrid();

	drawBallHow = DRAW_BALL;
	DrawBoingBall();

	glPopMatrix();
	glFlush();
}


/*****************************************************************************
* reshape()
*****************************************************************************/
void reshape(GLFWwindow* window, int w, int h)
{
	mat4x4 projection, view;

	glViewport(0, 0, (GLsizei)w, (GLsizei)h);

	glMatrixMode(GL_PROJECTION);
	mat4x4_perspective(projection,
		2.f * (float)atan2(RADIUS, 200.0f),
		(float)w / (float)h,
		0.f, VIEW_SCENE_DIST);
	glLoadMatrixf((const GLfloat*)projection);

	glMatrixMode(GL_MODELVIEW);
	{
		vec3 eye = { 0.f, 0.f, VIEW_SCENE_DIST };
		vec3 center = { 0.f, 0.f, 0.f };
		vec3 up = { 0.f, -1.f, 0.f };
		mat4x4_look_at(view, eye, center, up);
	}
	glLoadMatrixf((const GLfloat*)view);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	std::cout << key << std::endl;
	if (key == 61){
		distance_change += 0.1;
	}
	else if (key == 45){
		distance_change -= 0.1;
	}
	else if (key == 119){
		translation.y -= 100;
	}
	else if (key == 115){
		translation.y += 100;
	}
	else if (key == 97){
		translation.x += 100;
	}
	else if (key == 100){
		translation.x -= 100;
	}

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}

static void set_ball_pos(GLfloat x, GLfloat y)
{
	ball_x = (width / 2) - x;
	ball_y = y - (height / 2);
	dx = x_win_min - x;
	dy = y_win_min - y;

	float dx1 = -(float)(x - mouse_old_x)/100;
	float dy1 = (float)(y - mouse_old_y) / 100;



	rotate.x += -dy1 * 0.2f;
	rotate.y += dx1 * 0.2f;

	node_selected = closest_Node;

}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button != GLFW_MOUSE_BUTTON_LEFT)
		//override_pos = GL_TRUE;
		return;

	if (action == GLFW_PRESS)
	{
		override_pos = GL_TRUE;
		mouse_old_x = cursor_x;
		mouse_old_y = cursor_y;
		changeNode = true;
		set_ball_pos(cursor_x, cursor_y);
	}
	else
	{

		override_pos = GL_FALSE;
		changeNode = false;
		dx = dy = 0;


	}
}

void cursor_position_callback(GLFWwindow* window, double x, double y)
{
	cursor_x = (float)x;
	cursor_y = (float)y;
	mouse_old_x = cursor_x;
	mouse_old_x = cursor_y;
	//std::cout <<"coursor x " << cursor_x << std::endl;
	//std::cout << "coursor y " <<  cursor_y << std::endl;
	if (override_pos)
		set_ball_pos(cursor_x, cursor_y);
}

/*****************************************************************************
* Draw the Boing ball.
*
* The Boing ball is sphere in which each facet is a rectangle.
* Facet colors alternate between red and white.
* The ball is built by stacking latitudinal circles.  Each circle is composed
* of a widely-separated set of points, so that each facet is noticably large.
*****************************************************************************/
void DrawBoingBall(void)
{
	GLfloat lon_deg;     /* degree of longitude */
	double dt_total, dt2;

	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);

	/*
	* Another relative Z translation to separate objects.
	*/
	glTranslatef(0.0, 0.0, DIST_BALL);

	/* Update ball position and rotation (iterate if necessary) */
	dt_total = dt;
	while (dt_total > 0.0)
	{
		dt2 = dt_total > MAX_DELTA_T ? MAX_DELTA_T : dt_total;
		dt_total -= dt2;
		BounceBall(dt2);
		deg_rot_y = TruncateDeg(deg_rot_y + deg_rot_y_inc*((float)dt2*ANIMATION_SPEED));
	}

	/* Set ball position */
	glTranslatef(ball_x, ball_y, 0.0);

	/*
	* Offset the shadow.
	*/
	if (drawBallHow == DRAW_BALL_SHADOW)
	{
		glTranslatef(SHADOW_OFFSET_X,
			SHADOW_OFFSET_Y,
			SHADOW_OFFSET_Z);
	}

	/*
	* Tilt the ball.
	*/
	glRotatef(-20.0, 0.0, 0.0, 1.0);

	/*
	* Continually rotate ball around Y axis.
	*/
	glRotatef(deg_rot_y, 0.0, 1.0, 0.0);

	/*
	* Set OpenGL state for Boing ball.
	*/
	glCullFace(GL_FRONT);
	glEnable(GL_CULL_FACE);
	glEnable(GL_NORMALIZE);

	/*
	* Build a faceted latitude slice of the Boing ball,
	* stepping same-sized vertical bands of the sphere.
	*/
	for (lon_deg = 0;
		lon_deg < 180;
		lon_deg += STEP_LONGITUDE)
	{
		/*
		* Draw a latitude circle at this longitude.
		*/
		DrawBoingBallBand(lon_deg,
			lon_deg + STEP_LONGITUDE);
	}

	glPopMatrix();

	return;
}


/*****************************************************************************
* Bounce the ball.
*****************************************************************************/
void BounceBall(double delta_t)
{
	GLfloat sign;
	GLfloat deg;

	if (override_pos)
		return;

	/* Bounce on walls */
	if (ball_x > (BOUNCE_WIDTH / 2 + WALL_R_OFFSET))
	{
		ball_x_inc = -0.5f - 0.75f * (GLfloat)rand() / (GLfloat)RAND_MAX;
		deg_rot_y_inc = -deg_rot_y_inc;
	}
	if (ball_x < -(BOUNCE_HEIGHT / 2 + WALL_L_OFFSET))
	{
		ball_x_inc = 0.5f + 0.75f * (GLfloat)rand() / (GLfloat)RAND_MAX;
		deg_rot_y_inc = -deg_rot_y_inc;
	}

	/* Bounce on floor / roof */
	if (ball_y > BOUNCE_HEIGHT / 2)
	{
		ball_y_inc = -0.75f - 1.f * (GLfloat)rand() / (GLfloat)RAND_MAX;
	}
	if (ball_y < -BOUNCE_HEIGHT / 2 * 0.85)
	{
		ball_y_inc = 0.75f + 1.f * (GLfloat)rand() / (GLfloat)RAND_MAX;
	}

	/* Update ball position */
	ball_x += ball_x_inc * ((float)delta_t*ANIMATION_SPEED);
	ball_y += ball_y_inc * ((float)delta_t*ANIMATION_SPEED);

	/*
	* Simulate the effects of gravity on Y movement.
	*/
	if (ball_y_inc < 0) sign = -1.0; else sign = 1.0;

	deg = (ball_y + BOUNCE_HEIGHT / 2) * 90 / BOUNCE_HEIGHT;
	if (deg > 80) deg = 80;
	if (deg < 10) deg = 10;

	ball_y_inc = sign * 4.f * (float)sin_deg(deg);
}


/*****************************************************************************
* Draw a faceted latitude band of the Boing ball.
*
* Parms:   long_lo, long_hi
*          Low and high longitudes of slice, resp.
*****************************************************************************/
void DrawBoingBallBand(GLfloat long_lo,
	GLfloat long_hi)
{
	vertex_t vert_ne;            /* "ne" means south-east, so on */
	vertex_t vert_nw;
	vertex_t vert_sw;
	vertex_t vert_se;
	vertex_t vert_norm;
	GLfloat  lat_deg;
	static int colorToggle = 0;

	/*
	* Iterate thru the points of a latitude circle.
	* A latitude circle is a 2D set of X,Z points.
	*/
	for (lat_deg = 0;
		lat_deg <= (360 - STEP_LATITUDE);
		lat_deg += STEP_LATITUDE)
	{
		/*
		* Color this polygon with red or white.
		*/
		if (colorToggle)
			glColor3f(0.8f, 0.1f, 0.1f);
		else
			glColor3f(0.95f, 0.95f, 0.95f);
#if 0
		if (lat_deg >= 180)
			if (colorToggle)
				glColor3f(0.1f, 0.8f, 0.1f);
			else
				glColor3f(0.5f, 0.5f, 0.95f);
#endif
		colorToggle = !colorToggle;

		/*
		* Change color if drawing shadow.
		*/
		if (drawBallHow == DRAW_BALL_SHADOW)
			glColor3f(0.35f, 0.35f, 0.35f);

		/*
		* Assign each Y.
		*/
		vert_ne.y = vert_nw.y = (float)cos_deg(long_hi) * RADIUS;
		vert_sw.y = vert_se.y = (float)cos_deg(long_lo) * RADIUS;

		/*
		* Assign each X,Z with sin,cos values scaled by latitude radius indexed by longitude.
		* Eg, long=0 and long=180 are at the poles, so zero scale is sin(longitude),
		* while long=90 (sin(90)=1) is at equator.
		*/
		vert_ne.x = (float)cos_deg(lat_deg) * (RADIUS * (float)sin_deg(long_lo + STEP_LONGITUDE));
		vert_se.x = (float)cos_deg(lat_deg) * (RADIUS * (float)sin_deg(long_lo));
		vert_nw.x = (float)cos_deg(lat_deg + STEP_LATITUDE) * (RADIUS * (float)sin_deg(long_lo + STEP_LONGITUDE));
		vert_sw.x = (float)cos_deg(lat_deg + STEP_LATITUDE) * (RADIUS * (float)sin_deg(long_lo));

		vert_ne.z = (float)sin_deg(lat_deg) * (RADIUS * (float)sin_deg(long_lo + STEP_LONGITUDE));
		vert_se.z = (float)sin_deg(lat_deg) * (RADIUS * (float)sin_deg(long_lo));
		vert_nw.z = (float)sin_deg(lat_deg + STEP_LATITUDE) * (RADIUS * (float)sin_deg(long_lo + STEP_LONGITUDE));
		vert_sw.z = (float)sin_deg(lat_deg + STEP_LATITUDE) * (RADIUS * (float)sin_deg(long_lo));

		/*
		* Draw the facet.
		*/
		glBegin(GL_POLYGON);

		CrossProduct(vert_ne, vert_nw, vert_sw, &vert_norm);
		glNormal3f(vert_norm.x, vert_norm.y, vert_norm.z);

		glVertex3f(vert_ne.x, vert_ne.y, vert_ne.z);
		glVertex3f(vert_nw.x, vert_nw.y, vert_nw.z);
		glVertex3f(vert_sw.x, vert_sw.y, vert_sw.z);
		glVertex3f(vert_se.x, vert_se.y, vert_se.z);

		glEnd();

#if BOING_DEBUG
		printf("----------------------------------------------------------- \n");
		printf("lat = %f  long_lo = %f  long_hi = %f \n", lat_deg, long_lo, long_hi);
		printf("vert_ne  x = %.8f  y = %.8f  z = %.8f \n", vert_ne.x, vert_ne.y, vert_ne.z);
		printf("vert_nw  x = %.8f  y = %.8f  z = %.8f \n", vert_nw.x, vert_nw.y, vert_nw.z);
		printf("vert_se  x = %.8f  y = %.8f  z = %.8f \n", vert_se.x, vert_se.y, vert_se.z);
		printf("vert_sw  x = %.8f  y = %.8f  z = %.8f \n", vert_sw.x, vert_sw.y, vert_sw.z);
#endif

	}

	/*
	* Toggle color so that next band will opposite red/white colors than this one.
	*/
	colorToggle = !colorToggle;

	/*
	* This circular band is done.
	*/
	return;
}


/*****************************************************************************
* Draw the purple grid of lines, behind the Boing ball.
* When the Workbench is dropped to the bottom, Boing shows 12 rows.
*****************************************************************************/
void DrawGrid(void)
{
	int              row, col;
	const int        rowTotal = 12;                   /* must be divisible by 2 */
	const int        colTotal = rowTotal;             /* must be same as rowTotal */
	const GLfloat    widthLine = 2.0;                  /* should be divisible by 2 */
	const GLfloat    sizeCell = GRID_SIZE / rowTotal;
	const GLfloat    z_offset = -40.0;
	GLfloat          xl, xr;
	GLfloat          yt, yb;

	glPushMatrix();
	glDisable(GL_CULL_FACE);

	/*
	* Another relative Z translation to separate objects.
	*/
	glTranslatef(0.0, 0.0, DIST_BALL);

	/*
	* Draw vertical lines (as skinny 3D rectangles).
	*/
	for (col = 0; col <= colTotal; col++)
	{
		/*
		* Compute co-ords of line.
		*/
		xl = -GRID_SIZE / 2 + col * sizeCell;
		xr = xl + widthLine;

		yt = GRID_SIZE / 2;
		yb = -GRID_SIZE / 2 - widthLine;

		glBegin(GL_POLYGON);

		glColor3f(0.2f, 0.1f, 0.6f);               /* purple */

		glVertex3f(xr, yt, z_offset);       /* NE */
		glVertex3f(xl, yt, z_offset);       /* NW */
		glVertex3f(xl, yb, z_offset);       /* SW */
		glVertex3f(xr, yb, z_offset);       /* SE */

		glEnd();
	}

	/*
	* Draw horizontal lines (as skinny 3D rectangles).
	*/
	for (row = 0; row <= rowTotal; row++)
	{
		/*
		* Compute co-ords of line.
		*/
		yt = GRID_SIZE / 2 - row * sizeCell;
		yb = yt - widthLine;

		xl = -GRID_SIZE / 2;
		xr = GRID_SIZE / 2 + widthLine;

		glBegin(GL_POLYGON);

		glColor3f(0.6f, 0.1f, 0.6f);               /* purple */

		glVertex3f(xr, yt, z_offset);       /* NE */
		glVertex3f(xl, yt, z_offset);       /* NW */
		glVertex3f(xl, yb, z_offset);       /* SW */
		glVertex3f(xr, yb, z_offset);       /* SE */

		glEnd();


		glBegin(GL_LINE_LOOP);

		glColor3f(0.6f, 0.1f, 0.6f);               /* purple */

		glVertex3f(35.0 - 200.0 + t, 359.0 - 200.0, z_offset);       /* NE */
		glVertex3f(353.0 - 200.0, 364.0 - 200.0, z_offset);       /* NW */
		glVertex3f(322.0 - 200.0, 34.0 - 200.0, z_offset);       /* SW */


		glEnd();
	}

	glPopMatrix();

	return;
}

void drawMesh(Geometry *p){
	int numElem = p->return_numElems();
	double x, y, z;
	double x_win, y_win;
	int node_considered = 0;
	double distance_old = INFINITY;
	double distance;
	int shortest_node = 0;
	for (int i = 0; i < numElem; i++){

		//glBegin(GL_TRIANGLES);

		//for (int j = 0; j < 4; j++){
		//	node_considered = p->node_number_inElem(i, j);
		//	x = p->return_x(node_considered);
		//	y = p->return_y(node_considered);
		//	z = p->return_z(node_considered);
		//	glColor3f(x, y, (float)i / numElem);
		//	glVertex3f(x * 400 - 200, y * 400 - 200, z*400);       /* NE */

		//	x_win = -(x * 400 - 400);
		//	y_win = y * 400;
		//	distance = ((x_win - cursor_x)*(x_win - cursor_x) + (y_win - cursor_y)*(y_win - cursor_y));
		//	if (distance_old > distance){
		//		distance_old = distance;
		//		shortest_node = node_considered;

		//	}
		int node_considered4 = 0;

		//}
		//glEnd();
		int node_considered1 = p->node_number_inElem(i, 0);
		int node_considered2 = p->node_number_inElem(i, 1);
		int node_considered3 = p->node_number_inElem(i, 2);
		if (p->return_dim() == 3){
			node_considered4 = p->node_number_inElem(i, 3);
		}
		
		
		
		
		x = p->return_x(node_considered1);
		y = p->return_y(node_considered1);
		if (p->return_dim()== 3){
			z = p->return_z(node_considered1);
		}
		
		if (p->return_dim() == 3){
			glLineWidth(2);
			glColor4f(1, y, z, 0.5);
			glBegin(GL_LINE_LOOP);
			glVertex3f(p->return_x(node_considered1) * 400 - 200, p->return_y(node_considered1) * 400 - 200, p->return_z(node_considered1) * 400);       /* NE */
			glVertex3f(p->return_x(node_considered2) * 400 - 200, p->return_y(node_considered2) * 400 - 200, p->return_z(node_considered2) * 400);       /* NE */
			glVertex3f(p->return_x(node_considered3) * 400 - 200, p->return_y(node_considered3) * 400 - 200, p->return_z(node_considered3) * 400);       /* NE */

			glEnd();

			glBegin(GL_LINE_LOOP);
			glVertex3f(p->return_x(node_considered2) * 400 - 200, p->return_y(node_considered2) * 400 - 200, p->return_z(node_considered2) * 400);
			glVertex3f(p->return_x(node_considered3) * 400 - 200, p->return_y(node_considered3) * 400 - 200, p->return_z(node_considered3) * 400);
			glVertex3f(p->return_x(node_considered4) * 400 - 200, p->return_y(node_considered4) * 400 - 200, p->return_z(node_considered4) * 400);
			glEnd();


			glBegin(GL_LINE_LOOP);
			glVertex3f(p->return_x(node_considered2) * 400 - 200, p->return_y(node_considered2) * 400 - 200, p->return_z(node_considered2) * 400);
			glVertex3f(p->return_x(node_considered4) * 400 - 200, p->return_y(node_considered4) * 400 - 200, p->return_z(node_considered4) * 400);
			glVertex3f(p->return_x(node_considered1) * 400 - 200, p->return_y(node_considered1) * 400 - 200, p->return_z(node_considered1) * 400);
			glEnd();

			glBegin(GL_LINE_LOOP);
			glVertex3f(p->return_x(node_considered1) * 400 - 200, p->return_y(node_considered1) * 400 - 200, p->return_z(node_considered1) * 400);
			glVertex3f(p->return_x(node_considered4) * 400 - 200, p->return_y(node_considered4) * 400 - 200, p->return_z(node_considered4) * 400);
			glVertex3f(p->return_x(node_considered3) * 400 - 200, p->return_y(node_considered3) * 400 - 200, p->return_z(node_considered3) * 400);
			glEnd();
		}
		else if (p->return_dim() == 2){
			//glColor4f(p->global_stress_mises[i]*10.0, 0.2, 0.5, 0.5);
			glColor3f(p->global_stress_mises[i] * 10.0, p->global_stress_mises[i] * 2.0, p->global_stress_mises[i] * 5.0);
			//glColor3f(1.0, 0.0, 1.0);
			glPolygonMode(GL_FRONT_AND_BACK, GL_TRIANGLES);
			glBegin(GL_TRIANGLES);
			glVertex3f(p->return_x(node_considered1) * 200 - 200, p->return_y(node_considered1) * 200 - 200, p->return_z(node_considered1) * 200);       /* NE */
			glVertex3f(p->return_x(node_considered2) * 200 - 200, p->return_y(node_considered2) * 200 - 200, p->return_z(node_considered2) * 200);       /* NE */
			glVertex3f(p->return_x(node_considered3) * 200 - 200, p->return_y(node_considered3) * 200 - 200, p->return_z(node_considered3) * 200);       /* NE */
			glColor3f(p->global_stress_mises[i] * 10.0, p->global_stress_mises[i] * 2.0, p->global_stress_mises[i] * 5.0);
			glEnd();
			glBegin(GL_LINE_LOOP);
			glVertex3f(p->return_x(node_considered1) * 200 - 200, p->return_y(node_considered1) * 200 - 200, p->return_z(node_considered1) * 200);
			glVertex3f(p->return_x(node_considered2) * 200 - 200, p->return_y(node_considered2) * 200 - 200, p->return_z(node_considered2) * 200);
			glVertex3f(p->return_x(node_considered3) * 200 - 200, p->return_y(node_considered3) * 200 - 200, p->return_z(node_considered3) * 200);
			glEnd();
			
		}
		x_win = -(x * 400 - 400);
			y_win = y * 400;
			distance = ((x_win - cursor_x)*(x_win - cursor_x) + (y_win - cursor_y)*(y_win - cursor_y));
			if (distance_old > distance){
				distance_old = distance;
				shortest_node = node_considered;

		}
	
		//glColor4f(x, y, (float)i / numElem,0.5 );
		//glColor3f(1.0, 0.0, 1.0);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_TRIANGLES);
		//glBegin(GL_TRIANGLES);
		//glVertex3f(p->return_x(node_considered1) * 400 - 200, p->return_y(node_considered1) * 400 - 200, p->return_z(node_considered1) * 400-200);       /* NE */
		//glVertex3f(p->return_x(node_considered2) * 400 - 200, p->return_y(node_considered2) * 400 - 200, p->return_z(node_considered2) * 400 - 200);       /* NE */
		//glVertex3f(p->return_x(node_considered3) * 400 - 200, p->return_y(node_considered3) * 400 - 200, p->return_z(node_considered3) * 400 - 200);       /* NE */

		//glEnd();

		//glBegin(GL_TRIANGLES);
		//glVertex3f(p->return_x(node_considered2) * 400 - 200, p->return_y(node_considered2) * 400 - 200, p->return_z(node_considered2) * 400 - 200);
		//glVertex3f(p->return_x(node_considered3) * 400 - 200, p->return_y(node_considered3) * 400 - 200, p->return_z(node_considered3) * 400 - 200);
		//glVertex3f(p->return_x(node_considered4) * 400 - 200, p->return_y(node_considered4) * 400 - 200, p->return_z(node_considered4) * 400 - 200);
		//glEnd();

		//
		//glBegin(GL_TRIANGLES);
		//glVertex3f(p->return_x(node_considered2) * 400 - 200, p->return_y(node_considered2) * 400 - 200, p->return_z(node_considered2) * 400 - 200);
		//glVertex3f(p->return_x(node_considered4) * 400 - 200, p->return_y(node_considered4) * 400 - 200, p->return_z(node_considered4) * 400 - 200);
		//glVertex3f(p->return_x(node_considered1) * 400 - 200, p->return_y(node_considered1) * 400 - 200, p->return_z(node_considered1) * 400 - 200);
		//glEnd();

		//glBegin(GL_TRIANGLES);
		//glVertex3f(p->return_x(node_considered1) * 400 - 200, p->return_y(node_considered1) * 400 - 200, p->return_z(node_considered1) * 400 - 200);
		//glVertex3f(p->return_x(node_considered4) * 400 - 200, p->return_y(node_considered4) * 400 - 200, p->return_z(node_considered4) * 400 - 200);
		//glVertex3f(p->return_x(node_considered3) * 400 - 200, p->return_y(node_considered3) * 400 - 200, p->return_z(node_considered3) * 400 - 200);
		//glEnd();
		



		//glBegin(GL_LINE_LOOP);
		//glColor3f(1.0, 1.0, 0.0);
		//for (int j = 0; j < 4; j++){
		//	node_considered = p->node_number_inElem(i, j);
		//	x = p->return_x(node_considered);
		//	y = p->return_y(node_considered);
		//	z = p->return_z(node_considered);
		//	glVertex3f(x * 400 - 200, y * 400 - 200, z*400);       /* NE */

		//	x_win = -(x * 400 - 400);
		//	y_win = y * 400;
		//	distance = ((x_win - cursor_x)*(x_win - cursor_x) + (y_win - cursor_y)*(y_win - cursor_y));
		//	if (distance_old > distance){
		//		distance_old = distance;
		//		shortest_node = node_considered;

		//	}


		//}
		//glEnd();
	}
	glColor3f(0.6f, 1.0f, 0.6f);
	glPointSize(10.0);

	
	glBegin(GL_POINTS);
	glVertex3f(p->return_x(0) * 400 - 200, p->return_y(0) * 400 - 200, p->return_z(0)*400 );

	glColor3f(0.0f, 1.0f, 1.6f);
	glVertex3f(p->return_x(20) * 200 - 200, p->return_y(20) * 200 - 200, p->return_z(20) * 400);

	glEnd();

	//std::cout << " X _ win: " << -(p->return_x(shortest_node) * 400-400) << " Y _ WIN : " << p->return_y(shortest_node) * 400 << std::endl;
	if (!changeNode){
		closest_Node = shortest_node;
	}
	x_win_min = -(p->return_x(closest_Node) * 400 - 400);
	y_win_min = p->return_y(closest_Node) * 400;

}

/*======================================================================*
* main()
*======================================================================*/

int draw_things(Geometry *p)
{
	GLFWwindow* window;

	/* Init GLFW */
	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_DEPTH_BITS, 16);

	window = glfwCreateWindow(400, 400, "FEM TEST", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetFramebufferSizeCallback(window, reshape);
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glfwGetFramebufferSize(window, &width, &height);
	reshape(window, width, height);

	glfwSetTime(0.0);

	init();
	t = 0;
	/* Main loop */
	//p->initilizeMatrices();

	//----------opencmiss



	glfwMakeContextCurrent(window);

	glfwPollEvents();

	///-cmiss
	double duration_K;
	bool cuda_init = false;
	int display_counter = 0;
	//initilizing all of the vectors
	if (p->get_dynamic())
		p->initialize_dynamic();
	p->set_beta1(0.9); // if beta_2 >= beta1 and beta > 1/2 then the time stepping scheme is unconditionally stable.
	p->set_beta2(0.9);
	p->set_dt(0.05);
	p->set_dynamic_alpha(0.2);
	p->set_dynamic_xi(0.23);
	p->initialize_zerovector(9);
	//next we set what nodes we want to make stable
	int points[9];
	for (int i = 0; i < 9; i++){
		points[i] = i;

	}

	p->set_zero_nodes(points);
	if (!p->get_dynamic()){
		for (;;){



			///* Timing */
			//t = glfwGetTime();
			//dt = t - t_old;
			//t_old = t;
			//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			//glPushMatrix();
			//glRotatef(rotate.x, 1.0, 0.0, 0.0);
			//glRotatef(rotate.y, 0.0, 100.0, 0.0);
			//glScalef(distance_change, distance_change, distance_change);
			//glTranslatef(translation.x, translation.y, translation.z);
			///* Draw one frame */
			////display();
			////DrawGrid();
			//drawMesh(p);
			//glPopMatrix();
			//glFlush();
			///* Swap buffers */
			//glfwSwapBuffers(window);
			//glfwPollEvents();



			////Solve the 2D FEM in each frame
			//if (display_counter < 100){
			//	p->setSudoNode(20);
			//	p->setSudoForcex(0);
			//	p->setSudoForcey(0);
			//}
			//else {
			//	p->setSudoNode(20);
			//	p->setSudoForcex(0);
			//	p->setSudoForcey(0);
			//}


			//display_counter++;
			//if (p->return_dim() == 3){
			//	p->Linear3DBarycentric_B_CUDA_host();
			//}


			//p->make_K_matrix();

			////p->make_surface_f();



			//if (!cuda_init){
			//	p->initialize_CUDA();

			//	cuda_init = true;
			//}
			//std::clock_t start_K;
			//start_K = std::clock();
			//p->tt();
			//duration_K = (std::clock() - start_K) / (double)CLOCKS_PER_SEC;


			////std::cout << " change status : " << changeNode << std::endl;

			//std::cout << "Solver time ms:  " << duration_K << std::endl;
			////std::cout << " closet node : " << closest_Node << std::endl;
			//t++;


			/* Timing */
			t = glfwGetTime();
			dt = t - t_old;
			t_old = t;
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glPushMatrix();
			glRotatef(rotate.x, 1.0, 0.0, 0.0);
			glRotatef(rotate.y, 0.0, 100.0, 0.0);
			glScalef(distance_change, distance_change, distance_change);
			glTranslatef(translation.x, translation.y, translation.z);
			/* Draw one frame */
			//display();
			//DrawGrid();
			drawMesh(p);
			


			////////-----------------------CMISS

		

			////////////-------------------------------
			glPopMatrix();
			glFlush();
			
			/* Swap buffers */
			glfwSwapBuffers(window);
			
			glfwPollEvents();
			
			std::clock_t start_K;
			start_K = std::clock();

			//Solve the 2D FEM in each frame
			p->setSudoNode(200);
			p->setSudoForcex( 6.0);
			p->setSudoForcey( 6.0);
			
			/*if (p->return_dim() == 3){
				p->Linear3DBarycentric_B_CUDA_host();
			}
*/

			p->make_K_matrix();

			//p->make_surface_f();



			if (!cuda_init){
				p->initialize_CUDA();
				cuda_init = true;
			}
		
			p->tt();
			duration_K = (std::clock() - start_K) / (double)CLOCKS_PER_SEC;


			//std::cout << " change status : " << changeNode << std::endl;

			std::cout << "Solver time ms:  " << duration_K << std::endl;
			//std::cout << " closet node : " << closest_Node << std::endl;
			t++;
			/* Check if we are still running */
			if (glfwWindowShouldClose(window))
				break;
		}
	}
	else{
		for (;;){
			if (display_counter < 1){
				p->setSudoNode(900);
				p->setSudoForcex(3000.0);
				p->setSudoForcey(2000.0);
			}
			else {
				p->setSudoNode(120);
				p->setSudoForcex(0);
				p->setSudoForcey(0);
			}
		
			/*if (display_counter == 500){
				p->setSudoNode(20);
				p->setSudoForcex(-100);
				p->setSudoForcey(-100);
			}*/
			display_counter++;
			t = glfwGetTime();
			dt = t - t_old;
			t_old = t;
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glPushMatrix();
			glRotatef(rotate.x, 1.0, 0.0, 0.0);
			glRotatef(rotate.y, 0.0, 100.0, 0.0);
			glScalef(distance_change, distance_change, distance_change);
			glTranslatef(translation.x, translation.y, translation.z);
			/* Draw one frame */
			//display();
			//DrawGrid();
			drawMesh(p);
			glPopMatrix();
			glFlush();
			/* Swap buffers */
			glfwSwapBuffers(window);
			glfwPollEvents();

			///dynamic calculations
		
			p->make_K_matrix();
			
			p->find_b();
			std::clock_t start_K;
			start_K = std::clock();
			p->update_vector();
			duration_K = (std::clock() - start_K) / (double)CLOCKS_PER_SEC;
			std::cout << "Solver time ms:  " << duration_K << std::endl;
			p->update_dynamic_vectors();
			p->update_dynamic_xyz();
			t++;
			
			/* Check if we are still running */
			if (glfwWindowShouldClose(window))
				break;
		}
	}

	glfwTerminate();
	exit(EXIT_SUCCESS);
}
