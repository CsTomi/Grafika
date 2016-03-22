//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

using namespace std;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 130
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	in vec2 vertexPosition;		// variable input from Attrib Array selected by glBindAttribLocation
	in vec3 vertexColor;	    // variable input from Attrib Array selected by glBindAttribLocation
	out vec3 color;				// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 130
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 0.8f); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	//vektor * mátrix
	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	// vektro + vektor
	vec4 operator+(const vec4& other) {
		vec4 result;

		for (int i = 0; i < 3; i++)
		{
			result.v[i] = v[i] + other.v[i];
		}

		result.v[3] = v[3];

		return result;
	}

	vec4 operator-(const vec4& other) {
		vec4 result;

		for (int i = 0; i < 3; i++)
		{
			result.v[i] = v[i] - other.v[i];
		}

		result.v[3] = v[3];

		return result;
	}

	vec4 operator*(const float& skalar) {
		vec4 result;

		for (int i = 0; i < 4; i++)
		{
			result.v[i] *= skalar;
		}

		return result;
	}

};

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	// view matrix: translates the center to the origin
	mat4 V() { 
		return mat4(1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
				-wCx, -wCy, 0, 1);
	}

	// projection matrix: scales it to be a square of edge length 2
	mat4 P() { 
		float tx = 2 / wWx;
		float ty = 2 / wWy;
		return mat4(tx, 0, 0, 0,
					0, ty, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1);
	}

	// inverse view matrix
	mat4 Vinv() { 
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	// inverse projection matrix 
	mat4 Pinv()	{ 
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0;
		wCy = 0;
		wWx = 20; // width
		wWy = 20; // height of the World
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	float vertexes[6];
	float colors[9]; // rgb
public:
	Triangle() {
		Animate(0);
	}


	void setCoordinates(const float* coords){
		for (unsigned int i = 0; i < 6; i++)
		{
			vertexes[i] = coords[i];
		}
	}

	/*
	* The array must has 9 elements!
	*/
	void setColor(const float* color) {
		for (unsigned int i = 0; i < 9; i++)
		{
			colors[i] = color[i];
		}
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array

		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexes),  // number of the vbo in bytes
			vertexes,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			3, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

		// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array

		glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1;// 3 * sinf(t);
		sy = 1;//2 * cosf(t*3);
		wTx = 0; //4 * cosf(t / 2);
		wTy = 0; //4 * sinf(t / 2);
	}

	void Draw() {
		mat4 M(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

class LineStrip {
	const float tension = 0.8f;
	const static int nSplinePoint = 20;     // amount of the CR spline point between two CP's
	GLuint vao, spline_vao, vbo, spline_vbo; // vertex array object, vertex buffer object
	int   nVertices;                // number of vertices
	float vertexData[100];         // interleaved data of coordinates and colors
	float time[20];               // CP's time
	float spline[5 * nSplinePoint];  // points between the CP's

public:
	LineStrip() {
		nVertices = 0;
	}

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object for the CP's
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glGenVertexArrays(1, &spline_vao);
		glBindVertexArray(spline_vao);

		glGenBuffers(1, &spline_vbo); // Generate 1 vertex buffer object for the spline
		glBindBuffer(GL_ARRAY_BUFFER, spline_vbo);

		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		glEnableVertexAttribArray(2);  // attribute array 2
		glEnableVertexAttribArray(3);  // attribute array 3

		// Map attribute array 0 to the vertex data of the interleaved vbo
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
		
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	//Kész, nem kell bántani
	void AddPoint(float cX, float cY, float t) {
		if (nVertices >= 20) return;

		time[nVertices] = t;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = wVertex.v[0]; // x
		vertexData[5 * nVertices + 1] = wVertex.v[1]; // y
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;
		printf("\nPoint: (%f,%f)", vertexData[5 * (nVertices-1)], vertexData[5 * (nVertices-1)]);
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	//Bántani kell
	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();
			vec4 splinePoint;
			float ti = 0.01f; // time interval

			for (int i = 0; i < nVertices; i++) // itarete over the CP's
			{
				float t = time[i]; // i. CP's time

				// every spline point has a x and y coordinate
				for (int j = 0; j < nSplinePoint && t < time[i+1]; j++)
				{
					//TODO
					splinePoint = getSplinePoint(i, t);
					spline[5 * j] = splinePoint.v[0];     // x
					spline[5 * j + 1] = splinePoint.v[1]; // y
					spline[5 * j + 2] = 0.0f;
					spline[5 * j + 3] = 1.0f;
					spline[5 * j + 4] = 1.0f;
					t += ti;
				}

				glBindBuffer(GL_ARRAY_BUFFER, spline_vbo);
				glBufferData(GL_ARRAY_BUFFER, nSplinePoint * 5 * sizeof(float), spline, GL_DYNAMIC_DRAW);

				int location = glGetUniformLocation(shaderProgram, "MVP");
				if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
				else printf("uniform MVP cannot be set\n");

				glBindVertexArray(spline_vao);
				glDrawArrays(GL_LINES, 0, nSplinePoint);
			}
		}
	}

	vec4 getSplinePoint(const int index, const float t) {
		return Hermite(index, t);
	}

	vec4 Hermite(const int index, const float t_t0) {
		float nevezo = time[index + 1] - time[index];
		
		float Px = vertexData[5 * index];
		float Px2 = vertexData[5 * (index + 1)];
		
		float Py = vertexData[5 * index + 1];
		float Py2 = vertexData[5 * (index + 1) + 1];

		vec4 V1 = getVelocity(index);
		vec4 V2 = getVelocity(index + 1);

		vec4 a0(Px, Py);
		vec4 a1 = V1;
		
		vec4 a2( (3 * Px2 - Px) * (1 / (nevezo * nevezo)) - (V2.v[0] + 2 * V1.v[0]) * (1 / nevezo), /* x */
				 (3 * Py2 - Py) * (1 / (nevezo * nevezo)) - (V2.v[1] + 2 * V1.v[1]) * (1 / nevezo));/* y */
		
		vec4 a3( (2*Px-Px2) / (1 / (nevezo * nevezo * nevezo)) + (V2.v[0]+V1.v[0]) / (1 / (nevezo * nevezo)),
			     (2*Py-Py2) / (1 / (nevezo * nevezo * nevezo)) + (V2.v[1] + V1.v[1]) / (1 / (nevezo * nevezo)));
		
		return (a0 + a1 * t_t0 + a2 * t_t0 * t_t0 + a3 * t_t0 * t_t0 * t_t0);
	}

	vec4 getVelocity(const int index) {
		vec4 temp;
		if (index == 0 || index >= nVertices)
			return temp;

		temp = (vertexData[5 * (index + 1)] - vertexData[index])*(1.0 / (time[index + 1] - time[index])) +
			(vertexData[5 * index] - vertexData[ 5 * (index - 1)])*(1.0 / (time[index] - time[index - 1]));
		temp = temp * 0.5f;

		return temp;
	}
};

// The virtual world: collection of two objects
Triangle triangle;
LineStrip lineStrip;

//My global variables and contans
bool isAllowToMove;
const float G = 10.0f;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	
	isAllowToMove = false;

	float tmpCoord[] = { -2, -2, 4, 4, -8, -8, -2, -2, 4, 4, 8, 8 };
	triangle.setCoordinates(tmpCoord);

	float tmpColors[] = { 1.0f, 1.0f, 0.0f, 0.8f, 0.1f, 0.1f, 0.0f, 1.0f, 1.0f };
	triangle.setColor(tmpColors);


	// Create objects by setting up their vertex data on the GPU
	triangle.Create();
	lineStrip.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(shaderProgram, 1, "vertexColor");    // vertexColor gets values from Attrib Array 1

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	triangle.Draw();
	lineStrip.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		isAllowToMove = !isAllowToMove;
		printf("SPACE: %d",isAllowToMove);
		glutPostRedisplay();     // redraw
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		lineStrip.AddPoint(cX, cY, (float)glutGet(GLUT_ELAPSED_TIME)/1000.0f);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate( (isAllowToMove)? sec:0 );					// animate the camera
	triangle.Animate(sec);					// animate the triangle object
	glutPostRedisplay();					// redraw the scene
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

