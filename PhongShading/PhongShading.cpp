#include <Windows.h>
#include <iostream>
#include <limits>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>

#define GLFW_INCLUDE_GLU
#define GLFW_DLL
#include <GLFW/glfw3.h>
#include <vector>

#define GLM_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

using namespace glm;

// -------------------------------------------------
// Global Variables
// -------------------------------------------------
int Width = 512;
int Height = 512;
std::vector<float> OutputImage;
// -------------------------------------------------
//resterization
//--------------------------------------------------
int gNumVertices = 0;    // Number of 3D vertices.
int gNumTriangles = 0;    // Number of triangles.
int* gIndexBuffer = NULL; // Vertex indices for the triangles.
vec3* gVertexBuffer = NULL; // Vertex positions for the vertices.
// -------------------------------------------------

// -------------------------------------------------
// Classes
// -------------------------------------------------

class Material {
public:
    vec3 ka, kd, ks;
    float specularPower;
    Material(const vec3& a, const vec3& d, const vec3& s, float sp)
        : ka(a), kd(d), ks(s), specularPower(sp) {
    }
};

class Sphere {
public:
    vec3 center;
    float radius;
    Material material;
    Sphere(const vec3& c, float r, const Material& mat)
        : material(mat), center(c), radius(r) {
    }

};

class Light {
public:
    vec3 position;
    vec3 color;
    Light(const vec3& p, const vec3& c) : position(p), color(c) {}
};

class Camera {
public:
    vec3 position, direction;
    Camera(const vec3& pos, const vec3& dir)
        : position(pos), direction(normalize(dir)) {
    }
};


class Image {
public:
    void set(int x, int y, vec3 color) {
        OutputImage[(y * Width + x) * 3] = color.x;
        OutputImage[(y * Width + x) * 3 + 1] = color.y;
        OutputImage[(y * Width + x) * 3 + 2] = color.z;
    }
};

void create_scene()
{
    int width = 32;
    int height = 16;

    float theta, phi;
    int t;

    gNumVertices = (height - 2) * width + 2;
    gNumTriangles = (height - 2) * (width - 1) * 2;

    //TODO: Allocate an array for gNumVertices vertices.
    gVertexBuffer = new vec3[gNumVertices];

    gIndexBuffer = new int[3 * gNumTriangles];

    t = 0;
    for (int j = 1; j < height - 1; ++j)//1~15
    {
        for (int i = 0; i < width; ++i)//0~31
        {
            theta = (float)j / (height - 1) * M_PI;
            phi = (float)i / (width - 1) * M_PI * 2;

            float   x = sinf(theta) * cosf(phi);
            float   y = cosf(theta);
            float   z = -sinf(theta) * sinf(phi);

            //Set vertex t in the vertex array to {x, y, z}.
            gVertexBuffer[t] = vec3(x, y, z);

            t++;
        }
    }

    //Set vertex t in the vertex array to {0, 1, 0}.
    gVertexBuffer[t] = vec3(0, 1, 0);
    t++;

    //Set vertex t in the vertex array to {0, -1, 0}.
    gVertexBuffer[t] = vec3(0, -1, 0);
    t++;

    t = 0;
    for (int j = 0; j < height - 3; ++j)
    {
        for (int i = 0; i < width - 1; ++i)
        {
            gIndexBuffer[t++] = j * width + i;
            gIndexBuffer[t++] = (j + 1) * width + (i + 1);
            gIndexBuffer[t++] = j * width + (i + 1);
            gIndexBuffer[t++] = j * width + i;
            gIndexBuffer[t++] = (j + 1) * width + i;
            gIndexBuffer[t++] = (j + 1) * width + (i + 1);
        }
    }
    for (int i = 0; i < width - 1; ++i)
    {
        gIndexBuffer[t++] = (height - 2) * width;
        gIndexBuffer[t++] = i;
        gIndexBuffer[t++] = i + 1;
        gIndexBuffer[t++] = (height - 2) * width + 1;
        gIndexBuffer[t++] = (height - 3) * width + (i + 1);
        gIndexBuffer[t++] = (height - 3) * width + i;
    }
}

mat4 cal_modelTransform(Sphere sphere) {
    //1. Modeling Transform
    mat4 S(1.0f), T(1.0f);
    S[0][0] = sphere.radius;
    S[1][1] = sphere.radius;
    S[2][2] = sphere.radius;
    T[3][0] = sphere.center.x;
    T[3][1] = sphere.center.y;
    T[3][2] = sphere.center.z;
    return T * S;
}

mat4 cal_mvp(Camera camera, mat4 Mm) {
    mat4 Mcam(1.0f);

    mat4 P(0.0f);
    float l = -0.1f, r = 0.1f, b = -0.1f, t = 0.1f, n = 0.1f, f = 1000.0f;
    P[0][0] = (2.0f * n) / (r - l);
    P[1][1] = (2.0f * n) / (t - b);
    P[2][0] = (r + l) / (r - l);
    P[2][1] = (t + b) / (t - b);
    P[2][2] = (f + n) / (n - f);
    P[2][3] = -1.0f;

    P[3][2] = -(2.0f * f * n) / (f - n);

    // 4. Viewport Transformation
    mat4 Vp(1.0f);
    Vp[0][0] = Width / 2.0f;
    Vp[1][1] = Height / 2.0f;
    Vp[3][0] = (Width - 1.0f) / 2.0f;
    Vp[3][1] = (Height - 1.0f) / 2.0f;

    // final MVP
    return Vp * P * Mcam * Mm;
}


void cal_barycentric(mat4 MVP, Image image, Sphere sphere, const Light& light, const Camera& camera, mat4 Mm) {
    std::vector<float> depthBuffer(Width * Height, std::numeric_limits<float>::infinity());
    vec3 ambientLightIntensity = vec3(0.2f);
    float gamma_correct = 2.2f;

    // Precompute world positions and normals
    std::vector<vec3> vertexNormals(gNumVertices);
    std::vector<vec3> worldPositions(gNumVertices);

    for (int i = 0; i < gNumVertices; i++) {
        vec3 worldPos = vec3(Mm * vec4(gVertexBuffer[i], 1.0f));
        worldPositions[i] = worldPos;
        vertexNormals[i] = normalize(worldPos - sphere.center);
    }

    for (int i = 0; i < gNumTriangles; i++) {
        int k0 = gIndexBuffer[3 * i + 0];
        int k1 = gIndexBuffer[3 * i + 1];
        int k2 = gIndexBuffer[3 * i + 2];

        vec4 v0 = MVP * vec4(gVertexBuffer[k0], 1.0f);
        vec4 v1 = MVP * vec4(gVertexBuffer[k1], 1.0f);
        vec4 v2 = MVP * vec4(gVertexBuffer[k2], 1.0f);

        v0 /= v0.w;
        v1 /= v1.w;
        v2 /= v2.w;

        vec2 p0 = vec2(v0.x, v0.y);
        vec2 p1 = vec2(v1.x, v1.y);
        vec2 p2 = vec2(v2.x, v2.y);

        float z0 = v0.z;
        float z1 = v1.z;
        float z2 = v2.z;

        vec3 n0 = vertexNormals[k0];
        vec3 n1 = vertexNormals[k1];
        vec3 n2 = vertexNormals[k2];

        vec3 wp0 = worldPositions[k0];
        vec3 wp1 = worldPositions[k1];
        vec3 wp2 = worldPositions[k2];

        if (z0 < -1.0f && z1 < -1.0f && z2 < -1.0f) continue;
        if (z0 > 1.0f && z1 > 1.0f && z2 > 1.0f) continue;

        int minX = (int)floor(std::min({ p0.x, p1.x, p2.x }));
        int maxX = (int)ceil(std::max({ p0.x, p1.x, p2.x }));
        int minY = (int)floor(std::min({ p0.y, p1.y, p2.y }));
        int maxY = (int)ceil(std::max({ p0.y, p1.y, p2.y }));

        float beta_n = ((p0.y - p2.y) * p1.x + (p2.x - p0.x) * p1.y + p0.x * p2.y - p2.x * p0.y);
        float gamma_n = ((p0.y - p1.y) * p2.x + (p1.x - p0.x) * p2.y + p0.x * p1.y - p1.x * p0.y);
        float beta_x = (p0.y - p2.y) / beta_n;
        float beta_y = (p2.x - p0.x) / beta_n;
        float gamma_x = (p0.y - p1.y) / gamma_n;
        float gamma_y = (p1.x - p0.x) / gamma_n;
        float beta = ((p0.y - p2.y) * minX + (p2.x - p0.x) * minY + p0.x * p2.y - p2.x * p0.y) / beta_n;
        float gamma = ((p0.y - p1.y) * minX + (p1.x - p0.x) * minY + p0.x * p1.y - p1.x * p0.y) / gamma_n;
        int k = (maxX - minX) + 1;

        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                if (x >= 0 && x < Width && y >= 0 && y < Height) {
                    if (beta >= 0.0f && gamma >= 0.0f && (beta + gamma) <= 1.0f) {
                        float alpha = 1.0f - beta - gamma;
                        float z = alpha * z0 + beta * z1 + gamma * z2;

                        int index = y * Width + x;
                        if (z < depthBuffer[index]) {
                            depthBuffer[index] = z;

                            // Interpolate normal and world position
                            vec3 normal = normalize(alpha * n0 + beta * n1 + gamma * n2);
                            vec3 worldPos = alpha * wp0 + beta * wp1 + gamma * wp2;

                            // Per-pixel Phong shading
                            vec3 l = normalize(light.position - worldPos);
                            vec3 v = normalize(camera.position - worldPos);
                            vec3 h = normalize(l + v);

                            vec3 ambient = sphere.material.ka * ambientLightIntensity;
                            vec3 diffuse = sphere.material.kd * light.color * std::max(0.0f, dot(normal, l));
                            vec3 specular = sphere.material.ks * light.color * pow(std::max(0.0f, dot(normal, h)), sphere.material.specularPower);

                            vec3 color = ambient + diffuse + specular;
                            color = pow(color, vec3(1.0f / gamma_correct)); // Gamma correction

                            image.set(x, y, color);
                        }
                    }
                }
                beta += beta_x;
                gamma += gamma_x;
            }
            beta += beta_y - k * beta_x;
            gamma += gamma_y - k * gamma_x;
        }
    }
}



void render() {
    OutputImage.resize(Width * Height * 3, 0.0f);

    Material mat = Material(vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.5f, 0.0f), vec3(0.5f, 0.5f, 0.5f), 32.0f); // material define
    Sphere sphere(vec3(0.0f, 0.0f, -7.0f), 2.0f, mat); // sphere define
    Light light(vec3(-4.0f, 4.0f, -3.0f), vec3(1.0f, 1.0f, 1.0f)); // light define

    Camera camera(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -1.0f));//camera define
    Image image;
    create_scene();  //buffer is created in create_scene() in global variable

    //1 cal Mm
    mat4 Mm = cal_modelTransform(sphere);

    //2,3,4 calculate MVP
    mat4 MVP = cal_mvp(camera, Mm);

    //Baycentric algoritm
    cal_barycentric(MVP, image, sphere, light, camera, Mm);

    //delete buffer
    delete[] gVertexBuffer;
    delete[] gIndexBuffer;
}


void resize_callback(GLFWwindow*, int nw, int nh)
{
    //This is called in response to the window resizing.
    //The new width and height are passed in so we make 
    //any necessary changes:
    Width = nw;
    Height = nh;
    //Tell the viewport to use all of our screen estate
    glViewport(0, 0, nw, nh);

    //This is not necessary, we're just working in 2d so
    //why not let our spaces reflect it?
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(0.0, static_cast<double>(Width)
        , 0.0, static_cast<double>(Height)
        , 1.0, -1.0);

    //Reserve memory for our render so that we don't do 
    //excessive allocations and render the image
    OutputImage.reserve(Width * Height * 3);
    render();
}

int main(int argc, char* argv[])
{
    // -------------------------------------------------
    // Initialize Window
    // -------------------------------------------------

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(Width, Height, "OpenGL Viewer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    //We have an opengl context now. Everything from here on out 
    //is just managing our window or opengl directly.

    //Tell the opengl state machine we don't want it to make 
    //any assumptions about how pixels are aligned in memory 
    //during transfers between host and device (like glDrawPixels(...) )
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    //We call our resize function once to set everything up initially
    //after registering it as a callback with glfw
    glfwSetFramebufferSizeCallback(window, resize_callback);
    resize_callback(NULL, Width, Height);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        //Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        // -------------------------------------------------------------
        //Rendering begins!
        glDrawPixels(Width, Height, GL_RGB, GL_FLOAT, &OutputImage[0]);
        //and ends.
        // -------------------------------------------------------------

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

        //Close when the user hits 'q' or escape
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS
            || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

