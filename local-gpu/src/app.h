#pragma once
#include "util/gl_utils.h"
#include "sph/sph_system.h"
#include <GLFW/glfw3.h>

class App {
public:
    void init(GLFWwindow* window);
    void update(float dt);
    void render();
    void destroy();

    void onKey(int key, int action);
    void onMouse(double x, double y);
    void onMouseButton(int button, int action);
    void onScroll(double yoffset);
    void onResize(int w, int h);

    // Actions (match web app3d.js exactly)
    void startPour();
    void toggleDrain();
    void triggerShake();
    void triggerWave();
    void resetSim();
    void applyGravity();
    void changeParticleCount(int newN);
    void recalcSPHParams();

    // Particle radius from current targetN (fills ~40% box)
    float particleRadius() const;

    SPHSystem sph;
    bool paused = false;
    int  targetN = 750;
    static constexpr int MAX_PARTICLES = 20000;

    // Rendering
    GLuint particleVAO = 0;
    GLuint particleVBO = 0;
    GLuint speedVBO    = 0;
    GLuint shaderProg  = 0;

    // Camera — match web: FOV 32, position (2.8, 2.2, 2.8)
    int winW = 1600, winH = 900;
    float camDist  = 4.5f;
    float camPhi   = 0.78f;   // polar angle from Y axis
    float camTheta = 0.78f;   // azimuthal

    // Mouse
    double lastMX = 0, lastMY = 0;
    bool rightDrag  = false;
    bool leftDrag   = false;
    bool mouseInWindow = true;

    // Shake state (match web)
    float shakeTimer = 0;
    float shakeAngle = 0;
    float simTime    = 0;

    // Wave state (match web: setTimeout 900ms)
    float waveTimer  = 0;

    // Room tilt visual (match web roomGroup.rotation)
    float roomRotZ = 0;
    float roomRotX = 0;
    float roomOffY = 0;

    // Pour/drain visual state
    bool pouring  = false;
    bool draining = false;

private:
    void setupGL();
    GLFWwindow* window = nullptr;
};
