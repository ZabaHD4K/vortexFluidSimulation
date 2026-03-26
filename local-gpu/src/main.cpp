#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>

#include "app.h"
#include "util/cuda_utils.cuh"

static App app;

static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(w, GLFW_TRUE);
    app.onKey(key, action);
}

static void cursorCallback(GLFWwindow* w, double x, double y) {
    app.onMouse(x, y);
}

static void mouseButtonCallback(GLFWwindow* w, int button, int action, int mods) {
    app.onMouseButton(button, action);
}

static void scrollCallback(GLFWwindow* w, double xoffset, double yoffset) {
    app.onScroll(yoffset);
}

static void cursorEnterCallback(GLFWwindow* w, int entered) {
    app.mouseInWindow = (entered != 0);
    if (!entered) {
        // Match web mouseleave: stop repel
        app.sph.params.repelStr = 0.0f;
    }
}

static void resizeCallback(GLFWwindow* w, int width, int height) {
    app.onResize(width, height);
}

int main() {
    // Print CUDA device info
    printDeviceInfo();

    // GLFW init
    if (!glfwInit()) {
        fprintf(stderr, "Failed to init GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(1600, 900, "Vortex GPU — CUDA SPH Fluid Simulation", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to init GLAD\n");
        return -1;
    }

    printf("OpenGL: %s\n", glGetString(GL_RENDERER));
    printf("Version: %s\n", glGetString(GL_VERSION));

    // Callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, resizeCallback);
    glfwSetCursorEnterCallback(window, cursorEnterCallback);

    glEnable(GL_MULTISAMPLE);

    // Init app
    app.init(window);

    // Main loop
    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = (float)(now - lastTime);
        lastTime = now;

        glfwPollEvents();

        app.update(dt);
        app.render();

        glfwSwapBuffers(window);
    }

    app.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
