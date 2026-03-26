#pragma once
#include "../util/gl_utils.h"
#include "../sph/sph_system.h"
#include <GLFW/glfw3.h>

class App;  // forward declaration

namespace UI {
    void init(GLFWwindow* window);
    void draw(SPHSystem& sph, App& app);
    void shutdown();
}
