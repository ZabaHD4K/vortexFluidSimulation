#include "app.h"
#include "ui/ui.h"
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// Shaders — match web COLOR_MODES exactly
// Deep:   #001f6e -> #22eeff   Tropic: #003322 -> #00ffcc
// Magma:  #5a0800 -> #ff8800   Void:   #180040 -> #cc44ff
//
// Web material: roughness 0.08, metalness 0.05, opacity 0.88
// Web lighting: ambient #0a1520 i=5, sun #99ddff i=4 @(2,5,2),
//               fill #002244 i=1.5 @(-3,-1,-2), causticA #22aaff i=6, causticB #0044dd i=4
// Web fog: FogExp2(#030610, 0.10)
// Web background: #020508
// ---------------------------------------------------------------------------

static const char* VERT_SRC = R"(
#version 460 core
layout(location = 0) in vec4 aPos;
layout(location = 1) in float aSpeed;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform float uPointSize;
uniform int uColorMode;
uniform vec3 uCamPos;

out vec3 vColor;
out float vDist;  // for fog

vec3 colorMap(float t, int mode) {
    if (mode == 0) return mix(vec3(0.0, 0.122, 0.431), vec3(0.133, 0.933, 1.0), t);
    if (mode == 1) return mix(vec3(0.0, 0.200, 0.133), vec3(0.0, 1.0, 0.8), t);
    if (mode == 2) return mix(vec3(0.353, 0.031, 0.0), vec3(1.0, 0.533, 0.0), t);
    return mix(vec3(0.094, 0.0, 0.251), vec3(0.8, 0.267, 1.0), t);
}

void main() {
    vec4 worldPos = uModel * vec4(aPos.xyz, 1.0);
    gl_Position = uMVP * vec4(aPos.xyz, 1.0);
    gl_PointSize = uPointSize / gl_Position.w;
    vColor = colorMap(clamp(aSpeed * 0.12, 0.0, 1.0), uColorMode);
    vDist = length(worldPos.xyz - uCamPos);
}
)";

static const char* FRAG_SRC = R"(
#version 460 core
in vec3 vColor;
in float vDist;
out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;

    // Sphere normal for lighting
    float z = sqrt(1.0 - r2);
    vec3 normal = vec3(coord, z);

    // Match web multi-light setup
    // Sun: #99ddff from (2,5,2) normalized
    vec3 sunDir = normalize(vec3(2.0, 5.0, 2.0));
    float sun = max(dot(normal, sunDir), 0.0) * 0.55;

    // Fill: #002244 from (-3,-1,-2) normalized
    vec3 fillDir = normalize(vec3(-3.0, -1.0, -2.0));
    float fill = max(dot(normal, fillDir), 0.0) * 0.15;

    // Ambient: match web #0a1520 intensity 5 -> soft ambient
    float ambient = 0.35;

    // Specular highlight — bright glossy shine
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 halfDir = normalize(sunDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 24.0) * 0.45;

    // Rim light — bright edge glow so particles pop against each other
    float rim = 1.0 - max(dot(normal, viewDir), 0.0);
    rim = pow(rim, 2.5) * 0.35;

    float light = ambient + sun + fill + spec + rim;
    vec3 col = vColor * light;

    // Specular is white-ish, not tinted
    col += vec3(0.7, 0.85, 1.0) * spec * 0.5;

    // Bloom-like glow
    col += vColor * 0.12;

    // Fog: match web FogExp2(#030610, 0.10)
    vec3 fogColor = vec3(0.012, 0.024, 0.063);
    float fogFactor = exp(-0.10 * vDist * 0.10 * vDist);
    col = mix(fogColor, col, clamp(fogFactor, 0.0, 1.0));

    fragColor = vec4(col, 0.88);
}
)";

// Box wireframe — match web edge color: #1a4a2a with opacity 0.45
static const char* BOX_VERT = R"(
#version 460 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
)";

static const char* BOX_FRAG = R"(
#version 460 core
out vec4 fragColor;
void main() { fragColor = vec4(0.102, 0.29, 0.165, 0.45); }
)";

// Floor grid — match web #162a1a, opacity 0.55
static const char* GRID_FRAG = R"(
#version 460 core
out vec4 fragColor;
void main() { fragColor = vec4(0.086, 0.165, 0.102, 0.55); }
)";

// Floor plane — match web floor: color #0e1a12
static const char* FLOOR_FRAG = R"(
#version 460 core
out vec4 fragColor;
void main() { fragColor = vec4(0.055, 0.102, 0.071, 0.85); }
)";

// Corner dots — match web #22d3ee, opacity 0.55
static const char* DOT_FRAG = R"(
#version 460 core
out vec4 fragColor;
void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0) discard;
    fragColor = vec4(0.133, 0.827, 0.933, 0.55);
}
)";

// Drain ring shader — match web drain glow #22d3ee
static const char* DRAIN_FRAG = R"(
#version 460 core
out vec4 fragColor;
void main() { fragColor = vec4(0.133, 0.827, 0.933, 0.5); }
)";

static GLuint boxVAO, boxVBO, boxShader;
static GLuint gridVAO, gridVBO, gridShader;
static GLuint floorVAO, floorVBO, floorShader;
static GLuint dotVAO, dotVBO, dotShader;
static GLuint drainVAO, drainVBO, drainShader;
static int gridVertCount = 0;
static int drainVertCount = 0;

// Slightly tighter than web: r = 0.022 * cbrt(1500/N)
float App::particleRadius() const {
    return 0.022f * cbrtf(1500.0f / (float)targetN);
}

void App::recalcSPHParams() {
    float r = particleRadius();
    sph.params.particleRadius = r;

    // Tighter interaction: h = 0.08 * cbrt(1500/N)
    float h = 0.08f * cbrtf(1500.0f / (float)targetN);
    h = fmaxf(0.03f, fminf(0.25f, h));
    sph.params.h = h;
    sph.params.gridDim = (int)ceilf(sph.params.boxSize * 2.0f / h) + 2;
    sph.params.gridTotal = sph.params.gridDim * sph.params.gridDim * sph.params.gridDim;

    // Reset density calibration when params change
    sph.params.densityReady = 0;

    // Pour rate scales with N
    sph.params.pourRate = (int)fmaxf(1.0f, (float)targetN / 180.0f);
}

void App::changeParticleCount(int newN) {
    newN = (int)fmaxf(20.0f, fminf((float)MAX_PARTICLES, (float)newN));
    targetN = newN;
    recalcSPHParams();
    startPour();
}

void App::init(GLFWwindow* w) {
    window = w;

    sph.init(MAX_PARTICLES);
    sph.params.activeN = 0;
    recalcSPHParams();

    setupGL();
    UI::init(w);

    // Auto-start pour like web
    startPour();

    printf("App initialized. Controls match web version.\n");
}

void App::setupGL() {
    shaderProg = createProgram(VERT_SRC, FRAG_SRC);

    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);
    glGenBuffers(1, &speedVBO);

    glBindVertexArray(particleVAO);

    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, sph.params.maxParticles * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, speedVBO);
    glBufferData(GL_ARRAY_BUFFER, sph.params.maxParticles * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), nullptr);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    sph.registerGLBuffer(particleVBO);

    float B = sph.params.boxSize;

    // Box wireframe — 12 edges
    float boxVerts[] = {
        -B,-B,-B,  B,-B,-B,   B,-B,-B,  B, B,-B,   B, B,-B, -B, B,-B,  -B, B,-B, -B,-B,-B,
        -B,-B, B,  B,-B, B,   B,-B, B,  B, B, B,   B, B, B, -B, B, B,  -B, B, B, -B,-B, B,
        -B,-B,-B, -B,-B, B,   B,-B,-B,  B,-B, B,   B, B,-B,  B, B, B,  -B, B,-B, -B, B, B,
    };
    boxShader = createProgram(BOX_VERT, BOX_FRAG);
    glGenVertexArrays(1, &boxVAO);
    glGenBuffers(1, &boxVBO);
    glBindVertexArray(boxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(boxVerts), boxVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // Floor grid — match web 8x8
    {
        float W = B * 2.0f;
        float floorY = -B + 0.002f;
        std::vector<float> lines;
        for (int t = 0; t <= 8; t++) {
            float v = -B + (t / 8.0f) * W;
            lines.insert(lines.end(), {-B, floorY, v,  B, floorY, v});
            lines.insert(lines.end(), {v, floorY, -B,  v, floorY, B});
        }
        gridVertCount = (int)(lines.size() / 3);
        gridShader = createProgram(BOX_VERT, GRID_FRAG);
        glGenVertexArrays(1, &gridVAO);
        glGenBuffers(1, &gridVBO);
        glBindVertexArray(gridVAO);
        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
        glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    // Floor plane — match web floor mesh: PlaneGeometry(W,W) at y=-BOX+0.001
    {
        float fy = -B + 0.001f;
        float floorVerts[] = {
            -B, fy, -B,   B, fy, -B,   B, fy, B,
            -B, fy, -B,   B, fy, B,    -B, fy, B,
        };
        floorShader = createProgram(BOX_VERT, FLOOR_FRAG);
        glGenVertexArrays(1, &floorVAO);
        glGenBuffers(1, &floorVBO);
        glBindVertexArray(floorVAO);
        glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(floorVerts), floorVerts, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    // Corner dots — match web: cyan spheres at all 8 corners
    {
        float dots[] = {
            -B,-B,-B,  B,-B,-B,  -B, B,-B,  B, B,-B,
            -B,-B, B,  B,-B, B,  -B, B, B,  B, B, B,
        };
        dotShader = createProgram(VERT_SRC, DOT_FRAG); // reuse VERT for gl_PointSize
        // Actually use simpler shader — just needs MVP + point size
        dotShader = createProgram(R"(
#version 460 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
uniform float uPointSize;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize / gl_Position.w;
}
)", DOT_FRAG);
        glGenVertexArrays(1, &dotVAO);
        glGenBuffers(1, &dotVBO);
        glBindVertexArray(dotVAO);
        glBindBuffer(GL_ARRAY_BUFFER, dotVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(dots), dots, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    // Drain ring — circle of line segments at floor center
    {
        float dr = sph.params.drainRadius;
        float dy = -B + 0.003f;
        std::vector<float> ring;
        int segs = 32;
        for (int i = 0; i < segs; i++) {
            float a0 = (float)i / segs * 6.283185f;
            float a1 = (float)(i + 1) / segs * 6.283185f;
            ring.insert(ring.end(), {cosf(a0)*dr, dy, sinf(a0)*dr});
            ring.insert(ring.end(), {cosf(a1)*dr, dy, sinf(a1)*dr});
        }
        drainVertCount = (int)(ring.size() / 3);
        drainShader = createProgram(BOX_VERT, DRAIN_FRAG);
        glGenVertexArrays(1, &drainVAO);
        glGenBuffers(1, &drainVBO);
        glBindVertexArray(drainVAO);
        glBindBuffer(GL_ARRAY_BUFFER, drainVBO);
        glBufferData(GL_ARRAY_BUFFER, ring.size() * sizeof(float), ring.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void App::applyGravity() {
    // Match web applyGravity(): normalize (tiltX, 1, tiltZ) then scale by gravBase
    float tx = sph.params.gravTiltX;
    float tz = sph.params.gravTiltZ;
    float len = sqrtf(tx * tx + 1.0f + tz * tz);
    float s = sph.params.gravBase / len;
    sph.params.gravity = make_float3(tx * s, -s, tz * s);
}

void App::startPour() {
    // Match web startPour(): new SPH, startPour, show funnel, pouring=true
    sph.reset();
    recalcSPHParams();
    sph.params.pouring = 1;
    sph.params.gravTiltX = 0;
    sph.params.gravTiltZ = 0;
    pouring = true;
    applyGravity();
}

void App::toggleDrain() {
    // Match web toggleDrain()
    draining = !draining;
    sph.params.draining = draining ? 1 : 0;
    if (sph.params.activeN == 0) {
        draining = false;
        sph.params.draining = 0;
    }
}

void App::triggerShake() {
    // Match web triggerShake()
    shakeTimer = 1.4f;
    shakeAngle = ((float)rand() / RAND_MAX) * 6.283185f;
}

void App::triggerWave() {
    // Match web triggerWave()
    float ang = ((float)rand() / RAND_MAX) * 6.283185f;
    sph.params.gravTiltX = cosf(ang) * 1.6f;
    sph.params.gravTiltZ = sinf(ang) * 1.6f;
    applyGravity();
    waveTimer = 0.9f;
}

void App::resetSim() {
    // Match web resetSim(): new SPH(N) -> spawns N particles in spiral pile
    sph.resetWithParticles(targetN);
    sph.params.gravTiltX = 0;
    sph.params.gravTiltZ = 0;
    shakeTimer = 0;
    pouring = false;
    draining = false;
    applyGravity();
}

void App::update(float dt) {
    simTime += dt;

    // Wave timer — match web: setTimeout(() => { gravTiltX = gravTiltZ = 0 }, 900)
    if (waveTimer > 0) {
        waveTimer -= dt;
        if (waveTimer <= 0) {
            sph.params.gravTiltX = 0;
            sph.params.gravTiltZ = 0;
            applyGravity();
        }
    }

    // Room tilt animation — match web roomGroup.rotation
    if (shakeTimer > 0) {
        float env = sinf((shakeTimer / 1.4f) * 3.14159f);
        float osc = sinf(simTime * 13.0f);

        // Match web: roomGroup.rotation.z += (cos(shakeAngle)*osc*env*0.05 - rotation.z)*0.35
        float targetZ = cosf(shakeAngle) * osc * env * 0.05f;
        roomRotZ += (targetZ - roomRotZ) * 0.35f;

        // Match web: shakeOffY = osc * env * 0.045
        float shakeOffY = osc * env * 0.045f;
        roomOffY += (shakeOffY - roomOffY) * 0.25f;
    } else {
        // Match web: roomGroup.rotation.z += (-gravTiltX * 0.18 - rotation.z) * 0.12
        roomRotZ += (-sph.params.gravTiltX * 0.18f - roomRotZ) * 0.12f;
        // Match web: roomGroup.rotation.x += (gravTiltZ * 0.18 - rotation.x) * 0.12
        roomRotX += (sph.params.gravTiltZ * 0.18f - roomRotX) * 0.12f;
    }

    // Match web: drainDip = draining ? -0.045 : 0
    float drainDip = draining ? -0.045f : 0.0f;
    roomOffY += (drainDip - roomOffY) * 0.25f;

    if (paused) return;

    // Shake physics — match web exactly
    if (shakeTimer > 0) {
        shakeTimer -= dt;
        float env = sinf((shakeTimer / 1.4f) * 3.14159f);
        float osc = sinf(simTime * 13.0f);
        float gyShake = -(sph.params.gravBase * (1.0f + osc * env * 2.4f));
        applyGravity();
        sph.params.gravity.y = gyShake;
    } else {
        applyGravity();
    }

    // Pour: match web — stop when activeN >= N
    if (pouring) {
        if (sph.params.activeN >= targetN) {
            sph.params.pouring = 0;
            pouring = false;
        }
    }

    // Drain: auto-stop when empty — match web
    if (draining && sph.params.activeN == 0) {
        draining = false;
        sph.params.draining = 0;
    }

    sph.pourTick();
    sph.drainTick();
    sph.step();

    // Upload speed data for coloring
    if (sph.params.activeN > 0) {
        float* speedHost = new float[sph.params.activeN];
        cudaMemcpy(speedHost, sph.d_speed, sph.params.activeN * sizeof(float), cudaMemcpyDeviceToHost);
        glBindBuffer(GL_ARRAY_BUFFER, speedVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sph.params.activeN * sizeof(float), speedHost);
        delete[] speedHost;
    }
}

void App::render() {
    // Background: match web scene.background = 0x020508
    glClearColor(0.008f, 0.02f, 0.031f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Camera — match web: position (2.8, 2.2, 2.8), lookAt (0, -0.1, 0), FOV 32
    float cx = camDist * sinf(camPhi) * cosf(camTheta);
    float cy = camDist * cosf(camPhi);
    float cz = camDist * sinf(camPhi) * sinf(camTheta);

    glm::vec3 camPos(cx, cy, cz);
    glm::vec3 lookAt(0, -0.1f, 0);

    glm::mat4 view = glm::lookAt(camPos, lookAt, glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(
        glm::radians(32.0f), (float)winW / winH, 0.01f, 50.0f
    );

    // Room tilt model matrix — match web roomGroup rotation
    glm::mat4 roomModel = glm::mat4(1.0f);
    roomModel = glm::translate(roomModel, glm::vec3(0, roomOffY, 0));
    roomModel = glm::rotate(roomModel, roomRotZ, glm::vec3(0, 0, 1));
    roomModel = glm::rotate(roomModel, roomRotX, glm::vec3(1, 0, 0));

    glm::mat4 mvp = proj * view * roomModel;

    // Draw floor plane — match web floor mesh
    glUseProgram(floorShader);
    glUniformMatrix4fv(glGetUniformLocation(floorShader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glBindVertexArray(floorVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Draw floor grid
    glUseProgram(gridShader);
    glUniformMatrix4fv(glGetUniformLocation(gridShader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glBindVertexArray(gridVAO);
    glDrawArrays(GL_LINES, 0, gridVertCount);

    // Draw box wireframe
    glUseProgram(boxShader);
    glUniformMatrix4fv(glGetUniformLocation(boxShader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glBindVertexArray(boxVAO);
    glDrawArrays(GL_LINES, 0, 24);

    // Draw corner dots — match web cyan dots at 8 corners
    glUseProgram(dotShader);
    glUniformMatrix4fv(glGetUniformLocation(dotShader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(glGetUniformLocation(dotShader, "uPointSize"), 120.0f);
    glBindVertexArray(dotVAO);
    glDrawArrays(GL_POINTS, 0, 8);

    // Draw drain ring — visible only when draining, match web drainRing/drainGlow
    if (draining) {
        glUseProgram(drainShader);
        glUniformMatrix4fv(glGetUniformLocation(drainShader, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
        glLineWidth(2.0f);
        glBindVertexArray(drainVAO);
        glDrawArrays(GL_LINES, 0, drainVertCount);
        glLineWidth(1.0f);
    }

    // Draw particles
    int N = sph.params.activeN;
    if (N > 0) {
        // Dynamic radius: fills ~40% of box volume
        float visualR = particleRadius();
        glUseProgram(shaderProg);
        glUniformMatrix4fv(glGetUniformLocation(shaderProg, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(glGetUniformLocation(shaderProg, "uModel"), 1, GL_FALSE, glm::value_ptr(roomModel));
        glUniform1f(glGetUniformLocation(shaderProg, "uPointSize"), 1200.0f * visualR);
        glUniform1i(glGetUniformLocation(shaderProg, "uColorMode"), sph.params.colorMode);
        glUniform3f(glGetUniformLocation(shaderProg, "uCamPos"), cx, cy, cz);

        glBindVertexArray(particleVAO);
        glDrawArrays(GL_POINTS, 0, N);
    }

    // UI
    UI::draw(sph, *this);
}

void App::onKey(int key, int action) {
    // Match web: keydown only
    if (action != GLFW_PRESS) return;
    switch (key) {
        case GLFW_KEY_P:       startPour(); break;
        case GLFW_KEY_D:       toggleDrain(); break;
        case GLFW_KEY_R:       resetSim(); break;
        case GLFW_KEY_SPACE:   triggerShake(); break;
        case GLFW_KEY_W:       triggerWave(); break;
        // Match web arrow tilt: +-0.25, clamped to +-1.8
        case GLFW_KEY_LEFT:
            sph.params.gravTiltX = fmaxf(-1.8f, sph.params.gravTiltX - 0.25f);
            applyGravity(); break;
        case GLFW_KEY_RIGHT:
            sph.params.gravTiltX = fminf(1.8f, sph.params.gravTiltX + 0.25f);
            applyGravity(); break;
        case GLFW_KEY_UP:
            sph.params.gravTiltZ = fmaxf(-1.8f, sph.params.gravTiltZ - 0.25f);
            applyGravity(); break;
        case GLFW_KEY_DOWN:
            sph.params.gravTiltZ = fminf(1.8f, sph.params.gravTiltZ + 0.25f);
            applyGravity(); break;
    }
}

void App::onMouse(double x, double y) {
    double dx = x - lastMX;
    double dy = y - lastMY;
    lastMX = x;
    lastMY = y;

    if (ImGui::GetIO().WantCaptureMouse) return;

    // Match web: RIGHT drag = orbit (OrbitControls RIGHT = ROTATE)
    if (rightDrag) {
        camTheta += (float)dx * 0.005f;
        camPhi   -= (float)dy * 0.005f;
        // Match web orbit limits: minPolarAngle = PI/8, maxPolarAngle = PI/2.1
        camPhi = fmaxf(3.14159f / 8.0f, fminf(3.14159f / 2.1f, camPhi));
    }

    // Match web: mousemove ALWAYS repels lightly, click repels strongly
    // Web: sim.repelFrom(hitPoint.x, hitPoint.y, hitPoint.z, 0.15 + cursorVel*0.2, 0.20)
    // Approximate: project mouse to world plane
    float ndcX = (float)(2.0 * x / winW - 1.0);
    float ndcY = (float)(1.0 - 2.0 * y / winH);
    // Simple world-space approximation
    float repX = ndcX * camDist * 0.35f;
    float repZ = ndcY * camDist * 0.35f;
    sph.params.repelPos = make_float3(repX, 0.0f, repZ);

    if (leftDrag) {
        // Match web click: strong repel (1.0, 0.22)
        sph.params.repelStr = 15.0f;
        sph.params.repelRad = 0.22f;
    } else if (mouseInWindow) {
        // Match web hover: light repel (0.15 + vel*0.2, 0.20)
        float vel = sqrtf((float)(dx*dx + dy*dy)) * 0.1f;
        sph.params.repelStr = 2.0f + vel * 3.0f;
        sph.params.repelRad = 0.20f;
    }
}

void App::onMouseButton(int button, int action) {
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_RIGHT)
        rightDrag = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        leftDrag = (action == GLFW_PRESS);
        if (action == GLFW_RELEASE) {
            // Back to hover-strength repel
            sph.params.repelStr = 2.0f;
        }
    }
}

void App::onScroll(double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    camDist -= (float)yoffset * 0.3f;
    // Match web: minDistance 1.0, maxDistance 8.0
    camDist = fmaxf(1.0f, fminf(8.0f, camDist));
}

void App::onResize(int w, int h) {
    winW = w; winH = h;
    glViewport(0, 0, w, h);
}

void App::destroy() {
    sph.destroy();
    UI::shutdown();
    glDeleteVertexArrays(1, &particleVAO);
    glDeleteBuffers(1, &particleVBO);
    glDeleteBuffers(1, &speedVBO);
    glDeleteProgram(shaderProg);
    glDeleteVertexArrays(1, &boxVAO);
    glDeleteBuffers(1, &boxVBO);
    glDeleteProgram(boxShader);
    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    glDeleteProgram(gridShader);
    glDeleteVertexArrays(1, &floorVAO);
    glDeleteBuffers(1, &floorVBO);
    glDeleteProgram(floorShader);
    glDeleteVertexArrays(1, &dotVAO);
    glDeleteBuffers(1, &dotVBO);
    glDeleteProgram(dotShader);
    glDeleteVertexArrays(1, &drainVAO);
    glDeleteBuffers(1, &drainVBO);
    glDeleteProgram(drainShader);
}
