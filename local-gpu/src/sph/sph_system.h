#pragma once
#include "params.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <glad/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class SPHSystem {
public:
    SPHSystem();
    ~SPHSystem();

    void init(int maxN);
    void destroy();

    // Per-frame simulation (runs substeps on GPU)
    void step();

    // Particle management (host-side, uploads to GPU)
    void pourTick();
    void drainTick();
    void reset();
    void resetWithParticles(int n);

    // CUDA-GL interop
    void registerGLBuffer(unsigned int vbo);
    void unregisterGLBuffer();

    float4* mapPositions();
    void    unmapPositions();

    SimParams params;

    // Device buffers
    float4* d_pos         = nullptr;
    float4* d_vel         = nullptr;
    float4* d_force       = nullptr;
    float*  d_density     = nullptr;
    float*  d_nearDensity = nullptr;   // dual-density SPH
    float*  d_pressure    = nullptr;
    float*  d_speed       = nullptr;

    // Grid
    unsigned int* d_cellHash    = nullptr;
    unsigned int* d_particleIdx = nullptr;
    unsigned int* d_sortedHash  = nullptr;
    unsigned int* d_sortedIdx   = nullptr;
    unsigned int* d_cellStart   = nullptr;
    unsigned int* d_cellEnd     = nullptr;

    // GL interop
    cudaGraphicsResource* cudaVBOResource = nullptr;
    bool glInteropActive = false;

private:
    void allocateBuffers(int maxN);
    void freeBuffers();
    void buildGrid();
    void computeDensityPressure();
    void computeForces();
    void integrate();
    void pbd();              // PBD hard-sphere collision (web step 8)
    void calibrateDensity(); // measure rest density from first frame

    bool initialized = false;
    int maxGridTotal = 0;
};
