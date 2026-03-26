#pragma once
#include <cuda_runtime.h>

struct SimParams {
    // SPH — dual-density model (matches web sph.js exactly)
    float h;              // interaction radius
    float particleRadius; // visual radius, used for PBD collision diameter
    float stiffness;      // far-field pressure (web: 12)
    float nearStiffness;  // near-field anti-clustering (web: 8)
    float viscosity;      // XSPH velocity matching (web: 0.0005)
    float cohesion;       // surface tension / cohesion (web: 0.2)
    float friction;       // velocity damping per step (web: 0.996)
    float restitution;    // wall bounce (web: 0.10)
    float restDensity;    // calibrated from first frame
    int   densityReady;   // 0 = needs calibration

    // Time
    float dt;
    int   substeps;

    // World
    float3 gravity;
    float  boxSize;

    // Grid
    int   gridDim;
    int   gridTotal;

    // Particles
    int   maxParticles;
    int   activeN;

    // Pour / Drain
    int   pouring;
    int   draining;
    int   pourRate;
    float pourY;
    float pourSpread;
    float drainRadius;
    float drainPull;

    // Repel (cursor)
    float3 repelPos;
    float  repelStr;
    float  repelRad;

    // Color mode
    int   colorMode;

    // Gravity tilt
    float gravBase;
    float gravTiltX;
    float gravTiltZ;
};

inline SimParams defaultParams() {
    SimParams p{};
    // === Match web sph.js exactly ===
    p.h              = 0.11f;    // 0.09 * cbrt(1500/750)
    p.particleRadius = 0.031f;   // 0.025 * cbrt(1500/750)
    p.stiffness      = 12.0f;    // web: this.stiffness = 12
    p.nearStiffness  = 8.0f;     // web: this.nearStiffness = 8
    p.viscosity      = 0.0005f;  // web: this.viscosity = 0.0005
    p.cohesion       = 0.2f;     // web: this.cohesion = 0.2
    p.friction       = 0.996f;   // web: this.FRICTION = 0.996
    p.restitution    = 0.10f;    // web: this.RESTITUTION = 0.10
    p.restDensity    = 0.0f;     // calibrated at runtime
    p.densityReady   = 0;
    p.dt             = 0.003f;   // web: const DT = 0.003
    p.substeps       = 4;        // web: 4 substeps per frame
    p.gravity        = make_float3(0, -5.0f, 0);
    p.boxSize        = 0.58f;    // web: this.BOX = 0.58
    p.gridDim        = 0;
    p.gridTotal      = 0;
    p.maxParticles   = 1048576;
    p.activeN        = 0;
    p.pouring        = 0;
    p.draining       = 0;
    p.pourRate       = 4;        // web: POUR_RATE = 4
    p.pourY          = 0.55f;
    p.pourSpread     = 0.02f;    // web: POUR_SPREAD = 0.02
    p.drainRadius    = 0.09f;    // web: DRAIN_RADIUS = 0.09
    p.drainPull      = 0.8f;
    p.repelPos       = make_float3(0, 0, 0);
    p.repelStr       = 0.0f;
    p.repelRad       = 0.0f;
    p.colorMode      = 0;
    p.gravBase       = 5.0f;
    p.gravTiltX      = 0.0f;
    p.gravTiltZ      = 0.0f;
    return p;
}
