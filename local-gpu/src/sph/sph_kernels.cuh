#pragma once
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Wendland C2 kernel (3D)
// W(r,h) = (21 / 2*pi*h^3) * (1 - r/h)^4 * (1 + 4r/h)   for r < h
// ---------------------------------------------------------------------------

__device__ inline float wendlandC2(float r, float h) {
    if (r >= h) return 0.0f;
    float q  = r / h;
    float t  = 1.0f - q;
    float t2 = t * t;
    float alpha = 21.0f / (2.0f * 3.14159265f * h * h * h);
    return alpha * t2 * t2 * (1.0f + 4.0f * q);
}

// Gradient magnitude (scalar, multiply by direction separately)
// dW/dr = alpha * (-4)(1-q)^3 * (1+4q)/h + alpha * (1-q)^4 * 4/h
//       = alpha/h * (1-q)^3 * [-4(1+4q) + 4(1-q)]
//       = alpha/h * (1-q)^3 * [-4 - 16q + 4 - 4q]
//       = alpha/h * (1-q)^3 * (-20q)
__device__ inline float wendlandC2Grad(float r, float h) {
    if (r >= h || r < 1e-8f) return 0.0f;
    float q  = r / h;
    float t  = 1.0f - q;
    float alpha = 21.0f / (2.0f * 3.14159265f * h * h * h);
    return alpha / h * t * t * t * (-20.0f * q);
}

// Spiky gradient kernel (for pressure — sharper near center)
__device__ inline float spikyGrad(float r, float h) {
    if (r >= h || r < 1e-8f) return 0.0f;
    float t = h - r;
    float alpha = -45.0f / (3.14159265f * h * h * h * h * h * h);
    return alpha * t * t;
}

// Cubic spline (alternative, widely used)
__device__ inline float cubicSpline(float r, float h) {
    float q = r / h;
    float sigma = 8.0f / (3.14159265f * h * h * h);
    if (q <= 0.5f) {
        return sigma * (6.0f * (q * q * q - q * q) + 1.0f);
    } else if (q <= 1.0f) {
        float t = 1.0f - q;
        return sigma * 2.0f * t * t * t;
    }
    return 0.0f;
}

// ---------------------------------------------------------------------------
// Helper: float3 operations
// ---------------------------------------------------------------------------
__device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ inline float3 operator*(float3 a, float s)  { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ inline float3 operator*(float s, float3 a)  { return a * s; }
__device__ inline float  dot3(float3 a, float3 b)      { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline float  length3(float3 a)              { return sqrtf(dot3(a, a)); }
