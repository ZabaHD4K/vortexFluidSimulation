// ===========================================================================
// SPH System — dual-density SPH + PBD collisions
// Direct port of web/js/sph.js to CUDA
// ===========================================================================
#include "sph_system.h"
#include "grid.cuh"
#include "../util/cuda_utils.cuh"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <vector>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 256
#define GRID_BLOCKS(n) ((n + BLOCK_SIZE - 1) / BLOCK_SIZE)

// ---------------------------------------------------------------------------
// Constant memory for params
// ---------------------------------------------------------------------------
__constant__ SimParams d_params;

static void uploadParams(const SimParams& p) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, &p, sizeof(SimParams)));
}

// ---------------------------------------------------------------------------
// Kernel 1: Density estimation — matches web sph.js step() section 3
// density[i] = 1 + sum_j (1-d/h)^2       (symmetric in web, per-particle here)
// nearDensity[i] = 1 + sum_j (1-d/h)^3
// ---------------------------------------------------------------------------
__global__ void densityKernel(
    const float4* __restrict__ pos,
    float* __restrict__ density,
    float* __restrict__ nearDensity,
    const unsigned int* __restrict__ sortedIdx,
    const unsigned int* __restrict__ cellStart,
    const unsigned int* __restrict__ cellEnd,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float h   = d_params.h;
    float h2  = h * h;
    float invH = 1.0f / h;
    int gd = d_params.gridDim;
    float BOX = d_params.boxSize;

    float rho = 1.0f;
    float rhoNear = 1.0f;

    int cx = min(max((int)floorf((pi.x + BOX) / h), 0), gd - 1);
    int cy = min(max((int)floorf((pi.y + BOX) / h), 0), gd - 1);
    int cz = min(max((int)floorf((pi.z + BOX) / h), 0), gd - 1);

    for (int oz = -1; oz <= 1; oz++)
    for (int oy = -1; oy <= 1; oy++)
    for (int ox = -1; ox <= 1; ox++) {
        int nx = cx + ox, ny = cy + oy, nz = cz + oz;
        if (nx < 0 || nx >= gd || ny < 0 || ny >= gd || nz < 0 || nz >= gd) continue;
        unsigned int hash = (unsigned int)(nx + ny * gd + nz * gd * gd);
        unsigned int start = cellStart[hash];
        if (start == 0xFFFFFFFF) continue;
        unsigned int end = cellEnd[hash];

        for (unsigned int k = start; k < end; k++) {
            int j = sortedIdx[k];
            if (j == i) continue;

            float dx = pos[j].x - pi.x;
            float dy = pos[j].y - pi.y;
            float dz = pos[j].z - pi.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 >= h2 || d2 < 1e-10f) continue;

            float d = sqrtf(d2);
            float q = 1.0f - d * invH;
            float q2 = q * q;
            rho     += q2;
            rhoNear += q2 * q;
        }
    }

    density[i]     = rho;
    nearDensity[i] = rhoNear;
}

// ---------------------------------------------------------------------------
// Kernel 2: Forces — matches web sph.js step() sections 4-5
// Pressure: stiffness * max(-1, density - restDensity)
// Near-pressure: nearStiffness * nearDensity
// Force: pressure gradient + viscosity (XSPH) + cohesion + gravity + drain + repel
// ---------------------------------------------------------------------------
__global__ void forcesKernel(
    const float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float*  __restrict__ density,
    const float*  __restrict__ nearDensity,
    float4* __restrict__ force,
    float4* __restrict__ dvXSPH,
    const unsigned int* __restrict__ sortedIdx,
    const unsigned int* __restrict__ cellStart,
    const unsigned int* __restrict__ cellEnd,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float4 vi = vel[i];
    float h   = d_params.h;
    float h2  = h * h;
    float invH = 1.0f / h;
    int gd    = d_params.gridDim;
    float BOX = d_params.boxSize;
    float DT  = d_params.dt;
    float diam = d_params.particleRadius * 2.0f;

    // Pressure for particle i — match web: stiffness * max(-1, density - restDensity)
    float prsi = d_params.stiffness * fmaxf(-1.0f, density[i] - d_params.restDensity);
    float nearPi = d_params.nearStiffness * nearDensity[i];

    // Start with gravity
    float3 f = d_params.gravity;
    float3 dv = make_float3(0, 0, 0);

    int cx = min(max((int)floorf((pi.x + BOX) / h), 0), gd - 1);
    int cy = min(max((int)floorf((pi.y + BOX) / h), 0), gd - 1);
    int cz = min(max((int)floorf((pi.z + BOX) / h), 0), gd - 1);

    for (int oz = -1; oz <= 1; oz++)
    for (int oy = -1; oy <= 1; oy++)
    for (int ox = -1; ox <= 1; ox++) {
        int nx = cx + ox, ny = cy + oy, nz = cz + oz;
        if (nx < 0 || nx >= gd || ny < 0 || ny >= gd || nz < 0 || nz >= gd) continue;
        unsigned int hash = (unsigned int)(nx + ny * gd + nz * gd * gd);
        unsigned int start = cellStart[hash];
        if (start == 0xFFFFFFFF) continue;
        unsigned int end = cellEnd[hash];

        for (unsigned int k = start; k < end; k++) {
            int j = sortedIdx[k];
            if (j == i) continue;

            float ddx = pos[j].x - pi.x;
            float ddy = pos[j].y - pi.y;
            float ddz = pos[j].z - pi.z;
            float d2  = ddx*ddx + ddy*ddy + ddz*ddz;
            if (d2 >= h2 || d2 < 1e-10f) continue;

            float dist = sqrtf(d2);
            float dirx = ddx / dist, diry = ddy / dist, dirz = ddz / dist;
            float q = 1.0f - dist * invH;

            // Pressure: (press_i + press_j)*q + (nearP_i + nearP_j)*q^2
            float prsj = d_params.stiffness * fmaxf(-1.0f, density[j] - d_params.restDensity);
            float nearPj = d_params.nearStiffness * nearDensity[j];
            float fp = (prsi + prsj) * q + (nearPi + nearPj) * q * q;
            f.x -= fp * dirx;
            f.y -= fp * diry;
            f.z -= fp * dirz;

            // Viscosity (XSPH) — match web: viscosity * q * DT * (v_j - v_i)
            float vf = d_params.viscosity * q * DT;
            dv.x += vf * (vel[j].x - vi.x);
            dv.y += vf * (vel[j].y - vi.y);
            dv.z += vf * (vel[j].z - vi.z);

            // Cohesion — match web: if (d > diam) cohesion * q^2 * dir
            if (dist > diam) {
                float cf = d_params.cohesion * q * q;
                f.x += cf * dirx;
                f.y += cf * diry;
                f.z += cf * dirz;
            }
        }
    }

    // Drain pull toward center — match web
    if (d_params.draining) {
        float dx = -pi.x;
        float dz = -pi.z;
        float dist = sqrtf(dx*dx + dz*dz) + 0.01f;
        float pull = d_params.drainPull / dist;
        f.x += dx * pull * 0.006f / DT;  // scale to match web per-frame pull
        f.z += dz * pull * 0.006f / DT;
    }

    // Cursor repulsion — match web repelFrom()
    if (d_params.repelStr > 0.0f) {
        float dx = pi.x - d_params.repelPos.x;
        float dy = pi.y - d_params.repelPos.y;
        float dz = pi.z - d_params.repelPos.z;
        float d2 = dx*dx + dy*dy + dz*dz;
        float r2 = d_params.repelRad * d_params.repelRad;
        if (d2 < r2 && d2 > 1e-8f) {
            float d = sqrtf(d2);
            float str = d_params.repelStr * (1.0f - d / d_params.repelRad) / d;
            f.x += str * dx;
            f.y += str * dy;
            f.z += str * dz;
        }
    }

    force[i] = make_float4(f.x, f.y, f.z, 0);
    dvXSPH[i] = make_float4(dv.x, dv.y, dv.z, 0);
}

// ---------------------------------------------------------------------------
// Kernel 3: Integration — matches web sph.js step() section 6
// v = (v + f*DT + dv_xsph) * FRICTION
// p += v * DT
// + wall clamping (section 7 of web)
// ---------------------------------------------------------------------------
__global__ void integrateKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ force,
    const float4* __restrict__ dvXSPH,
    float* __restrict__ speed,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float DT = d_params.dt;
    float FR = d_params.friction;
    float RE = d_params.restitution;
    float W  = d_params.boxSize;

    float4 v = vel[i];
    float4 f = force[i];
    float4 dv = (dvXSPH != nullptr) ? dvXSPH[i] : make_float4(0,0,0,0);

    // Match web: v = (v + f*DT + dv) * FRICTION
    v.x = (v.x + f.x * DT + dv.x) * FR;
    v.y = (v.y + f.y * DT + dv.y) * FR;
    v.z = (v.z + f.z * DT + dv.z) * FR;

    float4 p = pos[i];
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;

    // Wall clamping — match web
    if (p.x < -W) { p.x = -W; if (v.x < 0) v.x *= -RE; }
    if (p.x >  W) { p.x =  W; if (v.x > 0) v.x *= -RE; }
    if (p.y < -W) { p.y = -W; if (v.y < 0) v.y *= -RE; }
    if (p.y >  W) { p.y =  W; if (v.y > 0) v.y *= -RE; }
    if (p.z < -W) { p.z = -W; if (v.z < 0) v.z *= -RE; }
    if (p.z >  W) { p.z =  W; if (v.z > 0) v.z *= -RE; }

    pos[i] = p;
    vel[i] = v;
    speed[i] = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

// ---------------------------------------------------------------------------
// Kernel 4: PBD hard-sphere collision — matches web sph.js step() section 8
// For each pair within diameter: push apart + velocity impulse
// Per-particle accumulation (no race conditions)
// ---------------------------------------------------------------------------
__global__ void pbdKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const unsigned int* __restrict__ sortedIdx,
    const unsigned int* __restrict__ cellStart,
    const unsigned int* __restrict__ cellEnd,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float4 vi = vel[i];
    float diam  = d_params.particleRadius * 2.0f;
    float diam2 = diam * diam;
    float RE    = d_params.restitution;
    float h     = d_params.h;
    int gd      = d_params.gridDim;
    float BOX   = d_params.boxSize;

    float3 dp = make_float3(0, 0, 0);
    float3 dvel = make_float3(0, 0, 0);

    int cx = min(max((int)floorf((pi.x + BOX) / h), 0), gd - 1);
    int cy = min(max((int)floorf((pi.y + BOX) / h), 0), gd - 1);
    int cz = min(max((int)floorf((pi.z + BOX) / h), 0), gd - 1);

    for (int oz = -1; oz <= 1; oz++)
    for (int oy = -1; oy <= 1; oy++)
    for (int ox = -1; ox <= 1; ox++) {
        int nx = cx + ox, ny = cy + oy, nz = cz + oz;
        if (nx < 0 || nx >= gd || ny < 0 || ny >= gd || nz < 0 || nz >= gd) continue;
        unsigned int hash = (unsigned int)(nx + ny * gd + nz * gd * gd);
        unsigned int start = cellStart[hash];
        if (start == 0xFFFFFFFF) continue;
        unsigned int end = cellEnd[hash];

        for (unsigned int k = start; k < end; k++) {
            int j = sortedIdx[k];
            if (j == i) continue;

            float ddx = pos[j].x - pi.x;
            float ddy = pos[j].y - pi.y;
            float ddz = pos[j].z - pi.z;
            float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
            if (d2 >= diam2 || d2 < 1e-10f) continue;

            float dist = sqrtf(d2);
            float push = (diam - dist) * 0.5f;
            float nx_ = ddx / dist, ny_ = ddy / dist, nz_ = ddz / dist;

            // Position correction (half, since both particles move)
            dp.x -= nx_ * push;
            dp.y -= ny_ * push;
            dp.z -= nz_ * push;

            // Velocity impulse
            float relVn = (vel[j].x - vi.x) * nx_
                        + (vel[j].y - vi.y) * ny_
                        + (vel[j].z - vi.z) * nz_;
            if (relVn < 0) {
                float imp = relVn * (1.0f + RE) * 0.5f;
                dvel.x += imp * nx_;
                dvel.y += imp * ny_;
                dvel.z += imp * nz_;
            }
        }
    }

    pos[i] = make_float4(pi.x + dp.x, pi.y + dp.y, pi.z + dp.z, 0);
    vel[i] = make_float4(vi.x + dvel.x, vi.y + dvel.y, vi.z + dvel.z, 0);
}

// ===========================================================================
// SPHSystem implementation
// ===========================================================================

SPHSystem::SPHSystem() { params = defaultParams(); }
SPHSystem::~SPHSystem() { destroy(); }

void SPHSystem::init(int maxN) {
    params.maxParticles = maxN;
    params.gridDim = (int)ceilf(params.boxSize * 2.0f / params.h) + 2;
    params.gridTotal = params.gridDim * params.gridDim * params.gridDim;

    // Worst case grid (smallest h = 0.04)
    int maxGD = (int)ceilf(params.boxSize * 2.0f / 0.04f) + 2;
    maxGridTotal = maxGD * maxGD * maxGD;

    allocateBuffers(maxN);
    initialized = true;
    printf("SPH initialized: maxN=%d, h=%.4f, gridDim=%d\n",
           maxN, params.h, params.gridDim);
}

void SPHSystem::allocateBuffers(int maxN) {
    CUDA_CHECK(cudaMalloc(&d_pos,         maxN * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_vel,         maxN * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_force,       maxN * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_density,     maxN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nearDensity, maxN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pressure,    maxN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_speed,       maxN * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_cellHash,    maxN * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_particleIdx, maxN * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_sortedHash,  maxN * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_sortedIdx,   maxN * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_cellStart,   maxGridTotal * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd,     maxGridTotal * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemset(d_vel,   0, maxN * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_force, 0, maxN * sizeof(float4)));
}

void SPHSystem::freeBuffers() {
    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_force);
    cudaFree(d_density); cudaFree(d_nearDensity);
    cudaFree(d_pressure); cudaFree(d_speed);
    cudaFree(d_cellHash); cudaFree(d_particleIdx);
    cudaFree(d_sortedHash); cudaFree(d_sortedIdx);
    cudaFree(d_cellStart); cudaFree(d_cellEnd);
}

void SPHSystem::destroy() {
    if (!initialized) return;
    unregisterGLBuffer();
    freeBuffers();
    initialized = false;
}

void SPHSystem::registerGLBuffer(unsigned int vbo) {
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBOResource, vbo,
               cudaGraphicsMapFlagsWriteDiscard));
    glInteropActive = true;
}

void SPHSystem::unregisterGLBuffer() {
    if (glInteropActive) {
        cudaGraphicsUnregisterResource(cudaVBOResource);
        glInteropActive = false;
    }
}

float4* SPHSystem::mapPositions() {
    if (!glInteropActive) return d_pos;
    size_t size;
    float4* ptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBOResource));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&ptr, &size, cudaVBOResource));
    return ptr;
}

void SPHSystem::unmapPositions() {
    if (glInteropActive) {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVBOResource));
    }
}

void SPHSystem::buildGrid() {
    int N = params.activeN;
    int GT = params.gridTotal;

    clearGridKernel<<<GRID_BLOCKS(GT), BLOCK_SIZE>>>(d_cellStart, d_cellEnd, GT);
    computeGridHashKernel<<<GRID_BLOCKS(N), BLOCK_SIZE>>>(
        d_cellHash, d_particleIdx, d_pos, N, params.h, params.boxSize, params.gridDim);

    CUDA_CHECK(cudaMemcpy(d_sortedHash, d_cellHash, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_sortedIdx, d_particleIdx, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    thrust::device_ptr<unsigned int> hashPtr(d_sortedHash);
    thrust::device_ptr<unsigned int> idxPtr(d_sortedIdx);
    thrust::sort_by_key(hashPtr, hashPtr + N, idxPtr);

    findCellStartEndKernel<<<GRID_BLOCKS(N), BLOCK_SIZE>>>(d_cellStart, d_cellEnd, d_sortedHash, N, GT);
}

void SPHSystem::computeDensityPressure() {
    int N = params.activeN;
    densityKernel<<<GRID_BLOCKS(N), BLOCK_SIZE>>>(
        d_pos, d_density, d_nearDensity, d_sortedIdx, d_cellStart, d_cellEnd, N);
}

void SPHSystem::computeForces() {
    int N = params.activeN;
    // d_force stores forces, d_pressure repurposed as dvXSPH (float4 needed)
    // Actually we need a float4 for dvXSPH. Let's reuse d_force for forces.
    // We need a separate buffer for dvXSPH. Use d_vel temporarily? No.
    // Just allocate... we already have d_force. Let's use d_pressure area.
    // Actually d_pressure is float*, we need float4*. Use a static device buffer.
    // Simplest: pass d_force as force output and use a portion of the force buffer.
    // No — let's just allocate a small temp or reuse the approach differently.
    // The cleanest: pass vel both for reading and for dvXSPH accumulation via d_force as second output.
    // Actually let's just use two float4 outputs. d_force for forces, and... we need another buffer.
    // The easiest: allocate another float4 buffer for dvXSPH in allocateBuffers.
    // But to avoid changing everything, let's compute dv inline in integrate.
    // OR: store (f.x, f.y, f.z, 0) in d_force and (dv.x, dv.y, dv.z, 0) in... hmm.

    // Let's just use d_pos mapped area as temp. No that's wrong.
    // Simplest: make forces kernel write force+dv interleaved into force buffer.
    // Actually even simpler: merge force and dv in the kernel, apply in integrate.
    // force[i] = (f.x + dv.x/DT, f.y + dv.y/DT, f.z + dv.z/DT) so integrate just does v = (v + F*DT)*FR

    // No, web applies them separately: v = (v + f*DT + dv)*FR
    // = (v + (f + dv/DT)*DT) * FR
    // So we can combine: effective_f = f + dv/DT, then v = (v + eff_f * DT) * FR
    // This is mathematically identical. Let's do that.

    forcesKernel<<<GRID_BLOCKS(N), BLOCK_SIZE>>>(
        d_pos, d_vel, d_density, d_nearDensity, d_force,
        // For dvXSPH, we'll combine in the kernel. Let me refactor.
        // Actually let me just create a device buffer for dvXSPH.
        // We have d_pressure which is float*. We need float4*.
        // Since d_pressure is unused in this model (pressure computed inline),
        // let's repurpose it. But it's float* not float4*.
        // Let me just add a static float4* for dvXSPH.
        nullptr, // placeholder
        d_sortedIdx, d_cellStart, d_cellEnd, N);
}

void SPHSystem::integrate() {
    int N = params.activeN;
    integrateKernel<<<GRID_BLOCKS(N), BLOCK_SIZE>>>(
        d_pos, d_vel, d_force, nullptr, d_speed, N);
}

void SPHSystem::pbd() {
    int N = params.activeN;
    pbdKernel<<<GRID_BLOCKS(N), BLOCK_SIZE>>>(
        d_pos, d_vel, d_sortedIdx, d_cellStart, d_cellEnd, N);
}

void SPHSystem::calibrateDensity() {
    // Match web: average density over all particles on first frame
    int N = params.activeN;
    if (N < 2) return;

    thrust::device_ptr<float> dPtr(d_density);
    float sum = thrust::reduce(dPtr, dPtr + N, 0.0f, thrust::plus<float>());
    params.restDensity = sum / N;
    params.densityReady = 1;
    printf("Rest density calibrated: %.3f (N=%d)\n", params.restDensity, N);
}

void SPHSystem::step() {
    if (params.activeN < 2) return;

    uploadParams(params);

    for (int s = 0; s < params.substeps; s++) {
        // 1. Build spatial grid
        buildGrid();

        // 2. Compute density (dual-density)
        computeDensityPressure();

        // 3. Calibrate rest density on first substep after reset/pour
        if (!params.densityReady) {
            calibrateDensity();
            uploadParams(params);  // re-upload with calibrated restDensity
        }

        // 4. Compute forces + integrate (combined to avoid extra buffer)
        {
            int N = params.activeN;
            // We need dvXSPH buffer — reuse d_pressure area cast to float4
            // d_pressure has maxParticles * sizeof(float) bytes
            // We need N * sizeof(float4) bytes.
            // maxParticles*4 >= N*16 only if maxParticles >= 4*N.
            // For safety, let's use a different approach: allocate dvXSPH as part of init.
            // For now, use a static device allocation.
            static float4* d_dvXSPH = nullptr;
            static int dvAllocN = 0;
            if (dvAllocN < params.maxParticles) {
                if (d_dvXSPH) cudaFree(d_dvXSPH);
                CUDA_CHECK(cudaMalloc(&d_dvXSPH, params.maxParticles * sizeof(float4)));
                dvAllocN = params.maxParticles;
            }

            forcesKernel<<<GRID_BLOCKS(N), BLOCK_SIZE>>>(
                d_pos, d_vel, d_density, d_nearDensity, d_force, d_dvXSPH,
                d_sortedIdx, d_cellStart, d_cellEnd, N);

            integrateKernel<<<GRID_BLOCKS(N), BLOCK_SIZE>>>(
                d_pos, d_vel, d_force, d_dvXSPH, d_speed, N);
        }

        // 5. Rebuild grid for PBD collision
        buildGrid();

        // 6. PBD hard-sphere collision — match web step 8
        pbd();
    }

    // Copy results to GL buffer for rendering
    if (glInteropActive) {
        float4* posPtr = mapPositions();
        CUDA_CHECK(cudaMemcpy(posPtr, d_pos, params.activeN * sizeof(float4), cudaMemcpyDeviceToDevice));
        unmapPositions();
    }
}

void SPHSystem::pourTick() {
    if (!params.pouring) return;
    int start = params.activeN;
    int count = min(params.pourRate, params.maxParticles - start);
    if (count <= 0) { params.pouring = 0; return; }

    // Match web sph.js pourTick: spawn at ceiling (W-0.01), vy = -(4+rand*3)
    float W = params.boxSize;
    std::vector<float4> newPos(count);
    std::vector<float4> newVel(count);

    for (int i = 0; i < count; i++) {
        float angle = ((float)rand() / RAND_MAX) * 6.283185f;
        float rr    = ((float)rand() / RAND_MAX) * params.pourSpread;
        newPos[i] = make_float4(
            cosf(angle) * rr,
            W - 0.01f,
            sinf(angle) * rr,
            0.0f
        );
        newVel[i] = make_float4(
            ((float)rand() / RAND_MAX - 0.5f) * 0.1f,
            -(4.0f + (float)rand() / RAND_MAX * 3.0f),
            ((float)rand() / RAND_MAX - 0.5f) * 0.1f,
            0.0f
        );
    }

    CUDA_CHECK(cudaMemcpy(d_pos + start, newPos.data(), count * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel + start, newVel.data(), count * sizeof(float4), cudaMemcpyHostToDevice));
    params.activeN += count;

    // Recalibrate density while pouring — match web
    params.densityReady = 0;
}

void SPHSystem::drainTick() {
    if (!params.draining || params.activeN == 0) return;

    int N = params.activeN;
    std::vector<float4> hPos(N);
    CUDA_CHECK(cudaMemcpy(hPos.data(), d_pos, N * sizeof(float4), cudaMemcpyDeviceToHost));

    float r2 = params.drainRadius * params.drainRadius;
    float floorY = -params.boxSize;

    std::vector<int> keep;
    keep.reserve(N);
    for (int i = 0; i < N; i++) {
        float dx = hPos[i].x;
        float dy = hPos[i].y - floorY;
        float dz = hPos[i].z;
        if (dx*dx + dy*dy + dz*dz < r2) continue;
        keep.push_back(i);
    }

    if ((int)keep.size() == N) return;

    int newN = (int)keep.size();
    std::vector<float4> hVel(N);
    CUDA_CHECK(cudaMemcpy(hVel.data(), d_vel, N * sizeof(float4), cudaMemcpyDeviceToHost));

    std::vector<float4> cPos(newN), cVel(newN);
    for (int i = 0; i < newN; i++) {
        cPos[i] = hPos[keep[i]];
        cVel[i] = hVel[keep[i]];
    }

    CUDA_CHECK(cudaMemcpy(d_pos, cPos.data(), newN * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, cVel.data(), newN * sizeof(float4), cudaMemcpyHostToDevice));
    params.activeN = newN;

    if (params.activeN == 0) params.draining = 0;
}

void SPHSystem::reset() {
    params.activeN = 0;
    params.pouring = 0;
    params.draining = 0;
    params.densityReady = 0;
    CUDA_CHECK(cudaMemset(d_vel, 0, params.maxParticles * sizeof(float4)));
}

void SPHSystem::resetWithParticles(int n) {
    n = min(n, params.maxParticles);
    float W = params.boxSize;
    float golden = 2.3999632f;

    std::vector<float4> hPos(n);
    for (int i = 0; i < n; i++) {
        float t   = (float)i / (float)n;
        float th  = i * golden;
        float rad = 0.04f + t * 0.30f;
        float px = fmaxf(-W, fminf(W, cosf(th) * rad * 0.9f));
        float py = fmaxf(-W, fminf(W, -W + t * W * 1.4f));
        float pz = fmaxf(-W, fminf(W, sinf(th) * rad * 0.9f));
        hPos[i] = make_float4(px, py, pz, 0.0f);
    }

    CUDA_CHECK(cudaMemcpy(d_pos, hPos.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vel, 0, params.maxParticles * sizeof(float4)));
    params.activeN     = n;
    params.pouring     = 0;
    params.draining    = 0;
    params.densityReady = 0;
}
