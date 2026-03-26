# Vortex GPU — Real-Time CUDA SPH Fluid Simulator

High-performance 3D fluid simulation running **dual-density SPH** (Clavet et al. 2005) entirely on NVIDIA GPUs via CUDA, with zero-copy OpenGL rendering through CUDA-GL interop. Direct port of the [web version](../README.md) (sph.js) to native CUDA, matching its physics model, visual appearance, and UI pixel-for-pixel.

---

## Physics Model

### Dual-Density SPH

The simulation implements the **Clavet-inspired dual-density SPH** model, ported 1:1 from the web `sph.js` implementation. This replaces traditional Tait EOS / Wendland kernel approaches with a simpler, more visually appealing formulation:

- **Density kernel**: `ρᵢ = 1 + Σⱼ (1 - dᵢⱼ/h)²` — quadratic falloff
- **Near-density kernel**: `ρ_nearᵢ = 1 + Σⱼ (1 - dᵢⱼ/h)³` — cubic falloff for anti-clustering
- **Pressure**: `P = stiffness × max(-1, ρ - ρ_rest)` — clamped linear pressure
- **Near-pressure**: `P_near = nearStiffness × ρ_near` — repulsive at close range

### Forces

| Force | Implementation | Parameter |
|-------|---------------|-----------|
| Pressure gradient | Symmetric pairwise: `(P_i + P_j) × (1-d/h) × dir` | `stiffness = 12` |
| Near-pressure | Anti-clustering: `(Pn_i + Pn_j) × (1-d/h)² × dir` | `nearStiffness = 8` |
| XSPH viscosity | Velocity averaging: `viscosity × (1-d/h) × (vⱼ-vᵢ)` | `viscosity = 0.0005` |
| Cohesion | Surface tension when `d > diameter`: `cohesion × (1-d/h) × dir` | `cohesion = 0.2` |
| Gravity | Constant field with tilt support | `g = (0, -5, 0)` default |
| Cursor repel | Radial force from mouse position | `strength = 2–15`, `radius = 0.20–0.22` |
| Drain pull | Downward pull near drain hole center | `drainPull = 0.8` |

### PBD Hard-Sphere Collisions

After integration, a **Position-Based Dynamics** (PBD) pass resolves inter-particle overlaps:

1. Rebuild spatial hash grid (second grid build per substep)
2. For each particle pair with `d < diameter`:
   - Position correction: push apart by `0.5 × (diameter - d)` along contact normal
   - Velocity impulse: reflect relative velocity component along normal (`restitution = 0.10`)

### Rest Density Calibration

Rest density (`ρ₀`) is **auto-calibrated from the first simulation frame**:

1. Run density kernel on initial particle configuration
2. `thrust::reduce` computes sum of all densities on GPU
3. `ρ₀ = sum / activeN`
4. Recalibration triggers on pour start and particle count change (`densityReady` flag)

### Simulation Pipeline (per substep)

```
buildGrid() → computeDensityPressure() → calibrateDensity() → computeForces() → integrate() → buildGrid() → pbd()
```

- **4 substeps per frame** at `dt = 0.003`
- Two grid builds per substep (one for SPH forces, one for PBD collisions)
- Friction damping: `v *= 0.996` per substep
- Wall boundary: position clamped to `[-boxSize+r, boxSize-r]` with velocity reflection × restitution

---

## CUDA Architecture

### Kernel Dispatch

All physics kernels launch with `(N + 255) / 256` blocks of 256 threads:

| Kernel | Purpose | Grid Access |
|--------|---------|-------------|
| `computeGridHashKernel` | Hash particle positions to cell IDs | Write |
| `findCellStartEndKernel` | Build cell start/end arrays from sorted hashes | Write |
| `clearGridKernel` | Reset cell arrays to `0xFFFFFFFF` | Write |
| `densityKernel` | Dual-density SPH: `ρ` and `ρ_near` per particle | 27-cell read |
| `forcesKernel` | Pressure + viscosity + cohesion + repel + drain | 27-cell read |
| `integrateKernel` | Symplectic Euler + XSPH + friction + wall clamp | None |
| `pbdKernel` | Position-based collision resolution | 27-cell read |

### Constant Memory

`SimParams` struct (POD, no default initializers) resides in CUDA `__constant__` memory for broadcast access across all threads:

```cpp
__constant__ SimParams d_params;
// Updated each substep via cudaMemcpyToSymbol()
```

### Spatial Hash Grid

Uniform grid with cell size = `h` (interaction radius):

1. **Hash**: `cellId = x + y × gridDim + z × gridDim²` (clamped to `[0, gridDim-1]³`)
2. **Sort**: `thrust::sort_by_key(d_cellHash, d_particleIdx)` — particles sorted by cell
3. **Cell ranges**: `d_cellStart[cell]` / `d_cellEnd[cell]` for O(1) neighbor cell lookup
4. **27-cell neighborhood**: each kernel iterates `[-1,0,+1]³` neighbor cells

Grid arrays allocated for worst-case `h = 0.04` (`gridDim ≈ 31`, `gridTotal ≈ 29791`) to support dynamic particle count changes without reallocation.

### Memory Layout

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `d_pos` | `float4` | `maxN × 16B` | Positions (xyz) + padding |
| `d_vel` | `float4` | `maxN × 16B` | Velocities (xyz) + padding |
| `d_force` | `float4` | `maxN × 16B` | Accumulated forces |
| `d_density` | `float` | `maxN × 4B` | SPH density |
| `d_nearDensity` | `float` | `maxN × 4B` | Near-density (anti-clustering) |
| `d_pressure` | `float` | `maxN × 4B` | Computed pressure |
| `d_speed` | `float` | `maxN × 4B` | Speed magnitude (for color) |
| `d_dvXSPH` | `float4` | `maxN × 16B` | XSPH velocity correction (static) |
| `d_cellHash` | `uint` | `maxN × 4B` | Per-particle cell hash |
| `d_particleIdx` | `uint` | `maxN × 4B` | Per-particle index |
| `d_sortedHash` | `uint` | `maxN × 4B` | Sorted cell hashes |
| `d_sortedIdx` | `uint` | `maxN × 4B` | Sorted particle indices |
| `d_cellStart` | `uint` | `gridTotal × 4B` | Cell range start |
| `d_cellEnd` | `uint` | `gridTotal × 4B` | Cell range end |

---

## Rendering

### CUDA-GL Interop (Zero-Copy)

Particle positions are written directly from CUDA to an OpenGL VBO without host-side copies:

```
cudaGraphicsMapResources() → cudaGraphicsResourceGetMappedPointer() → CUDA writes d_pos → cudaGraphicsUnmapResources() → OpenGL draws VBO
```

- `cudaGraphicsGLRegisterBuffer()` registers the VBO at init
- Each frame: map → `cudaMemcpy(vboPtr, d_pos, ...)` → unmap → draw

### Point-Sprite Sphere Rendering

Particles rendered as `GL_POINTS` with `gl_PointSize` set per-vertex. Fragment shader computes per-pixel sphere normals from `gl_PointCoord`:

```glsl
vec3 N = vec3(gl_PointCoord * 2.0 - 1.0, 0.0);
float r2 = dot(N.xy, N.xy);
if (r2 > 1.0) discard;
N.z = sqrt(1.0 - r2);
```

**Lighting model** (multi-light):

| Light | Direction | Color | Contribution |
|-------|-----------|-------|-------------|
| Sun | `(0.3, 1.0, 0.5)` | `#99ddff` | Diffuse + specular |
| Fill | `(-0.5, -0.2, -0.3)` | `#002244` | Diffuse (soft fill) |
| Ambient | — | — | `0.35` constant |

- **Specular**: Blinn-Phong, `pow(NdotH, 24)`, strength `0.45`, white-ish tint
- **Rim light**: Fresnel-like `pow(1.0 - NdotV, 2.5)` × `0.35`
- **Fog**: Exponential² (`exp(-fog × dist²)`), color `#030610`, density `0.10`
- **Bloom glow**: Additive `0.12` on final color
- **Opacity**: `0.88` alpha blending

### Color Modes

4 gradient color modes mapped from particle speed, matching web CSS:

| Mode | Name | Low Speed | High Speed |
|------|------|-----------|------------|
| 0 | Deep | `#001f6e` | `#22eeff` |
| 1 | Tropic | `#003322` | `#00ffcc` |
| 2 | Magma | `#5a0800` | `#ff8800` |
| 3 | Void | `#180040` | `#cc44ff` |

### Scene Elements

- **Floor plane**: `#0e1a12`, y = -boxSize
- **Grid**: 8×8 lines, `#162a1a`
- **Box wireframe**: 12 edges, `#1a4a2a`
- **Corner dots**: 8 vertices, `#22d3ee` (accent color)
- **Drain ring**: 32-segment circle at floor center, visible when draining

---

## Dynamic Particle Count

Particle count adjustable from **20 to 20,000** via UI slider. All SPH parameters scale dynamically:

```
scaleFactor = cbrt(1500 / N)
particleRadius = 0.022 × scaleFactor    // visual sphere size
h = 0.08 × scaleFactor                  // SPH interaction radius
```

Ensures particles fill roughly the same volume fraction regardless of count. Maximum radius constrained so particles occupy slightly less than half the box volume.

On count change: parameters recalculated → simulation reset → pour restarted → rest density recalibrated.

---

## UI

ImGui panel styled to match web CSS (glassmorphism design):

- **Panel**: 268px wide, right-aligned, `rgba(1, 6, 18, 0.78)` background, 18px border-radius
- **Colors**: `--text: #e2eaf2`, `--muted: #6b8096`, `--accent: #22d3ee`, `--surface: rgba(255,255,255,0.04)`
- **Header**: "VORTEX 3D" in accent cyan + subtitle with live particle count
- **Particles slider**: 20–20,000 range with styled track/thumb
- **Color buttons**: 2×2 grid (Deep / Tropic / Magma / Void) with per-mode gradient backgrounds
- **Action buttons**: Pause (primary) + Reset (row), Pour / Drain / Shake / Wave (full-width tidal style)
- **FPS badge**: bottom-right corner
- **Controls hint**: bottom-center

---

## Controls

| Input | Action |
|-------|--------|
| **Left click** | Strong repel (force=15, radius=0.22) |
| **Hover** | Light repel (force=2+vel×3, radius=0.20) |
| **Right drag** | Orbit camera (θ: π/8 → π/2.1, φ: free) |
| **Scroll** | Zoom (distance 1.2 → 4.0) |
| **P** | Start pour (resets + pours from ceiling) |
| **D** | Toggle drain (hole at floor center) |
| **R** | Reset simulation (spiral pile) |
| **Space** | Pause / Resume |
| **S** | Shake (1.4s gravity tilt) |
| **W** | Wave (gravity tilt + 0.9s return) |
| **1–4** | Color modes (Deep / Tropic / Magma / Void) |
| **Esc** | Quit |

---

## Project Structure

```
local-gpu/
├── CMakeLists.txt              # Build config (CUDA 13.2, SM 12.0, FetchContent)
├── build_and_run.bat           # One-click build+run (Ninja + MSVC)
├── extern/glad/                # GLAD2 OpenGL 4.6 core loader
│   ├── include/glad/gl.h
│   └── src/gl.c
├── src/
│   ├── main.cpp                # GLFW window, input callbacks, main loop
│   ├── app.h / app.cpp         # Application state, actions, camera, shaders
│   ├── sph/
│   │   ├── params.h            # SimParams struct (__constant__ memory layout)
│   │   ├── sph_system.h        # SPHSystem class (buffers, pipeline methods)
│   │   ├── sph_system.cu       # All CUDA kernels + host-side simulation logic
│   │   └── grid.cuh            # Spatial hash grid kernels
│   ├── render/
│   │   ├── renderer.cpp        # Main render orchestration
│   │   ├── fluid_renderer.cpp  # Point-sprite particle rendering
│   │   ├── box_renderer.cpp    # Box wireframe + floor + grid
│   │   └── shaders/            # GLSL vertex/fragment shaders
│   ├── ui/
│   │   ├── ui.h                # UI class declaration
│   │   └── ui.cpp              # ImGui panel (web CSS replica)
│   └── util/
│       ├── gl_utils.h          # Shader compilation helpers
│       └── gl_utils.cpp
```

---

## Build Requirements

| Dependency | Version | Source |
|------------|---------|--------|
| CUDA Toolkit | 13.2+ | [nvidia.com](https://developer.nvidia.com/cuda-downloads) |
| NVIDIA GPU | SM 12.0 (RTX 5060) | Configurable in CMakeLists.txt |
| CMake | 3.24+ | [cmake.org](https://cmake.org/download/) |
| MSVC | 2022 (v143) | Visual Studio Build Tools |
| Ninja | 1.11+ | Bundled with VS or [ninja-build.org](https://ninja-build.org/) |
| GLFW | 3.4 | FetchContent (auto-downloaded) |
| GLM | 1.0.1 | FetchContent (auto-downloaded) |
| Dear ImGui | 1.91.8 | FetchContent (auto-downloaded) |
| GLAD2 | OpenGL 4.6 core | Vendored in `extern/glad/` |

### Build (Ninja + MSVC)

```bash
# Use the provided build script:
./build_and_run.bat

# Or manually:
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/vortex_gpu.exe
```

> **Note**: `build_and_run.bat` sets up MSVC environment variables and paths automatically. For manual builds, ensure `vcvarsall.bat x64` has been sourced.

### Changing GPU Architecture

Edit `CMakeLists.txt` line 6:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 120)  # Change to your GPU's SM version
```

Common values: `86` (RTX 3000), `89` (RTX 4000), `100` (RTX 5000), `120` (RTX 5060).

---

## Web Version Parity

This CUDA implementation is a direct port of the web version (`sph.js` + `app3d.js`). The following are matched exactly:

- Dual-density SPH kernel weights and force formulation
- All physical constants (stiffness, nearStiffness, viscosity, cohesion, friction, restitution)
- Substep count (4) and timestep (0.003)
- Box size (0.58) and gravity (-5)
- Pour behavior (ceiling spawn, rate 4, spread 0.02)
- Drain mechanics (floor hole, radius 0.09, pull 0.8)
- Reset pattern (spiral pile)
- Room tilt animations (shake, wave, drain dip)
- Color modes and gradients
- UI layout, styling, and controls
- Camera defaults (FOV 32, orbit limits)
- Cursor interaction (hover repel + click strong repel)
