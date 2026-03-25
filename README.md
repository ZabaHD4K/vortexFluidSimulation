# Vortex Fluid Simulation

> Two real-time fluid simulations running entirely in the browser — no server, no build step, no dependencies to install.

A portfolio project demonstrating real-time physics simulation, GPU programming, and clean software architecture — all from scratch in vanilla JavaScript and WebGL2.

---

## Live demos

Open `index.html` in any modern browser. Switch between simulations using the **3D SPH / 2D Fluid** buttons in the top-right panel.

---

## Simulations

### 3D SPH — particle fluid in a glass box

Water-like particles simulated with real physics equations inside a tiltable 3D box. You can rotate the camera, push particles with the cursor, shake the box vertically, and tilt it with arrow keys.

### 2D Navier-Stokes — GPU fluid solver

An incompressible fluid simulation running entirely on the GPU via WebGL2 compute shaders. Tens of thousands of bubble particles float through a stone chamber with caustics, god rays, and foam at high velocity.

Both simulations share a single page. Switching modes pauses the inactive engine and resumes the active one — zero re-initialization cost.

---

## Technical stack

| Layer | Technology |
|---|---|
| 3D Render | Three.js r160 — InstancedMesh, UnrealBloom, ACES tonemapping |
| 3D Physics | Custom SPH engine — TypedArrays, spatial hash grid |
| 2D Fluid | Custom WebGL2 Navier-Stokes solver — GLSL compute shaders |
| Module system | Native ES Modules + Import Maps — zero bundler |
| UI | Glassmorphism panel — pure CSS, no framework |

---

## How the physics works

### 3D: Smoothed Particle Hydrodynamics

Each particle interacts with its neighbours through three forces:

**Viscosity** — velocity-matching between neighbours. Low value = water. High value = honey.

**Cohesion** — gentle attraction just beyond the hard-contact radius, keeping the blob together like surface tension.

**PBD collisions** — Position-Based Dynamics resolves hard-sphere overlaps with immediate position correction and a low restitution coefficient. This gives the inelastic, splashy feel without numerical explosion.

Integration runs at **3 substeps per frame** (effective dt ≈ 1/180 s) for stability.

#### Spatial hash grid — O(N) scaling

The naive pairwise loop is O(N²). At 400 particles that is ~80 000 pair checks per substep.

The solution is a **uniform spatial grid**:

```
1. Divide the simulation box into cells of size h (interaction radius)
2. Each frame: sort all particles into cells using a prefix-sum sort
3. For each particle, only check the 27 neighbouring cells
```

At 400 particles, each particle checks ~10 neighbours instead of 399 — an **18× speedup**. The grid uses pre-allocated `Int32Array` buffers, zero heap allocations per frame.

#### Volume-conserving radius

When changing particle count the radius scales as:

```
r = 0.075 × (40 / N)^(1/3)
```

This keeps the packing density constant. 400 small spheres fill the same volume as 40 large ones.

#### Spawn from above

Changing particle count triggers a **stream spawn**: particles start near the ceiling with strong downward velocities, spread in a golden-ratio spiral cone. They pour in and pile up rather than teleporting into position.

#### Vertical shake

The Space key oscillates the Y component of gravity rapidly between strongly positive and strongly negative using a bell-envelope:

```
gy = -(gravBase × (1 + sin(t × 13) × envelope × 2.4))
```

Particles slam between floor and ceiling. The box nudges in Y for physical feedback.

---

### 2D: Incompressible Navier-Stokes on the GPU

The velocity field lives in a pair of floating-point render targets (double-buffered FBOs). Each frame runs entirely on the GPU:

```
┌─────────────────────────────────────────────┐
│  1. Vorticity curl         (1 draw call)    │
│  2. Vorticity confinement  (1 draw call)    │  keeps swirls sharp
│  3. Divergence             (1 draw call)    │
│  4. Pressure solve         (30 iterations)  │  Jacobi relaxation
│  5. Gradient subtract      (1 draw call)    │  enforces ∇·u = 0
│  6. Velocity advection     (1 draw call)    │
│  7. Dye advection          (1 draw call)    │
└─────────────────────────────────────────────┘
```

Bubble particles (up to 65 536) are stored in a GPU texture. Their positions are updated by a GLSL fragment shader that reads the velocity field — no JavaScript per-particle cost.

The display shader adds caustics, god rays, foam at high velocity, wall ambient occlusion, and a surface shimmer — all in one pass.

---

## Architecture

```
vortex/
├── index.html      Single entry point — hosts both simulations
├── fluid.html      Standalone 2D page (alternative entry point)
├── style.css       Shared glassmorphism UI
└── js/
    ├── sph.js      Pure physics engine (no render dependency)
    ├── app3d.js    Three.js renderer + input + UI wiring
    └── vortex.js   WebGL2 fluid solver + bubble system + UI wiring
```

`sph.js` has zero rendering dependency — importable into any Three.js, Babylon.js, or WebGPU project as-is.

`vortex.js` is self-contained. All GLSL shaders are inlined as template literals, no extra file requests.

Both engines communicate through a `CustomEvent('vortexmode')` dispatched from the mode-switcher script. Neither module knows about the other.

---

## Getting started

No build step. No npm. Just open a file.

```bash
# Recommended — local server avoids ES module CORS issues
npx serve .
# or
python -m http.server 8080
```

Then open `http://localhost:8080` (or simply double-click `index.html` in Chrome/Edge).

Requires a browser with WebGL2 support: Chrome 56+, Firefox 51+, Edge 79+, Safari 15+.

---

## Controls

### 3D simulation

| Input | Action |
|---|---|
| Left drag | Push particles |
| Right drag | Rotate camera |
| Scroll | Zoom |
| Arrow keys | Tilt the box |
| Space | Vertical shake |
| W | Wave — tilt in random direction |
| R | Reset simulation |

### 2D simulation

| Input | Action |
|---|---|
| Click & drag | Paint fluid flow |
| Space | Tidal wave through the pillars |
| Touch | Fully supported |

---

## UI sliders

### 3D

| Slider | What it controls |
|---|---|
| Particles | 40 – 400 spheres. Count change spawns a stream from above. |
| Friction | Velocity-matching strength. Low = water, high = honey. |
| Bounce | Wall restitution. Low = soft splat, high = bouncy. |
| Gravity | Downward acceleration magnitude. |

### 2D

| Slider | What it controls |
|---|---|
| Bubbles | 4 096 – 65 536 GPU bubble particles. |
| Vorticity | Vorticity confinement — keeps swirls sharp. |
| Dissipation | How quickly dye fades. Near 1.0 = colours linger. |
| Splat Size | Radius of each painted splat. |

---

## Performance

| Particles (3D) | Discrete GPU | Integrated GPU |
|---|---|---|
| 40 | 60 fps | 60 fps |
| 150 | 60 fps | 60 fps |
| 250 | 60 fps | 45 – 60 fps |
| 400 | 55 – 60 fps | 30 – 45 fps |

The 2D simulation is GPU-bound. On any machine with hardware WebGL2, 16 384 – 65 536 bubbles run at 60 fps.

---

## Author

**Alejandro Zabaleta**

Built as a portfolio project — real-time physics and GPU programming in the browser, from scratch, with no external physics or shader libraries.
