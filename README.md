# Vortex Fluid Simulation

> Two real-time fluid simulations running entirely in the browser — no server, no build step, no dependencies to install.

A portfolio project demonstrating real-time physics simulation, GPU programming, and clean software architecture — all from scratch in vanilla JavaScript and WebGL2.

---

## Live demos

Run `start.bat` (Windows) or start a local server and open `index.html`. Switch between simulations using the **3D SPH / 2D Fluid** buttons in the top-right panel.

---

## Simulations

### 3D SPH — particle fluid in a glass box

Water-like particles simulated with real SPH pressure physics inside a tiltable 3D box. Particles flow like a liquid — they resist compression, fill gaps, slosh on tilt, and splash on impact. You can rotate the camera, push particles with the cursor, shake the box, and tilt it with arrow keys.

### 2D Navier-Stokes — GPU fluid solver

An incompressible fluid simulation running entirely on the GPU via WebGL2 compute shaders. Tens of thousands of bubble particles float through a stone chamber with caustics, god rays, and foam at high velocity.

Both simulations share a single page. Switching modes pauses the inactive engine and resumes the active one — zero re-initialization cost.

---

## Technical stack

| Layer | Technology |
|---|---|
| 3D Render | Three.js r170 — InstancedMesh, UnrealBloom, ACES tonemapping |
| 3D Physics | Custom dual-density SPH engine — TypedArrays, spatial hash grid |
| 2D Fluid | Custom WebGL2 Navier-Stokes solver — GLSL compute shaders |
| Module system | Native ES Modules + Import Maps (Three.js via CDN) — zero bundler |
| UI | Glassmorphism panel — pure CSS, no framework |

---

## How the physics works

### 3D: Dual-Density Smoothed Particle Hydrodynamics

The 3D simulation implements a Clavet-inspired dual-density SPH model. Each frame runs through a full fluid dynamics pipeline:

#### SPH pipeline (per sub-step)

```
1. Gravity seed               — initialize force accumulators
2. Spatial grid build          — sort particles into uniform cells
3. Density estimation          — compute ρ and ρ_near from kernel sums
4. Pressure computation        — equation of state: P = k(ρ - ρ₀)
5. Pairwise forces             — pressure gradient + viscosity + cohesion
6. Velocity & position update  — explicit Euler integration
7. Grid rebuild                — re-sort for collision detection
8. PBD collision resolution    — hard-sphere overlap correction (2 iterations)
9. Speed magnitude             — for colour mapping
```

#### Density estimation

Each particle's density is computed by summing SPH kernel contributions from all neighbours within the interaction radius h:

```
Far density:   ρᵢ  = Σⱼ (1 - d/h)²    — quadratic kernel
Near density:  ρnᵢ = Σⱼ (1 - d/h)³    — cubic kernel (steeper, short-range)
```

Rest density (ρ₀) is calibrated automatically from the first frame after reset, representing the natural packing density.

#### Pressure forces

The dual-density approach provides two complementary pressure forces:

**Far pressure** — enforces incompressibility. When particles are compressed above rest density, they repel. When rarefied below rest density, they attract to fill gaps:

```
P = k × (ρ - ρ₀)
```

**Near pressure** — prevents particle clustering. A steeper kernel provides strong short-range repulsion that keeps particles from collapsing into each other:

```
Pnear = k_near × ρ_near
```

The combined force per pair uses spiky-like kernel gradients:

```
F = (Pᵢ + Pⱼ) × q + (Pnear_i + Pnear_j) × q²
```

where q = 1 - d/h. The linear term (far) handles bulk pressure; the quadratic term (near) handles close-range repulsion.

#### Viscosity (XSPH)

Velocity-matching between neighbours smooths the flow:

```
Δv = viscosity × q × Δt × (vⱼ - vᵢ)
```

Low values produce water-like flow. High values produce honey-like viscosity.

#### Surface tension (cohesion)

A gentle attraction beyond contact range keeps the fluid blob together at the free surface:

```
F_cohesion = cohesion × q² × n̂    (only when d > 2r)
```

This acts only at the boundary of the fluid, not in the bulk (where pressure already maintains density).

#### Spatial hash grid — O(N) scaling

The naive pairwise loop is O(N²). At 400 particles that is ~80 000 pair checks per substep.

The solution is a **uniform spatial grid**:

```
1. Divide the simulation box into cells of size h (interaction radius)
2. Each frame: sort all particles into cells using a prefix-sum sort
3. For each particle, only check the 27 neighbouring cells
```

At 400 particles, each particle checks ~15–25 neighbours instead of 399 — an order-of-magnitude speedup. The grid uses pre-allocated `Int32Array` buffers, zero heap allocations per frame.

**Note:** The density pass and force pass both traverse the grid, so there are two neighbour iterations per sub-step. This is inherent to SPH (density must be fully computed before pressure forces can be evaluated).

#### Volume-conserving radius

When changing particle count the radius scales as:

```
r = 0.075 × (40 / N)^(1/3)
```

This keeps the packing density constant. 400 small spheres fill the same volume as 40 large ones.

#### Integration

4 sub-steps per frame (DT = 0.004 per step, total ~0.016 s/frame at 60 fps). The extra sub-step (vs 3 in simpler models) is needed for stability with pressure forces.

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
├── start.bat       One-click launcher (starts server + opens browser)
├── style.css       Shared glassmorphism UI
└── js/
    ├── sph.js      Pure SPH physics engine (no render dependency)
    ├── app3d.js    Three.js renderer + input + UI wiring
    └── vortex.js   WebGL2 fluid solver + bubble system + UI wiring
```

`sph.js` has zero rendering dependency — importable into any Three.js, Babylon.js, or WebGPU project as-is.

`vortex.js` is self-contained. All GLSL shaders are inlined as template literals, no extra file requests.

Both engines communicate through a `CustomEvent('vortexmode')` dispatched from the mode-switcher script. Neither module knows about the other.

---

## Getting started

No build step. No npm. Just open a file.

**Windows (easiest):**
```
Double-click start.bat
```

**Manual:**
```bash
npx serve -l 3000
# or
python -m http.server 3000
```

Then open `http://localhost:3000`.

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
| Friction | XSPH viscosity. Low = water, high = honey. |
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

Both renderers request `powerPreference: 'high-performance'` to force the discrete GPU on dual-GPU laptops.

---

## Author

**Alejandro Zabaleta**

Built as a portfolio project — real-time physics and GPU programming in the browser, from scratch, with no external physics or shader libraries.
