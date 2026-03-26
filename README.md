# Vortex — Fluid Simulation

Two implementations of real-time fluid simulation:

## `web-sin-gpu/` — Browser (WebGL)
SPH 3D + Navier-Stokes 2D running in the browser. No install needed.
- 750 particles, CPU physics, Three.js rendering
- Open `index.html` or run `start.bat`

## `local-gpu/` — Native CUDA (RTX GPU)
Professional SPH simulator using CUDA compute + OpenGL.
- 500K+ particles, full GPU physics
- Requires CUDA Toolkit + CMake + NVIDIA GPU
- See `local-gpu/README.md` for build instructions
