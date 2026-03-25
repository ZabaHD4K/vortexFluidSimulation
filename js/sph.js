'use strict';

// ---------------------------------------------------------------------------
// SPH — Smoothed Particle Hydrodynamics (3-D, hybrid PBD + SPH)
//
// Physics model
//   • Gravity          — configurable direction + strength (tiltable box)
//   • Viscosity        — velocity-matching between neighbours (0 = gas, 2 = honey)
//   • Cohesion         — gentle attraction just beyond hard-contact range
//   • PBD collisions   — immediate position correction, low restitution
//
// Performance
//   • Uniform spatial grid  — O(N) neighbour queries instead of O(N²)
//   • Pre-allocated typed arrays everywhere — zero GC pressure per frame
//   • Radius scales as N^(-1/3) so fill-factor stays constant across counts
//
// Public API
//   new SPH(N)        — construct with N particles
//   .reset()          — place particles in a stable pile at bottom
//   .spawnFromTop()   — launch all particles downward from the ceiling
//   .step()           — advance one sub-step (call 3× per frame)
//   .repelFrom(x,y,z,str,radius) — push particles away from a world point
//   .setGravityStrength(g)       — rescale gravity magnitude, keep direction
// ---------------------------------------------------------------------------

export class SPH {
  constructor(N = 40) {
    this.N   = N;

    // Volume-conserving radius: same packing density regardless of N.
    // Derived from: N * (4/3)π r³ = const  →  r ∝ N^(-1/3)
    this.r   = 0.075 * Math.cbrt(40 / N);

    this.BOX = 0.58;
    this.W   = this.BOX - this.r;   // effective wall (sphere centre limit)
    this.h   = this.r * 4.2;        // SPH interaction radius (~4 diameters)

    // Gravity vector — set externally by app3d.js
    this.gx = 0; this.gy = -9; this.gz = 0;

    // Fluid parameters exposed to UI sliders
    this.viscosity   = 0.55;   // velocity-matching strength
    this.cohesion    = 3.5;    // surface tension analogue
    this.FRICTION    = 0.9978; // per-substep air drag
    this.RESTITUTION = 0.10;   // wall bounce (low = soft splat)

    // Particle state — Float32 for cache-friendly sequential access
    this.px    = new Float32Array(N);
    this.py    = new Float32Array(N);
    this.pz    = new Float32Array(N);
    this.vx    = new Float32Array(N);
    this.vy    = new Float32Array(N);
    this.vz    = new Float32Array(N);
    this.speed = new Float32Array(N);   // magnitude, for colour mapping

    // Per-frame accumulators (reused each step, no allocation)
    this._fx  = new Float32Array(N);
    this._fy  = new Float32Array(N);
    this._fz  = new Float32Array(N);
    this._dvx = new Float32Array(N);
    this._dvy = new Float32Array(N);
    this._dvz = new Float32Array(N);

    // ----- Spatial hash grid -----
    // Cell size = h so one layer of neighbouring cells covers all interactions.
    // Grid spans [-BOX, BOX] in each axis.
    const gd    = Math.ceil(this.BOX * 2 / this.h) + 2; // +2 safety margin
    this._gd    = gd;
    this._ic    = 1.0 / this.h;          // inverse cell size
    const total = gd * gd * gd;
    this._cnt   = new Int32Array(total); // particles per cell
    this._sta   = new Int32Array(total); // start index of each cell in _idx
    this._tmp   = new Int32Array(total); // write-cursor per cell (build phase)
    this._idx   = new Int32Array(N);     // sorted particle indices
    this._cell  = new Int32Array(N);     // cell index for each particle

    this.reset();
  }

  // -------------------------------------------------------------------------
  // Spatial grid helpers
  // -------------------------------------------------------------------------

  _cellId(x, y, z) {
    const { _gd: gd, _ic, BOX } = this;
    const cx = Math.max(0, Math.min(gd - 1, (x + BOX) * _ic | 0));
    const cy = Math.max(0, Math.min(gd - 1, (y + BOX) * _ic | 0));
    const cz = Math.max(0, Math.min(gd - 1, (z + BOX) * _ic | 0));
    return cx + cy * gd + cz * gd * gd;
  }

  _buildGrid() {
    const { N, _gd: gd, _cnt, _sta, _tmp, _idx, _cell } = this;
    const total = gd * gd * gd;

    // 1. Count particles per cell
    _cnt.fill(0); _tmp.fill(0);
    for (let i = 0; i < N; i++) {
      const c = this._cellId(this.px[i], this.py[i], this.pz[i]);
      _cell[i] = c;
      _cnt[c]++;
    }

    // 2. Prefix-sum → cell start offsets
    _sta[0] = 0;
    for (let c = 1; c < total; c++) _sta[c] = _sta[c - 1] + _cnt[c - 1];

    // 3. Fill sorted index array
    for (let i = 0; i < N; i++) {
      const c = _cell[i];
      _idx[_sta[c] + _tmp[c]++] = i;
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Place particles in a tightly-packed spiral pile at the bottom. */
  reset() {
    const W = this.W, golden = 2.3999632;
    for (let i = 0; i < this.N; i++) {
      const t   = i / this.N;
      const θ   = i * golden;
      const rad = 0.04 + t * 0.30;
      this.px[i] = Math.max(-W, Math.min(W, Math.cos(θ) * rad * 0.9));
      this.py[i] = Math.max(-W, Math.min(W, -W + t * W * 1.4));
      this.pz[i] = Math.max(-W, Math.min(W, Math.sin(θ) * rad * 0.9));
      this.vx[i] = this.vy[i] = this.vz[i] = 0;
      this.speed[i] = 0;
    }
  }

  /**
   * Launch all particles from the top of the box with strong downward velocity,
   * spread in a golden-spiral cone — looks like water pouring in from above.
   */
  spawnFromTop() {
    const W = this.W, golden = 2.3999632;
    for (let i = 0; i < this.N; i++) {
      const t   = i / this.N;
      const θ   = i * golden;
      // Radial spread grows with index so they fan outward like a stream
      const rad = Math.sqrt(t) * W * 0.40;
      this.px[i] = Math.max(-W * 0.9, Math.min(W * 0.9, Math.cos(θ) * rad));
      // Stagger Y slightly so they cascade rather than all hitting at once
      this.py[i] = W * (0.60 + t * 0.30 + Math.random() * 0.08);
      this.pz[i] = Math.max(-W * 0.9, Math.min(W * 0.9, Math.sin(θ) * rad));
      // Strong downward velocity; faster for particles nearer the top
      this.vy[i] = -(3.0 + (1.0 - t) * 5.0);
      this.vx[i] = (Math.random() - 0.5) * 0.6;
      this.vz[i] = (Math.random() - 0.5) * 0.6;
      this.speed[i] = 0;
    }
  }

  /** Advance one physics sub-step (≈ 1/180 s at 60 fps, 3 sub-steps/frame). */
  step() {
    const DT = 0.0055;
    const { N, r, W, h, gx, gy, gz,
            viscosity, cohesion, FRICTION, RESTITUTION,
            _gd: gd, _ic, _cnt, _sta, _idx } = this;
    const diam  = r * 2;
    const diam2 = diam * diam;
    const h2    = h * h;
    const { _fx, _fy, _fz, _dvx, _dvy, _dvz } = this;
    const BOX = this.BOX;

    // ------------------------------------------------------------------
    // 1. Gravity seed
    // ------------------------------------------------------------------
    for (let i = 0; i < N; i++) {
      _fx[i] = gx; _fy[i] = gy; _fz[i] = gz;
      _dvx[i] = _dvy[i] = _dvz[i] = 0;
    }

    // ------------------------------------------------------------------
    // 2. Spatial grid (pre-integration positions)
    // ------------------------------------------------------------------
    this._buildGrid();

    // ------------------------------------------------------------------
    // 3. Pairwise forces: viscosity + cohesion  [O(N) with grid]
    // ------------------------------------------------------------------
    for (let i = 0; i < N; i++) {
      const cxI = Math.max(0, Math.min(gd - 1, (this.px[i] + BOX) * _ic | 0));
      const cyI = Math.max(0, Math.min(gd - 1, (this.py[i] + BOX) * _ic | 0));
      const czI = Math.max(0, Math.min(gd - 1, (this.pz[i] + BOX) * _ic | 0));

      for (let oz = -1; oz <= 1; oz++) {
        const czN = czI + oz; if (czN < 0 || czN >= gd) continue;
        for (let oy = -1; oy <= 1; oy++) {
          const cyN = cyI + oy; if (cyN < 0 || cyN >= gd) continue;
          for (let ox = -1; ox <= 1; ox++) {
            const cxN = cxI + ox; if (cxN < 0 || cxN >= gd) continue;
            const c   = cxN + cyN * gd + czN * gd * gd;
            const end = _sta[c] + _cnt[c];
            for (let k = _sta[c]; k < end; k++) {
              const j = _idx[k];
              if (j <= i) continue;   // each pair once

              const ddx = this.px[j] - this.px[i];
              const ddy = this.py[j] - this.py[i];
              const ddz = this.pz[j] - this.pz[i];
              const d2  = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d2 >= h2 || d2 < 1e-10) continue;

              const d  = Math.sqrt(d2);
              const nx = ddx / d, ny = ddy / d, nz = ddz / d;
              const q  = 1.0 - d / h;          // kernel weight [0,1]

              // Viscosity: velocity-matching impulse
              const vf  = viscosity * q * DT;
              const dvx = this.vx[j] - this.vx[i];
              const dvy = this.vy[j] - this.vy[i];
              const dvz = this.vz[j] - this.vz[i];
              _dvx[i] += vf * dvx; _dvy[i] += vf * dvy; _dvz[i] += vf * dvz;
              _dvx[j] -= vf * dvx; _dvy[j] -= vf * dvy; _dvz[j] -= vf * dvz;

              // Cohesion: attraction beyond hard-contact range
              if (d > diam) {
                const cf = cohesion * q * q;
                _fx[i] += cf * nx; _fy[i] += cf * ny; _fz[i] += cf * nz;
                _fx[j] -= cf * nx; _fy[j] -= cf * ny; _fz[j] -= cf * nz;
              }
            }
          }
        }
      }
    }

    // ------------------------------------------------------------------
    // 4. Integrate velocity and position
    // ------------------------------------------------------------------
    for (let i = 0; i < N; i++) {
      this.vx[i] = (this.vx[i] + _fx[i] * DT + _dvx[i]) * FRICTION;
      this.vy[i] = (this.vy[i] + _fy[i] * DT + _dvy[i]) * FRICTION;
      this.vz[i] = (this.vz[i] + _fz[i] * DT + _dvz[i]) * FRICTION;
      this.px[i] += this.vx[i] * DT;
      this.py[i] += this.vy[i] * DT;
      this.pz[i] += this.vz[i] * DT;
    }

    // ------------------------------------------------------------------
    // 5. Rebuild grid (post-integration) for collision detection
    // ------------------------------------------------------------------
    this._buildGrid();

    // ------------------------------------------------------------------
    // 6. PBD hard-sphere collision resolution  [O(N) with grid, 2 iters]
    //    Gauss-Seidel sequential relaxation — converges in 2 passes.
    // ------------------------------------------------------------------
    for (let iter = 0; iter < 2; iter++) {
      // Ball–ball
      for (let i = 0; i < N; i++) {
        const cxI = Math.max(0, Math.min(gd - 1, (this.px[i] + BOX) * _ic | 0));
        const cyI = Math.max(0, Math.min(gd - 1, (this.py[i] + BOX) * _ic | 0));
        const czI = Math.max(0, Math.min(gd - 1, (this.pz[i] + BOX) * _ic | 0));

        for (let oz = -1; oz <= 1; oz++) {
          const czN = czI + oz; if (czN < 0 || czN >= gd) continue;
          for (let oy = -1; oy <= 1; oy++) {
            const cyN = cyI + oy; if (cyN < 0 || cyN >= gd) continue;
            for (let ox = -1; ox <= 1; ox++) {
              const cxN = cxI + ox; if (cxN < 0 || cxN >= gd) continue;
              const c   = cxN + cyN * gd + czN * gd * gd;
              const end = _sta[c] + _cnt[c];
              for (let k = _sta[c]; k < end; k++) {
                const j = _idx[k];
                if (j <= i) continue;

                const ddx = this.px[j] - this.px[i];
                const ddy = this.py[j] - this.py[i];
                const ddz = this.pz[j] - this.pz[i];
                const d2  = ddx * ddx + ddy * ddy + ddz * ddz;
                if (d2 >= diam2 || d2 < 1e-10) continue;

                const d    = Math.sqrt(d2);
                const push = (diam - d) * 0.5;
                const nx = ddx / d, ny = ddy / d, nz = ddz / d;

                // Equal-mass position correction
                this.px[i] -= nx * push; this.py[i] -= ny * push; this.pz[i] -= nz * push;
                this.px[j] += nx * push; this.py[j] += ny * push; this.pz[j] += nz * push;

                // Inelastic velocity impulse (restitution = 0.30 feel, heavy fluid)
                const relVn = (this.vx[j] - this.vx[i]) * nx
                            + (this.vy[j] - this.vy[i]) * ny
                            + (this.vz[j] - this.vz[i]) * nz;
                if (relVn < 0) {
                  const imp = relVn * (1 + 0.30) * 0.5;
                  this.vx[i] += imp * nx; this.vy[i] += imp * ny; this.vz[i] += imp * nz;
                  this.vx[j] -= imp * nx; this.vy[j] -= imp * ny; this.vz[j] -= imp * nz;
                }
              }
            }
          }
        }
      }

      // Wall clamp
      for (let i = 0; i < N; i++) {
        if      (this.px[i] < -W) { this.px[i] = -W; if (this.vx[i] < 0) this.vx[i] *= -RESTITUTION; }
        else if (this.px[i] >  W) { this.px[i] =  W; if (this.vx[i] > 0) this.vx[i] *= -RESTITUTION; }
        if      (this.py[i] < -W) { this.py[i] = -W; if (this.vy[i] < 0) this.vy[i] *= -RESTITUTION; }
        else if (this.py[i] >  W) { this.py[i] =  W; if (this.vy[i] > 0) this.vy[i] *= -RESTITUTION; }
        if      (this.pz[i] < -W) { this.pz[i] = -W; if (this.vz[i] < 0) this.vz[i] *= -RESTITUTION; }
        else if (this.pz[i] >  W) { this.pz[i] =  W; if (this.vz[i] > 0) this.vz[i] *= -RESTITUTION; }
      }
    }

    // ------------------------------------------------------------------
    // 7. Speed magnitude — used by renderer for colour mapping
    // ------------------------------------------------------------------
    for (let i = 0; i < N; i++) {
      const vx = this.vx[i], vy = this.vy[i], vz = this.vz[i];
      this.speed[i] = Math.sqrt(vx * vx + vy * vy + vz * vz);
    }
  }

  /** Push particles away from world-space point (cursor repulsion / click burst). */
  repelFrom(x, y, z, str, radius) {
    const r2 = radius * radius;
    for (let i = 0; i < this.N; i++) {
      const dx = this.px[i] - x, dy = this.py[i] - y, dz = this.pz[i] - z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < r2 && d2 > 1e-8) {
        const d = Math.sqrt(d2), f = str * (1 - d / radius) / d;
        this.vx[i] += f * dx; this.vy[i] += f * dy; this.vz[i] += f * dz;
      }
    }
  }

  /** Rescale gravity to new magnitude while preserving its current direction. */
  setGravityStrength(g) {
    const len = Math.sqrt(this.gx * this.gx + this.gy * this.gy + this.gz * this.gz) || 1;
    const sc  = Math.abs(g) / len;
    this.gx *= sc; this.gy *= sc; this.gz *= sc;
  }
}
