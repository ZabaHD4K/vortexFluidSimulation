'use strict';

// ---------------------------------------------------------------------------
// SPH — Smoothed Particle Hydrodynamics (3-D, dual-density SPH + PBD)
//
// Physics model (Clavet-inspired dual-density SPH)
//   - Density estimation — per-particle density from SPH kernel sums
//   - Pressure forces   — repel when compressed (clamped, no tension)
//   - Near-pressure     — short-range repulsion prevents particle clustering
//   - Viscosity         — XSPH velocity-matching between neighbours
//   - Cohesion          — surface tension at the free surface
//   - PBD collisions    — position correction for hard-sphere contacts
//   - Gravity           — configurable direction + strength (tiltable box)
//
// Performance
//   - Uniform spatial grid  — O(N) neighbour queries instead of O(N^2)
//   - Pre-allocated typed arrays everywhere — zero GC pressure per frame
//   - Visual radius decoupled from SPH interaction radius
//
// Public API
//   new SPH(N)        — construct with N particles
//   .reset()          — place particles in a stable pile at bottom
//   .spawnFromTop()   — launch all particles downward from the ceiling
//   .startPour()      — deactivate all particles for gradual pouring
//   .pourTick(count, x, y, z, spread) — activate more particles at a point
//   .step()           — advance one sub-step
//   .repelFrom(x,y,z,str,radius) — push particles away from a world point
//   .setGravityStrength(g)       — rescale gravity magnitude, keep direction
// ---------------------------------------------------------------------------

export class SPH {
  constructor(N = 40) {
    this.N = N;
    this.activeN = N;

    // Visual radius — small spheres, scales with N^(-1/3)
    this.r = 0.025 * Math.cbrt(1500 / N);

    this.BOX = 0.58;
    this.W   = this.BOX - this.r;

    // SPH interaction radius — decoupled from visual radius
    // Calibrated for ~25 neighbours when particles settle in bottom half of box
    this.h = 0.09 * Math.cbrt(1500 / N);

    // Gravity vector — set externally by app3d.js
    this.gx = 0; this.gy = -9; this.gz = 0;

    // Fluid parameters
    this.viscosity   = 0.0005;  // water — nearly zero viscosity
    this.cohesion    = 0.2;     // light surface tension
    this.FRICTION    = 0.996;   // light drag
    this.RESTITUTION = 0.10;    // wall bounce (low = soft splat)

    // SPH pressure parameters
    this.stiffness     = 12;    // far-field pressure
    this.nearStiffness = 8;     // near-field pressure (anti-clustering)
    this._restDensity  = 0;
    this._densityReady = false;

    // Particle state — Float32 for cache-friendly sequential access
    this.px    = new Float32Array(N);
    this.py    = new Float32Array(N);
    this.pz    = new Float32Array(N);
    this.vx    = new Float32Array(N);
    this.vy    = new Float32Array(N);
    this.vz    = new Float32Array(N);
    this.speed = new Float32Array(N);   // magnitude, for colour mapping

    // SPH density fields
    this.density     = new Float32Array(N);
    this.nearDensity = new Float32Array(N);
    this.press       = new Float32Array(N);

    // Per-frame accumulators (reused each step, no allocation)
    this._fx  = new Float32Array(N);
    this._fy  = new Float32Array(N);
    this._fz  = new Float32Array(N);
    this._dvx = new Float32Array(N);
    this._dvy = new Float32Array(N);
    this._dvz = new Float32Array(N);

    // ----- Spatial hash grid -----
    const gd    = Math.ceil(this.BOX * 2 / this.h) + 2;
    this._gd    = gd;
    this._ic    = 1.0 / this.h;
    const total = gd * gd * gd;
    this._cnt   = new Int32Array(total);
    this._sta   = new Int32Array(total);
    this._tmp   = new Int32Array(total);
    this._idx   = new Int32Array(N);
    this._cell  = new Int32Array(N);

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
    const { activeN: N, _gd: gd, _cnt, _sta, _tmp, _idx, _cell } = this;
    const total = gd * gd * gd;

    _cnt.fill(0); _tmp.fill(0);
    for (let i = 0; i < N; i++) {
      const c = this._cellId(this.px[i], this.py[i], this.pz[i]);
      _cell[i] = c;
      _cnt[c]++;
    }

    _sta[0] = 0;
    for (let c = 1; c < total; c++) _sta[c] = _sta[c - 1] + _cnt[c - 1];

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
      const th  = i * golden;
      const rad = 0.04 + t * 0.30;
      this.px[i] = Math.max(-W, Math.min(W, Math.cos(th) * rad * 0.9));
      this.py[i] = Math.max(-W, Math.min(W, -W + t * W * 1.4));
      this.pz[i] = Math.max(-W, Math.min(W, Math.sin(th) * rad * 0.9));
      this.vx[i] = this.vy[i] = this.vz[i] = 0;
      this.speed[i] = 0;
    }
    this.activeN = this.N;
    this._densityReady = false;
  }

  /** Launch all particles from the top of the box with strong downward velocity. */
  spawnFromTop() {
    const W = this.W, golden = 2.3999632;
    for (let i = 0; i < this.N; i++) {
      const t   = i / this.N;
      const th  = i * golden;
      const rad = Math.sqrt(t) * W * 0.40;
      this.px[i] = Math.max(-W * 0.9, Math.min(W * 0.9, Math.cos(th) * rad));
      this.py[i] = W * (0.60 + t * 0.30 + Math.random() * 0.08);
      this.pz[i] = Math.max(-W * 0.9, Math.min(W * 0.9, Math.sin(th) * rad));
      this.vy[i] = -(3.0 + (1.0 - t) * 5.0);
      this.vx[i] = (Math.random() - 0.5) * 0.6;
      this.vz[i] = (Math.random() - 0.5) * 0.6;
      this.speed[i] = 0;
    }
    this.activeN = this.N;
    this._densityReady = false;
  }

  /** Deactivate all particles. Use pourTick() to add them gradually. */
  startPour() {
    for (let i = 0; i < this.N; i++) {
      this.px[i] = 0; this.py[i] = 10; this.pz[i] = 0;
      this.vx[i] = this.vy[i] = this.vz[i] = 0;
      this.speed[i] = 0;
    }
    this.activeN = 0;
    this._densityReady = false;
  }

  /** Activate `count` more particles at position (x,y,z) with small random spread. */
  pourTick(count, x, y, z, spread) {
    const start = this.activeN;
    const end   = Math.min(start + count, this.N);
    for (let i = start; i < end; i++) {
      const angle = Math.random() * Math.PI * 2;
      const rr    = Math.random() * spread;
      this.px[i] = x + Math.cos(angle) * rr;
      this.py[i] = this.W - 0.01;   // spawn at ceiling
      this.pz[i] = z + Math.sin(angle) * rr;
      this.vx[i] = (Math.random() - 0.5) * 0.1;
      this.vy[i] = -(4.0 + Math.random() * 3.0);
      this.vz[i] = (Math.random() - 0.5) * 0.1;
      this.speed[i] = 0;
    }
    this.activeN = end;
    // Keep recalibrating rest density while pouring
    this._densityReady = false;
  }

  /** Advance one physics sub-step. */
  step() {
    const DT = 0.003;
    const { activeN: N, r, W, h, gx, gy, gz,
            viscosity, cohesion, FRICTION, RESTITUTION,
            stiffness, nearStiffness,
            _gd: gd, _ic, _cnt, _sta, _idx } = this;
    if (N < 2) return;

    const diam  = r * 2;
    const diam2 = diam * diam;
    const h2    = h * h;
    const invH  = 1.0 / h;
    const { _fx, _fy, _fz, _dvx, _dvy, _dvz } = this;
    const { density, nearDensity, press } = this;
    const BOX = this.BOX;

    // ------------------------------------------------------------------
    // 1. Gravity seed
    // ------------------------------------------------------------------
    for (let i = 0; i < N; i++) {
      _fx[i] = gx; _fy[i] = gy; _fz[i] = gz;
      _dvx[i] = _dvy[i] = _dvz[i] = 0;
    }

    // ------------------------------------------------------------------
    // 2. Spatial grid
    // ------------------------------------------------------------------
    this._buildGrid();

    // ------------------------------------------------------------------
    // 3. Density estimation
    // ------------------------------------------------------------------
    for (let i = 0; i < N; i++) { density[i] = 1.0; nearDensity[i] = 1.0; }

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
              if (d2 >= h2 || d2 < 1e-10) continue;

              const d  = Math.sqrt(d2);
              const q  = 1.0 - d * invH;
              const q2 = q * q;
              density[i] += q2;     density[j] += q2;
              nearDensity[i] += q2 * q; nearDensity[j] += q2 * q;
            }
          }
        }
      }
    }

    // Calibrate rest density from first frame after reset/spawn/pour
    if (!this._densityReady) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += density[i];
      this._restDensity = sum / N;
      this._densityReady = true;
    }

    // ------------------------------------------------------------------
    // 4. Pressure from density (clamped: no tension -> no collapse)
    // ------------------------------------------------------------------
    const rho0 = this._restDensity;
    for (let i = 0; i < N; i++) {
      press[i] = stiffness * Math.max(-1, density[i] - rho0);
    }

    // ------------------------------------------------------------------
    // 5. Pairwise forces: pressure + viscosity + cohesion
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
              if (j <= i) continue;

              const ddx = this.px[j] - this.px[i];
              const ddy = this.py[j] - this.py[i];
              const ddz = this.pz[j] - this.pz[i];
              const d2  = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d2 >= h2 || d2 < 1e-10) continue;

              const d  = Math.sqrt(d2);
              const nx = ddx / d, ny = ddy / d, nz = ddz / d;
              const q  = 1.0 - d * invH;

              // Pressure gradient (dual-density SPH)
              const nearPi = nearStiffness * nearDensity[i];
              const nearPj = nearStiffness * nearDensity[j];
              const fp = (press[i] + press[j]) * q + (nearPi + nearPj) * q * q;
              _fx[i] -= fp * nx; _fy[i] -= fp * ny; _fz[i] -= fp * nz;
              _fx[j] += fp * nx; _fy[j] += fp * ny; _fz[j] += fp * nz;

              // Viscosity (XSPH)
              const vf  = viscosity * q * DT;
              const dvx = this.vx[j] - this.vx[i];
              const dvy = this.vy[j] - this.vy[i];
              const dvz = this.vz[j] - this.vz[i];
              _dvx[i] += vf * dvx; _dvy[i] += vf * dvy; _dvz[i] += vf * dvz;
              _dvx[j] -= vf * dvx; _dvy[j] -= vf * dvy; _dvz[j] -= vf * dvz;

              // Cohesion (surface tension at free surface)
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
    // 6. Integrate velocity and position
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
    // 7. Rebuild grid for collision detection
    // ------------------------------------------------------------------
    this._buildGrid();

    // ------------------------------------------------------------------
    // 8. PBD hard-sphere collision resolution (1 iteration)
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
              if (j <= i) continue;

              const ddx = this.px[j] - this.px[i];
              const ddy = this.py[j] - this.py[i];
              const ddz = this.pz[j] - this.pz[i];
              const d2  = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d2 >= diam2 || d2 < 1e-10) continue;

              const d    = Math.sqrt(d2);
              const push = (diam - d) * 0.5;
              const nx = ddx / d, ny = ddy / d, nz = ddz / d;

              this.px[i] -= nx * push; this.py[i] -= ny * push; this.pz[i] -= nz * push;
              this.px[j] += nx * push; this.py[j] += ny * push; this.pz[j] += nz * push;

              const relVn = (this.vx[j] - this.vx[i]) * nx
                          + (this.vy[j] - this.vy[i]) * ny
                          + (this.vz[j] - this.vz[i]) * nz;
              if (relVn < 0) {
                const imp = relVn * (1 + RESTITUTION) * 0.5;
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

    // ------------------------------------------------------------------
    // 9. Speed magnitude — used by renderer for colour mapping
    // ------------------------------------------------------------------
    for (let i = 0; i < N; i++) {
      const vx = this.vx[i], vy = this.vy[i], vz = this.vz[i];
      this.speed[i] = Math.sqrt(vx * vx + vy * vy + vz * vz);
    }
  }

  /**
   * Remove particles near a drain hole. Swaps with last active particle.
   * Returns number of particles drained this call.
   */
  drain(cx, cy, cz, radius) {
    const r2 = radius * radius;
    let drained = 0;
    let i = 0;
    while (i < this.activeN) {
      const dx = this.px[i] - cx;
      const dy = this.py[i] - cy;
      const dz = this.pz[i] - cz;
      if (dx * dx + dy * dy + dz * dz < r2) {
        // Swap with last active particle
        const last = this.activeN - 1;
        if (i !== last) {
          this.px[i] = this.px[last]; this.py[i] = this.py[last]; this.pz[i] = this.pz[last];
          this.vx[i] = this.vx[last]; this.vy[i] = this.vy[last]; this.vz[i] = this.vz[last];
          this.speed[i] = this.speed[last];
        }
        this.activeN--;
        drained++;
        // Don't increment i — check the swapped particle
      } else {
        i++;
      }
    }
    return drained;
  }

  /** Push particles away from world-space point. */
  repelFrom(x, y, z, str, radius) {
    const r2 = radius * radius;
    for (let i = 0; i < this.activeN; i++) {
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
