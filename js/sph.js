'use strict';

// Hybrid fluid: PBD hard-sphere collision + SPH viscosity + cohesion.
// Viscosity = velocity matching between neighbours → smooth liquid flow.
// Cohesion   = gentle attraction just beyond contact → blob stays together.
// PBD        = immediate position correction, low restitution → no jitter.

export class SPH {
  constructor(N = 40) {
    this.N   = N;
    this.r   = 0.075;
    this.BOX = 0.58;
    this.W   = this.BOX - this.r;   // effective wall
    this.h   = this.r * 4.2;        // interaction radius (~4 sphere diameters)

    // 3-D gravity (tiltable)
    this.gx = 0; this.gy = -9; this.gz = 0;

    // Fluid parameters (all exposed to UI sliders)
    this.viscosity  = 0.55;  // velocity-matching strength (0=gas, 2=honey)
    this.cohesion   = 3.5;   // attraction just beyond contact (keeps blob together)
    this.FRICTION   = 0.9978; // per-substep air drag
    this.RESTITUTION = 0.10;  // wall bounce coefficient (low = soft)

    this.px    = new Float32Array(N);
    this.py    = new Float32Array(N);
    this.pz    = new Float32Array(N);
    this.vx    = new Float32Array(N);
    this.vy    = new Float32Array(N);
    this.vz    = new Float32Array(N);
    this.speed = new Float32Array(N);

    // Force / velocity-correction accumulators (pre-allocated)
    this._fx  = new Float32Array(N);
    this._fy  = new Float32Array(N);
    this._fz  = new Float32Array(N);
    this._dvx = new Float32Array(N);
    this._dvy = new Float32Array(N);
    this._dvz = new Float32Array(N);

    this.reset();
  }

  reset() {
    const W      = this.W;
    const golden = 2.3999632; // golden-angle spiral for even spacing
    for (let i = 0; i < this.N; i++) {
      const θ   = i * golden;
      const rad = 0.04 + (i / this.N) * 0.30;
      this.px[i] = Math.max(-W, Math.min(W, Math.cos(θ) * rad * 0.9));
      this.py[i] = Math.max(-W, Math.min(W, -W + (i / this.N) * W * 1.4));
      this.pz[i] = Math.max(-W, Math.min(W, Math.sin(θ) * rad * 0.9));
      this.vx[i] = this.vy[i] = this.vz[i] = 0;
      this.speed[i] = 0;
    }
  }

  step() {
    const DT   = 0.0055; // ≈ 1/(3×60) — three substeps per frame
    const {N, r, W, h, gx, gy, gz, viscosity, cohesion, FRICTION, RESTITUTION} = this;
    const diam = r * 2;
    const diam2 = diam * diam;
    const h2   = h * h;
    const {_fx, _fy, _fz, _dvx, _dvy, _dvz} = this;

    // ---- 1. Gravity --------------------------------------------------------
    for (let i = 0; i < N; i++) {
      _fx[i] = gx; _fy[i] = gy; _fz[i] = gz;
      _dvx[i] = _dvy[i] = _dvz[i] = 0;
    }

    // ---- 2. Pairwise: viscosity + cohesion --------------------------------
    for (let i = 0; i < N - 1; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = this.px[j]-this.px[i];
        const dy = this.py[j]-this.py[i];
        const dz = this.pz[j]-this.pz[i];
        const d2 = dx*dx + dy*dy + dz*dz;
        if (d2 >= h2 || d2 < 1e-10) continue;

        const d  = Math.sqrt(d2);
        const nx = dx/d, ny = dy/d, nz = dz/d;
        const q  = 1.0 - d/h; // 0→1 as distance decreases

        // Viscosity: accumulate velocity-matching correction
        const vf  = viscosity * q * DT;
        const dvx = this.vx[j]-this.vx[i];
        const dvy = this.vy[j]-this.vy[i];
        const dvz = this.vz[j]-this.vz[i];
        _dvx[i] += vf * dvx; _dvy[i] += vf * dvy; _dvz[i] += vf * dvz;
        _dvx[j] -= vf * dvx; _dvy[j] -= vf * dvy; _dvz[j] -= vf * dvz;

        // Cohesion: gentle attraction when beyond contact but within h
        if (d > diam) {
          const cf_q = q * q; // stronger when closer
          const cf = cohesion * cf_q;
          _fx[i] += cf * nx; _fy[i] += cf * ny; _fz[i] += cf * nz;
          _fx[j] -= cf * nx; _fy[j] -= cf * ny; _fz[j] -= cf * nz;
        }
      }
    }

    // ---- 3. Integrate velocity + apply corrections -------------------------
    for (let i = 0; i < N; i++) {
      this.vx[i] = (this.vx[i] + _fx[i] * DT + _dvx[i]) * FRICTION;
      this.vy[i] = (this.vy[i] + _fy[i] * DT + _dvy[i]) * FRICTION;
      this.vz[i] = (this.vz[i] + _fz[i] * DT + _dvz[i]) * FRICTION;

      this.px[i] += this.vx[i] * DT;
      this.py[i] += this.vy[i] * DT;
      this.pz[i] += this.vz[i] * DT;
    }

    // ---- 4. PBD collision resolution (2 iterations for stability) ---------
    for (let iter = 0; iter < 2; iter++) {
      // Ball-ball
      for (let i = 0; i < N - 1; i++) {
        for (let j = i + 1; j < N; j++) {
          const dx = this.px[j]-this.px[i];
          const dy = this.py[j]-this.py[i];
          const dz = this.pz[j]-this.pz[i];
          const d2 = dx*dx + dy*dy + dz*dz;
          if (d2 >= diam2 || d2 < 1e-10) continue;
          const d    = Math.sqrt(d2);
          const push = (diam - d) * 0.5;
          const nx = dx/d, ny = dy/d, nz = dz/d;
          // Position correction
          this.px[i] -= nx*push; this.py[i] -= ny*push; this.pz[i] -= nz*push;
          this.px[j] += nx*push; this.py[j] += ny*push; this.pz[j] += nz*push;
          // Velocity impulse — low restitution (0.3), heavy inelastic feel
          const relVn = (this.vx[j]-this.vx[i])*nx
                      + (this.vy[j]-this.vy[i])*ny
                      + (this.vz[j]-this.vz[i])*nz;
          if (relVn < 0) {
            const imp = relVn * (1 + 0.30) * 0.5;
            this.vx[i] += imp*nx; this.vy[i] += imp*ny; this.vz[i] += imp*nz;
            this.vx[j] -= imp*nx; this.vy[j] -= imp*ny; this.vz[j] -= imp*nz;
          }
        }
      }

      // Walls
      for (let i = 0; i < N; i++) {
        if (this.px[i] < -W) { this.px[i]=-W; if(this.vx[i]<0) this.vx[i]*=-RESTITUTION; }
        else if(this.px[i] > W) { this.px[i]=W;  if(this.vx[i]>0) this.vx[i]*=-RESTITUTION; }
        if (this.py[i] < -W) { this.py[i]=-W; if(this.vy[i]<0) this.vy[i]*=-RESTITUTION; }
        else if(this.py[i] > W) { this.py[i]=W;  if(this.vy[i]>0) this.vy[i]*=-RESTITUTION; }
        if (this.pz[i] < -W) { this.pz[i]=-W; if(this.vz[i]<0) this.vz[i]*=-RESTITUTION; }
        else if(this.pz[i] > W) { this.pz[i]=W;  if(this.vz[i]>0) this.vz[i]*=-RESTITUTION; }
      }
    }

    // ---- 5. Speed for color -----------------------------------------------
    for (let i = 0; i < N; i++) {
      const vx=this.vx[i], vy=this.vy[i], vz=this.vz[i];
      this.speed[i] = Math.sqrt(vx*vx + vy*vy + vz*vz);
    }
  }

  repelFrom(x, y, z, str, radius) {
    const r2 = radius * radius;
    for (let i = 0; i < this.N; i++) {
      const dx=this.px[i]-x, dy=this.py[i]-y, dz=this.pz[i]-z;
      const d2=dx*dx+dy*dy+dz*dz;
      if (d2 < r2 && d2 > 1e-8) {
        const d=Math.sqrt(d2), f=str*(1-d/radius)/d;
        this.vx[i]+=f*dx; this.vy[i]+=f*dy; this.vz[i]+=f*dz;
      }
    }
  }

  setGravityStrength(g) {
    const len = Math.sqrt(this.gx*this.gx + this.gy*this.gy + this.gz*this.gz) || 1;
    const sc  = Math.abs(g) / len;
    this.gx *= sc; this.gy *= sc; this.gz *= sc;
  }
}
