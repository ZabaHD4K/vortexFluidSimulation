'use strict';

// ---------------------------------------------------------------------------
// SPH — WebGPU Compute Engine
// Dual-density SPH running entirely on the GPU via compute shaders.
// Same physics as sph.js but parallelised for 10K-50K particles.
// ---------------------------------------------------------------------------

const WG = 64; // workgroup size

// ---- WGSL Shader Module (all compute passes in one module) ----
const WGSL = /* wgsl */`

struct Params {
  activeN:       u32,
  gridDim:       u32,
  gridTotal:     u32,
  _pad0:         u32,
  r:             f32,
  h:             f32,
  invH:          f32,
  h2:            f32,
  W:             f32,
  BOX:           f32,
  diam:          f32,
  diam2:         f32,
  gx:            f32,
  gy:            f32,
  gz:            f32,
  DT:            f32,
  viscosity:     f32,
  cohesion:      f32,
  friction:      f32,
  restitution:   f32,
  stiffness:     f32,
  nearStiffness: f32,
  restDensity:   f32,
  drainPull:     f32,
  repelX:        f32,
  repelY:        f32,
  repelZ:        f32,
  repelStrength: f32,
  repelRadius:   f32,
  drainX:        f32,
  drainY:        f32,
  drainZ:        f32,
};

@group(0) @binding(0)  var<uniform>             params:       Params;
@group(0) @binding(1)  var<storage, read_write>  pos:          array<vec4f>;
@group(0) @binding(2)  var<storage, read_write>  vel:          array<vec4f>;
@group(0) @binding(3)  var<storage, read_write>  force:        array<vec4f>;
@group(0) @binding(4)  var<storage, read_write>  densityArr:   array<f32>;
@group(0) @binding(5)  var<storage, read_write>  nearDenArr:   array<f32>;
@group(0) @binding(6)  var<storage, read_write>  speedArr:     array<f32>;
@group(0) @binding(7)  var<storage, read_write>  cellCount:    array<atomic<u32>>;
@group(0) @binding(8)  var<storage, read_write>  cellStart:    array<u32>;
@group(0) @binding(9)  var<storage, read_write>  sortedIdx:    array<u32>;

fn getCellId(x: f32, y: f32, z: f32) -> u32 {
  let gd = params.gridDim;
  let ic = params.invH;
  let cx = min(u32(max(0.0, (x + params.BOX) * ic)), gd - 1u);
  let cy = min(u32(max(0.0, (y + params.BOX) * ic)), gd - 1u);
  let cz = min(u32(max(0.0, (z + params.BOX) * ic)), gd - 1u);
  return cx + cy * gd + cz * gd * gd;
}

// ---- 1. Clear grid counters ----
@compute @workgroup_size(${WG})
fn clearGrid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.gridTotal) { return; }
  atomicStore(&cellCount[i], 0u);
}

// ---- 2. Count particles per cell ----
@compute @workgroup_size(${WG})
fn buildGridCount(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.activeN) { return; }
  let p = pos[i];
  let c = getCellId(p.x, p.y, p.z);
  atomicAdd(&cellCount[c], 1u);
}

// ---- 3. Prefix sum (serial — fine for <100K cells) ----
@compute @workgroup_size(1)
fn prefixSum() {
  var sum = 0u;
  let gt = params.gridTotal;
  for (var i = 0u; i < gt; i++) {
    let cnt = atomicLoad(&cellCount[i]);
    cellStart[i] = sum;
    sum += cnt;
    atomicStore(&cellCount[i], 0u);   // reset for scatter
  }
  cellStart[gt] = sum;                // sentinel
}

// ---- 4. Scatter particles into sorted order ----
@compute @workgroup_size(${WG})
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.activeN) { return; }
  let p = pos[i];
  let c = getCellId(p.x, p.y, p.z);
  let offset = atomicAdd(&cellCount[c], 1u);
  sortedIdx[cellStart[c] + offset] = i;
}

// ---- 5. Density estimation (dual-density SPH kernels) ----
@compute @workgroup_size(${WG})
fn computeDensity(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.activeN) { return; }
  let pi = pos[i];
  var d: f32 = 1.0;
  var dn: f32 = 1.0;
  let gd = params.gridDim;
  let ic = params.invH;
  let cxI = min(u32(max(0.0, (pi.x + params.BOX) * ic)), gd - 1u);
  let cyI = min(u32(max(0.0, (pi.y + params.BOX) * ic)), gd - 1u);
  let czI = min(u32(max(0.0, (pi.z + params.BOX) * ic)), gd - 1u);

  for (var oz: i32 = -1; oz <= 1; oz++) {
    let czN = i32(czI) + oz;
    if (czN < 0 || czN >= i32(gd)) { continue; }
    for (var oy: i32 = -1; oy <= 1; oy++) {
      let cyN = i32(cyI) + oy;
      if (cyN < 0 || cyN >= i32(gd)) { continue; }
      for (var ox: i32 = -1; ox <= 1; ox++) {
        let cxN = i32(cxI) + ox;
        if (cxN < 0 || cxN >= i32(gd)) { continue; }
        let c = u32(cxN) + u32(cyN) * gd + u32(czN) * gd * gd;
        let cS = cellStart[c];
        let cE = cellStart[c + 1u];
        for (var k = cS; k < cE; k++) {
          let j = sortedIdx[k];
          if (j == i) { continue; }
          let pj = pos[j];
          let dx = pj.x - pi.x;
          let dy = pj.y - pi.y;
          let dz = pj.z - pi.z;
          let dist2 = dx*dx + dy*dy + dz*dz;
          if (dist2 >= params.h2 || dist2 < 1e-10) { continue; }
          let dist = sqrt(dist2);
          let q = 1.0 - dist * params.invH;
          let q2 = q * q;
          d  += q2;
          dn += q2 * q;
        }
      }
    }
  }
  densityArr[i] = d;
  nearDenArr[i] = dn;
}

// ---- 6. Forces: gravity + pressure + viscosity + cohesion + drain pull ----
@compute @workgroup_size(${WG})
fn computeForces(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.activeN) { return; }
  let pi = pos[i];
  let vi = vel[i];
  let di = densityArr[i];
  let dni = nearDenArr[i];
  let prI = params.stiffness * max(-1.0, di - params.restDensity);

  var fx = params.gx;
  var fy = params.gy;
  var fz = params.gz;

  let gd = params.gridDim;
  let ic = params.invH;
  let cxI = min(u32(max(0.0, (pi.x + params.BOX) * ic)), gd - 1u);
  let cyI = min(u32(max(0.0, (pi.y + params.BOX) * ic)), gd - 1u);
  let czI = min(u32(max(0.0, (pi.z + params.BOX) * ic)), gd - 1u);

  for (var oz: i32 = -1; oz <= 1; oz++) {
    let czN = i32(czI) + oz;
    if (czN < 0 || czN >= i32(gd)) { continue; }
    for (var oy: i32 = -1; oy <= 1; oy++) {
      let cyN = i32(cyI) + oy;
      if (cyN < 0 || cyN >= i32(gd)) { continue; }
      for (var ox: i32 = -1; ox <= 1; ox++) {
        let cxN = i32(cxI) + ox;
        if (cxN < 0 || cxN >= i32(gd)) { continue; }
        let c = u32(cxN) + u32(cyN) * gd + u32(czN) * gd * gd;
        let cS = cellStart[c];
        let cE = cellStart[c + 1u];
        for (var k = cS; k < cE; k++) {
          let j = sortedIdx[k];
          if (j == i) { continue; }
          let pj = pos[j];
          let vj = vel[j];
          let ddx = pj.x - pi.x;
          let ddy = pj.y - pi.y;
          let ddz = pj.z - pi.z;
          let dist2 = ddx*ddx + ddy*ddy + ddz*ddz;
          if (dist2 >= params.h2 || dist2 < 1e-10) { continue; }
          let dist = sqrt(dist2);
          let nx = ddx / dist;
          let ny = ddy / dist;
          let nz = ddz / dist;
          let q = 1.0 - dist * params.invH;

          // Pressure (dual-density)
          let prJ = params.stiffness * max(-1.0, densityArr[j] - params.restDensity);
          let nPi = params.nearStiffness * dni;
          let nPj = params.nearStiffness * nearDenArr[j];
          let fp = (prI + prJ) * q + (nPi + nPj) * q * q;
          fx -= fp * nx;
          fy -= fp * ny;
          fz -= fp * nz;

          // Viscosity (as force)
          let vf = params.viscosity * q;
          fx += vf * (vj.x - vi.x);
          fy += vf * (vj.y - vi.y);
          fz += vf * (vj.z - vi.z);

          // Cohesion
          if (dist > params.diam) {
            let cf = params.cohesion * q * q;
            fx += cf * nx;
            fy += cf * ny;
            fz += cf * nz;
          }
        }
      }
    }
  }

  // Drain pull toward center
  if (params.drainPull > 0.0) {
    let dx = params.drainX - pi.x;
    let dz = params.drainZ - pi.z;
    let dist = sqrt(dx*dx + dz*dz) + 0.01;
    let pull = params.drainPull / dist;
    fx += dx * pull;
    fz += dz * pull;
  }

  force[i] = vec4f(fx, fy, fz, 0.0);
}

// ---- 7. Integrate + wall clamp + speed ----
@compute @workgroup_size(${WG})
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.activeN) { return; }
  var v = vel[i];
  let f = force[i];
  let DT = params.DT;
  let FR = params.friction;
  let RE = params.restitution;
  let W  = params.W;

  v = vec4f(
    (v.x + f.x * DT) * FR,
    (v.y + f.y * DT) * FR,
    (v.z + f.z * DT) * FR,
    0.0
  );

  var p = pos[i];
  p = vec4f(p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT, 0.0);

  if (p.x < -W) { p.x = -W; if (v.x < 0.0) { v.x *= -RE; } }
  else if (p.x >  W) { p.x =  W; if (v.x > 0.0) { v.x *= -RE; } }
  if (p.y < -W) { p.y = -W; if (v.y < 0.0) { v.y *= -RE; } }
  else if (p.y >  W) { p.y =  W; if (v.y > 0.0) { v.y *= -RE; } }
  if (p.z < -W) { p.z = -W; if (v.z < 0.0) { v.z *= -RE; } }
  else if (p.z >  W) { p.z =  W; if (v.z > 0.0) { v.z *= -RE; } }

  pos[i] = p;
  vel[i] = v;
  speedArr[i] = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

// ---- 8. Cursor repulsion ----
@compute @workgroup_size(${WG})
fn repelFrom(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.activeN || params.repelStrength == 0.0) { return; }
  let p = pos[i];
  let dx = p.x - params.repelX;
  let dy = p.y - params.repelY;
  let dz = p.z - params.repelZ;
  let d2 = dx*dx + dy*dy + dz*dz;
  let r2 = params.repelRadius * params.repelRadius;
  if (d2 < r2 && d2 > 1e-8) {
    let d = sqrt(d2);
    let f = params.repelStrength * (1.0 - d / params.repelRadius) / d;
    var v = vel[i];
    v = vec4f(v.x + f*dx, v.y + f*dy, v.z + f*dz, 0.0);
    vel[i] = v;
  }
}
`;

// ---------------------------------------------------------------------------
// SPHCompute class
// ---------------------------------------------------------------------------
export class SPHCompute {

  static async isSupported() {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    return !!adapter;
  }

  async init(N) {
    this.N = N;
    this.activeN = 0;

    // Physics params (same as CPU SPH)
    const BOX = 0.58;
    this.r            = 0.025 * Math.cbrt(1500 / N);
    this.h            = 0.15 * Math.cbrt(1500 / N);
    this.BOX          = BOX;
    this.W            = BOX - this.r;
    this.viscosity    = 0.0005;
    this.cohesion     = 0.2;
    this.friction     = 0.996;
    this.restitution  = 0.10;
    this.stiffness    = 12;
    this.nearStiffness = 8;
    this.restDensity  = 0;
    this._densityReady = false;
    this.DT           = 0.003;

    this.gridDim   = Math.ceil(BOX * 2 / this.h) + 2;
    this.gridTotal = this.gridDim ** 3;

    // CPU readback arrays
    this.px    = new Float32Array(N);
    this.py    = new Float32Array(N);
    this.pz    = new Float32Array(N);
    this.speed = new Float32Array(N);

    // Gravity
    this.gx = 0; this.gy = -5; this.gz = 0;

    // Drain state
    this.drainPull = 0;
    this.drainX = 0; this.drainY = 0; this.drainZ = 0;

    // Repel state
    this.repelX = 0; this.repelY = 0; this.repelZ = 0;
    this.repelStrength = 0; this.repelRadius = 0;

    // ---- WebGPU setup ----
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    this.device = await adapter.requestDevice();
    this.queue  = this.device.queue;

    // Log GPU
    const info = await adapter.requestAdapterInfo();
    console.log(`WebGPU: ${info.vendor} ${info.device} (${info.description})`);

    this._createBuffers();
    this._createPipelines();
    this._createBindGroup();
  }

  _createBuffers() {
    const d = this.device;
    const N = this.N;
    const GT = this.gridTotal;
    const S  = GPUBufferUsage.STORAGE;
    const CS = GPUBufferUsage.COPY_SRC;
    const CD = GPUBufferUsage.COPY_DST;

    this.posBuffer         = d.createBuffer({ size: N * 16,      usage: S | CS | CD, label: 'pos' });
    this.velBuffer         = d.createBuffer({ size: N * 16,      usage: S | CD,      label: 'vel' });
    this.forceBuffer       = d.createBuffer({ size: N * 16,      usage: S,           label: 'force' });
    this.densityBuffer     = d.createBuffer({ size: N * 4,       usage: S | CS,      label: 'density' });
    this.nearDensityBuffer = d.createBuffer({ size: N * 4,       usage: S,           label: 'nearDensity' });
    this.speedBuffer       = d.createBuffer({ size: N * 4,       usage: S | CS,      label: 'speed' });
    this.cellCountBuffer   = d.createBuffer({ size: GT * 4,      usage: S,           label: 'cellCount' });
    this.cellStartBuffer   = d.createBuffer({ size: (GT + 1) * 4, usage: S,          label: 'cellStart' });
    this.sortedIdxBuffer   = d.createBuffer({ size: N * 4,       usage: S,           label: 'sortedIdx' });

    this.paramsBuffer = d.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | CD, label: 'params' });

    // Readback buffers
    const MR = GPUBufferUsage.MAP_READ;
    this.readbackPos     = d.createBuffer({ size: N * 16, usage: MR | CD, label: 'rb_pos' });
    this.readbackSpeed   = d.createBuffer({ size: N * 4,  usage: MR | CD, label: 'rb_speed' });
    this.readbackDensity = d.createBuffer({ size: N * 4,  usage: MR | CD, label: 'rb_density' });
  }

  _createPipelines() {
    const d = this.device;
    const module = d.createShaderModule({ code: WGSL, label: 'SPH' });

    // Explicit bind group layout (all passes share same layout)
    this.bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ]
    });

    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    const make = (entry) => d.createComputePipeline({ layout, compute: { module, entryPoint: entry } });

    this.pipelines = {
      clearGrid:      make('clearGrid'),
      buildGridCount: make('buildGridCount'),
      prefixSum:      make('prefixSum'),
      scatter:        make('scatter'),
      density:        make('computeDensity'),
      forces:         make('computeForces'),
      integrate:      make('integrate'),
      repel:          make('repelFrom'),
    };
  }

  _createBindGroup() {
    this.bindGroup = this.device.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.posBuffer } },
        { binding: 2, resource: { buffer: this.velBuffer } },
        { binding: 3, resource: { buffer: this.forceBuffer } },
        { binding: 4, resource: { buffer: this.densityBuffer } },
        { binding: 5, resource: { buffer: this.nearDensityBuffer } },
        { binding: 6, resource: { buffer: this.speedBuffer } },
        { binding: 7, resource: { buffer: this.cellCountBuffer } },
        { binding: 8, resource: { buffer: this.cellStartBuffer } },
        { binding: 9, resource: { buffer: this.sortedIdxBuffer } },
      ]
    });
  }

  // Upload the params uniform buffer
  _uploadParams() {
    const buf = new ArrayBuffer(128);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0]  = this.activeN;
    u[1]  = this.gridDim;
    u[2]  = this.gridTotal;
    u[3]  = 0;
    f[4]  = this.r;
    f[5]  = this.h;
    f[6]  = 1.0 / this.h;
    f[7]  = this.h * this.h;
    f[8]  = this.W;
    f[9]  = this.BOX;
    f[10] = this.r * 2;
    f[11] = this.r * 2 * this.r * 2;
    f[12] = this.gx;
    f[13] = this.gy;
    f[14] = this.gz;
    f[15] = this.DT;
    f[16] = this.viscosity;
    f[17] = this.cohesion;
    f[18] = this.friction;
    f[19] = this.restitution;
    f[20] = this.stiffness;
    f[21] = this.nearStiffness;
    f[22] = this.restDensity;
    f[23] = this.drainPull;
    f[24] = this.repelX;
    f[25] = this.repelY;
    f[26] = this.repelZ;
    f[27] = this.repelStrength;
    f[28] = this.repelRadius;
    f[29] = this.drainX;
    f[30] = this.drainY;
    f[31] = this.drainZ;
    this.queue.writeBuffer(this.paramsBuffer, 0, buf);
  }

  // ---- Public API ----

  /** Start empty (for pour mode). */
  startPour() {
    this.activeN = 0;
    this._densityReady = false;
  }

  /** Add particles at (x,y,z) with small spread. */
  pourTick(count, x, y, z, spread) {
    const start = this.activeN;
    const end   = Math.min(start + count, this.N);
    const n     = end - start;
    if (n <= 0) return;

    const posData = new Float32Array(n * 4);
    const velData = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const angle = Math.random() * Math.PI * 2;
      const rr    = Math.random() * spread;
      posData[i * 4]     = x + Math.cos(angle) * rr;
      posData[i * 4 + 1] = y + (Math.random() - 0.5) * 0.02;
      posData[i * 4 + 2] = z + Math.sin(angle) * rr;
      posData[i * 4 + 3] = 0;
      velData[i * 4]     = (Math.random() - 0.5) * 0.15;
      velData[i * 4 + 1] = -(2.0 + Math.random() * 2.0);
      velData[i * 4 + 2] = (Math.random() - 0.5) * 0.15;
      velData[i * 4 + 3] = 0;
    }
    this.queue.writeBuffer(this.posBuffer, start * 16, posData);
    this.queue.writeBuffer(this.velBuffer, start * 16, velData);
    this.activeN = end;
    this._densityReady = false;
  }

  /** Remove particles near (cx,cy,cz). Uses CPU readback data. */
  drain(cx, cy, cz, radius) {
    const r2 = radius * radius;
    let i = 0;
    while (i < this.activeN) {
      const dx = this.px[i] - cx;
      const dy = this.py[i] - cy;
      const dz = this.pz[i] - cz;
      if (dx * dx + dy * dy + dz * dz < r2) {
        const last = this.activeN - 1;
        if (i !== last) {
          // Swap last particle to position i on GPU (pos + vel)
          const pData = new Float32Array([this.px[last], this.py[last], this.pz[last], 0]);
          this.queue.writeBuffer(this.posBuffer, i * 16, pData);
          // vel: we don't have vel on CPU, write zero vel (particle will re-accelerate)
          this.queue.writeBuffer(this.velBuffer, i * 16, new Float32Array(4));
          // Swap CPU readback too
          this.px[i] = this.px[last];
          this.py[i] = this.py[last];
          this.pz[i] = this.pz[last];
          this.speed[i] = this.speed[last];
        }
        this.activeN--;
      } else {
        i++;
      }
    }
  }

  /** Set cursor repulsion for next step. */
  setRepel(x, y, z, strength, radius) {
    this.repelX = x; this.repelY = y; this.repelZ = z;
    this.repelStrength = strength; this.repelRadius = radius;
  }

  /** Clear repulsion after step. */
  clearRepel() {
    this.repelStrength = 0;
  }

  /** Dispatch all compute passes for N substeps. */
  step(substeps = 4) {
    if (this.activeN < 2) return;
    this._uploadParams();

    const enc = this.device.createCommandEncoder();
    const wg  = (n) => Math.ceil(n / WG);
    const N   = this.activeN;
    const GT  = this.gridTotal;

    const dispatch = (pipeline, groups) => {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(groups);
      pass.end();
    };

    for (let s = 0; s < substeps; s++) {
      dispatch(this.pipelines.clearGrid,      wg(GT));
      dispatch(this.pipelines.buildGridCount, wg(N));
      dispatch(this.pipelines.prefixSum,      1);
      dispatch(this.pipelines.scatter,        wg(N));
      dispatch(this.pipelines.density,        wg(N));
      dispatch(this.pipelines.forces,         wg(N));
      dispatch(this.pipelines.integrate,      wg(N));
    }

    // Repel (once after all substeps)
    if (this.repelStrength > 0) {
      dispatch(this.pipelines.repel, wg(N));
    }

    // Copy to readback
    enc.copyBufferToBuffer(this.posBuffer,   0, this.readbackPos,   0, N * 16);
    enc.copyBufferToBuffer(this.speedBuffer, 0, this.readbackSpeed, 0, N * 4);

    // Also copy density for calibration if needed
    if (!this._densityReady && this.activeN > 20) {
      enc.copyBufferToBuffer(this.densityBuffer, 0, this.readbackDensity, 0, N * 4);
    }

    this.queue.submit([enc.finish()]);
  }

  /** Read back positions + speeds to CPU. Calibrates restDensity on first call. */
  async readback() {
    const N = this.activeN;
    if (N < 1) return;

    const promises = [
      this.readbackPos.mapAsync(GPUMapMode.READ, 0, N * 16),
      this.readbackSpeed.mapAsync(GPUMapMode.READ, 0, N * 4),
    ];
    const needCalibrate = !this._densityReady && N > 20;
    if (needCalibrate) {
      promises.push(this.readbackDensity.mapAsync(GPUMapMode.READ, 0, N * 4));
    }
    await Promise.all(promises);

    // Positions (deinterleave vec4 → px, py, pz)
    const posSrc = new Float32Array(this.readbackPos.getMappedRange(0, N * 16));
    for (let i = 0; i < N; i++) {
      this.px[i]    = posSrc[i * 4];
      this.py[i]    = posSrc[i * 4 + 1];
      this.pz[i]    = posSrc[i * 4 + 2];
    }
    this.readbackPos.unmap();

    // Speed
    const spdSrc = new Float32Array(this.readbackSpeed.getMappedRange(0, N * 4));
    this.speed.set(spdSrc.subarray(0, N));
    this.readbackSpeed.unmap();

    // Density calibration
    if (needCalibrate) {
      const denSrc = new Float32Array(this.readbackDensity.getMappedRange(0, N * 4));
      let sum = 0;
      for (let i = 0; i < N; i++) sum += denSrc[i];
      this.restDensity = sum / N;
      this._densityReady = true;
      this.readbackDensity.unmap();
    }
  }

  /** Place particles in bottom pile (like CPU SPH.reset). */
  reset() {
    const N = this.N;
    const W = this.W;
    const golden = 2.3999632;
    const posData = new Float32Array(N * 4);
    const velData = new Float32Array(N * 4); // all zero

    for (let i = 0; i < N; i++) {
      const t   = i / N;
      const th  = i * golden;
      const rad = 0.04 + t * 0.30;
      posData[i * 4]     = Math.max(-W, Math.min(W, Math.cos(th) * rad * 0.9));
      posData[i * 4 + 1] = Math.max(-W, Math.min(W, -W + t * W * 1.4));
      posData[i * 4 + 2] = Math.max(-W, Math.min(W, Math.sin(th) * rad * 0.9));
    }

    this.queue.writeBuffer(this.posBuffer, 0, posData);
    this.queue.writeBuffer(this.velBuffer, 0, velData);
    this.activeN = N;
    this._densityReady = false;
  }

  setGravity(gx, gy, gz) {
    this.gx = gx; this.gy = gy; this.gz = gz;
  }
}
