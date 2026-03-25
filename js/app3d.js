import * as THREE from 'three';
import { OrbitControls }   from 'three/addons/controls/OrbitControls.js';
import { EffectComposer }  from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SPH }             from './sph.js';

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
const BOX     = 0.58;

let useGPU    = false;
let N         = 750;
let sim       = new SPH(N);
sim.startPour();
sim.activeN = 0;
let colorMode = 0;
let paused    = false;
let hidden    = false;
let gravBase  = 5;
let gravTiltX = 0;
let gravTiltZ = 0;

// Shake state
let shakeTimer = 0;
let shakeAngle = 0;

// Pour state
let pouring  = false;
// Drain state
let draining = false;
const POUR_RATE   = 4;
const POUR_Y      = 0.55;
const POUR_SPREAD = 0.02;

// GPU async guard
let computing = false;
let GPU_POUR_RATE = 40;

// ---------------------------------------------------------------------------
// GRAVITY HELPER
// ---------------------------------------------------------------------------
function applyGravity(gyOverride = null) {
  const len = Math.sqrt(gravTiltX * gravTiltX + 1.0 + gravTiltZ * gravTiltZ);
  const s   = gravBase / len;
  const gx = gravTiltX * s;
  const gy = (gyOverride !== null) ? gyOverride : -s;
  const gz = gravTiltZ * s;
  if (useGPU) {
    sim.setGravity(gx, gy, gz);
  } else {
    sim.gx = gx; sim.gy = gy; sim.gz = gz;
  }
}
applyGravity();

// ---------------------------------------------------------------------------
// RENDERER
// ---------------------------------------------------------------------------
const canvas   = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: 'high-performance',
  stencil: false,
  depth: true,
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping         = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.5;
renderer.outputColorSpace    = THREE.SRGBColorSpace;

// Log GPU in use
const glCtx = renderer.getContext();
const dbg = glCtx.getExtension('WEBGL_debug_renderer_info');
if (dbg) console.log('GPU:', glCtx.getParameter(dbg.UNMASKED_RENDERER_WEBGL));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020508);
scene.fog = new THREE.FogExp2(0x030610, 0.10);

// ---------------------------------------------------------------------------
// CAMERA
// ---------------------------------------------------------------------------
const camera = new THREE.PerspectiveCamera(32, innerWidth / innerHeight, 0.01, 50);
camera.position.set(2.8, 2.2, 2.8);
camera.lookAt(0, -0.1, 0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance   = 1.0;
controls.maxDistance   = 8.0;
controls.minPolarAngle = Math.PI / 8;
controls.maxPolarAngle = Math.PI / 2.1;
controls.target.set(0, -0.1, 0);
controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };

// ---------------------------------------------------------------------------
// LIGHTING
// ---------------------------------------------------------------------------
scene.add(new THREE.AmbientLight(0x0a1520, 5.0));

const sunLight = new THREE.DirectionalLight(0x99ddff, 4.0);
sunLight.position.set(2, 5, 2);
scene.add(sunLight);

const fillLight = new THREE.DirectionalLight(0x002244, 1.5);
fillLight.position.set(-3, -1, -2);
scene.add(fillLight);

const causticA = new THREE.PointLight(0x22aaff, 6.0, 3.5);
causticA.position.set(0.3, 0.3, 0.3);
scene.add(causticA);

const causticB = new THREE.PointLight(0x0044dd, 4.0, 3.5);
causticB.position.set(-0.3, -0.2, -0.2);
scene.add(causticB);

// ---------------------------------------------------------------------------
// CONTAINER BOX (tiltable group)
// ---------------------------------------------------------------------------
const roomGroup = new THREE.Group();
scene.add(roomGroup);

(function buildRoom() {
  const W = BOX * 2;

  roomGroup.add(new THREE.Mesh(
    new THREE.BoxGeometry(W, W, W),
    new THREE.MeshStandardMaterial({
      color: 0x0b1510, roughness: 0.94, metalness: 0.04, side: THREE.BackSide,
    })
  ));

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(W, W),
    new THREE.MeshStandardMaterial({ color: 0x0e1a12, roughness: 0.90, metalness: 0.06 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -BOX + 0.001;
  roomGroup.add(floor);

  const lines = [];
  for (let t = 0; t <= 8; t++) {
    const v = -BOX + (t / 8) * W;
    lines.push(-BOX, -BOX + 0.002, v,    BOX, -BOX + 0.002, v);
    lines.push(v,    -BOX + 0.002, -BOX,  v,  -BOX + 0.002,  BOX);
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(lines), 3));
  roomGroup.add(new THREE.LineSegments(lg,
    new THREE.LineBasicMaterial({ color: 0x162a1a, transparent: true, opacity: 0.55 })));

  roomGroup.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(W, W, W)),
    new THREE.LineBasicMaterial({ color: 0x1a4a2a, transparent: true, opacity: 0.45 })
  ));

  const dotG = new THREE.SphereGeometry(0.010, 6, 5);
  const dotM = new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.55 });
  for (const x of [-BOX, BOX])
    for (const y of [-BOX, BOX])
      for (const z of [-BOX, BOX]) {
        const m = new THREE.Mesh(dotG, dotM);
        m.position.set(x, y, z);
        roomGroup.add(m);
      }
})();

// ---------------------------------------------------------------------------
// FUNNEL — visible during pour
// ---------------------------------------------------------------------------
const funnelGroup = new THREE.Group();

(function buildFunnel() {
  const funnelMat = new THREE.MeshStandardMaterial({
    color: 0x44aacc, transparent: true, opacity: 0.12,
    side: THREE.DoubleSide, roughness: 0.1, metalness: 0.5
  });

  // Truncated cone: wide top (mouth) → narrow bottom (spout)
  const coneGeo = new THREE.CylinderGeometry(0.03, 0.18, 0.22, 12, 1, true);
  funnelGroup.add(new THREE.Mesh(coneGeo, funnelMat));

  // Wireframe outline
  funnelGroup.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(coneGeo),
    new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.35 })
  ));

  // Narrow spout cylinder
  const spoutGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.06, 8, 1, true);
  const spout = new THREE.Mesh(spoutGeo, funnelMat.clone());
  spout.position.y = -0.14;
  funnelGroup.add(spout);

  // Spout wireframe
  funnelGroup.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(spoutGeo),
    new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.25 })
  ).translateY(-0.14));

  // Position so spout exit is at POUR_Y
  funnelGroup.position.y = POUR_Y + 0.17;
  funnelGroup.visible = false;
  roomGroup.add(funnelGroup);
})();

// ---------------------------------------------------------------------------
// DRAIN HOLE — small ring on the floor centre, visible when draining
// ---------------------------------------------------------------------------
const DRAIN_RADIUS = 0.09;
const drainRing = new THREE.Mesh(
  new THREE.RingGeometry(0.01, DRAIN_RADIUS, 16),
  new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.DoubleSide })
);
drainRing.rotation.x = -Math.PI / 2;
drainRing.position.y = -BOX + 0.003;
drainRing.visible = false;
roomGroup.add(drainRing);

// Glow ring around the hole
const drainGlow = new THREE.Mesh(
  new THREE.RingGeometry(DRAIN_RADIUS, DRAIN_RADIUS + 0.015, 16),
  new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.5, side: THREE.DoubleSide })
);
drainGlow.rotation.x = -Math.PI / 2;
drainGlow.position.y = -BOX + 0.003;
drainGlow.visible = false;
roomGroup.add(drainGlow);

// ---------------------------------------------------------------------------
// WATER SPHERES — InstancedMesh coloured by speed
// ---------------------------------------------------------------------------
const COLOR_MODES = [
  [new THREE.Color(0x001f6e), new THREE.Color(0x22eeff)],  // Deep
  [new THREE.Color(0x003322), new THREE.Color(0x00ffcc)],  // Tropic
  [new THREE.Color(0x5a0800), new THREE.Color(0xff8800)],  // Magma
  [new THREE.Color(0x180040), new THREE.Color(0xcc44ff)],  // Void
];

const sphereMat = new THREE.MeshStandardMaterial({
  roughness: 0.08, metalness: 0.05, transparent: true, opacity: 0.88,
});

let sphereMesh = makeSpheres(N);
scene.add(sphereMesh);

function sphereSegments(n) {
  if (n >= 5000)  return [4,  2];
  if (n >= 1200) return [5,  3];
  if (n >= 800)  return [6,  4];
  if (n >= 400)  return [8,  6];
  if (n >= 200)  return [10, 7];
  if (n >= 100)  return [14, 10];
  return [20, 14];
}

function makeSpheres(n) {
  const [seg, ring] = sphereSegments(n);
  const m = new THREE.InstancedMesh(
    new THREE.SphereGeometry(sim.r, seg, ring),
    sphereMat,
    n
  );
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  m.frustumCulled = false;
  m.count = sim.activeN;
  return m;
}

const _m4 = new THREE.Matrix4();
const _c  = new THREE.Color();

function syncSpheres() {
  const [slow, fast] = COLOR_MODES[colorMode];
  const AN = sim.activeN;
  for (let i = 0; i < AN; i++) {
    _m4.setPosition(sim.px[i], sim.py[i], sim.pz[i]);
    sphereMesh.setMatrixAt(i, _m4);
    _c.lerpColors(slow, fast, Math.min(sim.speed[i] * 0.12, 1.0));
    sphereMesh.setColorAt(i, _c);
  }
  sphereMesh.count = AN;
  sphereMesh.instanceMatrix.needsUpdate = true;
  if (sphereMesh.instanceColor) sphereMesh.instanceColor.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// CURSOR VISUAL
// ---------------------------------------------------------------------------
const cursorMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.040, 14, 10),
  new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.0 })
);
cursorMesh.visible = false;
scene.add(cursorMesh);

const ringMesh = new THREE.Mesh(
  new THREE.TorusGeometry(0.10, 0.005, 6, 32),
  new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.0 })
);
ringMesh.rotation.x = Math.PI / 2;
ringMesh.visible = false;
scene.add(ringMesh);

// ---------------------------------------------------------------------------
// POST-PROCESSING — Unreal Bloom
// ---------------------------------------------------------------------------
const bloomRes = new THREE.Vector2(Math.floor(innerWidth / 2), Math.floor(innerHeight / 2));
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(bloomRes, 1.0, 0.6, 0.45));

// ---------------------------------------------------------------------------
// MOUSE
// ---------------------------------------------------------------------------
const raycaster  = new THREE.Raycaster();
const _mouse     = new THREE.Vector2();
const _hitPoint  = new THREE.Vector3();
const _iPlane    = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let isClicking   = false;
let ringPulse    = 0;
let prevHitX     = 0, prevHitZ = 0;
let cursorVel    = 0;
let cursorActive = false;

function updateCursor(mx, my) {
  _mouse.set((mx / innerWidth) * 2 - 1, -(my / innerHeight) * 2 + 1);
  raycaster.setFromCamera(_mouse, camera);

  let avgY = 0;
  const AN = sim.activeN;
  for (let i = 0; i < AN; i++) avgY += sim.py[i];
  _iPlane.constant = AN > 0 ? -(avgY / AN) : 0;

  if (!raycaster.ray.intersectPlane(_iPlane, _hitPoint)) return;
  const w = sim.W;
  _hitPoint.x = Math.max(-w, Math.min(w, _hitPoint.x));
  _hitPoint.z = Math.max(-w, Math.min(w, _hitPoint.z));

  cursorVel = Math.hypot(_hitPoint.x - prevHitX, _hitPoint.z - prevHitZ) * 60;
  prevHitX  = _hitPoint.x;
  prevHitZ  = _hitPoint.z;

  cursorMesh.position.copy(_hitPoint);
  cursorMesh.visible = true;
  ringMesh.position.copy(_hitPoint);
  ringMesh.visible = true;
  cursorActive = true;

  if (useGPU) {
    sim.setRepel(_hitPoint.x, _hitPoint.y, _hitPoint.z,
      isClicking ? 1.0 : 0.15 + cursorVel * 0.2,
      isClicking ? 0.22 : 0.20);
  } else {
    sim.repelFrom(_hitPoint.x, _hitPoint.y, _hitPoint.z, 0.15 + cursorVel * 0.2, 0.20);
    if (isClicking) {
      sim.repelFrom(_hitPoint.x, _hitPoint.y, _hitPoint.z, 1.0, 0.22);
    }
  }

  if (isClicking) {
    ringPulse = 1.0;
  }
}

canvas.addEventListener('mousemove',  e => updateCursor(e.clientX, e.clientY));
canvas.addEventListener('mousedown',  e => {
  if (e.button === 0) {
    isClicking = true;
    const burst = 1.5 + cursorVel * 0.5;
    if (useGPU) {
      sim.setRepel(_hitPoint.x, _hitPoint.y, _hitPoint.z, Math.min(burst, 4.0), 0.22);
    } else {
      sim.repelFrom(_hitPoint.x, _hitPoint.y, _hitPoint.z, Math.min(burst, 4.0), 0.22);
    }
    ringPulse = 1.0;
  }
});
canvas.addEventListener('mouseup',    e => { if (e.button === 0) isClicking = false; });
canvas.addEventListener('mouseleave', () => {
  cursorMesh.visible = false;
  ringMesh.visible   = false;
  isClicking         = false;
  cursorActive       = false;
  if (useGPU) sim.clearRepel();
});
canvas.addEventListener('contextmenu', e => e.preventDefault());

// Pause rendering when the other simulation is active
window.addEventListener('vortexmode', e => { hidden = (e.detail.mode !== '3d'); });

// Keyboard
window.addEventListener('keydown', e => {
  switch (e.code) {
    case 'Space':
      triggerShake(); e.preventDefault(); break;
    case 'KeyW':
      triggerWave(); break;
    case 'ArrowLeft':
      gravTiltX = Math.max(-1.8, gravTiltX - 0.25); applyGravity(); e.preventDefault(); break;
    case 'ArrowRight':
      gravTiltX = Math.min( 1.8, gravTiltX + 0.25); applyGravity(); e.preventDefault(); break;
    case 'ArrowUp':
      gravTiltZ = Math.max(-1.8, gravTiltZ - 0.25); applyGravity(); e.preventDefault(); break;
    case 'ArrowDown':
      gravTiltZ = Math.min( 1.8, gravTiltZ + 0.25); applyGravity(); e.preventDefault(); break;
    case 'KeyP':
      startPour(); break;
    case 'KeyD':
      toggleDrain(); break;
    case 'KeyR':
      resetSim(); break;
  }
});

// ---------------------------------------------------------------------------
// ACTIONS
// ---------------------------------------------------------------------------

function resetSim() {
  scene.remove(sphereMesh);
  sim        = new SPH(N);
  gravTiltX  = gravTiltZ = 0;
  shakeTimer = 0;
  pouring    = false;
  funnelGroup.visible = false;
  applyGravity();
  sphereMesh = makeSpheres(N);
  scene.add(sphereMesh);
}

function startPour() {
  scene.remove(sphereMesh);
  sim = new SPH(N);
  applyGravity();
  sim.startPour();
  sphereMesh = makeSpheres(N);
  scene.add(sphereMesh);
  funnelGroup.visible = true;
  pouring = true;
}

function toggleDrain() {
  draining = !draining;
  drainRing.visible = draining;
  drainGlow.visible = draining;
}

function triggerShake() {
  shakeTimer = 1.4;
  shakeAngle = Math.random() * Math.PI * 2;
}

function triggerWave() {
  const ang = Math.random() * Math.PI * 2;
  gravTiltX = Math.cos(ang) * 1.6;
  gravTiltZ = Math.sin(ang) * 1.6;
  applyGravity();
  setTimeout(() => { gravTiltX = gravTiltZ = 0; applyGravity(); }, 900);
}

// ---------------------------------------------------------------------------
// MAIN LOOP
// ---------------------------------------------------------------------------
let lastT = 0, simT = 0, fpsFrames = 0, fpsAcc = 0;
const fpsBadge = document.getElementById('fpsBadge');

function animate(ts) {
  requestAnimationFrame(animate);
  if (hidden) return;

  const dt = Math.min((ts - lastT) * 0.001, 0.05);
  lastT = ts;
  simT += dt;

  // FPS counter
  fpsFrames++; fpsAcc += dt;
  if (fpsAcc >= 0.5) {
    fpsBadge.textContent = Math.round(fpsFrames / fpsAcc) + ' fps';
    fpsFrames = 0; fpsAcc = 0;
  }

  // ------------------------------------------------------------------
  // Pour: add particles gradually through the funnel
  // ------------------------------------------------------------------
  if (pouring && !paused) {
    if (sim.activeN < sim.N) {
      sim.pourTick(useGPU ? GPU_POUR_RATE : POUR_RATE, 0, POUR_Y, 0, POUR_SPREAD);
    } else {
      pouring = false;
      setTimeout(() => { funnelGroup.visible = false; }, 1500);
    }
  }

  // ------------------------------------------------------------------
  // Drain: remove particles near the floor hole + subtle funnel gravity
  // ------------------------------------------------------------------
  if (draining && !paused) {
    const floorY = -sim.W;
    sim.drain(0, floorY, 0, DRAIN_RADIUS);
    if (useGPU) {
      sim.drainPull = 0.8;
      sim.drainX = 0; sim.drainY = floorY; sim.drainZ = 0;
    } else {
      // Gentle pull toward centre so water flows to the hole
      for (let i = 0; i < sim.activeN; i++) {
        const dx = -sim.px[i];
        const dz = -sim.pz[i];
        const dist = Math.sqrt(dx * dx + dz * dz) + 0.01;
        const pull = 0.8 / dist;  // stronger near centre
        sim.vx[i] += dx * pull * 0.006;
        sim.vz[i] += dz * pull * 0.006;
      }
    }
    if (sim.activeN === 0) {
      draining = false;
      drainRing.visible = false;
      drainGlow.visible = false;
    }
  } else if (useGPU && sim.drainPull) {
    sim.drainPull = 0;
  }

  // ------------------------------------------------------------------
  // Vertical shake
  // ------------------------------------------------------------------
  let shakeOffY = 0;
  if (shakeTimer > 0) {
    shakeTimer -= dt;
    const env = Math.sin((shakeTimer / 1.4) * Math.PI);
    const osc = Math.sin(simT * 13.0);

    const gyShake = -(gravBase * (1.0 + osc * env * 2.4));
    applyGravity(gyShake);

    shakeOffY = osc * env * 0.045;
    roomGroup.rotation.z += (Math.cos(shakeAngle) * osc * env * 0.05
                             - roomGroup.rotation.z) * 0.35;
  } else {
    applyGravity();
    roomGroup.rotation.z += (-gravTiltX * 0.18 - roomGroup.rotation.z) * 0.12;
    roomGroup.rotation.x += ( gravTiltZ * 0.18 - roomGroup.rotation.x) * 0.12;
  }

  // Subtle dip when draining — box sinks slightly to suggest funnel
  const drainDip = draining ? -0.045 : 0;
  roomGroup.position.y += (shakeOffY + drainDip - roomGroup.position.y) * 0.25;

  // ------------------------------------------------------------------
  // Simulation substeps
  // ------------------------------------------------------------------
  if (!paused) {
    if (useGPU) {
      if (!computing) {
        computing = true;
        sim.step(4);
        sim.clearRepel();
        sim.readback().then(() => {
          syncSpheres();
          computing = false;
        }).catch(() => { computing = false; });
      }
    } else {
      for (let s = 0; s < 4; s++) sim.step();
      syncSpheres();
    }
  }

  // Animate caustic lights
  causticA.position.set(
    Math.sin(simT * 0.73) * 0.38,
    0.18 + Math.cos(simT * 0.55) * 0.18,
    Math.cos(simT * 0.67) * 0.38
  );

  // Cursor ring pulse
  if (ringPulse > 0) {
    ringPulse -= dt * 2.8;
    const rp = Math.max(ringPulse, 0);
    ringMesh.scale.setScalar(1.0 + (1.0 - rp) * 2.5);
    ringMesh.material.opacity   = rp * 0.75;
    cursorMesh.material.opacity = 0.55;
  } else {
    cursorMesh.material.opacity = isClicking ? 0.55 : 0.28;
    ringMesh.scale.setScalar(1.0);
    ringMesh.material.opacity   = isClicking ? 0.45 : 0.20;
  }

  controls.update();
  composer.render();
}

// ---------------------------------------------------------------------------
// RESIZE
// ---------------------------------------------------------------------------
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
(function setupUI() {
  const $ = id => document.getElementById(id);

  document.querySelectorAll('#colorModes3d .color-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#colorModes3d .color-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      colorMode = +btn.dataset.mode;
    });
  });

  const pauseBtn = $('pauseBtn');
  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'Resume' : 'Pause';
    pauseBtn.classList.toggle('active', paused);
  });

  $('resetBtn').addEventListener('click', resetSim);
  $('pourBtn').addEventListener('click', startPour);
  $('drainBtn').addEventListener('click', toggleDrain);
  $('shakeBtn').addEventListener('click', triggerShake);
  $('waveBtn').addEventListener('click', triggerWave);

  $('toggleUI').addEventListener('click', () => {
    const ui = $('ui');
    ui.classList.toggle('collapsed');
    $('toggleUI').textContent = ui.classList.contains('collapsed') ? '+' : '\u2212';
  });
})();

// ---------------------------------------------------------------------------
// START — CPU runs immediately, WebGPU upgrades in background
// ---------------------------------------------------------------------------
requestAnimationFrame(animate);

// Try WebGPU upgrade (non-blocking — CPU sim is already running)
import('./sph-gpu.js').then(async ({ SPHCompute }) => {
  if (!await SPHCompute.isSupported()) return;
  try {
    const gpuN = 20000;
    const gpu  = new SPHCompute();
    await gpu.init(gpuN);

    // Hot-swap: preserve current pour/drain state
    const wasPouring  = pouring;
    const wasDraining = draining;

    // Transfer active particles from CPU to GPU
    gpu.startPour();
    gpu.activeN = 0;

    // Swap sim
    useGPU = true;
    N = gpuN;
    sim = gpu;
    applyGravity();

    // Rebuild spheres for higher count
    scene.remove(sphereMesh);
    sphereMesh = makeSpheres(N);
    scene.add(sphereMesh);

    // Restart pour if it was active
    if (wasPouring) {
      pouring = true;
      funnelGroup.visible = true;
    }
    if (wasDraining) {
      draining = true;
      drainRing.visible = true;
      drainGlow.visible = true;
    }

    console.log(`Upgraded to WebGPU SPH with ${N} particles`);
  } catch (e) {
    console.warn('WebGPU upgrade failed, staying on CPU:', e);
  }
}).catch(e => {
  console.warn('sph-gpu.js load failed, staying on CPU:', e);
});
