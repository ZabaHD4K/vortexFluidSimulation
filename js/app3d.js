import * as THREE from 'three';
import { OrbitControls }   from 'three/addons/controls/OrbitControls.js';
import { EffectComposer }  from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SPH }             from './sph.js';

// ---------------------------------------------------------------------------
// PARTICLE PRESETS
// Radius scales automatically as N^(-1/3) inside SPH — no manual tuning needed.
// ---------------------------------------------------------------------------
const PCOUNTS = [40, 80, 150, 250, 400];
const PLABELS = ['40', '80', '150', '250', '400'];
const BOX     = 0.58;

let N         = PCOUNTS[2];   // default: 150 particles
let sim       = new SPH(N);
let colorMode = 0;
let paused    = false;
let hidden    = false;   // true when 2D mode is active — skip rendering entirely
let gravBase  = 8;
let gravTiltX = 0;
let gravTiltZ = 0;

// Shake state
let shakeTimer = 0;
let shakeAngle = 0;

// ---------------------------------------------------------------------------
// GRAVITY HELPER
// Computes gx/gz from tilt; gy from normal gravity or an override value
// (used by the vertical shake to oscillate the Y component independently).
// ---------------------------------------------------------------------------
function applyGravity(gyOverride = null) {
  const len = Math.sqrt(gravTiltX * gravTiltX + 1.0 + gravTiltZ * gravTiltZ);
  const s   = gravBase / len;
  sim.gx = gravTiltX * s;
  sim.gy = (gyOverride !== null) ? gyOverride : -s;
  sim.gz = gravTiltZ * s;
}
applyGravity();

// ---------------------------------------------------------------------------
// RENDERER
// ---------------------------------------------------------------------------
const canvas   = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping         = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.5;
renderer.outputColorSpace    = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020508);
scene.fog = new THREE.FogExp2(0x030610, 0.10);

// ---------------------------------------------------------------------------
// CAMERA — isometric perspective
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
// Left-drag = push water (custom raycaster), Right-drag = orbit, Middle = zoom
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
// CONTAINER BOX  (tiltable group)
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

  // Floor tile grid
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

  // Corner accent dots
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
// WATER SPHERES — InstancedMesh coloured by speed
// Polygon count scales down with N to keep draw calls cheap at 400 particles.
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
  if (n >= 300) return [9,  7];
  if (n >= 150) return [13, 10];
  if (n >=  80) return [18, 13];
  return [24, 16];
}

function makeSpheres(n) {
  const [seg, ring] = sphereSegments(n);
  const m = new THREE.InstancedMesh(
    new THREE.SphereGeometry(sim.r, seg, ring),
    sphereMat,
    n
  );
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  return m;
}

const _m4 = new THREE.Matrix4();
const _c  = new THREE.Color();

function syncSpheres() {
  const [slow, fast] = COLOR_MODES[colorMode];
  for (let i = 0; i < sim.N; i++) {
    _m4.setPosition(sim.px[i], sim.py[i], sim.pz[i]);
    sphereMesh.setMatrixAt(i, _m4);
    _c.lerpColors(slow, fast, Math.min(sim.speed[i] * 0.12, 1.0));
    sphereMesh.setColorAt(i, _c);
  }
  sphereMesh.instanceMatrix.needsUpdate = true;
  if (sphereMesh.instanceColor) sphereMesh.instanceColor.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// CURSOR VISUAL  (hover glow + click ring)
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
// POST-PROCESSING — Unreal Bloom for that liquid glow
// ---------------------------------------------------------------------------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 1.0, 0.6, 0.45));

// ---------------------------------------------------------------------------
// MOUSE — hover repels water; left-click bursts; right-drag = orbit
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

  // Project onto a plane at the average particle height
  let avgY = 0;
  for (let i = 0; i < sim.N; i++) avgY += sim.py[i];
  _iPlane.constant = -(avgY / sim.N);

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

  // Gentle hover repulsion, scales with cursor speed
  sim.repelFrom(_hitPoint.x, _hitPoint.y, _hitPoint.z, 0.7 + cursorVel * 1.0, 0.30);

  if (isClicking) {
    sim.repelFrom(_hitPoint.x, _hitPoint.y, _hitPoint.z, 5.0, 0.28);
    ringPulse = 1.0;
  }
}

canvas.addEventListener('mousemove',  e => updateCursor(e.clientX, e.clientY));
canvas.addEventListener('mousedown',  e => {
  if (e.button === 0) {
    isClicking = true;
    const burst = 8.0 + cursorVel * 3.0;
    sim.repelFrom(_hitPoint.x, _hitPoint.y, _hitPoint.z, Math.min(burst, 20.0), 0.32);
    ringPulse = 1.0;
  }
});
canvas.addEventListener('mouseup',    e => { if (e.button === 0) isClicking = false; });
canvas.addEventListener('mouseleave', () => {
  cursorMesh.visible = false;
  ringMesh.visible   = false;
  isClicking         = false;
  cursorActive       = false;
});
canvas.addEventListener('contextmenu', e => e.preventDefault());

// Pause rendering when the other simulation is active
window.addEventListener('vortexmode', e => { hidden = (e.detail.mode !== '3d'); });

// Arrow keys tilt the box; Space = vertical shake; W = wave; R = reset
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
    case 'KeyR':
      resetSim(); break;
  }
});

// ---------------------------------------------------------------------------
// ACTIONS
// ---------------------------------------------------------------------------

/** Full reset: new sim, particles settle from bottom. */
function resetSim() {
  scene.remove(sphereMesh);
  sim        = new SPH(N);
  gravTiltX  = gravTiltZ = 0;
  shakeTimer = 0;
  applyGravity();
  sphereMesh = makeSpheres(N);
  scene.add(sphereMesh);
}

/**
 * Particle-count change: spawn new particles pouring from the top.
 * More visually engaging than an instant teleport.
 */
function respawnFromTop() {
  scene.remove(sphereMesh);
  sim = new SPH(N);
  applyGravity();
  sim.spawnFromTop();
  sphereMesh = makeSpheres(N);
  scene.add(sphereMesh);
}

/** Vertical shake — oscillates the Y gravity component up and down. */
function triggerShake() {
  shakeTimer = 1.4;
  shakeAngle = Math.random() * Math.PI * 2;
}

/** Wave — tilt the box in a random horizontal direction, then return. */
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
  if (hidden) return;   // 2D mode active — skip entirely

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
  // Vertical shake: oscillates Y gravity so particles slam floor/ceiling.
  // The room gets a subtle up/down nudge for physical feedback.
  // ------------------------------------------------------------------
  let shakeOffY = 0;
  if (shakeTimer > 0) {
    shakeTimer -= dt;
    const env = Math.sin((shakeTimer / 1.4) * Math.PI);   // bell envelope
    const osc = Math.sin(simT * 13.0);                     // fast oscillation

    // Physics: gy flips between strongly positive and strongly negative
    const gyShake = -(gravBase * (1.0 + osc * env * 2.4));
    applyGravity(gyShake);

    // Visuals: box nudges in Y and a tiny X roll for drama
    shakeOffY = osc * env * 0.045;
    roomGroup.rotation.z += (Math.cos(shakeAngle) * osc * env * 0.05
                             - roomGroup.rotation.z) * 0.35;
  } else {
    applyGravity(); // restore normal gravity
    // Smooth tilt from arrow keys
    roomGroup.rotation.z += (-gravTiltX * 0.18 - roomGroup.rotation.z) * 0.12;
    roomGroup.rotation.x += ( gravTiltZ * 0.18 - roomGroup.rotation.x) * 0.12;
  }

  // Vertical position of the whole room during shake
  roomGroup.position.y += (shakeOffY - roomGroup.position.y) * 0.25;

  // ------------------------------------------------------------------
  // Simulation substeps (3× per frame for stability)
  // ------------------------------------------------------------------
  if (!paused) {
    for (let s = 0; s < 3; s++) sim.step();
    syncSpheres();
  }

  // Animate caustic point-lights
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

  // Particle count — triggers spawn-from-top animation
  $('particleCount').addEventListener('input', e => {
    const idx = +e.target.value;
    $('particleVal').textContent = PLABELS[idx];
    N = PCOUNTS[idx];
    respawnFromTop();
  });

  // Sync initial label with default slider value
  $('particleVal').textContent = PLABELS[+$('particleCount').value];

  $('viscosity').addEventListener('input', e => {
    const v = +e.target.value;
    sim.viscosity = v;
    $('viscosityVal').textContent = v.toFixed(2);
  });

  $('pressure').addEventListener('input', e => {
    const v = +e.target.value;
    sim.RESTITUTION = v;
    $('pressureVal').textContent = v.toFixed(2);
  });

  $('gravity').addEventListener('input', e => {
    gravBase = +e.target.value;
    $('gravityVal').textContent = e.target.value;
    applyGravity();
  });

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
  $('shakeBtn').addEventListener('click', triggerShake);
  $('waveBtn').addEventListener('click', triggerWave);

  $('toggleUI').addEventListener('click', () => {
    const ui = $('ui');
    ui.classList.toggle('collapsed');
    $('toggleUI').textContent = ui.classList.contains('collapsed') ? '+' : '−';
  });

  // Gradient fill for range sliders (3D panel only)
  document.querySelectorAll('#panel3d input[type="range"]').forEach(inp => {
    const upd = () => {
      const pct = ((+inp.value - +inp.min) / (+inp.max - +inp.min)) * 100;
      inp.style.setProperty('--pct', pct + '%');
    };
    inp.addEventListener('input', upd); upd();
  });
})();

requestAnimationFrame(animate);
