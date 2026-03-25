'use strict';

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
const BUBBLE_COUNTS  = [4096, 8192, 16384, 32768, 65536];
const BUBBLE_LABELS  = ['4 096', '8 192', '16 384', '32 768', '65 536'];

const cfg = {
  SIM_RES:         256,
  DYE_RES:         1024,
  DISSIPATION:     0.985,
  VEL_DISSIPATION: 0.99,
  PRESSURE_ITER:   30,
  CURL:            25,
  SPLAT_RADIUS:    0.0025,
  COLOR_MODE:      0,
  BUBBLE_IDX:      2,
  PAUSED:          false,
};

// ---------------------------------------------------------------------------
// SHADERS
// ---------------------------------------------------------------------------
const VS_QUAD = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main() { vUv = aPos*.5+.5; gl_Position = vec4(aPos,0.,1.); }`;

const FS_ADVECT = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uVelocity, uSource;
uniform float uDt, uDissipation;
uniform vec2 uTexelSize;
void main() {
  vec2 vel = texture(uVelocity, vUv).xy;
  o = uDissipation * texture(uSource, vUv - vel * uDt * uTexelSize);
}`;

const FS_DIVERGENCE = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uVelocity; uniform vec2 uTexelSize;
void main() {
  float L=texture(uVelocity,vUv-vec2(uTexelSize.x,0)).x;
  float R=texture(uVelocity,vUv+vec2(uTexelSize.x,0)).x;
  float B=texture(uVelocity,vUv-vec2(0,uTexelSize.y)).y;
  float T=texture(uVelocity,vUv+vec2(0,uTexelSize.y)).y;
  o=vec4(.5*(R-L+T-B),0,0,1);
}`;

const FS_PRESSURE = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uPressure, uDivergence; uniform vec2 uTexelSize;
void main() {
  float L=texture(uPressure,vUv-vec2(uTexelSize.x,0)).r;
  float R=texture(uPressure,vUv+vec2(uTexelSize.x,0)).r;
  float B=texture(uPressure,vUv-vec2(0,uTexelSize.y)).r;
  float T=texture(uPressure,vUv+vec2(0,uTexelSize.y)).r;
  float d=texture(uDivergence,vUv).r;
  o=vec4((L+R+B+T-d)*.25,0,0,1);
}`;

const FS_GRADIENT = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uPressure, uVelocity; uniform vec2 uTexelSize;
void main() {
  float L=texture(uPressure,vUv-vec2(uTexelSize.x,0)).r;
  float R=texture(uPressure,vUv+vec2(uTexelSize.x,0)).r;
  float B=texture(uPressure,vUv-vec2(0,uTexelSize.y)).r;
  float T=texture(uPressure,vUv+vec2(0,uTexelSize.y)).r;
  o=vec4(texture(uVelocity,vUv).xy-.5*vec2(R-L,T-B),0,1);
}`;

const FS_VORTICITY = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uVelocity; uniform vec2 uTexelSize;
void main() {
  float L=texture(uVelocity,vUv-vec2(uTexelSize.x,0)).y;
  float R=texture(uVelocity,vUv+vec2(uTexelSize.x,0)).y;
  float B=texture(uVelocity,vUv-vec2(0,uTexelSize.y)).x;
  float T=texture(uVelocity,vUv+vec2(0,uTexelSize.y)).x;
  o=vec4(.5*(R-L-T+B),0,0,1);
}`;

const FS_VORTICITY_FORCE = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uVelocity, uVorticity; uniform vec2 uTexelSize;
uniform float uCurl, uDt;
void main() {
  float L=abs(texture(uVorticity,vUv-vec2(uTexelSize.x,0)).r);
  float R=abs(texture(uVorticity,vUv+vec2(uTexelSize.x,0)).r);
  float B=abs(texture(uVorticity,vUv-vec2(0,uTexelSize.y)).r);
  float T=abs(texture(uVorticity,vUv+vec2(0,uTexelSize.y)).r);
  float C=texture(uVorticity,vUv).r;
  vec2 f=normalize(vec2(abs(T)-abs(B),abs(R)-abs(L))+1e-5)*uCurl*C;
  f.y*=-1.; o=vec4(texture(uVelocity,vUv).xy+f*uDt,0,1);
}`;

const FS_ENFORCE_BOUNDARY = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uVelocity, uBoundary;
void main() {
  float b = texture(uBoundary, vUv).r;
  o = vec4(texture(uVelocity, vUv).xy * b, 0.0, 1.0);
}`;

const FS_SPLAT = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uTarget; uniform vec2 uPoint; uniform vec3 uColor;
uniform float uRadius, uAspect;
void main() {
  vec2 p=vUv-uPoint; p.x*=uAspect;
  o=vec4(texture(uTarget,vUv).rgb+exp(-dot(p,p)/uRadius)*uColor,1);
}`;

// ---- WATER DISPLAY SHADER ----
const FS_DISPLAY_WATER = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform sampler2D uDye, uVelocity, uBoundary;
uniform float uTime;
uniform int uColorMode;

float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p) {
  vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}

vec3 stoneColor(vec2 uv) {
  float row = floor(uv.y * 9.0);
  vec2 brick = vec2(floor(uv.x * 15.0 + row * 0.5), row);
  float mx = smoothstep(0.87, 1.0, fract(uv.x * 15.0 + row * 0.5));
  float my = smoothstep(0.87, 1.0, fract(uv.y * 9.0));
  float mortar = max(mx, my);
  float n  = hash(brick) * 0.2;
  float n2 = noise(uv * 28.0) * 0.07;
  float base = 0.17 + n + n2;
  return mix(vec3(base*1.1, base, base*0.88), vec3(0.07,0.065,0.06), mortar);
}

float caustics(vec2 uv, float t) {
  vec2 p = uv * 10.0;
  float c = sin(p.x*1.3 + sin(p.y*0.9+t*0.5) + t*0.4);
  c += sin(p.x*0.7 + sin(p.y*1.4-t*0.35) - t*0.25);
  c += sin((p.x+p.y)*0.75 + t*0.38);
  c += sin(p.x*1.6 - p.y*0.5 + t*0.55);
  return pow(max(0.0, c*0.25+0.35), 3.0);
}

float godRays(vec2 uv, float t) {
  vec2 lp  = vec2(0.5, 1.08);
  vec2 d   = uv - lp;
  float ang = atan(d.x, -d.y);
  float dist = length(d);
  float r = (sin(ang*11.0+t*0.38)*0.5+0.5) * (sin(ang*7.0-t*0.27)*0.5+0.5);
  return r * exp(-dist * 2.0) * 0.7;
}

float wallAO(vec2 uv) {
  float s = 0.0;
  s += texture(uBoundary, uv + vec2( 0.012,  0.0)).r;
  s += texture(uBoundary, uv + vec2(-0.012,  0.0)).r;
  s += texture(uBoundary, uv + vec2( 0.0,    0.012)).r;
  s += texture(uBoundary, uv + vec2( 0.0,   -0.012)).r;
  s += texture(uBoundary, uv + vec2( 0.008,  0.008)).r;
  s += texture(uBoundary, uv + vec2(-0.008, -0.008)).r;
  s += texture(uBoundary, uv + vec2( 0.008, -0.008)).r;
  s += texture(uBoundary, uv + vec2(-0.008,  0.008)).r;
  return 1.0 - clamp(s / 8.0, 0.0, 1.0);
}

void main() {
  float b = texture(uBoundary, vUv).r;

  // ---------- WALLS / PILLARS ----------
  if (b < 0.5) {
    vec3 wall = stoneColor(vUv);
    float depth = vUv.y;
    // Wet darkening at bottom
    float wet = (1.0 - depth) * 0.5 + texture(uDye, vUv).r * 0.25;
    wall *= 0.55 + wet * 0.3;
    // Moss on lower sections
    float moss = (1.0 - depth) * 0.35 * noise(vUv * 18.0 + 0.7);
    wall = mix(wall, vec3(0.05, 0.095, 0.04), clamp(moss, 0.0, 1.0));
    // Caustic shimmer on walls from water above
    float caust = caustics(vUv, uTime) * 0.25 * depth;
    wall += vec3(0.01, 0.03, 0.07) * caust;
    o = vec4(wall, 1.0); return;
  }

  // ---------- WATER ----------
  vec3 dye   = texture(uDye, vUv).rgb;
  vec2 vel   = texture(uVelocity, vUv).xy;
  float speed = length(vel);
  float depth = vUv.y;

  // Base water depth gradient
  vec3 waterColor;
  if (uColorMode == 0) {
    // Deep Ocean
    vec3 c0 = vec3(0.003, 0.015, 0.050);
    vec3 c1 = vec3(0.008, 0.048, 0.115);
    vec3 c2 = vec3(0.025, 0.110, 0.210);
    waterColor = mix(c0, mix(c1, c2, depth), depth);
  } else if (uColorMode == 1) {
    // Tropic
    vec3 c0 = vec3(0.005, 0.040, 0.080);
    vec3 c1 = vec3(0.015, 0.130, 0.180);
    vec3 c2 = vec3(0.030, 0.250, 0.280);
    waterColor = mix(c0, mix(c1, c2, depth), depth);
  } else if (uColorMode == 2) {
    // Thermal — hot spring
    vec3 c0 = vec3(0.060, 0.015, 0.005);
    vec3 c1 = vec3(0.130, 0.055, 0.010);
    vec3 c2 = vec3(0.200, 0.140, 0.030);
    waterColor = mix(c0, mix(c1, c2, depth), depth);
  } else {
    // Bioluminescent
    vec3 c0 = vec3(0.002, 0.008, 0.015);
    vec3 c1 = vec3(0.005, 0.030, 0.050);
    vec3 c2 = vec3(0.010, 0.060, 0.080);
    waterColor = mix(c0, mix(c1, c2, depth), depth);
  }

  // Caustics (stronger near surface where light enters)
  float caust = caustics(vUv, uTime) * (0.15 + depth * 0.85);
  if (uColorMode == 0) waterColor += vec3(0.010, 0.048, 0.110) * caust;
  else if (uColorMode == 1) waterColor += vec3(0.010, 0.060, 0.080) * caust;
  else if (uColorMode == 2) waterColor += vec3(0.060, 0.025, 0.005) * caust;
  else waterColor += vec3(0.005, 0.060, 0.080) * caust;

  // God rays from top
  float rays = godRays(vUv, uTime);
  if (uColorMode == 0) waterColor += vec3(0.006, 0.028, 0.065) * rays;
  else if (uColorMode == 1) waterColor += vec3(0.005, 0.040, 0.060) * rays;
  else if (uColorMode == 2) waterColor += vec3(0.040, 0.015, 0.003) * rays;
  else waterColor += vec3(0.003, 0.050, 0.070) * rays;

  // Dye adds turbulence color swirls
  if (uColorMode == 3) {
    // Biolum: dye glows bright
    float glow = length(dye) * (1.0 + speed * 3.0);
    waterColor += vec3(0.0, 0.8, 0.6) * glow * 0.4;
    waterColor += vec3(0.1, 0.2, 1.0) * length(dye) * speed * 0.8;
  } else {
    waterColor += dye * 0.10 * (1.0 + speed * 2.0);
  }

  // Velocity / turbulence streaks
  float turb = speed * 3.5;
  if (uColorMode == 0) waterColor += vec3(0.012, 0.050, 0.100) * turb;
  else if (uColorMode == 1) waterColor += vec3(0.008, 0.060, 0.090) * turb;
  else if (uColorMode == 2) waterColor += vec3(0.080, 0.030, 0.005) * turb;
  else waterColor += vec3(0.000, 0.100, 0.080) * turb * (1.0 + length(dye));

  // Foam at high velocity
  float foam = smoothstep(0.05, 0.14, speed);
  vec3 foamColor = uColorMode == 2 ? vec3(1.0, 0.7, 0.4) : vec3(0.60, 0.85, 0.95);
  waterColor = mix(waterColor, foamColor, foam * 0.38);

  // Near-wall ambient occlusion (contact shadow)
  float ao = wallAO(vUv);
  waterColor *= (1.0 - ao * 0.65);

  // Surface shimmer at top edge
  float surfProx = smoothstep(0.82, 0.97, depth);
  float shimmer  = sin(vUv.x * 38.0 + uTime * 2.8) * sin(vUv.x * 24.0 - uTime * 2.0) * 0.5 + 0.5;
  vec3 shimCol = uColorMode == 2 ? vec3(0.9, 0.7, 0.4) : vec3(0.35, 0.72, 0.95);
  waterColor = mix(waterColor, shimCol, surfProx * shimmer * 0.28);

  // Room vignette — darker at bottom corners
  float vign = 1.0 - smoothstep(0.2, 0.8, length((vUv - vec2(0.5,0.6)) * vec2(1.0, 0.9)));
  waterColor = mix(waterColor * 0.35, waterColor, clamp(vign + 0.2, 0.0, 1.0));

  o = vec4(waterColor, 1.0);
}`;

// ---- BUBBLE SHADERS ----
const FS_BUBBLE_UPDATE = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform highp sampler2D uParticleTex, uVelocity, uBoundary;
uniform float uDt, uTime;

float rand(vec2 co) { return fract(sin(dot(co,vec2(12.9898,78.233)))*43758.5453); }

void main() {
  vec4 data  = texture(uParticleTex, vUv);
  vec2 pos   = data.xy;
  float life = data.z;
  float seed = data.w;

  life -= uDt * 0.18;
  bool dead   = life <= 0.0 || pos.y > 0.95;
  bool inWall = texture(uBoundary, pos).r < 0.5;

  if (dead || inWall) {
    vec2 s = vUv + mod(uTime * 0.00137 + seed, 1.0);
    // Spawn across floor, inside walls (x: 8%-92%, y: 5%-25%)
    float bx = 0.08 + rand(s)       * 0.84;
    float by = 0.05 + rand(s + 0.3) * 0.20;
    // Check spawn point is in fluid
    if (texture(uBoundary, vec2(bx, by)).r < 0.5) {
      bx = 0.5; by = 0.5; // fallback center
    }
    pos  = vec2(bx, by);
    life = 0.3 + rand(s + 0.8) * 0.7;
    seed = rand(s + 1.4);
  } else {
    vec2 vel    = texture(uVelocity, pos).xy;
    float wobble = sin(uTime * 2.5 + seed * 6.283) * 0.0018;
    // Buoyancy up + partial flow following + wobble
    pos += vec2(vel.x * 0.35 + wobble, vel.y * 0.25 + uDt * 0.045);
    // Don't enter walls
    if (texture(uBoundary, pos).r < 0.5) pos = data.xy;
  }
  o = vec4(pos, life, seed);
}`;

const VS_BUBBLE = `#version 300 es
precision highp float;
uniform highp sampler2D uParticleTex;
uniform ivec2 uTexSize;
out float vLife;
out float vSeed;
void main() {
  ivec2 coord = ivec2(gl_VertexID % uTexSize.x, gl_VertexID / uTexSize.x);
  vec4 data   = texelFetch(uParticleTex, coord, 0);
  vLife = data.z; vSeed = data.w;
  gl_Position  = vec4(data.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = max(1.0, 1.2 + vSeed * 2.5);
}`;

const FS_BUBBLE = `#version 300 es
precision highp float;
in float vLife; in float vSeed;
out vec4 o;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  if (d > 1.0) discard;
  float rim    = smoothstep(0.55, 0.95, d) * 0.7;
  float center = smoothstep(0.25, 0.0, d)  * 0.45;
  float alpha  = (rim + center) * vLife * 0.65;
  vec3 col     = mix(vec3(0.35, 0.72, 0.96), vec3(0.92, 0.97, 1.0), center * 2.0);
  o = vec4(col, alpha);
}`;

// ---------------------------------------------------------------------------
// GL SETUP
// ---------------------------------------------------------------------------
const canvas = document.getElementById('canvas-2d');
let gl;

function initGL() {
  gl = canvas.getContext('webgl2', { alpha:false, antialias:false, powerPreference:'high-performance' });
  if (!gl) { document.body.innerHTML='<p style="color:#fff;padding:2rem">WebGL2 not supported.</p>'; return false; }
  if (!gl.getExtension('EXT_color_buffer_float')) { document.body.innerHTML='<p style="color:#fff;padding:2rem">EXT_color_buffer_float not supported.</p>'; return false; }
  gl.getExtension('OES_texture_float_linear');
  return true;
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = Math.floor(canvas.clientWidth  * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

// ---------------------------------------------------------------------------
// SHADER UTILS
// ---------------------------------------------------------------------------
function compileShader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function mkProg(vsSrc, fsSrc, ...uniforms) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compileShader(gl.VERTEX_SHADER,   vsSrc));
  gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const u = {};
  for (const n of uniforms) u[n] = gl.getUniformLocation(prog, n);
  return { prog, u, use() { gl.useProgram(this.prog); } };
}

// ---------------------------------------------------------------------------
// VAOs
// ---------------------------------------------------------------------------
let quadVAO, bubbleVAO;

function initVAOs() {
  quadVAO = gl.createVertexArray();
  gl.bindVertexArray(quadVAO);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  bubbleVAO = gl.createVertexArray();
}

function drawQuad() {
  gl.bindVertexArray(quadVAO);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
}

// ---------------------------------------------------------------------------
// FBO UTILS
// ---------------------------------------------------------------------------
function makeFBO(w, h, iFmt, fmt, type, filter) {
  gl.activeTexture(gl.TEXTURE0);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, iFmt, w, h, 0, fmt, type, null);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0,0,w,h); gl.clear(gl.COLOR_BUFFER_BIT);
  return { tex, fbo, w, h, attach(s){gl.activeTexture(gl.TEXTURE0+s);gl.bindTexture(gl.TEXTURE_2D,tex);return s;} };
}

class DoubleFBO {
  constructor(w,h,i,f,t,fl){this.a=makeFBO(w,h,i,f,t,fl);this.b=makeFBO(w,h,i,f,t,fl);this.w=w;this.h=h;}
  get read(){return this.a;} get write(){return this.b;} swap(){[this.a,this.b]=[this.b,this.a];}
}

function makeBubbleFBO(size, data) {
  const make = (d) => {
    gl.activeTexture(gl.TEXTURE0);
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, size, size, 0, gl.RGBA, gl.FLOAT, d||null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.viewport(0,0,size,size); gl.clear(gl.COLOR_BUFFER_BIT);
    return { tex, fbo, w:size, h:size, attach(s){gl.activeTexture(gl.TEXTURE0+s);gl.bindTexture(gl.TEXTURE_2D,tex);return s;} };
  };
  const obj = { a:make(data), b:make(null), size };
  Object.defineProperty(obj,'read',  {get(){return this.a;}});
  Object.defineProperty(obj,'write', {get(){return this.b;}});
  obj.swap = function(){[this.a,this.b]=[this.b,this.a];};
  return obj;
}

// ---------------------------------------------------------------------------
// BOUNDARY TEXTURE  (room + stone pillars)
// ---------------------------------------------------------------------------
let boundaryObj;

function createBoundary() {
  const S = cfg.SIM_RES;
  const data = new Uint8Array(S * S);
  const WALL = 10; // px wall thickness

  // Three stone pillars at the bottom third
  const pillars = [
    { cx: 0.22, cy: 0.24, rx: 0.060, ry: 0.080 },
    { cx: 0.50, cy: 0.20, rx: 0.070, ry: 0.090 },
    { cx: 0.78, cy: 0.24, rx: 0.060, ry: 0.080 },
  ];

  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const idx = y * S + x;
      const nx = x / S, ny = y / S;

      if (x < WALL || x >= S-WALL || y < WALL || y >= S-WALL) { data[idx] = 0; continue; }

      let wall = false;
      for (const p of pillars) {
        const dx = (nx - p.cx) / p.rx;
        const dy = (ny - p.cy) / p.ry;
        if (dx*dx + dy*dy < 1.0) { wall = true; break; }
      }
      data[idx] = wall ? 0 : 255;
    }
  }

  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, S, S, 0, gl.RED, gl.UNSIGNED_BYTE, data);
  return { tex, attach(s){gl.activeTexture(gl.TEXTURE0+s);gl.bindTexture(gl.TEXTURE_2D,tex);return s;} };
}

// ---------------------------------------------------------------------------
// SIM STATE
// ---------------------------------------------------------------------------
let velocity, dye, pressure, divergence, vorticity, bubbles;
let programs = {};

function initSim() {
  const S=cfg.SIM_RES, D=cfg.DYE_RES, F=gl.FLOAT, L=gl.LINEAR, N=gl.NEAREST;
  velocity   = new DoubleFBO(S, S, gl.RG16F,  gl.RG,   F, L);
  dye        = new DoubleFBO(D, D, gl.RGBA16F, gl.RGBA, F, L);
  pressure   = new DoubleFBO(S, S, gl.R16F,   gl.RED,  F, N);
  divergence = makeFBO(S, S, gl.R16F, gl.RED, F, N);
  vorticity  = makeFBO(S, S, gl.R16F, gl.RED, F, N);
  boundaryObj = createBoundary();
  initBubbles();
}

function initBubbles() {
  const count = BUBBLE_COUNTS[cfg.BUBBLE_IDX];
  const size  = Math.ceil(Math.sqrt(count));
  const data  = new Float32Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    data[i*4]   = 0.08 + Math.random() * 0.84;  // x inside walls
    data[i*4+1] = 0.05 + Math.random() * 0.85;  // y spread through room
    data[i*4+2] = Math.random();                  // life
    data[i*4+3] = Math.random();                  // seed
  }
  bubbles = makeBubbleFBO(size, data);
  bubbles.count = count;
}

function initPrograms() {
  programs.advect     = mkProg(VS_QUAD, FS_ADVECT,            'aPos','uVelocity','uSource','uDt','uDissipation','uTexelSize');
  programs.divergence = mkProg(VS_QUAD, FS_DIVERGENCE,        'aPos','uVelocity','uTexelSize');
  programs.pressure   = mkProg(VS_QUAD, FS_PRESSURE,          'aPos','uPressure','uDivergence','uTexelSize');
  programs.gradient   = mkProg(VS_QUAD, FS_GRADIENT,          'aPos','uPressure','uVelocity','uTexelSize');
  programs.vorticity  = mkProg(VS_QUAD, FS_VORTICITY,         'aPos','uVelocity','uTexelSize');
  programs.vorticityF = mkProg(VS_QUAD, FS_VORTICITY_FORCE,   'aPos','uVelocity','uVorticity','uTexelSize','uCurl','uDt');
  programs.enforce    = mkProg(VS_QUAD, FS_ENFORCE_BOUNDARY,  'aPos','uVelocity','uBoundary');
  programs.splat      = mkProg(VS_QUAD, FS_SPLAT,             'aPos','uTarget','uPoint','uColor','uRadius','uAspect');
  programs.display    = mkProg(VS_QUAD, FS_DISPLAY_WATER,     'aPos','uDye','uVelocity','uBoundary','uTime','uColorMode');
  programs.bubbleUpd  = mkProg(VS_QUAD, FS_BUBBLE_UPDATE,     'aPos','uParticleTex','uVelocity','uBoundary','uDt','uTime');
  programs.bubbleRen  = mkProg(VS_BUBBLE, FS_BUBBLE,          'uParticleTex','uTexSize');
}

// ---------------------------------------------------------------------------
// ENFORCE BOUNDARY (zero velocity at walls)
// ---------------------------------------------------------------------------
function enforceBoundary() {
  programs.enforce.use();
  gl.uniform1i(programs.enforce.u.uVelocity, velocity.read.attach(0));
  gl.uniform1i(programs.enforce.u.uBoundary, boundaryObj.attach(1));
  gl.bindFramebuffer(gl.FRAMEBUFFER, velocity.write.fbo);
  gl.viewport(0,0,cfg.SIM_RES,cfg.SIM_RES);
  drawQuad();
  velocity.swap();
}

// ---------------------------------------------------------------------------
// SIMULATION STEP
// ---------------------------------------------------------------------------
function simStep(dt) {
  const S=cfg.SIM_RES, D=cfg.DYE_RES;
  const tS=[1/S,1/S], tD=[1/D,1/D];

  // Vorticity curl
  programs.vorticity.use();
  gl.uniform2fv(programs.vorticity.u.uTexelSize, tS);
  gl.uniform1i(programs.vorticity.u.uVelocity, velocity.read.attach(0));
  gl.bindFramebuffer(gl.FRAMEBUFFER, vorticity.fbo);
  gl.viewport(0,0,S,S); drawQuad();

  // Vorticity confinement
  programs.vorticityF.use();
  gl.uniform2fv(programs.vorticityF.u.uTexelSize, tS);
  gl.uniform1i(programs.vorticityF.u.uVelocity,  velocity.read.attach(0));
  gl.uniform1i(programs.vorticityF.u.uVorticity, vorticity.attach(1));
  gl.uniform1f(programs.vorticityF.u.uCurl, cfg.CURL);
  gl.uniform1f(programs.vorticityF.u.uDt,   dt);
  gl.bindFramebuffer(gl.FRAMEBUFFER, velocity.write.fbo);
  gl.viewport(0,0,S,S); drawQuad(); velocity.swap();
  enforceBoundary();

  // Divergence
  programs.divergence.use();
  gl.uniform2fv(programs.divergence.u.uTexelSize, tS);
  gl.uniform1i(programs.divergence.u.uVelocity, velocity.read.attach(0));
  gl.bindFramebuffer(gl.FRAMEBUFFER, divergence.fbo);
  gl.viewport(0,0,S,S); drawQuad();

  // Clear pressure
  gl.bindFramebuffer(gl.FRAMEBUFFER, pressure.read.fbo);
  gl.viewport(0,0,S,S); gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);

  // Pressure solve
  programs.pressure.use();
  gl.uniform2fv(programs.pressure.u.uTexelSize, tS);
  gl.uniform1i(programs.pressure.u.uDivergence, divergence.attach(1));
  for (let i=0;i<cfg.PRESSURE_ITER;i++) {
    gl.uniform1i(programs.pressure.u.uPressure, pressure.read.attach(0));
    gl.bindFramebuffer(gl.FRAMEBUFFER, pressure.write.fbo); drawQuad(); pressure.swap();
  }

  // Gradient subtract
  programs.gradient.use();
  gl.uniform2fv(programs.gradient.u.uTexelSize, tS);
  gl.uniform1i(programs.gradient.u.uPressure, pressure.read.attach(0));
  gl.uniform1i(programs.gradient.u.uVelocity, velocity.read.attach(1));
  gl.bindFramebuffer(gl.FRAMEBUFFER, velocity.write.fbo);
  gl.viewport(0,0,S,S); drawQuad(); velocity.swap();
  enforceBoundary();

  // Advect velocity
  programs.advect.use();
  gl.uniform2fv(programs.advect.u.uTexelSize, tS);
  gl.uniform1f(programs.advect.u.uDt, dt);
  gl.uniform1f(programs.advect.u.uDissipation, cfg.VEL_DISSIPATION);
  gl.uniform1i(programs.advect.u.uVelocity, velocity.read.attach(0));
  gl.uniform1i(programs.advect.u.uSource,   velocity.read.attach(0));
  gl.bindFramebuffer(gl.FRAMEBUFFER, velocity.write.fbo);
  gl.viewport(0,0,S,S); drawQuad(); velocity.swap();
  enforceBoundary();

  // Advect dye
  gl.uniform2fv(programs.advect.u.uTexelSize, tD);
  gl.uniform1f(programs.advect.u.uDissipation, cfg.DISSIPATION);
  gl.uniform1i(programs.advect.u.uVelocity, velocity.read.attach(0));
  gl.uniform1i(programs.advect.u.uSource,   dye.read.attach(1));
  gl.bindFramebuffer(gl.FRAMEBUFFER, dye.write.fbo);
  gl.viewport(0,0,D,D); drawQuad(); dye.swap();
}

// ---------------------------------------------------------------------------
// BUBBLE UPDATE
// ---------------------------------------------------------------------------
function updateBubbles(dt, time) {
  programs.bubbleUpd.use();
  gl.uniform1i(programs.bubbleUpd.u.uParticleTex, bubbles.read.attach(0));
  gl.uniform1i(programs.bubbleUpd.u.uVelocity,    velocity.read.attach(1));
  gl.uniform1i(programs.bubbleUpd.u.uBoundary,    boundaryObj.attach(2));
  gl.uniform1f(programs.bubbleUpd.u.uDt,   dt);
  gl.uniform1f(programs.bubbleUpd.u.uTime, time);
  gl.bindFramebuffer(gl.FRAMEBUFFER, bubbles.write.fbo);
  gl.viewport(0,0,bubbles.size,bubbles.size); drawQuad(); bubbles.swap();
}

// ---------------------------------------------------------------------------
// SPLAT
// ---------------------------------------------------------------------------
function splat(nx, ny, dx, dy, color) {
  const aspect = canvas.width / canvas.height;
  programs.splat.use();
  gl.uniform1f(programs.splat.u.uAspect, aspect);
  gl.uniform1f(programs.splat.u.uRadius, cfg.SPLAT_RADIUS);
  gl.uniform2f(programs.splat.u.uPoint, nx, ny);

  gl.uniform3f(programs.splat.u.uColor, dx, dy, 0);
  gl.uniform1i(programs.splat.u.uTarget, velocity.read.attach(0));
  gl.bindFramebuffer(gl.FRAMEBUFFER, velocity.write.fbo);
  gl.viewport(0,0,cfg.SIM_RES,cfg.SIM_RES); drawQuad(); velocity.swap();

  gl.uniform3fv(programs.splat.u.uColor, color);
  gl.uniform1i(programs.splat.u.uTarget, dye.read.attach(0));
  gl.bindFramebuffer(gl.FRAMEBUFFER, dye.write.fbo);
  gl.viewport(0,0,cfg.DYE_RES,cfg.DYE_RES); drawQuad(); dye.swap();
}

function hsvToRgb(h,s,v) {
  const i=Math.floor(h*6), f=h*6-i;
  const p=v*(1-s), q=v*(1-f*s), t=v*(1-(1-f)*s);
  return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6];
}

function randomSplat() {
  // Stay within fluid area (away from walls and pillars)
  const x = 0.15 + Math.random() * 0.70;
  const y = 0.35 + Math.random() * 0.50; // upper portion, away from pillars
  const a = Math.random() * Math.PI * 2;
  const str = 350 + Math.random() * 700;
  const col = hsvToRgb(0.52 + Math.random() * 0.15, 0.6 + Math.random() * 0.3, 0.8 + Math.random() * 0.2);
  splat(x, y, Math.cos(a)*str, Math.sin(a)*str, col);
}

function tidalWave() {
  // Horizontal wave that crashes through all 3 pillars
  const dir = Math.random() > 0.5 ? 1 : -1;
  const str = 1200;
  splat(0.5, 0.55, dir * str, 0, [0.02, 0.08, 0.18]);
  splat(0.5, 0.45, dir * str * 0.8, 30, [0.02, 0.06, 0.14]);
  splat(0.5, 0.65, dir * str * 0.6, -20, [0.02, 0.07, 0.16]);
}

// ---------------------------------------------------------------------------
// RENDER
// ---------------------------------------------------------------------------
let simTime = 0;

function render() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);

  // Water + room background
  programs.display.use();
  gl.uniform1i(programs.display.u.uDye,      dye.read.attach(0));
  gl.uniform1i(programs.display.u.uVelocity, velocity.read.attach(1));
  gl.uniform1i(programs.display.u.uBoundary, boundaryObj.attach(2));
  gl.uniform1f(programs.display.u.uTime,     simTime);
  gl.uniform1i(programs.display.u.uColorMode, cfg.COLOR_MODE);
  drawQuad();

  // Bubbles on top (additive blend)
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.bindVertexArray(bubbleVAO);
  programs.bubbleRen.use();
  gl.uniform1i(programs.bubbleRen.u.uParticleTex, bubbles.read.attach(0));
  gl.uniform2i(programs.bubbleRen.u.uTexSize, bubbles.size, bubbles.size);
  gl.drawArrays(gl.POINTS, 0, bubbles.count);
  gl.bindVertexArray(null);
  gl.disable(gl.BLEND);
}

// ---------------------------------------------------------------------------
// LOOP
// ---------------------------------------------------------------------------
let lastTime = 0, autoTimer = 0;
let frames = 0, fpsAcc = 0;
const fpsBadge = document.getElementById('fpsBadge');

function loop(ts) {
  requestAnimationFrame(loop);
  const now = ts * 0.001;
  const dt  = Math.min(now - lastTime, 0.05);
  lastTime  = now;

  frames++; fpsAcc += dt;
  if (fpsAcc >= 0.5) {
    if (!cfg.PAUSED) fpsBadge.textContent = Math.round(frames / fpsAcc) + ' fps';
    frames = 0; fpsAcc = 0;
  }

  if (cfg.PAUSED) return;

  resizeCanvas();
  simTime += dt;

  autoTimer += dt;
  if (autoTimer > 2.2) { autoTimer = 0; randomSplat(); }

  flushPointer();
  simStep(dt);
  updateBubbles(dt, simTime);
  render();
}

// ---------------------------------------------------------------------------
// INPUT
// ---------------------------------------------------------------------------
const pointers = new Map();
const pendingSplats = [];

function flushPointer() {
  for (const s of pendingSplats) splat(...s);
  pendingSplats.length = 0;
}

function norm(x, y) { return [x / canvas.clientWidth, 1.0 - y / canvas.clientHeight]; }

canvas.addEventListener('mousedown', e => {
  const [nx, ny] = norm(e.clientX, e.clientY);
  pointers.set('m', { nx, ny, down:true });
  // Single click splat
  const col = hsvToRgb(0.52 + Math.random()*0.15, 0.7, 0.9);
  pendingSplats.push([nx, ny, 0, 0, col]);
});
canvas.addEventListener('mousemove', e => {
  const p = pointers.get('m');
  if (!p?.down) return;
  const [nx, ny] = norm(e.clientX, e.clientY);
  const dx = (nx - p.nx) * canvas.clientWidth  * 7;
  const dy = (ny - p.ny) * canvas.clientHeight * 7;
  const col = hsvToRgb(0.52 + Math.random()*0.12, 0.65, 0.9);
  pendingSplats.push([nx, ny, dx, dy, col]);
  p.nx = nx; p.ny = ny;
});
canvas.addEventListener('mouseup',    () => { const p=pointers.get('m'); if(p) p.down=false; });
canvas.addEventListener('mouseleave', () => { const p=pointers.get('m'); if(p) p.down=false; });

canvas.addEventListener('touchstart', e => {
  e.preventDefault();
  for (const t of e.changedTouches) {
    const [nx,ny] = norm(t.clientX, t.clientY);
    pointers.set(t.identifier, {nx,ny});
  }
}, {passive:false});
canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  for (const t of e.changedTouches) {
    const p = pointers.get(t.identifier); if(!p) continue;
    const [nx,ny] = norm(t.clientX, t.clientY);
    const dx = (nx-p.nx)*canvas.clientWidth*9, dy=(ny-p.ny)*canvas.clientHeight*9;
    const col = hsvToRgb(0.52+Math.random()*0.12, 0.65, 0.9);
    pendingSplats.push([nx,ny,dx,dy,col]);
    p.nx=nx; p.ny=ny;
  }
}, {passive:false});
canvas.addEventListener('touchend', e => { for(const t of e.changedTouches) pointers.delete(t.identifier); });

// Tidal wave on spacebar (only when 2D mode is active)
window.addEventListener('keydown', e => {
  if (e.code === 'Space' && !cfg.PAUSED) { e.preventDefault(); tidalWave(); }
});

// Pause/resume when mode switches
window.addEventListener('vortexmode', e => {
  cfg.PAUSED = (e.detail.mode !== '2d');
});

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
function setupUI() {
  const $ = id => document.getElementById(id);

  const bubbleSlider = $('particleCount2d');
  $('particleVal2d').textContent = BUBBLE_LABELS[cfg.BUBBLE_IDX];
  bubbleSlider.addEventListener('input', () => {
    cfg.BUBBLE_IDX = +bubbleSlider.value;
    $('particleVal2d').textContent = BUBBLE_LABELS[cfg.BUBBLE_IDX];
    initBubbles();
  });

  const vortSlider = $('vorticity');
  vortSlider.addEventListener('input', () => {
    cfg.CURL = +vortSlider.value;
    $('vorticityVal').textContent = vortSlider.value;
  });

  const dissSlider = $('dissipation');
  dissSlider.addEventListener('input', () => {
    cfg.DISSIPATION = +dissSlider.value;
    $('dissipationVal').textContent = (+dissSlider.value).toFixed(2);
  });

  const splatSlider = $('splatSize');
  splatSlider.addEventListener('input', () => {
    cfg.SPLAT_RADIUS = +splatSlider.value * 0.003;
    $('splatVal').textContent = (+splatSlider.value).toFixed(2);
  });

  document.querySelectorAll('#colorModes2d .color-btn2d').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#colorModes2d .color-btn2d').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      cfg.COLOR_MODE = +btn.dataset.mode;
    });
  });

  const pauseBtn = $('pauseBtn2d');
  pauseBtn.addEventListener('click', () => {
    cfg.PAUSED = !cfg.PAUSED;
    pauseBtn.textContent = cfg.PAUSED ? 'Resume' : 'Pause';
    pauseBtn.classList.toggle('active', cfg.PAUSED);
  });

  $('resetBtn2d').addEventListener('click', () => {
    [velocity.read,velocity.write,dye.read,dye.write,
     pressure.read,pressure.write,divergence,vorticity].forEach(f => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, f.fbo);
      gl.viewport(0,0,f.w,f.h); gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
    });
    initBubbles();
    tidalWave(); tidalWave();
  });

  $('tidalBtn').addEventListener('click', tidalWave);

  const ui = $('ui');
  $('toggleUI').addEventListener('click', () => {
    ui.classList.toggle('collapsed');
    $('toggleUI').textContent = ui.classList.contains('collapsed') ? '+' : '−';
  });

  document.querySelectorAll('#panel2d input[type="range"]').forEach(inp => {
    const upd = () => {
      const pct = ((+inp.value - +inp.min) / (+inp.max - +inp.min)) * 100;
      inp.style.setProperty('--pct', pct + '%');
    };
    inp.addEventListener('input', upd); upd();
  });
}

// ---------------------------------------------------------------------------
// BOOT
// ---------------------------------------------------------------------------
(function boot() {
  resizeCanvas();
  if (!initGL()) return;
  initVAOs();
  initSim();
  initPrograms();
  setupUI();

  // Start paused — 3D mode is active by default.
  // The 'vortexmode' event will unpause when user switches to 2D.
  cfg.PAUSED = true;

  // Pre-warm the fluid so it's ready when the user first switches to 2D
  tidalWave();
  setTimeout(() => tidalWave(), 400);
  for (let i = 0; i < 5; i++) randomSplat();

  window.addEventListener('resize', resizeCanvas);
  requestAnimationFrame(loop);
})();
