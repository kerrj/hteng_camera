import * as THREE from 'three';
import { VRButton } from './vendor/VRButton.js';
import { VERT, FRAG } from './shaders.js';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const msg = document.getElementById('msg');
if (!window.isSecureContext) {
  msg.textContent = 'Not a secure context — open the tethered http://localhost URL, '
    + 'or accept the HTTPS cert on the wifi URL. WebXR will not start otherwise.';
} else if (!navigator.xr) {
  msg.textContent = 'WebXR not available in this browser.';
} else {
  navigator.xr.isSessionSupported('immersive-vr').then((ok) => {
    if (!ok) msg.textContent = 'immersive-vr not supported on this device.';
  });
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  80, window.innerWidth / window.innerHeight, 0.1, 1000);

function makeTexture() {
  const t = new THREE.Texture();
  t.flipY = false;                       // we flip v in the shader
  t.colorSpace = THREE.SRGBColorSpace;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  return t;
}

function makeEye(layer) {
  const tex = makeTexture();
  const mat = new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, side: THREE.BackSide,
    uniforms: {
      map: { value: tex },
      fx: { value: 1 }, fy: { value: 1 }, cx: { value: 0 }, cy: { value: 0 },
      k1: { value: 0 }, k2: { value: 0 }, k3: { value: 0 }, k4: { value: 0 },
      imgW: { value: 1 }, imgH: { value: 1 }, maxTheta: { value: 1.4 },
    },
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(500, 64, 48), mat);
  mesh.layers.set(layer);
  scene.add(mesh);
  return { tex, mat, mesh };
}
const eyes = { left: makeEye(1), right: makeEye(2) };

function applyCalib(calib) {
  for (const name of ['left', 'right']) {
    const u = eyes[name].mat.uniforms, c = calib[name];
    u.fx.value = c.fx; u.fy.value = c.fy; u.cx.value = c.cx; u.cy.value = c.cy;
    u.k1.value = c.dist[0]; u.k2.value = c.dist[1];
    u.k3.value = c.dist[2]; u.k4.value = c.dist[3];
    u.imgW.value = c.width; u.imgH.value = c.height;
    u.maxTheta.value = calib.maxTheta;
  }
  // Co-align the right eye: B * R^T * B (B = diag(1,-1,-1)), R row-major OpenCV.
  const R = calib.R;
  const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]; // transpose
  const b = [1, -1, -1];
  const m = new THREE.Matrix3();
  // (B*Rt*B)[i][j] = b[i]*Rt[i][j]*b[j]
  m.set(
    b[0]*Rt[0]*b[0], b[0]*Rt[1]*b[1], b[0]*Rt[2]*b[2],
    b[1]*Rt[3]*b[0], b[1]*Rt[4]*b[1], b[1]*Rt[5]*b[2],
    b[2]*Rt[6]*b[0], b[2]*Rt[7]*b[1], b[2]*Rt[8]*b[2]);
  const e = m.elements; // column-major
  const m4 = new THREE.Matrix4().set(
    e[0], e[3], e[6], 0,
    e[1], e[4], e[7], 0,
    e[2], e[5], e[8], 0,
    0,    0,    0,    1);
  eyes.right.mesh.quaternion.setFromRotationMatrix(m4);
}

function setEyeImage(name, bitmap) {
  const tex = eyes[name].tex;
  tex.image = bitmap;
  tex.needsUpdate = true;
}

// ---- WebSocket: text calib, then [eyeByte]+jpeg binary frames ----
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = async (ev) => {
    if (typeof ev.data === 'string') { applyCalib(JSON.parse(ev.data)); return; }
    const bytes = new Uint8Array(ev.data);
    const name = bytes[0] === 0 ? 'left' : 'right';
    const blob = new Blob([bytes.subarray(1)], { type: 'image/jpeg' });
    setEyeImage(name, await createImageBitmap(blob));
  };
  ws.onclose = () => setTimeout(connect, 1000);   // reconnect without restart
}
connect();

// ---- per-eye layer routing in XR ----
renderer.xr.addEventListener('sessionstart', () => {
  const xrCam = renderer.xr.getCamera();
  xrCam.cameras[0].layers.enable(1);
  xrCam.cameras[1].layers.enable(2);
});

// ---- keep spheres centered on the head (ignore translation, keep rotation) ----
const headPos = new THREE.Vector3();
function recenter(cam) {
  cam.getWorldPosition(headPos);
  eyes.left.mesh.position.copy(headPos);
  eyes.right.mesh.position.copy(headPos);
}

// ---- desktop (non-XR) preview: drag to look, render LEFT eye on both layers ----
let yaw = 0, pitch = 0, dragging = false, px = 0, py = 0;
camera.layers.enable(1);
addEventListener('pointerdown', (e) => { dragging = true; px = e.clientX; py = e.clientY; });
addEventListener('pointerup', () => { dragging = false; });
addEventListener('pointermove', (e) => {
  if (!dragging) return;
  yaw -= (e.clientX - px) * 0.005; pitch -= (e.clientY - py) * 0.005;
  pitch = Math.max(-1.5, Math.min(1.5, pitch));
  px = e.clientX; py = e.clientY;
  camera.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

renderer.setAnimationLoop(() => {
  const cam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
  recenter(cam);
  renderer.render(scene, camera);
});
