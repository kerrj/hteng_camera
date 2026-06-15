import * as THREE from 'three';
import { VRButton } from './vendor/VRButton.js';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  70, window.innerWidth / window.innerHeight, 0.1, 1000);

// Two big inward-facing spheres, one per eye, on layers 1 and 2.
function makeSphere(color, layer) {
  const geo = new THREE.SphereGeometry(500, 60, 40);
  const mat = new THREE.MeshBasicMaterial({ color, side: THREE.BackSide });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.layers.set(layer);
  scene.add(mesh);
  return mesh;
}
const leftSphere = makeSphere(0x208020, 1);   // green to LEFT eye
const rightSphere = makeSphere(0x802020, 2);  // red to RIGHT eye

renderer.xr.addEventListener('sessionstart', () => {
  const xrCam = renderer.xr.getCamera();
  xrCam.cameras[0].layers.enable(1);  // left eye renders layer 1
  xrCam.cameras[1].layers.enable(2);  // right eye renders layer 2
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => renderer.render(scene, camera));
