// Kannala-Brandt fisheye sampling. vDir is the sphere-local outward direction
// (sphere orientation encodes the lens frame). See plan "Convention reference".
export const VERT = `
varying vec3 vDir;
void main() {
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

export const FRAG = `
precision highp float;
varying vec3 vDir;
uniform sampler2D map;
uniform float fx, fy, cx, cy;
uniform float k1, k2, k3, k4;
uniform float imgW, imgH, maxTheta;
void main() {
  // Three local frame -> OpenCV lens frame (x right, y down, z forward).
  // Negate x: we view the texture on the *inside* of the sphere (BackSide),
  // which mirrors it horizontally; this flip restores correct handedness.
  vec3 d = normalize(vec3(-vDir.x, -vDir.y, -vDir.z));
  float rxy = length(d.xy);
  float theta = atan(rxy, d.z);
  if (theta > maxTheta) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  float t2 = theta * theta;
  float thetad = theta * (1.0 + k1*t2 + k2*t2*t2 + k3*t2*t2*t2 + k4*t2*t2*t2*t2);
  float scale = rxy > 1e-6 ? thetad / rxy : 1.0;
  float u = (fx * d.x * scale + cx) / imgW;
  float v = (fy * d.y * scale + cy) / imgH;
  if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return;     // outside captured image
  }
  gl_FragColor = texture2D(map, vec2(u, v));              // flipY=false: image row 0 is at v=0
}`;
