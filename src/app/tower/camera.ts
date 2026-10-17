// An orbit camera for the tower: it circles a target point and zooms toward
// whatever is under the pointer. Kept in double precision; the GPU only ever
// sees positions relative to the eye.

export interface View {
  target: [number, number, number];
  dist: number;
  yaw: number; // radians around the vertical axis
  pitch: number; // radians above the horizontal
}

export const FOV = (50 * Math.PI) / 180;

export function eyeOf(v: View): [number, number, number] {
  const cp = Math.cos(v.pitch);
  return [v.target[0] + v.dist * cp * Math.sin(v.yaw), v.target[1] + v.dist * Math.sin(v.pitch), v.target[2] + v.dist * cp * Math.cos(v.yaw)];
}

// The camera's axes: right, up, and forward (toward the target).
function basis(v: View) {
  const e = eyeOf(v);
  const f = norm([v.target[0] - e[0], v.target[1] - e[1], v.target[2] - e[2]]);
  const r = norm(cross(f, [0, 1, 0]));
  const u = cross(r, f);
  return { f, r, u };
}

// View rotation times a reversed-depth, infinitely far perspective, for
// positions already relative to the eye. Column-major.
export function viewProj(v: View, aspect: number): Float32Array {
  const { f, r, u } = basis(v);
  const near = Math.max(0.002, v.dist * 0.002);
  const t = 1 / Math.tan(FOV / 2);
  // view: rows are r, u, -f
  const V = [r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, 0, 0, 0, 1];
  const P = [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, 0, -1, 0, 0, near, 0];
  return new Float32Array(mul(P, V));
}

// The ray from the eye through a point on screen (in -1..1 coordinates).
export function ray(v: View, aspect: number, sx: number, sy: number) {
  const { f, r, u } = basis(v);
  const t = Math.tan(FOV / 2);
  const d = norm([f[0] + r[0] * sx * t * aspect + u[0] * sy * t, f[1] + r[1] * sx * t * aspect + u[1] * sy * t, f[2] + r[2] * sx * t * aspect + u[2] * sy * t]);
  return { origin: eyeOf(v), dir: d };
}

// Where a ray meets the horizontal plane at height y, if in front of the eye.
export function hitPlane(o: number[], d: number[], y: number) {
  if (Math.abs(d[1]) < 1e-9) return null;
  const s = (y - o[1]) / d[1];
  if (s <= 0) return null;
  return [o[0] + d[0] * s, y, o[2] + d[2] * s] as [number, number, number];
}

// Screen position (pixels) of a world point, or null if behind the eye.
export function project(v: View, aspect: number, width: number, height: number, p: number[]) {
  const e = eyeOf(v);
  const { f, r, u } = basis(v);
  const q = [p[0] - e[0], p[1] - e[1], p[2] - e[2]];
  const z = dot(q, f);
  if (z <= v.dist * 0.002) return null;
  const t = Math.tan(FOV / 2);
  const x = dot(q, r) / (z * t * aspect);
  const y = dot(q, u) / (z * t);
  return { x: ((x + 1) / 2) * width, y: ((1 - y) / 2) * height, depth: z };
}

const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function norm(a: number[]) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
function mul(a: number[], b: number[]) {
  const out = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return out;
}
