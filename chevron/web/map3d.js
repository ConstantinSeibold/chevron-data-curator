// Chevron — 3D latent walk.
//
// Ported from Spacewalker (https://github.com/ConstantinSeibold/Spacewalker),
// MIT License, Copyright (c) 2024 Lukas Heine — see the third-party notices in LICENSE.
// Generalised here so a point is an INSTANCE (a mask crop) rather than only a whole sample, and so
// that painting writes into Chevron's shared selection instead of a viewer-local one.
//
// An ES module loaded on demand: three.js is ~330 KB and most sessions never open 3D, so the import
// only happens on the first switch. No bundler — vendored ESM plus an import map.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COLORS = [0x4f8cff, 0x3fb27f, 0xe0a13d, 0xc163d8, 0xd8635f, 0x5bc8d8, 0x9aa3b2];
const SELECTED = 0xffffff;

export function createMap3D(canvas, { onSelectionChange, getSelected, colorOf }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0c10);
  const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
  camera.position.set(1.6, 1.2, 1.6);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 0.6);
  key.position.set(2, 3, 2);
  scene.add(key);

  // A grid box makes depth readable — a point cloud with no frame is impossible to orient in.
  const box = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
    new THREE.LineBasicMaterial({ color: 0x2a2f3a }));
  box.position.set(0.5, 0.5, 0.5);
  scene.add(box);

  let mesh = null, pts = [], radius = 0.012, brush = 0.10, paintMode = false, disposed = false;
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  function build(points) {
    pts = points || [];
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); mesh = null; }
    if (!pts.length) return;
    // ONE InstancedMesh for the whole cloud: 100k+ points stay interactive because there is a single
    // draw call, which is the reason to use three.js here rather than 100k sprites.
    mesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(1, 8, 6),
      new THREE.MeshLambertMaterial({ vertexColors: true }),
      pts.length);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(mesh);
    layout();
    recolor();
  }

  function layout() {
    if (!mesh) return;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      dummy.position.set(p.x, p.y, p.z ?? 0.5);
      dummy.scale.setScalar(radius);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }

  function recolor() {
    if (!mesh) return;
    const sel = getSelected();
    for (let i = 0; i < pts.length; i++) {
      const u = pts[i].iuid;
      color.set(sel.has(u) ? SELECTED : (colorOf ? colorOf(pts[i]) : COLORS[i % COLORS.length]));
      mesh.setColorAt(i, color);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  // ---- paint-select: drag with the brush to add, hold Alt to erase --------
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let dragging = false;

  function pickAround(ev, erase) {
    if (!mesh) return;
    const r = canvas.getBoundingClientRect();
    ndc.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
    ndc.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(ndc, camera);
    // Distance from each point to the pick RAY, in world units: a screen-space brush would change
    // meaning with zoom, which makes a 3D selection feel arbitrary.
    const sel = getSelected();
    let changed = false;
    const v = new THREE.Vector3();
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      v.set(p.x, p.y, p.z ?? 0.5);
      if (ray.ray.distanceToPoint(v) <= brush) {
        const u = p.iuid;
        if (erase ? sel.delete(u) : (!sel.has(u) && (sel.add(u), true))) changed = true;
      }
    }
    if (changed) { recolor(); onSelectionChange && onSelectionChange(); }
  }

  const onDown = (e) => { if (!paintMode || e.button !== 0) return;
    dragging = true; controls.enabled = false; pickAround(e, e.altKey); };
  const onMove = (e) => { if (dragging) pickAround(e, e.altKey); };
  const onUp = () => { dragging = false; controls.enabled = true; };
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  addEventListener("pointerup", onUp);

  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  (function loop() {
    if (disposed) return;
    requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, camera);
  })();

  return {
    build, recolor, resize,
    setPointSize(r) { radius = r; layout(); },
    setBrush(b) { brush = b; },
    setPaintMode(on) { paintMode = !!on; controls.enabled = !on || !dragging; },
    frame() { controls.target.set(0.5, 0.5, 0.5); camera.position.set(1.6, 1.2, 1.6); controls.update(); },
    // the query pin from P6: a marker for a point that is NOT one of the instances
    pin(p) {
      if (this._pin) { scene.remove(this._pin); this._pin.geometry.dispose(); this._pin.material.dispose(); }
      if (!p) { this._pin = null; return; }
      const m = new THREE.Mesh(new THREE.SphereGeometry(radius * 2.6, 12, 10),
                               new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true }));
      m.position.set(p.x, p.y, p.z ?? 0.5);
      scene.add(m); this._pin = m;
    },
    dispose() {
      disposed = true;
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      removeEventListener("pointerup", onUp);
      controls.dispose(); renderer.dispose();
    },
  };
}
