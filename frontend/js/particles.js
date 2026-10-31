(() => {
  const canvas = document.getElementById("flow-canvas");
  if (!canvas || typeof THREE === "undefined") return;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xf4f4f4, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0.4, 0.1, 6.2);

  const count = 5200;
  const positions = new Float32Array(count * 3);
  const seeds = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const u = Math.random();
    const v = Math.random();
    const theta = u * Math.PI;
    const phi = v * Math.PI * 2;
    const r = 1.55 + Math.pow(Math.random(), 1.8) * 1.8;
    const x = r * Math.sin(theta) * Math.cos(phi) + 1.8;
    const y = r * Math.cos(theta) * 1.15;
    const z = r * Math.sin(theta) * Math.sin(phi);
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    seeds[i] = Math.random() * Math.PI * 2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: 0x101010,
    size: 0.028,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.92,
  });
  const points = new THREE.Points(geometry, material);
  scene.add(points);

  const squareGeo = new THREE.PlaneGeometry(0.28, 0.34);
  const squareMat = new THREE.MeshBasicMaterial({ color: 0xd8d4cc, transparent: true, opacity: 0.55 });
  const square = new THREE.Mesh(squareGeo, squareMat);
  square.position.set(-1.6, -0.2, 1.4);
  square.rotation.z = -0.35;
  scene.add(square);

  const resize = () => {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  window.addEventListener("resize", resize);

  let mx = 0;
  let my = 0;
  window.addEventListener("pointermove", (e) => {
    mx = (e.clientX / window.innerWidth) * 2 - 1;
    my = (e.clientY / window.innerHeight) * 2 - 1;
  });

  const tick = (t) => {
    const pos = geometry.attributes.position.array;
    const time = t * 0.001;
    for (let i = 0; i < count; i++) {
      const s = seeds[i];
      pos[i * 3 + 1] += Math.sin(time * 0.6 + s) * 0.00035;
    }
    geometry.attributes.position.needsUpdate = true;
    points.rotation.y = time * 0.04 + mx * 0.12;
    points.rotation.x = my * 0.06;
    square.position.x = -1.6 + mx * 0.15;
    square.position.y = -0.2 - my * 0.1;
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  document.querySelectorAll("[data-viz]").forEach((el) => {
    const bars = 48;
    el.innerHTML = "";
    el.style.display = "flex";
    el.style.alignItems = "flex-end";
    el.style.gap = "4px";
    el.style.padding = "18px";
    for (let i = 0; i < bars; i++) {
      const b = document.createElement("span");
      const h = 12 + Math.abs(Math.sin(i * 0.45 + el.dataset.viz.length)) * 70;
      b.style.flex = "1";
      b.style.height = `${h}%`;
      b.style.background = i % 7 === 0 ? "#ececec" : "#6d6d6d";
      el.appendChild(b);
    }
  });
})();
