// static/js/avatar.js
(() => {
  // comprueba que Three esté disponible
  if (typeof THREE === "undefined") {
    console.error("Three.js no cargado");
    return;
  }

  const container = document.getElementById("three-container");
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Escena, cámara, renderer
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 1000);
  camera.position.set(0, 1.6, 3.5);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(width, height);
  renderer.outputEncoding = THREE.sRGBEncoding;
  container.appendChild(renderer.domElement);

  // Iluminación
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
  hemi.position.set(0, 20, 0);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(3, 10, 10);
  dir.castShadow = true;
  scene.add(dir);

  // Suave luz ambiental para dar "calor solar" (se puede cambiar según predicción)
  const sunLight = new THREE.PointLight(0xfff1c2, 0.3, 50);
  sunLight.position.set(-5, 6, -2);
  scene.add(sunLight);

  // Controls (opcional, para que puedas rotar manualmente)
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.2, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.enablePan = false;
  controls.minDistance = 1.8;
  controls.maxDistance = 6;

  // Piso sutil
  const planeGeo = new THREE.PlaneGeometry(10, 10);
  const planeMat = new THREE.MeshStandardMaterial({ color: 0xf5f5f5, roughness: 1, metalness: 0 });
  const plane = new THREE.Mesh(planeGeo, planeMat);
  plane.rotation.x = -Math.PI/2;
  plane.position.y = 0;
  plane.receiveShadow = true;
  scene.add(plane);

  // Cargar GLB
  const loader = new THREE.GLTFLoader();
  let avatar = null;
  let mixer = null;
  let bonesOrMeshes = [];
  loader.load(
    "C:\\Users\\Juan\\prediccion_produccion_solar\\static\\models\\RobotExpressive.glb",
    (gltf) => {
      avatar = gltf.scene || gltf.scenes[0];
      avatar.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          bonesOrMeshes.push(child);
          // habilita que el material pueda cambiarse con color/emissive si existe
          if (child.material) {
            child.material = child.material.clone();
            child.material.emissive = child.material.emissive || new THREE.Color(0x000000);
          }
        }
      });

      // Ajustes de escala / posición
      avatar.position.set(0, 0, 0);
      avatar.scale.set(1.0, 1.0, 1.0);

      scene.add(avatar);

      // Si hay animaciones, prepara el mixer
      if (gltf.animations && gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(avatar);
        gltf.animations.forEach((clip) => {
          mixer.clipAction(clip).play();
        });
      }

      // Primera reacción según predicción inicial (si existe)
      applyReaction(window.PREDICTION_VALUE);
    },
    (xhr) => {
      // progreso
      // console.log((xhr.loaded / xhr.total * 100) + '% loaded');
    },
    (err) => {
      console.error("Error cargando avatar.glb:", err);
    }
  );

  // Animación loop
  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    if (mixer) mixer.update(delta);

    // Si el avatar está cargado, haz un leve movimiento "respiración" para vida
    if (avatar) {
      const t = performance.now() * 0.001;
      avatar.position.y = 0.02 * Math.sin(t * 1.5);
      avatar.rotation.y = 0.06 * Math.sin(t * 0.6);
    }

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Actualiza tamaño al redimensionar
  window.addEventListener("resize", () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  // Función para aplicar reacción según valor (puedes ajustar umbrales y efectos)
  function applyReaction(value) {
    if (!avatar) return; // si el avatar aún no carga, no hacer nada

    // Umbrales ejemplo (W)
    const high = 800;
    const medium = 300;

    if (value === null || value === undefined) {
      // sin predicción: expresión neutra
      setEmissive(0x222222, 0.2);
      setScale(1.0);
      playAllAnimations(0.6);
      setSunIntensity(0.25);
      return;
    }

    if (value >= high) {
      // Muy buena producción → avatar contento, color cálido, animación animada
      setEmissive(0xffd27f, 0.55);
      setScale(1.06);
      playAllAnimations(1.4);
      setSunIntensity(0.55);
    } else if (value >= medium) {
      // Producción media → expresión positiva moderada
      setEmissive(0xa6e3a1, 0.35);
      setScale(1.02);
      playAllAnimations(1.0);
      setSunIntensity(0.35);
    } else {
      // Baja producción → avatar serio, colores fríos, animación lenta
      setEmissive(0x90caf9, 0.2);
      setScale(0.98);
      playAllAnimations(0.6);
      setSunIntensity(0.18);
    }
  }

  // Cambiar emissive/color en todos los meshes
  function setEmissive(hexColor, intensity = 0.4) {
    bonesOrMeshes.forEach((m) => {
      if (m.material) {
        try {
          m.material.emissive = new THREE.Color(hexColor);
          m.material.emissiveIntensity = intensity;
          // también cambiar color si el material tiene 'color'
          if (m.material.color) m.material.color.lerp(new THREE.Color(0xffffff), 0.05);
        } catch (e) {
          // algunos materiales no permiten cambios simples; ignorar
        }
      }
    });
  }

  // Escalar suavemente
  function setScale(target) {
    if (!avatar) return;
    // animación sencilla con tween manual
    const duration = 300;
    const start = performance.now();
    const initial = avatar.scale.x;
    function step(now) {
      const t = Math.min(1, (now - start) / duration);
      const v = initial + (target - initial) * (0.5 - 0.5 * Math.cos(Math.PI * t)); // ease
      avatar.scale.set(v, v, v);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // Control de animaciones (velocidad)
  function playAllAnimations(speed = 1.0) {
    if (!mixer) return;
    mixer.timeScale = speed;
    // ajustar peso de las acciones si fuera necesario
    mixer._actions && mixer._actions.forEach(a => a.setEffectiveTimeScale(speed));
  }

  // Ajustar intensidad de luz "solar" para efecto visual
  function setSunIntensity(val) {
    sunLight.intensity = val;
  }

  // Exponer función para cambiar reacción desde consola o desde otras partes del JS
  window.avatarReactTo = applyReaction;

  // Si la predicción cambia (por ejemplo, tras una nueva petición AJAX),
  // puedes llamar window.avatarReactTo(nuevo_valor)
})();
