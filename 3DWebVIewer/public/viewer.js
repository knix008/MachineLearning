import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

class Viewer3D {
  constructor() {
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.currentModel = null;
    this.gridHelper = null;
    this.wireframeMode = false;
    this.originalMaterials = [];
    
    this.init();
    this.setupEventListeners();
    this.animate();
  }

  init() {
    const container = document.getElementById('viewer');
    
    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a1a);
    
    // Camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(5, 5, 5);
    
    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);
    
    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 1000;
    this.controls.enableZoom = true;

    // Track camera distance for zoom display
    this.fitCameraDistance = this.camera.position.length();
    this.controls.addEventListener('change', () => this.updateZoomDisplay());
    
    // Lights (store as properties for later control)
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(this.ambientLight);
    
    this.directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    this.directionalLight1.position.set(5, 10, 7.5);
    this.scene.add(this.directionalLight1);
    
    this.directionalLight2 = new THREE.DirectionalLight(0xffffff, 2.0);
    this.directionalLight2.position.set(-5, 10, -7.5);
    this.scene.add(this.directionalLight2);
    
    // Grid
    this.gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    this.scene.add(this.gridHelper);
    
    // Resize handler
    window.addEventListener('resize', () => this.onWindowResize());

    // Axis indicator scene (rendered in bottom-left corner)
    this.renderer.autoClear = false;
    this.axisScene = new THREE.Scene();
    this.axisScene.background = new THREE.Color(0x0c0c1e);
    this.axisCamera = new THREE.PerspectiveCamera(80, 1, 0.1, 10);
    this.axisScene.add(new THREE.AxesHelper(1));
    this.axisScene.add(new THREE.AmbientLight(0xffffff, 3));
    this.axisScene.add(this.makeAxisLabel('X', '#ff5555', new THREE.Vector3(1.2, 0, 0)));
    this.axisScene.add(this.makeAxisLabel('Y', '#55ff55', new THREE.Vector3(0, 1.2, 0)));
    this.axisScene.add(this.makeAxisLabel('Z', '#5599ff', new THREE.Vector3(0, 0, 1.2)));
  }

  updateZoomDisplay() {
    const dist = this.camera.position.distanceTo(this.controls.target);
    const zoom = this.fitCameraDistance > 0 ? (this.fitCameraDistance / dist) : 1;
    const el = document.getElementById('currentScale');
    if (el) el.textContent = zoom.toFixed(2);
  }

  makeAxisLabel(text, color, position) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = color;
    ctx.font = 'bold 52px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 32, 32);
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, depthTest: false });
    const sprite = new THREE.Sprite(material);
    sprite.position.copy(position);
    sprite.scale.set(0.45, 0.45, 0.45);
    return sprite;
  }

  setupEventListeners() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    // File input change
    fileInput.addEventListener('change', (e) => {
      console.log('File selected:', e.target.files);
      if (e.target.files.length > 0) {
        this.loadFile(e.target.files[0]);
      }
    });
    
    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      console.log('Files dropped:', files);
      if (files.length > 0) {
        this.loadFile(files[0]);
      }
    });
    
    // Controls
    document.getElementById('resetCamera').addEventListener('click', () => {
      this.resetCamera();
    });
    
    document.getElementById('toggleWireframe').addEventListener('click', () => {
      this.toggleWireframe();
    });
    
    document.getElementById('toggleGrid').addEventListener('click', () => {
      this.toggleGrid();
    });
    
    // Settings
    document.getElementById('scaleSlider').addEventListener('input', (e) => {
      const scale = parseFloat(e.target.value);
      document.getElementById('scaleValue').textContent = scale.toFixed(1);
      this.setModelScale(scale);
    });
    
    document.getElementById('rotationSpeed').addEventListener('input', (e) => {
      const speed = parseFloat(e.target.value);
      document.getElementById('rotationSpeedValue').textContent = speed.toFixed(1);
      this.controls.rotateSpeed = speed;
    });
    
    document.getElementById('autoRotate').addEventListener('change', (e) => {
      this.controls.autoRotate = e.target.checked;
      this.controls.autoRotateSpeed = 2.0;
    });
    
    document.getElementById('bgColor').addEventListener('input', (e) => {
      this.scene.background = new THREE.Color(e.target.value);
    });
    
    // Lighting controls
    document.getElementById('ambientLight').addEventListener('input', (e) => {
      const intensity = parseFloat(e.target.value);
      document.getElementById('ambientLightValue').textContent = intensity.toFixed(1);
      this.ambientLight.intensity = intensity;
    });
    
    document.getElementById('directionalLight1').addEventListener('input', (e) => {
      const intensity = parseFloat(e.target.value);
      document.getElementById('directionalLight1Value').textContent = intensity.toFixed(1);
      this.directionalLight1.intensity = intensity;
    });
    
    document.getElementById('directionalLight2').addEventListener('input', (e) => {
      const intensity = parseFloat(e.target.value);
      document.getElementById('directionalLight2Value').textContent = intensity.toFixed(1);
      this.directionalLight2.intensity = intensity;
    });
    
    document.getElementById('lightColor').addEventListener('input', (e) => {
      const color = new THREE.Color(e.target.value);
      this.directionalLight1.color = color;
      this.directionalLight2.color = color;
    });
  }

  loadFile(file) {
    const fileName = file.name;
    const fileExtension = fileName.split('.').pop().toLowerCase();
    
    this.showLoading(true);
    this.updateFileInfo(file);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const contents = e.target.result;
      
      switch(fileExtension) {
        case 'obj':
          this.loadOBJ(contents, fileName);
          break;
        case 'glb':
        case 'gltf':
          this.loadGLTF(contents, fileExtension);
          break;
        case 'fbx':
          this.loadFBX(contents);
          break;
        case 'stl':
          this.loadSTL(contents);
          break;
        default:
          alert('지원하지 않는 파일 형식입니다.');
          this.showLoading(false);
      }
    };
    
    if (fileExtension === 'glb' || fileExtension === 'fbx' || fileExtension === 'stl') {
      reader.readAsArrayBuffer(file);
    } else {
      reader.readAsText(file);
    }
  }

  loadOBJ(contents, fileName) {
    const loader = new OBJLoader();
    try {
      const object = loader.parse(contents);
      this.addModelToScene(object);
      this.showLoading(false);
    } catch (error) {
      console.error('OBJ 로딩 오류:', error);
      alert('OBJ 파일을 로드하는데 실패했습니다.');
      this.showLoading(false);
    }
  }

  loadGLTF(contents, extension) {
    const loader = new GLTFLoader();
    try {
      loader.parse(contents, '', (gltf) => {
        this.addModelToScene(gltf.scene);
        this.showLoading(false);
      }, (error) => {
        console.error('GLTF 로딩 오류:', error);
        alert('GLTF/GLB 파일을 로드하는데 실패했습니다.');
        this.showLoading(false);
      });
    } catch (error) {
      console.error('GLTF 파싱 오류:', error);
      alert('GLTF/GLB 파일을 로드하는데 실패했습니다.');
      this.showLoading(false);
    }
  }

  loadFBX(contents) {
    const loader = new FBXLoader();
    try {
      const object = loader.parse(contents);
      this.addModelToScene(object);
      this.showLoading(false);
    } catch (error) {
      console.error('FBX 로딩 오류:', error);
      alert('FBX 파일을 로드하는데 실패했습니다.');
      this.showLoading(false);
    }
  }

  loadSTL(contents) {
    const loader = new STLLoader();
    try {
      const geometry = loader.parse(contents);
      const material = new THREE.MeshPhongMaterial({ 
        color: 0x888888,
        specular: 0x111111,
        shininess: 200
      });
      const mesh = new THREE.Mesh(geometry, material);
      this.addModelToScene(mesh);
      this.showLoading(false);
    } catch (error) {
      console.error('STL 로딩 오류:', error);
      alert('STL 파일을 로드하는데 실패했습니다.');
      this.showLoading(false);
    }
  }

  addModelToScene(model) {
    // Remove previous model
    if (this.currentModel) {
      this.scene.remove(this.currentModel);
      this.disposeObject(this.currentModel);
    }
    
    this.currentModel = model;
    this.originalMaterials = [];
    
    // Store original materials
    this.currentModel.traverse((child) => {
      if (child.isMesh) {
        this.originalMaterials.push({
          mesh: child,
          material: child.material
        });
      }
    });
    
    this.scene.add(this.currentModel);
    
    // Center and fit the model
    this.centerAndFitModel();
    
    // Update model statistics
    this.updateModelStats();
  }

  centerAndFitModel() {
    const box = new THREE.Box3().setFromObject(this.currentModel);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    
    // Center the model
    this.currentModel.position.sub(center);
    
    // Calculate camera distance
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this.camera.fov * (Math.PI / 180);
    let cameraDistance = Math.abs(maxDim / Math.sin(fov / 2)) * 1.5;
    
    // Update camera position
    this.camera.position.set(cameraDistance, cameraDistance, cameraDistance);
    this.camera.lookAt(0, 0, 0);
    this.controls.target.set(0, 0, 0);
    this.controls.update();

    // Reset zoom baseline
    this.fitCameraDistance = this.camera.position.distanceTo(this.controls.target);
    this.updateZoomDisplay();
  }

  resetCamera() {
    if (this.currentModel) {
      this.centerAndFitModel();
    } else {
      this.camera.position.set(5, 5, 5);
      this.camera.lookAt(0, 0, 0);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
      this.fitCameraDistance = this.camera.position.length();
      this.updateZoomDisplay();
    }
  }

  toggleWireframe() {
    this.wireframeMode = !this.wireframeMode;
    
    if (this.currentModel) {
      this.currentModel.traverse((child) => {
        if (child.isMesh) {
          if (this.wireframeMode) {
            child.material = new THREE.MeshBasicMaterial({ 
              color: 0x00ff00, 
              wireframe: true 
            });
          } else {
            const original = this.originalMaterials.find(item => item.mesh === child);
            if (original) {
              child.material = original.material;
            }
          }
        }
      });
    }
  }

  toggleGrid() {
    this.gridHelper.visible = !this.gridHelper.visible;
  }

  setModelScale(scale) {
    if (this.currentModel) {
      this.currentModel.scale.setScalar(scale);
    }
  }

  updateFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    const fileSize = (file.size / 1024).toFixed(2);
    const fileType = file.name.split('.').pop().toUpperCase();
    
    fileInfo.innerHTML = `
      <p><strong>파일명:</strong> ${file.name}</p>
      <p><strong>형식:</strong> ${fileType}</p>
      <p><strong>크기:</strong> ${fileSize} KB</p>
      <p><strong>수정일:</strong> ${new Date(file.lastModified).toLocaleDateString('ko-KR')}</p>
    `;
  }

  updateModelStats() {
    let vertexCount = 0;
    let triangleCount = 0;
    
    if (this.currentModel) {
      this.currentModel.traverse((child) => {
        if (child.isMesh && child.geometry) {
          const geometry = child.geometry;
          if (geometry.attributes.position) {
            vertexCount += geometry.attributes.position.count;
          }
          if (geometry.index) {
            triangleCount += geometry.index.count / 3;
          }
        }
      });
      
      const box = new THREE.Box3().setFromObject(this.currentModel);
      const size = box.getSize(new THREE.Vector3());
      
      document.getElementById('vertexCount').textContent = vertexCount.toLocaleString();
      document.getElementById('triangleCount').textContent = Math.floor(triangleCount).toLocaleString();
      document.getElementById('bboxX').textContent = size.x.toFixed(2);
      document.getElementById('bboxY').textContent = size.y.toFixed(2);
      document.getElementById('bboxZ').textContent = size.z.toFixed(2);
      
      document.getElementById('modelStats').style.display = 'block';
    } else {
      document.getElementById('modelStats').style.display = 'none';
    }
  }

  showLoading(show) {
    document.getElementById('loadingSpinner').style.display = show ? 'flex' : 'none';
  }

  disposeObject(object) {
    object.traverse((child) => {
      if (child.isMesh) {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(material => material.dispose());
          } else {
            child.material.dispose();
          }
        }
      }
    });
  }

  onWindowResize() {
    const container = document.getElementById('viewer');
    this.camera.aspect = container.clientWidth / container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();

    const w = this.renderer.domElement.clientWidth;
    const h = this.renderer.domElement.clientHeight;

    // Main scene (full viewport)
    this.renderer.setViewport(0, 0, w, h);
    this.renderer.setScissorTest(false);
    this.renderer.clear(true, true, true);
    this.renderer.render(this.scene, this.camera);

    // Axis indicator (bottom-left) — lookAt approach keeps axes always visible
    const inset = 190;
    const camDir = new THREE.Vector3();
    this.camera.getWorldDirection(camDir);
    this.axisCamera.position.copy(camDir.negate().setLength(1.8));
    this.axisCamera.up.copy(this.camera.up);
    this.axisCamera.lookAt(0, 0, 0);

    this.renderer.setScissorTest(true);
    this.renderer.setScissor(10, 10, inset, inset);
    this.renderer.setViewport(10, 10, inset, inset);
    this.renderer.clear(true, true, true);
    this.renderer.render(this.axisScene, this.axisCamera);
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, w, h);
  }
}

// Initialize viewer when page loads
window.addEventListener('DOMContentLoaded', () => {
  new Viewer3D();
});
