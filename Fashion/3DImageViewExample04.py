import torch
import numpy as np
from PIL import Image
import gradio as gr
import base64
from io import BytesIO
from transformers import DPTImageProcessor, DPTForDepthEstimation

# DPT 모델 준비
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

def to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def surface_html(image: Image.Image):
    img = image.convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    # 뎁스맵 추론
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    depth = outputs.predicted_depth.squeeze().cpu().numpy()
    # 원본 크기로 리사이즈
    depth_resized = np.array(Image.fromarray(depth).resize((w, h), Image.BILINEAR))
    # 정규화
    depth_norm = (depth_resized - np.min(depth_resized)) / (np.max(depth_resized) - np.min(depth_resized) + 1e-8)
    depth_norm = depth_norm * 80  # z 스케일

    # depth 값을 JS array로 변환
    depth_js = depth_norm.tolist()

    # 원본 이미지를 base64로 변환
    img_b64 = to_base64(img)

    # HTML 템플릿
    html = f"""
    <div id="container3d" style="width: 100%; height: 480px;"></div>
    <script type="module">
    import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.154.0/build/three.module.js';
    let container = document.getElementById('container3d');
    container.innerHTML = '';
    let scene = new THREE.Scene();
    let camera = new THREE.PerspectiveCamera(45, container.offsetWidth / 480, 0.1, 1000);
    let renderer = new THREE.WebGLRenderer({{antialias:true}});
    renderer.setSize(container.offsetWidth, 480);
    container.appendChild(renderer.domElement);

    // 조명
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    let dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(0, 0, 1);
    scene.add(dirLight);

    // 텍스처 로드
    let texture = new THREE.TextureLoader().load('data:image/png;base64,{img_b64}');
    texture.minFilter = THREE.LinearFilter;

    // 뎁스 데이터
    let width = {w};
    let height = {h};
    let depth = {depth_js};

    // geometry 생성
    let geometry = new THREE.PlaneGeometry(1, height/width, width, height);
    // z좌표를 depth로 변경
    let vertices = geometry.attributes.position.array;
    for(let i=0, idx=0; i<vertices.length; i+=3, idx++) {{
        let row = Math.floor(idx / (width+1));
        let col = idx % (width+1);
        if(row < height && col < width) {{
            vertices[i+2] = depth[row][col] * 0.01; // z축 (깊이)
        }}
    }}
    geometry.computeVertexNormals();

    // 매터리얼
    let material = new THREE.MeshPhongMaterial({{
        map: texture,
        side: THREE.DoubleSide,
        shininess: 30,
        flatShading: false,
    }});

    let mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    // 카메라 위치
    camera.position.set(0, -0.6, 1.6);
    camera.lookAt(0, 0, 0.2);

    // OrbitControls
    import {{OrbitControls}} from 'https://cdn.jsdelivr.net/npm/three@0.154.0/examples/jsm/controls/OrbitControls.min.js';
    let controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0.2);

    // 렌더루프
    function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }}
    animate();

    window.onresize = function() {{
        camera.aspect = container.offsetWidth / 480;
        camera.updateProjectionMatrix();
        renderer.setSize(container.offsetWidth, 480);
    }};
    </script>
    """
    return html

iface = gr.Interface(
    fn=surface_html,
    inputs=gr.Image(type="pil", label="2D 이미지 업로드"),
    outputs=gr.HTML(label="WebGL 3D 컬러 텍스처 뷰어"),
    title="DPT Hybrid 기반 2D→3D(컬러 텍스처) WebGL 변환 데모",
    description="원본 이미지를 3D 표면에 그대로 입혀서 WebGL로 실시간 회전·확대 가능하게 시각화합니다."
)

if __name__ == "__main__":
    iface.launch()