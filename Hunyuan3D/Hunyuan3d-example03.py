import gradio as gr
import tempfile
import os
from pathlib import Path
from PIL import Image
from rembg import remove
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# 전역으로 파이프라인 로드 (한 번만 로드)
print("Loading Hunyuan3D model...")
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
print("Model loaded successfully!")

def generate_3d_model(input_image):
    """
    입력 이미지로부터 3D 메시를 생성하고 OBJ와 GLB 파일로 저장
    
    Args:
        input_image: 입력 이미지 (PIL Image 또는 파일 경로)
    
    Returns:
        tuple: (배경제거이미지, obj_파일경로, glb_파일경로, 상태메시지)
    """
    if input_image is None:
        return None, None, None, "⚠️ 이미지를 업로드해주세요."
    
    try:
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        nobg_path = os.path.join(temp_dir, "nobg.png")
        obj_path = os.path.join(temp_dir, "output.obj")
        glb_path = os.path.join(temp_dir, "output.glb")
        
        # 1단계: 배경 제거
        print(f"Removing background...")
        input_pil = Image.open(input_image) if isinstance(input_image, str) else input_image
        output_pil = remove(input_pil)
        output_pil.save(nobg_path)
        print(f"Background removed and saved to {nobg_path}")
        
        # 2단계: 3D 메시 생성 (배경 제거된 이미지 사용)
        print(f"Generating 3D mesh from image...")
        mesh = pipeline(image=nobg_path)[0]
        
        # OBJ 및 GLB 형식으로 저장
        mesh.export(obj_path)
        mesh.export(glb_path)
        
        # 메시 정보
        status_msg = f"✅ 생성 완료!\n\n"
        status_msg += f"📋 처리 단계:\n"
        status_msg += f"1. ✓ 배경 제거 완료\n"
        status_msg += f"2. ✓ 3D 메시 생성 완료\n\n"
        status_msg += f"📊 메시 정보:\n"
        status_msg += f"- 정점(Vertices): {len(mesh.vertices):,}\n"
        status_msg += f"- 면(Faces): {len(mesh.faces):,}\n\n"
        status_msg += f"💾 파일:\n"
        status_msg += f"- 배경제거: {nobg_path}\n"
        status_msg += f"- OBJ: {obj_path}\n"
        status_msg += f"- GLB: {glb_path}"
        
        print(status_msg)
        
        return nobg_path, obj_path, glb_path, status_msg
    
    except Exception as e:
        error_msg = f"❌ 오류 발생: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, None, error_msg

# Gradio 인터페이스 구성
with gr.Blocks(title="Hunyuan3D 2 - 3D Model Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 Hunyuan3D 2 - 이미지에서 3D 모델 생성
    
    단일 이미지를 업로드하면 자동으로 3D 메시를 생성합니다.
    생성된 모델은 OBJ와 GLB 형식으로 다운로드할 수 있습니다.
    """)
    
    with gr.Row():
        # 왼쪽: 입력 이미지
        with gr.Column(scale=1):
            gr.Markdown("### 📥 입력 이미지")
            input_image = gr.Image(
                label="이미지 업로드",
                type="filepath",
                sources=["upload", "clipboard"],
                height=350
            )
            generate_btn = gr.Button("🚀 3D 모델 생성", variant="primary", size="lg")
        
        # 중간: 배경 제거 결과
        with gr.Column(scale=1):
            gr.Markdown("### 🎭 배경 제거")
            nobg_image = gr.Image(
                label="배경 제거된 이미지",
                type="filepath",
                height=350,
                interactive=False
            )
        
        # 오른쪽: 출력 3D 모델 (탭으로 OBJ/GLB 구분)
        with gr.Column(scale=1):
            gr.Markdown("### 📤 생성된 3D 모델")
            with gr.Tabs():
                with gr.Tab("OBJ 뷰어"):
                    output_obj = gr.Model3D(
                        label="OBJ 포맷",
                        height=350,
                        clear_color=[0.8, 0.8, 0.8, 1.0]
                    )
                
                with gr.Tab("GLB 뷰어"):
                    output_glb = gr.Model3D(
                        label="GLB 포맷",
                        height=350,
                        clear_color=[0.8, 0.8, 0.8, 1.0]
                    )
    
    # 상태 표시 (전체 폭)
    with gr.Row():
        status_text = gr.Textbox(
            label="처리 상태",
            lines=8,
            interactive=False,
            placeholder="이미지를 업로드하고 '3D 모델 생성' 버튼을 클릭하세요."
        )
    
    # 예제 이미지 (demo.png가 있다면 사용)
    if os.path.exists("demo.png"):
        gr.Examples(
            examples=[["demo.png"]],
            inputs=input_image,
            label="예제 이미지"
        )
    
    gr.Markdown("""
    ---
    ### 📝 사용 방법:
    1. 왼쪽에서 이미지를 업로드합니다
    2. '3D 모델 생성' 버튼을 클릭합니다
    3. 중간에 배경이 제거된 이미지를 확인합니다
    4. 오른쪽 탭에서 생성된 3D 모델을 확인하고 다운로드합니다
    
    ### ⚠️ 참고사항:
    - 배경 제거를 위해 rembg를 사용합니다 (투명 배경 생성)
    - 생성 시간은 이미지 크기와 시스템 성능에 따라 다를 수 있습니다
    - GPU가 있으면 더 빠르게 생성됩니다
    - 텍스처 페인팅 기능은 custom_rasterizer 확장이 필요합니다
    """)
    
    # 이벤트 연결
    generate_btn.click(
        fn=generate_3d_model,
        inputs=[input_image],
        outputs=[nobg_image, output_obj, output_glb, status_text]
    )

# 서버 실행
if __name__ == "__main__":
    demo.launch(inbrowser=True)