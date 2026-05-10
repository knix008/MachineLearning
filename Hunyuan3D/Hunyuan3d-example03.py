import gradio as gr
import tempfile
import os
import time
import signal
import sys
from pathlib import Path
from PIL import Image
from rembg import remove
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# 텍스처 파이프라인은 선택적으로 로드
texture_pipeline = None
TEXTURE_ENABLED = False

# 전역으로 파이프라인 로드 (한 번만 로드)
print("Loading Hunyuan3D models...")
print("  [1/2] Loading shape generation model...")
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")

try:
    print("  [2/2] Loading texture generation model...")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    TEXTURE_ENABLED = True
    print("✓ Models loaded successfully (with texture support)!")
except Exception as e:
    print(f"⚠️  Texture generation not available: {e}")
    print("✓ Shape generation model loaded (texture generation disabled)")
    print("💡 To enable texture: install custom_rasterizer and differentiable_renderer modules")

# CTRL+C 종료 시 자원 해제 함수
def cleanup_and_exit(signum=None, frame=None):
    """
    CTRL+C 종료 시 GPU 메모리 및 파이프라인 자원 해제
    """
    print("\n\n⚠️  종료 신호 감지... 자원을 해제합니다.")
    
    try:
        # 파이프라인 해제
        global shape_pipeline, texture_pipeline
        if shape_pipeline is not None:
            print("📦 Shape 파이프라인 모델 해제 중...")
            del shape_pipeline
            shape_pipeline = None
        if texture_pipeline is not None:
            print("📦 Texture 파이프라인 모델 해제 중...")
            del texture_pipeline
            texture_pipeline = None
        
        # GPU 캐시 정리 (PyTorch 사용 시)
        try:
            import torch
            if torch.cuda.is_available():
                print("🔧 GPU 메모리 해제 중...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        print("✅ 자원 해제 완료!")
        
    except Exception as e:
        print(f"⚠️  자원 해제 중 오류: {e}")
    
    finally:
        print("👋 프로그램을 종료합니다.")
        sys.exit(0)

# CTRL+C (SIGINT) 시그널 핸들러 등록
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

print("✓ 종료 핸들러 등록 완료 (CTRL+C로 안전하게 종료할 수 있습니다)")

def generate_3d_model(input_image):
    """
    입력 이미지로부터 배경을 제거하고 3D 메시를 생성
    Generator 함수로 중간 결과를 즉시 표시
    
    Args:
        input_image: 입력 이미지 (PIL Image 또는 파일 경로)
    
    Yields:
        tuple: (배경제거이미지, obj_파일경로, glb_파일경로, 상태메시지)
    """
    if input_image is None:
        yield None, None, None, "⚠️ 이미지를 업로드해주세요."
        return
    
    try:
        start_time = time.time()
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        nobg_path = os.path.join(temp_dir, "nobg.png")
        obj_path = os.path.join(temp_dir, "output.obj")
        glb_path = os.path.join(temp_dir, "output.glb")
        
        # 1단계: 배경 제거 (0% → 40%)
        print(f"[1/2] Removing background...")
        step1_start = time.time()
        
        input_pil = Image.open(input_image) if isinstance(input_image, str) else input_image
        output_pil = remove(input_pil)
        output_pil.save(nobg_path)
        
        step1_time = time.time() - step1_start
        print(f"✓ Background removed in {step1_time:.1f}s")
        
        # 배경 제거 완료 후 즉시 표시 (25% 완료)
        estimated_shape_time = step1_time * 3
        estimated_texture_time = step1_time * 3
        total_estimated = step1_time + estimated_shape_time + estimated_texture_time
        
        status_msg_step1 = "⏳ 처리 중...\n\n"
        status_msg_step1 += f"📊 진행률: 25% 완료\n"
        status_msg_step1 += f"⏱️ 경과 시간: {step1_time:.1f}초\n"
        status_msg_step1 += f"⏱️ 예상 남은 시간: 약 {estimated_shape_time + estimated_texture_time:.0f}초\n"
        status_msg_step1 += f"⏱️ 총 예상 시간: 약 {total_estimated:.0f}초\n\n"
        status_msg_step1 += "📋 처리 단계:\n"
        status_msg_step1 += "1. ✓ 배경 제거 완료\n"
        status_msg_step1 += "2. ⏳ 3D 메시 생성 중...\n"
        status_msg_step1 += "3. ⏸️ 텍스처 생성 대기 중..."
        
        # 3D 뷰어에 "생성 중" 상태 표시
        yield nobg_path, gr.update(value=None, label="OBJ 포맷 - ⏳ 생성 중..."), gr.update(value=None, label="GLB 포맷 - ⏳ 생성 중..."), status_msg_step1
        
        # 2단계: 3D 메시 생성 (25% → 60%)
        print(f"[2/3] Generating 3D mesh from image...")
        step2_start = time.time()
        
        mesh = shape_pipeline(image=nobg_path)[0]
        
        step2_time = time.time() - step2_start
        print(f"✓ 3D mesh generated in {step2_time:.1f}s")
        
        # 텍스처 생성 가능 여부 확인
        if TEXTURE_ENABLED and texture_pipeline is not None:
            # 메시 생성 완료 후 중간 상태 표시 (60% 완료)
            elapsed_time = time.time() - start_time
            estimated_remaining = step2_time * 1.5  # 텍스처는 메시 생성 시간의 약 1.5배
            
            status_msg_step2 = "⏳ 처리 중...\n\n"
            status_msg_step2 += f"📊 진행률: 60% 완료\n"
            status_msg_step2 += f"⏱️ 경과 시간: {elapsed_time:.1f}초\n"
            status_msg_step2 += f"⏱️ 예상 남은 시간: 약 {estimated_remaining:.0f}초\n\n"
            status_msg_step2 += "📋 처리 단계:\n"
            status_msg_step2 += "1. ✓ 배경 제거 완료\n"
            status_msg_step2 += "2. ✓ 3D 메시 생성 완료\n"
            status_msg_step2 += "3. ⏳ 텍스처 생성 중..."
            
            yield nobg_path, gr.update(value=None, label="OBJ 포맷 - ⏳ 텍스처 생성 중..."), gr.update(value=None, label="GLB 포맷 - ⏳ 텍스처 생성 중..."), status_msg_step2
            
            # 3단계: 텍스처 생성 (60% → 100%)
            print(f"[3/3] Generating texture for the mesh...")
            step3_start = time.time()
            
            mesh = texture_pipeline(mesh, image=nobg_path)
            
            step3_time = time.time() - step3_start
            print(f"✓ Texture generated in {step3_time:.1f}s")
        else:
            step3_time = 0
            print("⚠️  Texture generation skipped (not available)")
        
        # OBJ 및 GLB 형식으로 저장 (텍스처 포함 또는 미포함)
        mesh.export(obj_path)
        mesh.export(glb_path)
        
        total_time = time.time() - start_time
        
        # 메시 정보
        status_msg = f"✅ 생성 완료!\n\n"
        status_msg += f"📊 진행률: 100% 완료\n"
        status_msg += f"⏱️ 총 처리 시간: {total_time:.1f}초\n"
        status_msg += f"  - 배경 제거: {step1_time:.1f}초\n"
        status_msg += f"  - 3D 생성: {step2_time:.1f}초\n"
        if TEXTURE_ENABLED and step3_time > 0:
            status_msg += f"  - 텍스처 생성: {step3_time:.1f}초\n\n"
        else:
            status_msg += f"  - 텍스처: 비활성화됨\n\n"
        status_msg += f"📋 처리 단계:\n"
        status_msg += f"1. ✓ 배경 제거 완료\n"
        status_msg += f"2. ✓ 3D 메시 생성 완료\n"
        if TEXTURE_ENABLED and step3_time > 0:
            status_msg += f"3. ✓ 텍스처 생성 완료\n\n"
        else:
            status_msg += f"3. ⚠️  텍스처 생성 건너뜀 (모듈 미설치)\n\n"
        status_msg += f"📊 메시 정보:\n"
        status_msg += f"- 정점(Vertices): {len(mesh.vertices):,}\n"
        status_msg += f"- 면(Faces): {len(mesh.faces):,}\n\n"
        status_msg += f"💾 파일:\n"
        status_msg += f"- 배경제거: {nobg_path}\n"
        status_msg += f"- OBJ: {obj_path}\n"
        status_msg += f"- GLB: {glb_path}"
        
        print(status_msg)
        
        # 최종 결과 반환 (label 원래대로 복원)
        yield nobg_path, gr.update(value=obj_path, label="OBJ 포맷"), gr.update(value=glb_path, label="GLB 포맷"), status_msg
    
    except Exception as e:
        error_msg = f"❌ 오류 발생: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        yield None, None, None, error_msg

# Gradio 인터페이스 구성
with gr.Blocks(title="Hunyuan3D 2 - 3D Model Generator") as demo:
    texture_status = "✓ 텍스처 생성 활성화" if TEXTURE_ENABLED else "⚠️ 텍스처 생성 비활성화 (bare mesh만 생성)"
    
    gr.Markdown(f"""
    # 🎨 Hunyuan3D 2 - 이미지에서 3D 모델 생성
    
    단일 이미지를 업로드하면 자동으로 3D 메시를 생성합니다.
    
    **상태**: {texture_status}
    """)
    
    with gr.Row():
        # 왼쪽 컬럼: 입력 및 전처리
        with gr.Column(scale=1):
            gr.Markdown("### 📥 입력")
            input_image = gr.Image(
                label="이미지 업로드",
                type="filepath",
                sources=["upload", "clipboard"],
                height=300
            )
            
            generate_btn = gr.Button("🚀 3D 모델 생성", variant="primary", size="lg")
            
            gr.Markdown("### 🎭 배경 제거 결과")
            nobg_image = gr.Image(
                label="배경이 제거된 이미지",
                type="filepath",
                height=300,
                interactive=False
            )
            
            # 예제 이미지 (demo.png가 있다면 사용)
            if os.path.exists("demo.png"):
                gr.Examples(
                    examples=[["demo.png"]],
                    inputs=input_image,
                    label="예제"
                )
        
        # 오른쪽 컬럼: 3D 모델 출력
        with gr.Column(scale=1):
            gr.Markdown("### 📤 생성된 3D 모델")
            with gr.Tabs():
                with gr.Tab("OBJ"):
                    output_obj = gr.Model3D(
                        label="OBJ 포맷",
                        height=450,
                        clear_color=[0.8, 0.8, 0.8, 1.0]
                    )
                
                with gr.Tab("GLB"):
                    output_glb = gr.Model3D(
                        label="GLB 포맷",
                        height=450,
                        clear_color=[0.8, 0.8, 0.8, 1.0]
                    )
            
            gr.Markdown("### 📊 처리 상태")
            status_text = gr.Textbox(
                label="",
                lines=6,
                interactive=False,
                show_label=False,
                placeholder="이미지를 업로드하고 '3D 모델 생성' 버튼을 클릭하세요."
            )
    
    gr.Markdown("""
    ---
    ### 📝 사용 방법
    1. **이미지 업로드** - 왼쪽 상단에 이미지를 업로드합니다
    2. **생성 버튼 클릭** - '3D 모델 생성' 버튼을 클릭합니다
    3. **배경 제거 확인** - 왼쪽 하단에 배경이 제거된 이미지가 표시됩니다
    4. **3D 모델 확인** - 오른쪽에서 생성된 3D 모델을 회전하며 확인하고 다운로드합니다
    
    ### ⚠️ 참고사항
    - 배경 제거: rembg 라이브러리 사용 (투명 배경 생성)
    - 생성 시간: 이미지 크기와 시스템 성능에 따라 다름
    - GPU 사용 시 더 빠른 처리 가능
    """)
    
    # 이벤트 연결
    # 버튼 클릭 시 배경 제거 → 3D 생성을 순차적으로 실행
    generate_btn.click(
        fn=generate_3d_model,
        inputs=[input_image],
        outputs=[nobg_image, output_obj, output_glb, status_text]
    )

# 서버 실행
if __name__ == "__main__":
    try:
        print("🚀 Gradio 서버 시작 중...")
        print("💡 종료하려면 CTRL+C를 누르세요\n")
        demo.launch(inbrowser=True, theme=gr.themes.Soft())
    except KeyboardInterrupt:
        print("\n")
        cleanup_and_exit()
    except Exception as e:
        print(f"❌ 서버 실행 중 오류 발생: {e}")
        cleanup_and_exit()
    finally:
        cleanup_and_exit()