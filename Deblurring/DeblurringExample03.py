import cv2
import numpy as np
import gradio as gr
import os

# --- 설정 ---
# Super-Resolution 모델 파일 경로를 지정하세요.
# 예: SR_MODEL_PATH = "EDSR_x4.pb" (코드가 있는 폴더에 모델 파일이 있을 경우)
SR_MODEL_PATH = "EDSR_x4.pb" 
# SR 기능을 사용하지 않거나 모델 파일이 없다면, None으로 설정하세요.
# SR_MODEL_PATH = None
# --- 설정 끝 ---

def enhance_image_quality_gradio(image: np.ndarray, method: str) -> np.ndarray:
    """
    Gradio 웹 앱에서 이미지 화질 향상을 처리하는 핵심 함수.

    Args:
        image (np.ndarray): Gradio를 통해 업로드된 이미지 (NumPy 배열 형식).
        method (str): 사용자가 선택한 화질 향상 방법.

    Returns:
        np.ndarray: 처리된 후 화질이 향상된 이미지 (NumPy 배열 형식).
    """
    if image is None:
        return None

    # 원본 이미지를 복사하여 처리 중 원본이 변경되지 않도록 합니다.
    processed_image = image.copy()

    try:
        if method == '선명화':
            print("이미지 선명화 적용 중...")
            # 선명화 커널: 이미지의 경계를 강조하여 선명도를 높입니다.
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)

        elif method == '노이즈 제거 (가우시안)':
            print("가우시안 블러를 이용한 노이즈 제거 적용 중...")
            # 가우시안 블러: 노이즈를 부드럽게 제거하고 이미지를 부드럽게 만듭니다.
            processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)

        elif method == '노이즈 제거 (미디언)':
            print("미디언 블러를 이용한 노이즈 제거 적용 중...")
            # 미디언 블러: 솔트-앤-페퍼 노이즈와 같은 임펄스 노이즈 제거에 효과적입니다.
            processed_image = cv2.medianBlur(processed_image, 5)

        elif method == '초해상도 (Super-Resolution)':
            print("Super-Resolution 적용 중...")
            if not SR_MODEL_PATH or not os.path.exists(SR_MODEL_PATH):
                print(f"오류: Super-Resolution 모델 파일('{SR_MODEL_PATH}')을 찾을 수 없습니다.")
                print("초해상도 기능을 사용하려면 'SR_MODEL_PATH' 변수에 올바른 모델 파일 경로를 지정하고 모델을 다운로드해야 합니다.")
                return image # 모델이 없으면 원본 이미지 반환

            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(SR_MODEL_PATH)

            # 모델 이름과 스케일을 파일 이름에서 유추합니다.
            try:
                model_name = os.path.basename(SR_MODEL_PATH).split('_')[0].lower()
                scale = int(os.path.basename(SR_MODEL_PATH).split('x')[-1].split('.')[0])
                sr.setModel(model_name, scale)
                processed_image = sr.upsample(processed_image)
            except Exception as e:
                print(f"오류: SR 모델 설정 또는 업스케일링 중 문제가 발생했습니다: {e}")
                print("SR_MODEL_PATH가 올바른 모델 파일명 규칙(예: EDSR_x2.pb)을 따르는지 확인하세요.")
                return image # 오류 발생 시 원본 이미지 반환

        else:
            print(f"알 수 없는 화질 향상 방법입니다: {method}")
            return image

        return processed_image

    except Exception as e:
        print(f"이미지 처리 중 예상치 못한 오류 발생: {e}")
        return image # 오류 발생 시 원본 이미지 반환

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=enhance_image_quality_gradio, # 이미지 처리 함수 연결
    inputs=[
        gr.Image(type="numpy", label="여기에 이미지를 업로드하세요"), # 이미지 업로드 컴포넌트
        gr.Radio(
            ["선명화", "노이즈 제거 (가우시안)", "노이즈 제거 (미디언)", "초해상도 (Super-Resolution)"],
            label="화질 향상 방법 선택",
            value="선명화" # 앱 실행 시 기본 선택 값
        )
    ],
    outputs=gr.Image(type="numpy", label="향상된 이미지"), # 결과 이미지 출력 컴포넌트
    title="🌟 AI 이미지 화질 향상 도구 🌟",
    description="이미지를 업로드하고 원하는 화질 향상 기법을 선택하여 더욱 선명하고 깨끗한 이미지를 만들어 보세요!"
)

# Gradio 애플리케이션 실행
if __name__ == "__main__":
    if SR_MODEL_PATH and not os.path.exists(SR_MODEL_PATH):
        print(f"\n[경고] Super-Resolution 모델 파일이 '{SR_MODEL_PATH}' 경로에 없습니다.")
        print("초해상도 기능을 사용하려면 위 경로에 모델 파일을 다운로드하여 넣어주세요.")
        print("모델 다운로드 링크: https://github.com/opencv/opencv_extra/tree/master/testdata/dnn_superres\n")
        
    print("Gradio 앱을 시작합니다. 웹 브라우저에서 표시되는 URL을 열어주세요.")
    # share=True로 설정하면 임시 퍼블릭 링크가 생성됩니다 (공유 목적).
    # 개발 중에는 share=False (기본값)를 권장합니다.
    iface.launch(share=False)