#!/bin/bash

echo "========================================"
echo "Stable Diffusion 3.5 Style Transfer"
echo "========================================"
echo

# 패키지 설치 확인
echo "패키지 설치 확인 중..."
python3 -c "import gradio, torch, diffusers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "필요한 패키지를 설치합니다..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "패키지 설치에 실패했습니다."
        exit 1
    fi
fi

echo
echo "GPU 지원 확인 중..."
python3 -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available())"

echo
echo "어떤 버전을 실행하시겠습니까?"
echo "1. 전체 기능 버전 (포트 7860)"
echo "2. 간단한 버전 (포트 7861)"
echo
read -p "선택하세요 (1 또는 2): " choice

case $choice in
    1)
        echo "전체 기능 버전을 실행합니다..."
        python3 stablediffusion-3.5-style-transfer-gradio.py
        ;;
    2)
        echo "간단한 버전을 실행합니다..."
        python3 simple-style-transfer-gradio.py
        ;;
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac 