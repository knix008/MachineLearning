import os
import kagglehub
import matplotlib.pyplot as plt
from PIL import Image

def show_images_from_folder(folder, n_images=8, title=''):
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = images[:n_images]
    plt.figure(figsize=(16, 4))
    for i, img_name in enumerate(images):
        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path).convert("RGB")
        plt.subplot(1, n_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{img_name}', fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    # 1. KaggleHub에서 데이터 다운로드
    dataset_dir = kagglehub.dataset_download("arnaud58/selfie2anime")
    print("다운로드된 데이터 경로:", dataset_dir)

    # 2. 원본 데이터 위치 탐색 (selfie2anime 폴더 또는 images 폴더 내에 있음)
    possible_subdirs = ['selfie2anime', 'images']
    found = False
    for sub in possible_subdirs:
        candidate = os.path.join(dataset_dir, sub)
        if os.path.isdir(candidate):
            root_data = candidate
            found = True
            break
    if not found:
        root_data = dataset_dir  # 혹시 바로 그 안에 trainA 등이 있을 수도 있음

    # 3. trainA, trainB 폴더 경로
    selfie_train_dir = os.path.join(root_data, "trainA")
    anime_train_dir = os.path.join(root_data, "trainB")

    print("Selfie (A) 학습 데이터 예시:")
    show_images_from_folder(selfie_train_dir, n_images=8, title='trainA (Selfie)')

    print("Anime (B) 학습 데이터 예시:")
    show_images_from_folder(anime_train_dir, n_images=8, title='trainB (Anime)')

if __name__ == "__main__":
    main()