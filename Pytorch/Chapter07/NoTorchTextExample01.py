import os

from datasets import load_dataset
from torch.utils.data import DataLoader

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Hugging Face 허브에서 심볼릭 링크 경고 비활성화

# Hugging Face 허브에서 IMDB 영화 리뷰 데이터셋 로드
dataset = load_dataset("imdb")

# 데이터셋을 PyTorch의 DataLoader로 변환
train_dataloader = DataLoader(dataset["train"], batch_size=32, shuffle=True)

# 데이터 로더를 통해 배치 데이터 확인
for batch in train_dataloader:
    print(batch["text"])
    print(batch["label"])
    break