import kagglehub

# Download latest version
path = kagglehub.dataset_download("ajayrana/hymenoptera-data")

print("> Path to dataset files:", path)