import kagglehub

# Download latest version
path = kagglehub.dataset_download("whegedusich/president-bidens-state-of-the-union-2023")
print("Path to dataset files:", path)