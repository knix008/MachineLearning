import os
import hashlib
from collections import defaultdict

# Directory containing generated face images
IMAGE_DIR = 'generated_faces'

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

def get_file_hash(filepath, hash_func=hashlib.md5, chunk_size=8192):
    """Compute hash of a file."""
    h = hash_func()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def find_duplicate_images(directory):
    hash_dict = defaultdict(list)
    files = os.listdir(directory)
    total = len(files)
    count = 0
    for filename in files:
        count += 1
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            print(f"Processing {filename} ({count}/{total})...")
            filepath = os.path.join(directory, filename)
            file_hash = get_file_hash(filepath)
            hash_dict[file_hash].append(filename)
    print(f"Processed {count} files.")
    # Print duplicates
    found = False
    for files in hash_dict.values():
        if len(files) > 1:
            found = True
            print(f"Duplicate images: {files}")
    if not found:
        print("No duplicate images found.")
    print("Duplicate check complete.")

if __name__ == "__main__":
    find_duplicate_images(IMAGE_DIR)
