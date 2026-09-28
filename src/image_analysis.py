from pathlib import Path
from PIL import Image
from collections import Counter

DATA_DIR = Path("data/Training")

sizes = Counter()
modes = Counter()

for class_dir in DATA_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    for image_path in class_dir.iterdir():
        try:
            with Image.open(image_path) as img:
                sizes[img.size] += 1
                modes[img.mode] += 1
        except Exception as e:
            print(f"Problem with {image_path}: {e}")

print("Image sizes:")
for size, count in sizes.most_common(10):
    print(size, ":", count)

print("\nImage modes:")
for mode, count in modes.items():
    print(mode, ":", count)

print("\nNumber of unique sizes:", len(sizes))