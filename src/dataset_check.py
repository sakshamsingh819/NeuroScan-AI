from pathlib import Path

DATA_DIR = Path("data")

for split in ["Training", "Testing"]:
    split_dir = DATA_DIR / split

    print(f"\n{split}")
    print("-" * 30)

    total = 0

    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.iterdir()))
            print(f"{class_dir.name}: {count}")

            total += count

    print(f"Total: {total}")