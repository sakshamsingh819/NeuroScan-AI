from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = Path("data/Training")

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(classes):
    image_files = list((DATA_DIR / class_name).glob("*"))

    image = Image.open(image_files[0])

    plt.subplot(2, 2, i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()
plt.show()