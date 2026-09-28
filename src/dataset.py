from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

# Convert every image to RGB
rgb_transform = transforms.Lambda(lambda img: img.convert("RGB"))


# Training preprocessing + augmentation
train_transform = transforms.Compose([
    rgb_transform,
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and testing preprocessing
# No random augmentation
test_transform = transforms.Compose([
    rgb_transform,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

# Training dataset with augmentation
train_full = ImageFolder(
    "data/Training",
    transform=train_transform
)

# Same training images without augmentation
val_full = ImageFolder(
    "data/Training",
    transform=test_transform
)

# Separate testing dataset
test_dataset = ImageFolder(
    "data/Testing",
    transform=test_transform
)


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------

# Create a reproducible 80/20 split
train_size = int(0.8 * len(train_full))

generator = torch.Generator().manual_seed(42)
indices = torch.randperm(
    len(train_full),
    generator=generator
).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Apply the same indices to datasets with different transforms
train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(val_full, val_indices)


# --------------------------------------------------
# DATALOADERS
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# VERIFY DATASET
# --------------------------------------------------

print("Classes:", train_full.classes)
print("Class mapping:", train_full.class_to_idx)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))


# Verify one training batch
images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)