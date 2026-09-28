from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader
from transfer_model import create_model


# --------------------------------------------------
# SETUP
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

Path("models").mkdir(exist_ok=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = create_model(num_classes=4)

# Freeze all pretrained layers
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)


# --------------------------------------------------
# LOSS AND OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

# Train only the new classifier
optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

num_epochs = 5
best_val_accuracy = 0.0

for epoch in range(num_epochs):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        train_total += labels.size(0)
        train_correct += (predictions == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total


    # --------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/efficientnet_b0_best.pth"
        )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


print(f"\nBest validation accuracy: {best_val_accuracy:.2f}%")
print("Best model saved to models/efficientnet_b0_best.pth")