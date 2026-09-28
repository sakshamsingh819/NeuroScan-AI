import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader
from model import BrainTumorCNN


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = BrainTumorCNN(num_classes=4)
model = model.to(device)


# --------------------------------------------------
# LOSS FUNCTION
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

num_epochs = 10

for epoch in range(num_epochs):

    # Training mode
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Track loss
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total


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

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total


    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

torch.save(model.state_dict(), "models/brain_tumor_cnn.pth")

print("\nModel saved to models/brain_tumor_cnn.pth")