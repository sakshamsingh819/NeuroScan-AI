import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import test_loader
from model import BrainTumorCNN


# Choose device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load trained model
model = BrainTumorCNN(num_classes=4)

model.load_state_dict(
    torch.load(
        "models/brain_tumor_cnn.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Store predictions and actual labels
all_predictions = []
all_labels = []

# Evaluate on test data
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

# Calculate test accuracy
correct = sum(
    p == label
    for p, label in zip(all_predictions, all_labels)
)

accuracy = 100 * correct / len(all_labels)

print(f"\nTest Accuracy: {accuracy:.2f}%")

# Detailed performance for each class
class_names = test_loader.dataset.classes

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))