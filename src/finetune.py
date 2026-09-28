import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import test_loader
from transfer_model import create_model


# --------------------------------------------------
# SETUP
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD FINE-TUNED MODEL
# --------------------------------------------------

model = create_model(num_classes=4)

model.load_state_dict(
    torch.load(
        "models/efficientnet_b0_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# EVALUATE ON TEST DATA
# --------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.tolist()
        )


# --------------------------------------------------
# TEST ACCURACY
# --------------------------------------------------

correct = sum(
    prediction == label
    for prediction, label in zip(
        all_predictions,
        all_labels
    )
)

accuracy = 100 * correct / len(all_labels)

print(f"\nFine-tuned EfficientNet Test Accuracy: {accuracy:.2f}%")


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

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


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)