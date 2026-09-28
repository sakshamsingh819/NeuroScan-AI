import torch
import torch.nn as nn


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Test the model
if __name__ == "__main__":
    model = BrainTumorCNN()

    # Create a fake batch matching our DataLoader
    sample = torch.randn(32, 3, 224, 224)

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print(model)