import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# --------------------------------------------------
# EFFICIENTNET-B0 MODEL
# --------------------------------------------------

def create_model(num_classes=4):

    # Load EfficientNet-B0 with pretrained ImageNet weights
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    # Get the number of inputs to the final layer
    in_features = model.classifier[1].in_features

    # Replace the original classifier with our 4 classes
    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# --------------------------------------------------
# TEST THE MODEL
# --------------------------------------------------

if __name__ == "__main__":

    model = create_model()

    # Simulate a batch of 32 MRI images
    sample = torch.randn(32, 3, 224, 224)

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Model created successfully!")