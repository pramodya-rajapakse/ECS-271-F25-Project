import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class MAEViTModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.model = vit_b_16(weights=None)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def load_trained_weights(self, path, device="cpu"):
        state_dict = torch.load(path, map_location=device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print("\nLoaded with:")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

