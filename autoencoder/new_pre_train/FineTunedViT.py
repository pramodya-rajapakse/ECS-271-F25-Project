import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class FineTunedViT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = vit_b_16(weights=weights)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def load_trained_weights(self, path, device="cpu"):
        """
        Load pretrained weights from file.
        """
        state_dict = torch.load(path, map_location=device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print("\nLoaded weights with:")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        self.to(device)
        return self