import torch.nn as nn
from torchvision.models import resnet18


class JigsawResNet18(nn.Module):
    def __init__(self, num_permutations=50):
        super().__init__()
        base = resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, num_permutations)
        )

    def forward(self, patches):  # (batch, 9, C, H, W)
        batch_size = patches.size(0)
        patches = patches.view(-1, *patches.shape[2:])
        features = self.encoder(patches).flatten(1)
        features = features.view(batch_size, -1)
        return self.classifier(features)