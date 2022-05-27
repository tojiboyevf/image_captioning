import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained Densenet-161 and replace top classifier layer."""
        super(Encoder, self).__init__()
        densenet = models.densenet161(pretrained=True)
        modules = list(densenet.children())[:-1]
        self.densenet = nn.Sequential(*modules)
        self.embed = nn.Sequential(
            nn.Linear(densenet.classifier.in_features, embed_size),
            nn.Dropout(p=0.5),
        )
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.densenet(images)
            features = F.avg_pool2d(features, kernel_size=7).view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
