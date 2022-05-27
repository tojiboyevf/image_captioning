import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained vgg-16 and replace top fc layer."""
        super(Encoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier = vgg16.classifier[:-1]
        self.vgg16 = vgg16
        self.embed = nn.Linear(vgg16.classifier[-3].out_features, embed_size)  # FC-relu-dropout
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.vgg16(images)
        features = self.embed(features)
        features = self.bn(features)
        return features