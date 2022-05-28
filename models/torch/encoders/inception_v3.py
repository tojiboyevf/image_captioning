import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)
        modules = list(inception_v3.children())[:-1]
        self.inception_v3 = nn.Sequential(*modules)
        self.embed = nn.Sequential(
            nn.Linear(inception_v3.fc.in_features, embed_size),
            nn.Dropout(p=0.5),
        )
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        with torch.no_grad():
            features = self.inception_v3(images)
            features = F.relu(features, inplace=True).view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
