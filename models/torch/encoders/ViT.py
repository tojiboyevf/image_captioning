import torch
import timm
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.embed = nn.Linear(in_features=self.vit.head.out_features, out_features=embed_size)
    
    def forward(self, images):
        with torch.no_grad():
            features = self.vit(images)
            features = self.embed(features)
        return features

# model = ViT(1280)
# x = torch.randn((1, 3, 224, 224))
# print(model)
# print(model(x).shape)