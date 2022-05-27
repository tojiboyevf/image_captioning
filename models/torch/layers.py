import torch
import torch.nn as nn


def embedding_layer(trainable=True, embedding_matrix=None, **kwargs):
    emb_layer = nn.Embedding(**kwargs)
    if embedding_matrix is not None:
        emb_layer.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())
    trainable = (embedding_matrix is None) or trainable
    if not trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer
