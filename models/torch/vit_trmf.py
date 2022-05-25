import torch.nn as nn
from models.torch.decoders.transformer import Decoder
from models.torch.encoders.ViT import Encoder

class Captioner(nn.Module):
    def __init__(self, embed_size, vocab_size, n_heads=10, num_layers=6, embedding_matrix=None, train_embd=True):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, vocab_size, n_heads, num_layers,
                               embedding_matrix=embedding_matrix, train_embd=train_embd)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def sample(self, images, max_len=40, endseq_idx=-1):
        features = self.encoder(images)
        captions = self.decoder.sample(features=features, max_len=max_len, endseq_idx=endseq_idx)
        return captions