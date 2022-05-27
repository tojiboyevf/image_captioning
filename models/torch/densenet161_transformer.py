from torch import nn as nn

from .encoders.densenet161 import Encoder
from .decoders.transformer import Decoder

class Captioner(nn.Module):
    def __init__(self, num_heads, decoder_layers, embed_size, vocab_size, embedding_matrix=None, train_embd=True):
        super().__init__()
        self.name = 'densenet161_transformer'
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(num_heads, decoder_layers, embed_size, vocab_size,
                               embedding_matrix=embedding_matrix, train_embd=train_embd)

    def forward(self, images, captions):
        features = self.encoder(images)
        features = features.unsqueeze(1)
        output, padding_mask = self.decoder(features, captions)
        return output, padding_mask
    
    def sample(self, images, max_len=40, topk=3, start_token=59, end_token=57, pad_token=58):
        features = self.encoder(images)
        features = features.unsqueeze(1)
        captions = self.decoder.sample(features=features, max_len=max_len,
                                       start_token=start_token, pad_token=pad_token)
        return captions
