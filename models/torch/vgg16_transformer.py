from torch import nn as nn

from .encoders.vgg16 import Encoder
from .decoders.transformer import Decoder

class Captioner(nn.Module):
    def __init__(self, num_heads, decoder_layers, embed_size, vocab_size, embedding_matrix=None, train_embd=True):
        super().__init__()
        self.name = 'vgg16_transformer'
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(num_heads, decoder_layers, embed_size, vocab_size,
                               embedding_matrix=embedding_matrix, train_embd=train_embd)

    def forward(self, images, captions):
        features = self.encoder(images)
        features = features.unsqueeze(1)
        output, padding_mask = self.decoder(features, captions)
        return output, padding_mask
    
    def sample(self, images, max_len=40, start_token=59, pad_token=58):
        features = self.encoder(images)
        features = features.unsqueeze(1)
        captions = self.decoder.sample(features=features, max_len=max_len,
                                       start_token=start_token, pad_token=pad_token)
        return captions
