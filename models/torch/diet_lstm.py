import torch.nn as nn
from models.torch.decoders.monolstm import Decoder
from models.torch.encoders.diet import Encoder

class Captioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, embedding_matrix=None, train_embd=True):
        super().__init__()
        self.name = 'ViT_lstm'
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers,
                               embedding_matrix=embedding_matrix, train_embd=train_embd)

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    def sample(self, images, max_len=40, endseq_idx=-1):
        features = self.encoder(images)
        captions = self.decoder.sample(features=features, max_len=max_len, endseq_idx=endseq_idx)
        return captions

    def sample_beam_search(self, images, max_len=40, beam_width=5):
        features = self.encoder(images)
        captions = self.decoder.sample_beam_search(features=features, max_len=max_len, beam_width=beam_width)
        return captions
