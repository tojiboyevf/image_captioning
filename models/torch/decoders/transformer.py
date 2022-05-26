import torch
from torch import nn as nn
import numpy as np
from models.torch.layers import embedding_layer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 40):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(
        self,
        embed_size, 
        vocab_size,
        num_heads,
        decoder_layers,
        embedding_matrix=None, 
        train_embd=True, 
        max_len=40, 
        dropout=0.1
    ):
        super(Decoder, self).__init__()
        
        self.max_len = max_len
        self.embed_size = embed_size

        self.embed = embedding_layer(num_embeddings=vocab_size, embedding_dim=embed_size,
                                     embedding_matrix=embedding_matrix, trainable=train_embd)

        self.trg_position_embedding = PositionalEncoding(d_model=embed_size, max_len=max_len)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads) #8
        self.transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=decoder_layers) #6

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0     
        return mask

    def forward(self, featrues, tgt, tgt_mask=None, tgt_pad_mask=None):
        
        featrues = featrues.unsqueeze(1)

        tgt_seq_length = tgt.size(1)

        tgt = self.embed(tgt) * np.sqrt(self.embed_size)
        tgt_mask = self.get_tgt_mask(tgt_seq_length).to(tgt.device)

        tgt = tgt.permute(1, 0, 2)
        tgt = self.trg_position_embedding(tgt)
        featrues = featrues.permute(1, 0, 2)

        transformer_out = self.transformer(tgt, featrues, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.fc_out(transformer_out)
        return out

    def sample(self, inputs, max_len=40, endseq_idx=1):

        sampled_ids = []
        y_input = torch.tensor([[0]], dtype=torch.long, device=inputs.device)
        
        with torch.no_grad():
            for _ in range(max_len):

                outputs = self.forward(inputs, y_input)
                predicted = outputs.topk(1)[1].view(-1)[-1]

                sampled_ids.append(predicted.item())
                if (predicted.item() == endseq_idx):
                    break
                
                inputs = self.embed(predicted).unsqueeze(0)
                
                # predicted = torch.tensor([[predicted]], dtype=torch.long, device=inputs.device)
                predicted = predicted.unsqueeze(0)                
                y_input = torch.cat((y_input, predicted), dim=1)
        
        return sampled_ids


a = torch.randn((3, 3, 224, 224))