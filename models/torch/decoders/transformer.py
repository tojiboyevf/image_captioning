import torch
from torch import nn as nn
import math
from models.torch.layers import embedding_layer
from ..encoders.positional_encoder import PositionalEncoding


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads,
        decoder_layers,
        embed_size, 
        vocab_size, 
        embedding_matrix=None, 
        train_embd=True,  
        dropout=0.1,
    ):
        super(Decoder, self).__init__()
        
        self.embedding = embedding_layer(num_embeddings=vocab_size, embedding_dim=embed_size,
                                     embedding_matrix=embedding_matrix, trainable=train_embd)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=decoder_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embedding_size = embed_size
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool
    

    def forward(self, features, captions):
        features = features.permute(1,0,2)
        decoder_inp_embed = self.embedding(captions)* math.sqrt(self.embedding_size)
        decoder_inp_embed = decoder_inp_embed.permute(1,0,2)
        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)        
        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(captions.size(1), captions)
        decoder_input_mask = decoder_input_mask.to(features.device)
        decoder_input_pad_mask = decoder_input_pad_mask.to(features.device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(features.device)

        decoder_output = self.transformer(tgt = decoder_inp_embed, memory = features, tgt_mask = decoder_input_mask, tgt_key_padding_mask = decoder_input_pad_mask_bool)
        
        final_output = self.fc_out(decoder_output)
        
        return final_output,  decoder_input_pad_mask
    
    def sample(self, features, max_len=40, start_token=59, pad_token=58):
        input_seq = torch.ones(features.size(0), max_len).type(torch.LongTensor) * pad_token
        input_seq[:, 0] = start_token
        input_seq = input_seq.to(features.device)
        for i in range(max_len-1):
            output, _ = self.forward(features, input_seq)
            output = output[i, :, :]
            predicted = output.argmax(1)
            input_seq[:, i+1] = predicted
        return input_seq
