import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """PositionalEncoding.

    Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """Init PositionalEncoding.

        Args:
            d_model: hidden model embedding size
            dropout: dropout rate
            max_len: max length of input sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        ten_thousand = 10000.0
        div_term = (-math.log(ten_thousand) / d_model)
        div_term = torch.arange(0, d_model, 2).float() * (div_term)
        div_term = torch.exp(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, tensor):
        """Define the computation of the PositionalEncoding.

        Args:
            tensor: Tensor, shape [seq_len, batch_size, embedding_dim]

        Returns:
            tensor: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tensor = tensor + self.pe[:tensor.size(0), :]
        return self.dropout(tensor)
