import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 128):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 32,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids):
        # TODO(STUDENT): students need to implement 
        
        # 1. Token embedding
        x = self.token_embedding(input_ids) # [B, L, D]

        # 2. 加位置编码
        x = self.positional_encoding(x)

        # 3. 创建 padding mask (True = 忽略)
        padding_mask = input_ids.eq(self.padding_idx) # [B, L]

        # 4. Transformer 编码
        x = self.encoder(x, src_key_padding_mask=padding_mask) # [B, L, D]

        # 5. 均值池化（忽略 padding 位置）
        mask = (~padding_mask).unsqueeze(-1).float() # [B, L, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # [B, D]

        # 6. 线性投影
        x = self.proj(x) # [B, D]
        return x
