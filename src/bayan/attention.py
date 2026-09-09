"""Lab 2: scaled dot-product attention and multi-head attention."""

import math
import torch
import torch.nn as nn


def attention(q, k, v, mask=None):
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        attended = attention(q, k, v, mask)

        attended = attended.transpose(1, 2).contiguous()

        attended = attended.view(
            batch_size,
            seq_len,
            self.d_model
        )

        return self.out_proj(attended)