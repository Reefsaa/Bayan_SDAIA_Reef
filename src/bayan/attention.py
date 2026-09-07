"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math
import torch
import torch.nn as nn


def attention(q, k, v, mask=None):
    """Scaled dot-product attention."""

    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(weights, v)

    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.last_attention_weights = None

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        self.last_attention_weights = weights

        output = torch.matmul(weights, v)

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size, -1, self.d_model
        )

        output = self.out_proj(output)

        return output, weights