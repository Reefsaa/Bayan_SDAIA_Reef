"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math
import torch


def attention(q, k, v, mask=None):
    """
    Scaled dot-product attention.
    q, k, v shape:
    [batch, heads, seq_len, d_k]
    """

    d_k = q.size(-1)

    # QK^T / sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax over keys
    weights = torch.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(weights, v)

    return output


class MultiHeadAttention:
    def __init__(self, *args, **kwargs):
        # We will complete this in the next Lab 2 step.
        pass