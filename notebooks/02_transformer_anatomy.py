"""Lab 2 starter notebook-as-script.
Complete the marked sections, verify numerical equivalence, inspect parameter
accounting, causal masking, attention heads and pad-attention leakage.
"""


"""Lab 2: Anatomy of a Transformer."""

import math
import torch
import torch.nn.functional as F

from src.bayan.attention import attention, MultiHeadAttention


def main():
    torch.manual_seed(42)

    # 1. Numerical equivalence
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 3, 8)
    v = torch.randn(2, 3, 8)

    ours, ours_weights = attention(q, k, v)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    reference_weights = F.softmax(scores, dim=-1)
    reference = torch.matmul(reference_weights, v)

    assert torch.allclose(ours, reference, atol=1e-6)
    assert torch.allclose(ours_weights, reference_weights, atol=1e-6)

    print("✅ Attention numerical equivalence passed!")

    # 2. Multi-head attention
    mha = MultiHeadAttention(d_model=768, num_heads=12)

    x = torch.randn(1, 5, 768)
    output, weights = mha(x, x, x)

    print("MHA output shape:", tuple(output.shape))
    print("MHA weights shape:", tuple(weights.shape))

    assert output.shape == (1, 5, 768)
    assert weights.shape == (1, 12, 5, 5)

    print("✅ Multi-head attention passed!")

    # 3. Causal masking
    seq_len = 5

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    )

    test_scores = torch.randn(seq_len, seq_len)

    masked_scores = test_scores.masked_fill(
        ~causal_mask,
        float("-inf")
    )

    causal_weights = F.softmax(masked_scores, dim=-1)

    future_mass = causal_weights[~causal_mask].sum().item()

    print("Attention paid to future tokens:", future_mass)

    assert future_mass < 1e-6

    print("✅ Causal masking passed!")


if __name__ == "__main__":
    main()