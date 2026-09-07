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
    # 4. Pad-attention leakage
    x = torch.randn(1, 5, 768)

    pad_mask = torch.tensor(
        [[1, 1, 1, 0, 0]],
        dtype=torch.bool
    )

    mha_no_mask = MultiHeadAttention(d_model=768, num_heads=12)
    _, weights_no_mask = mha_no_mask(x, x, x)

    pad_mass_no_mask = weights_no_mask[..., 3:].sum().item()
    print("Pad attention mass without mask:", pad_mass_no_mask)

    mha_masked = MultiHeadAttention(d_model=768, num_heads=12)
    _, weights_masked = mha_masked(x, x, x, mask=pad_mask[:, None, None, :])

    pad_mass_masked = weights_masked[..., 3:].sum().item()
    print("Pad attention mass with mask:", pad_mass_masked)

    assert pad_mass_no_mask > 0
    assert pad_mass_masked < 1e-6

    print("✅ Pad-attention leakage test passed!")

if __name__ == "__main__":
    main()