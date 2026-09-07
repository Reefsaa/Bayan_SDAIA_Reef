"""Lab 2 transformer anatomy evidence script."""

import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from bayan.attention import attention, MultiHeadAttention


def attention_weights(q, k, mask=None):
    """Calculate scaled dot-product attention weights."""

    d_k = q.size(-1)

    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    ) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

    return torch.softmax(scores, dim=-1)


def main():
    torch.manual_seed(42)

    # =========================================================
    # 1. Numerical equivalence
    # =========================================================

    print("\n=== Numerical equivalence ===")

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    ours = attention(q, k, v)

    pytorch = F.scaled_dot_product_attention(
        q,
        k,
        v
    )

    max_diff = (ours - pytorch).abs().max().item()

    print("Max difference:", max_diff)

    print(
        "Equivalent:",
        torch.allclose(
            ours,
            pytorch,
            atol=1e-6
        )
    )

    # =========================================================
    # 2. Attention weight matrix
    # =========================================================

    print("\n=== Attention weight matrix ===")

    weights = attention_weights(q, k)

    print(weights[0, 0])

    # =========================================================
    # 3. Multi-Head Attention
    # =========================================================

    print("\n=== Multi-Head Attention ===")

    x = torch.randn(2, 5, 16)

    mha = MultiHeadAttention(
        d_model=16,
        num_heads=4
    )

    mha_output = mha(x)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "MHA output shape:",
        mha_output.shape
    )

    # =========================================================
    # 4. Causal mask
    # =========================================================

    print("\n=== Causal mask ===")

    seq_len = q.size(-2)

    causal_mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool
        )
    )

    causal_weights = attention_weights(
        q,
        k,
        causal_mask
    )

    print(causal_weights[0, 0])

    future_mass = torch.triu(
        causal_weights[0, 0],
        diagonal=1
    ).sum().item()

    print(
        "Future attention mass:",
        future_mass
    )

    print(
        "Causal mask valid:",
        future_mass == 0.0
    )

    print(
        "Model family:",
        "Decoder-style causal attention"
    )

    # =========================================================
    # 5. Simple masking evidence
    # =========================================================

    print("\n=== Masking evidence ===")

    padding_mask = torch.tensor(
        [[[[1, 1, 1, 0]]]],
        dtype=torch.bool
    )

    masked_weights = attention_weights(
        q,
        k,
        padding_mask
    )

    pad_mass = masked_weights[
        ...,
        -1
    ].sum().item()

    print(
        "Attention mass on masked PAD position:",
        pad_mass
    )

    # =========================================================
    # 6. Step 5: Attention-map diagnostics + PAD leak
    # =========================================================

    print(
        "\n=== Step 5: Attention-map diagnostics + PAD leak ==="
    )

    checkpoint = "bert-base-multilingual-cased"

    print(
        "Loading model:",
        checkpoint
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    model = AutoModel.from_pretrained(
        checkpoint,
        output_attentions=True,
        attn_implementation="eager"
    )

    model.eval()

    # Bayan Arabic/English examples
    bayan_examples = [
        "الخدمة ممتازة لكن الرد كان متأخر.",
        "The service was good but the response was delayed."
    ]

    encoded = tokenizer(
        bayan_examples,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # =========================================================
    # Run WITH correct attention mask
    # =========================================================

    with torch.no_grad():
        outputs_masked = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_attentions=True
        )

    # =========================================================
    # Run WITHOUT attention mask
    # =========================================================

    with torch.no_grad():
        outputs_unmasked = model(
            input_ids=encoded["input_ids"],
            output_attentions=True
        )

    masked_attention = outputs_masked.attentions[-1]
    unmasked_attention = outputs_unmasked.attentions[-1]

    # =========================================================
    # Display tokens
    # =========================================================

    print("\n=== Tokens ===")

    tokens = [
        tokenizer.convert_ids_to_tokens(ids)
        for ids in encoded["input_ids"]
    ]

    for i, token_list in enumerate(tokens):
        print(
            f"Example {i + 1}:",
            token_list
        )

    # =========================================================
    # PAD attention comparison
    # =========================================================

    print("\n=== PAD attention comparison ===")

    pad_positions = encoded["attention_mask"] == 0

    masked_pad_mass = 0.0
    unmasked_pad_mass = 0.0

    for batch_idx in range(
        encoded["input_ids"].size(0)
    ):

        pad_idx = torch.where(
            pad_positions[batch_idx]
        )[0]

        if len(pad_idx) > 0:

            masked_pad_mass += (
                masked_attention[
                    batch_idx,
                    :,
                    :,
                    pad_idx
                ]
                .sum()
                .item()
            )

            unmasked_pad_mass += (
                unmasked_attention[
                    batch_idx,
                    :,
                    :,
                    pad_idx
                ]
                .sum()
                .item()
            )

    print(
        "PAD attention mass WITHOUT mask:",
        unmasked_pad_mass
    )

    print(
        "PAD attention mass WITH mask:",
        masked_pad_mass
    )

    print(
        "PAD leak reduced:",
        masked_pad_mass < unmasked_pad_mass
    )

    # =========================================================
    # SEP sink behaviour
    # =========================================================

    print("\n=== [SEP] sink behaviour ===")

    sep_token_id = tokenizer.sep_token_id

    for batch_idx in range(
        encoded["input_ids"].size(0)
    ):

        sep_positions = torch.where(
            encoded["input_ids"][batch_idx]
            == sep_token_id
        )[0]

        if len(sep_positions) > 0:

            sep_idx = sep_positions[0].item()

            sep_mass = masked_attention[
                batch_idx,
                :,
                :,
                sep_idx
            ].mean().item()

            print(
                f"Example {batch_idx + 1} "
                f"mean attention to [SEP]:",
                sep_mass
            )

    # =========================================================
    # Adjacency-looking attention head
    # =========================================================

    print(
        "\n=== Adjacency-looking head diagnostic ==="
    )

    first_example_attention = masked_attention[0]

    best_head = None
    best_adj_mass = -1.0

    for head_idx in range(
        first_example_attention.size(0)
    ):

        head = first_example_attention[
            head_idx
        ]

        adjacency_values = torch.diagonal(
            head,
            offset=-1
        )

        if adjacency_values.numel() > 0:
            adjacency_mass = (
                adjacency_values.mean().item()
            )
        else:
            adjacency_mass = 0.0

        print(
            f"Head {head_idx} adjacency score:",
            adjacency_mass
        )

        if adjacency_mass > best_adj_mass:
            best_adj_mass = adjacency_mass
            best_head = head_idx

    print(
        "\nMost adjacency-looking head:",
        best_head
    )

    print(
        "Adjacency score:",
        best_adj_mass
    )

    print("\n=== Lab 2 Evidence Summary ===")

    print(
        "Numerical equivalence:",
        torch.allclose(
            ours,
            pytorch,
            atol=1e-6
        )
    )

    print(
        "Future attention mass:",
        future_mass
    )

    print(
        "Causal mask valid:",
        future_mass == 0.0
    )

    print(
        "PAD mass without mask:",
        unmasked_pad_mass
    )

    print(
        "PAD mass with mask:",
        masked_pad_mass
    )

    print(
        "Most adjacency-looking head:",
        best_head
    )

    print(
        "\nLab 2 transformer anatomy evidence complete."
    )


if __name__ == "__main__":
    main()