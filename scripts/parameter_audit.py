"""Lab 2: Parameter audit."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        n = param.numel()
        lname = name.lower()

        if "embeddings" in lname:
            buckets["embeddings"] += n

        elif "attention" in lname:
            buckets["attention"] += n

        elif (
            "intermediate" in lname
            or "output.dense" in lname
            or "ffn" in lname
        ):
            buckets["ffn"] += n

        elif (
            "layernorm" in lname
            or "layer_norm" in lname
            or "norm" in lname
        ):
            buckets["norms"] += n

        elif "pooler" in lname:
            buckets["pooler"] += n

        else:
            buckets["other"] += n

    total = sum(buckets.values())

    result = {
        **buckets,
        "total": total,
    }

    return result


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print(f"\nCheckpoint: {ckpt}")
        results = audit(ckpt)

        for category, count in results.items():
            print(f"{category:12s}: {count:,}")