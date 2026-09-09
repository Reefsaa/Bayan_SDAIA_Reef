"""Lab 5: build a versioned FAISS search index."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DATA_PATH = Path("data/search/bayan_cases.csv")
MODEL_NAME = "intfloat/multilingual-e5-base"


def build_index(
    prefix: str = "artifacts/search/case_index_v1",
    limit=None,
):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print("=== Lab 5: FAISS Index Build ===")
    print("Model:", MODEL_NAME)
    print("Preprocessing version:", PREPROC_VERSION)

    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit).copy()

    texts = [
        "passage: " + preprocess(str(text))
        for text in df["case_text"].fillna("")
    ]

    model = SentenceTransformer(MODEL_NAME)

    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(
        vectors,
        dtype="float32",
    )

    # Keep explicit normalization for FAISS contract.
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = Path(f"{prefix}.faiss")
    metadata_path = Path(f"{prefix}_metadata.json")
    manifest_path = Path(f"{prefix}_manifest.json")

    faiss.write_index(
        index,
        str(index_path),
    )

    metadata = df.to_dict(
        orient="records"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    manifest = {
        "model": MODEL_NAME,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
        "index_type": "IndexFlatIP",
        "normalization": "L2",
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "data_source": str(DATA_PATH),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Vectors:", index.ntotal)
    print("Dimension:", dim)
    print("Index:", index_path)
    print("Metadata:", metadata_path)
    print("Manifest:", manifest_path)

    return index


if __name__ == "__main__":
    build_index()