"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DATA_PATH = Path("data/search/bayan_cases.csv")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index(prefix: str, limit: int | None = None):
    prefix = Path(prefix)

    # 1. Load case corpus
    df = pd.read_csv(DATA_PATH)

    if "case_text" not in df.columns:
        raise ValueError(
            f"'case_text' column not found. Available columns: {list(df.columns)}"
        )

    if limit is not None:
        df = df.head(limit).copy()

    if len(df) == 0:
        raise ValueError("No cases available to index.")

    # 2. Apply shared preprocessing
    texts = [
        preprocess(str(text))
        for text in df["case_text"].fillna("")
    ]

    # 3. Load multilingual bi-encoder
    model = SentenceTransformer(MODEL_NAME)

    # 4. Encode corpus
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    vectors = np.asarray(vectors, dtype="float32")

    # 5. L2-normalise vectors
    faiss.normalize_L2(vectors)

    n_vectors, dim = vectors.shape

    # 6. Build FAISS index
    # Inner product on normalized vectors behaves like cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Make sure destination folder exists
    prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 7. Save FAISS index
    index_path = Path(f"{prefix}.faiss")

    faiss.write_index(
        index,
        str(index_path),
    )

    # 8. Save metadata
    metadata_path = Path(
        f"{prefix}_metadata.json"
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

    # 9. Save manifest
    manifest = {
        "model": MODEL_NAME,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(n_vectors),
        "dim": int(dim),
        "index_type": "IndexFlatIP",
        "normalization": "L2",
        "data_source": str(DATA_PATH),
    }

    manifest_path = Path(
        f"{prefix}_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== Lab 5: FAISS Index Build ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Preprocessing version: {PREPROC_VERSION}")
    print(f"Vectors: {n_vectors}")
    print(f"Dimension: {dim}")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")

    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "n_vectors": int(n_vectors),
        "dim": int(dim),
    }