"""Lab 5: two-stage bilingual case search."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str):
        self.prefix = Path(prefix)

        manifest_path = Path(f"{self.prefix}_manifest.json")
        metadata_path = Path(f"{self.prefix}_metadata.json")
        index_path = Path(f"{self.prefix}.faiss")

        # -------------------------
        # Check persisted files
        # -------------------------
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}"
            )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        # -------------------------
        # Load manifest
        # -------------------------
        self.manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        required_keys = [
            "model",
            "preproc_version",
            "n_vectors",
            "dim",
        ]

        for key in required_keys:
            if key not in self.manifest:
                raise ValueError(
                    f"Manifest missing required key: {key}"
                )

        # -------------------------
        # Verify preprocessing version
        # -------------------------
        if (
            self.manifest["preproc_version"]
            != PREPROC_VERSION
        ):
            raise ValueError(
                "Preprocessing version mismatch: "
                f"index={self.manifest['preproc_version']} "
                f"current={PREPROC_VERSION}"
            )

        # -------------------------
        # Load FAISS index
        # -------------------------
        self.index = faiss.read_index(
            str(index_path)
        )

        if (
            self.index.ntotal
            != self.manifest["n_vectors"]
        ):
            raise ValueError(
                "Index vector count does not match manifest."
            )

        if (
            self.index.d
            != self.manifest["dim"]
        ):
            raise ValueError(
                "Index dimension does not match manifest."
            )

        # -------------------------
        # Load metadata
        # -------------------------
        self.metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            len(self.metadata)
            != self.manifest["n_vectors"]
        ):
            raise ValueError(
                "Metadata count does not match manifest."
            )

        # -------------------------
        # Load models
        # -------------------------
        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = -1e9,
    ):
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        # IMPORTANT:
        # Same preprocessing convention used
        # when the FAISS index was built.
        query = preprocess(query)

        if not query.strip():
            return []

        if k <= 0:
            return []

        candidates = max(
            int(candidates),
            int(k),
        )

        candidates = min(
            candidates,
            self.index.ntotal,
        )

        # -------------------------
        # Bi-encoder query encoding
        # -------------------------
        query_vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_vector = np.asarray(
            query_vector,
            dtype="float32",
        )

        # Required for cosine-like
        # similarity with IndexFlatIP.
        faiss.normalize_L2(
            query_vector
        )

        # -------------------------
        # FAISS candidate retrieval
        # -------------------------
        bi_scores, indices = (
            self.index.search(
                query_vector,
                candidates,
            )
        )

        candidate_results = []

        for bi_score, idx in zip(
            bi_scores[0],
            indices[0],
        ):
            idx = int(idx)

            if idx < 0:
                continue

            metadata = self.metadata[idx]

            # Use the same information
            # represented in the index.
            searchable_text = " ".join(
                [
                    str(
                        metadata.get(
                            "topic",
                            "",
                        )
                    ),
                    str(
                        metadata.get(
                            "case_text",
                            "",
                        )
                    ),
                    str(
                        metadata.get(
                            "resolution",
                            "",
                        )
                    ),
                ]
            ).strip()

            candidate_results.append(
                {
                    "index": idx,
                    "case": metadata,
                    "text": searchable_text,
                    "bi_score": float(
                        bi_score
                    ),
                }
            )

        if not candidate_results:
            return []

        # -------------------------
        # Cross-encoder reranking
        # -------------------------
        rerank_pairs = [
            [
                query,
                item["text"],
            ]
            for item in candidate_results
        ]

        rerank_scores = (
            self.reranker.predict(
                rerank_pairs,
                show_progress_bar=False,
            )
        )

        rerank_scores = np.asarray(
            rerank_scores
        ).reshape(-1)

        for item, score in zip(
            candidate_results,
            rerank_scores,
        ):
            item["score"] = float(
                score
            )

        # Highest cross-encoder
        # score first.
        candidate_results.sort(
            key=lambda item: (
                -item["score"],
                item["index"],
            )
        )

        # -------------------------
        # Honest no-result filtering
        # -------------------------
        filtered_results = [
            item
            for item in candidate_results
            if item["score"] >= min_score
        ]

        results = []

        for item in filtered_results[:k]:
            case = dict(
                item["case"]
            )

            case["score"] = (
                item["score"]
            )

            case["bi_score"] = (
                item["bi_score"]
            )

            results.append(
                case
            )

        return results