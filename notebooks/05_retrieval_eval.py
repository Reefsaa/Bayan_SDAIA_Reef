"""Lab 5: labelled-query retrieval evaluation — QUICK TEST."""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch


QUERIES_PATH = Path("data/search/bayan_queries.jsonl")
INDEX_PREFIX = "artifacts/search/case_index_v1"

# Quick test only
QUICK_TEST_SIZE = 10


def load_queries():
    queries = []

    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                queries.append(json.loads(line))

    return queries


def reciprocal_rank(results, relevant_ids, k=10):
    relevant_ids = set(relevant_ids)

    for rank, result in enumerate(results[:k], start=1):
        if result.get("case_id") in relevant_ids:
            return 1.0 / rank

    return 0.0


def recall_at_k(results, relevant_ids, k=10):
    relevant_ids = set(relevant_ids)

    retrieved_ids = {
        result.get("case_id")
        for result in results[:k]
    }

    # Query-level Recall@10:
    # 1 if at least one relevant case appears in Top-10.
    return 1.0 if (relevant_ids & retrieved_ids) else 0.0


def bi_encoder_search(searcher, query, k=10):
    query = preprocess(query)

    query_vector = searcher.encoder.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32",
    )

    # Required L2 normalization.
    faiss.normalize_L2(query_vector)

    # QUICK FIX:
    # Retrieve only Top-k instead of all 20,000 vectors.
    scores, indices = searcher.index.search(
        query_vector,
        k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        case = dict(searcher.metadata[int(idx)])

        case["bi_score"] = float(score)

        results.append(case)

    return results


def evaluate_retrieval(searcher, queries):
    bi_recalls = []
    bi_mrrs = []

    rerank_recalls = []
    rerank_mrrs = []

    bi_latency = []
    rerank_latency = []

    by_lang = {
        "ar": {
            "recall": [],
            "mrr": [],
        },
        "en": {
            "recall": [],
            "mrr": [],
        },
    }

    print("\n=== QUICK Retrieval Evaluation ===")
    print(f"Queries: {len(queries)}")

    for i, q in enumerate(queries, start=1):
        query = q["query"]
        relevant = q["relevant_case_ids"]

        # -------------------------
        # Bi-encoder
        # -------------------------
        start = time.perf_counter()

        bi_results = bi_encoder_search(
            searcher,
            query,
            k=10,
        )

        bi_latency.append(
            time.perf_counter() - start
        )

        bi_recall = recall_at_k(
            bi_results,
            relevant,
            k=10,
        )

        bi_mrr = reciprocal_rank(
            bi_results,
            relevant,
            k=10,
        )

        bi_recalls.append(bi_recall)
        bi_mrrs.append(bi_mrr)

        # -------------------------
        # Cross-encoder reranking
        # -------------------------
        start = time.perf_counter()

        reranked_results = searcher.search(
            query=query,
            k=10,
            candidates=50,
            min_score=-1e9,
        )

        rerank_latency.append(
            time.perf_counter() - start
        )

        rerank_recall = recall_at_k(
            reranked_results,
            relevant,
            k=10,
        )

        rerank_mrr = reciprocal_rank(
            reranked_results,
            relevant,
            k=10,
        )

        rerank_recalls.append(rerank_recall)
        rerank_mrrs.append(rerank_mrr)

        lang = q.get("lang")

        if lang in by_lang:
            by_lang[lang]["recall"].append(
                rerank_recall
            )

            by_lang[lang]["mrr"].append(
                rerank_mrr
            )

        print(
            f"{q['query_id']} | "
            f"BI Hit={bool(bi_recall)} "
            f"BI RR={bi_mrr:.3f} | "
            f"Rerank Hit={bool(rerank_recall)} "
            f"Rerank RR={rerank_mrr:.3f}"
        )

    metrics = {
        "bi_recall_at_10": float(
            np.mean(bi_recalls)
        ),
        "bi_mrr_at_10": float(
            np.mean(bi_mrrs)
        ),
        "rerank_recall_at_10": float(
            np.mean(rerank_recalls)
        ),
        "rerank_mrr_at_10": float(
            np.mean(rerank_mrrs)
        ),
        "bi_latency_ms": float(
            np.mean(bi_latency) * 1000
        ),
        "rerank_latency_ms": float(
            np.mean(rerank_latency) * 1000
        ),
    }

    for lang in ["ar", "en"]:
        recall_values = by_lang[lang]["recall"]
        mrr_values = by_lang[lang]["mrr"]

        metrics[f"{lang}_recall_at_10"] = (
            float(np.mean(recall_values))
            if recall_values
            else 0.0
        )

        metrics[f"{lang}_mrr_at_10"] = (
            float(np.mean(mrr_values))
            if mrr_values
            else 0.0
        )

    metrics["cross_lingual_recall_gap"] = abs(
        metrics["ar_recall_at_10"]
        - metrics["en_recall_at_10"]
    )

    metrics["cross_lingual_mrr_gap"] = abs(
        metrics["ar_mrr_at_10"]
        - metrics["en_mrr_at_10"]
    )

    return metrics


def main():
    print("=== Lab 5 QUICK TEST ===")

    all_queries = load_queries()

    # Only answerable queries for quick retrieval test.
    answerable = [
        q
        for q in all_queries
        if not q.get("no_answer", False)
    ]

    queries = answerable[:QUICK_TEST_SIZE]

    print(
        f"Loaded {len(all_queries)} total queries"
    )

    print(
        f"Running QUICK TEST on "
        f"{len(queries)} answerable queries"
    )

    searcher = CaseSearch(INDEX_PREFIX)

    metrics = evaluate_retrieval(
        searcher,
        queries,
    )

    print("\n=== QUICK TEST RESULTS ===")

    print(
        "Recall@10 without reranking:",
        f"{metrics['bi_recall_at_10']:.4f}",
    )

    print(
        "MRR@10 without reranking:",
        f"{metrics['bi_mrr_at_10']:.4f}",
    )

    print(
        "Recall@10 with reranking:",
        f"{metrics['rerank_recall_at_10']:.4f}",
    )

    print(
        "MRR@10 with reranking:",
        f"{metrics['rerank_mrr_at_10']:.4f}",
    )

    print(
        "Arabic Recall@10:",
        f"{metrics['ar_recall_at_10']:.4f}",
    )

    print(
        "English Recall@10:",
        f"{metrics['en_recall_at_10']:.4f}",
    )

    print(
        "Cross-lingual Recall gap:",
        f"{metrics['cross_lingual_recall_gap']:.4f}",
    )

    print(
        "Average bi-encoder latency:",
        f"{metrics['bi_latency_ms']:.2f} ms/query",
    )

    print(
        "Average reranked latency:",
        f"{metrics['rerank_latency_ms']:.2f} ms/query",
    )

    print("\n=== QUICK TARGET CHECK ===")

    print(
        "Recall@10 >= 0.80:",
        metrics["rerank_recall_at_10"] >= 0.80,
    )

    print(
        "MRR@10 >= 0.70:",
        metrics["rerank_mrr_at_10"] >= 0.70,
    )


if __name__ == "__main__":
    main()