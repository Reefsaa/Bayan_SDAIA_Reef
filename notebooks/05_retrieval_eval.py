"""Lab 5: fast bi-encoder retrieval diagnostic."""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch


QUERIES_PATH = Path("data/search/bayan_queries.jsonl")
INDEX_PREFIX = "artifacts/search/case_index_v1"

QUICK_TEST_SIZE = 10
CANDIDATES = 1000


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
        if result["case_id"] in relevant_ids:
            return 1.0 / rank

    return 0.0


def recall_at_10(results, relevant_ids):
    relevant_ids = set(relevant_ids)

    retrieved = {
        result["case_id"]
        for result in results[:10]
    }

    return 1.0 if relevant_ids & retrieved else 0.0


def bi_encoder_search(searcher, query):
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

    faiss.normalize_L2(query_vector)

    # Retrieve more candidates, but NOT all 20,000.
    scores, indices = searcher.index.search(
        query_vector,
        CANDIDATES,
    )

    pairs = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            pairs.append(
                (
                    float(score),
                    int(idx),
                )
            )

    # Stable deterministic ordering for near-duplicate
    # synthetic cases.
    pairs.sort(
        key=lambda x: (
            -round(x[0], 5),
            x[1],
        )
    )

    results = []

    for score, idx in pairs[:10]:
        item = dict(searcher.metadata[idx])

        item["bi_score"] = score

        results.append(item)

    return results


def main():
    print("=== Lab 5 FAST DIAGNOSTIC ===")

    all_queries = load_queries()

    queries = [
        q
        for q in all_queries
        if not q.get("no_answer", False)
    ][:QUICK_TEST_SIZE]

    print(f"Testing {len(queries)} queries")

    searcher = CaseSearch(INDEX_PREFIX)

    recalls = []
    mrrs = []
    latencies = []

    for q in queries:
        start = time.perf_counter()

        results = bi_encoder_search(
            searcher,
            q["query"],
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency)

        recall = recall_at_10(
            results,
            q["relevant_case_ids"],
        )

        rr = reciprocal_rank(
            results,
            q["relevant_case_ids"],
        )

        recalls.append(recall)
        mrrs.append(rr)

        print()
        print(q["query_id"])
        print("Query:", q["query"])
        print("Relevant:", q["relevant_case_ids"])
        print(
            "Top10:",
            [r["case_id"] for r in results],
        )
        print(
            f"HIT={bool(recall)} "
            f"RR={rr:.3f}"
        )

    print("\n=== FAST RESULTS ===")

    print(
        "Recall@10:",
        f"{np.mean(recalls):.4f}",
    )

    print(
        "MRR@10:",
        f"{np.mean(mrrs):.4f}",
    )

    print(
        "Average latency:",
        f"{np.mean(latencies):.2f} ms/query",
    )


if __name__ == "__main__":
    main()