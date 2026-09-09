"""Lab 5: labelled-query retrieval evaluation."""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch


QUERIES_PATH = Path("data/search/bayan_queries.jsonl")
INDEX_PREFIX = "artifacts/search/case_index_v1"


def load_queries():
    queries = []

    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                queries.append(json.loads(line))

    return queries


def recall_at_10(results, relevant_ids):
    relevant = set(relevant_ids)

    retrieved = {
        result.get("case_id")
        for result in results[:10]
    }

    return 1.0 if relevant & retrieved else 0.0


def reciprocal_rank(results, relevant_ids):
    relevant = set(relevant_ids)

    for rank, result in enumerate(results[:10], start=1):
        if result.get("case_id") in relevant:
            return 1.0 / rank

    return 0.0


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

    # Required L2 normalization
    faiss.normalize_L2(query_vector)

    scores, indices = searcher.index.search(
        query_vector,
        k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        idx = int(idx)

        if idx < 0:
            continue

        case = dict(searcher.metadata[idx])
        case["bi_score"] = float(score)

        results.append(case)

    return results


def evaluate_retrieval(searcher, queries):
    answerable = [
        q
        for q in queries
        if not q.get("no_answer", False)
    ]

    bi_recalls = []
    bi_mrrs = []

    rerank_recalls = []
    rerank_mrrs = []

    bi_times = []
    rerank_times = []

    language = {
        "ar": {"recall": [], "mrr": []},
        "en": {"recall": [], "mrr": []},
    }

    print("\n=== Retrieval Evaluation ===")
    print(f"Answerable queries: {len(answerable)}")

    for i, q in enumerate(answerable, start=1):
        query = q["query"]
        relevant = q["relevant_case_ids"]

        # Bi-encoder
        start = time.perf_counter()

        bi_results = bi_encoder_search(
            searcher,
            query,
            k=10,
        )

        bi_times.append(
            time.perf_counter() - start
        )

        bi_recall = recall_at_10(
            bi_results,
            relevant,
        )

        bi_mrr = reciprocal_rank(
            bi_results,
            relevant,
        )

        bi_recalls.append(bi_recall)
        bi_mrrs.append(bi_mrr)

        # Bi-encoder + cross-encoder reranking
        start = time.perf_counter()

        reranked = searcher.search(
            query=query,
            k=10,
            candidates=50,
            min_score=-1e9,
        )

        rerank_times.append(
            time.perf_counter() - start
        )

        rerank_recall = recall_at_10(
            reranked,
            relevant,
        )

        rerank_mrr = reciprocal_rank(
            reranked,
            relevant,
        )

        rerank_recalls.append(rerank_recall)
        rerank_mrrs.append(rerank_mrr)

        lang = q.get("lang")

        if lang in language:
            language[lang]["recall"].append(
                rerank_recall
            )
            language[lang]["mrr"].append(
                rerank_mrr
            )

        if i % 10 == 0:
            print(
                f"Processed {i}/{len(answerable)} queries"
            )

    metrics = {
        "bi_recall": float(np.mean(bi_recalls)),
        "bi_mrr": float(np.mean(bi_mrrs)),
        "rerank_recall": float(np.mean(rerank_recalls)),
        "rerank_mrr": float(np.mean(rerank_mrrs)),
        "bi_latency_ms": float(np.mean(bi_times) * 1000),
        "rerank_latency_ms": float(np.mean(rerank_times) * 1000),
    }

    for lang in ["ar", "en"]:
        recalls = language[lang]["recall"]
        mrrs = language[lang]["mrr"]

        metrics[f"{lang}_recall"] = (
            float(np.mean(recalls))
            if recalls
            else 0.0
        )

        metrics[f"{lang}_mrr"] = (
            float(np.mean(mrrs))
            if mrrs
            else 0.0
        )

    metrics["cross_lingual_recall_gap"] = abs(
        metrics["ar_recall"]
        - metrics["en_recall"]
    )

    metrics["cross_lingual_mrr_gap"] = abs(
        metrics["ar_mrr"]
        - metrics["en_mrr"]
    )

    return metrics


def evaluate_no_answer(searcher, queries):
    no_answer = [
        q
        for q in queries
        if q.get("no_answer", False)
    ]

    print("\n=== No-answer Evaluation ===")
    print(f"No-answer queries: {len(no_answer)}")

    top_scores = []

    # Run the reranker once per no-answer query
    for i, q in enumerate(no_answer, start=1):
        results = searcher.search(
            query=q["query"],
            k=1,
            candidates=50,
            min_score=-1e9,
        )

        if results:
            top_scores.append(
                float(results[0]["score"])
            )
        else:
            top_scores.append(
                float("-inf")
            )

        print(
            f"Processed no-answer {i}/{len(no_answer)}"
        )

    thresholds = [
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    threshold_results = []

    print("\nNo-answer threshold behaviour:")

    for threshold in thresholds:
        correct = sum(
            score < threshold
            for score in top_scores
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "correct": correct,
            }
        )

        print(
            f"threshold={threshold:>4} "
            f"correct={correct}/{len(no_answer)}"
        )

    passing = [
        item
        for item in threshold_results
        if item["correct"] >= 17
    ]

    if passing:
        selected = min(
            passing,
            key=lambda x: x["threshold"],
        )
    else:
        selected = max(
            threshold_results,
            key=lambda x: x["correct"],
        )

    return {
        "threshold": selected["threshold"],
        "correct": selected["correct"],
        "total": len(no_answer),
    }


def main():
    print("=== Lab 5 Retrieval Evaluation ===")

    queries = load_queries()

    print(
        f"Loaded queries: {len(queries)}"
    )

    searcher = CaseSearch(
        INDEX_PREFIX
    )

    metrics = evaluate_retrieval(
        searcher,
        queries,
    )

    no_answer = evaluate_no_answer(
        searcher,
        queries,
    )

    print("\n=== Final Metrics ===")

    print(
        "Recall@10 without reranking:",
        f"{metrics['bi_recall']:.4f}",
    )

    print(
        "MRR@10 without reranking:",
        f"{metrics['bi_mrr']:.4f}",
    )

    print(
        "Recall@10 with reranking:",
        f"{metrics['rerank_recall']:.4f}",
    )

    print(
        "MRR@10 with reranking:",
        f"{metrics['rerank_mrr']:.4f}",
    )

    print(
        "Arabic Recall@10:",
        f"{metrics['ar_recall']:.4f}",
    )

    print(
        "English Recall@10:",
        f"{metrics['en_recall']:.4f}",
    )

    print(
        "Cross-lingual Recall gap:",
        f"{metrics['cross_lingual_recall_gap']:.4f}",
    )

    print(
        "Arabic MRR@10:",
        f"{metrics['ar_mrr']:.4f}",
    )

    print(
        "English MRR@10:",
        f"{metrics['en_mrr']:.4f}",
    )

    print(
        "Cross-lingual MRR gap:",
        f"{metrics['cross_lingual_mrr_gap']:.4f}",
    )

    print(
        "Average bi-encoder latency:",
        f"{metrics['bi_latency_ms']:.2f} ms/query",
    )

    print(
        "Average reranked latency:",
        f"{metrics['rerank_latency_ms']:.2f} ms/query",
    )

    print(
        "Selected no-answer threshold:",
        no_answer["threshold"],
    )

    print(
        "No-answer correctness:",
        f"{no_answer['correct']}/{no_answer['total']}",
    )

    print("\n=== Lab 5 Targets ===")

    print(
        "Recall@10 >= 0.80:",
        metrics["rerank_recall"] >= 0.80,
    )

    print(
        "MRR@10 >= 0.70:",
        metrics["rerank_mrr"] >= 0.70,
    )

    print(
        "No-answer >= 17/20:",
        no_answer["correct"] >= 17,
    )


if __name__ == "__main__":
    main()