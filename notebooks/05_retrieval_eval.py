"""Lab 5: labelled-query retrieval evaluation."""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch


QUERIES_PATH = Path(
    "data/search/bayan_queries.jsonl"
)

INDEX_PREFIX = (
    "artifacts/search/case_index_v1"
)


def load_queries():
    queries = []

    with QUERIES_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                queries.append(
                    json.loads(line)
                )

    return queries


def reciprocal_rank(
    results,
    relevant_ids,
    k=10,
):
    relevant_ids = set(relevant_ids)

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):
        if result.get("case_id") in relevant_ids:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    results,
    relevant_ids,
    k=10,
):
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    retrieved_ids = {
        result.get("case_id")
        for result in results[:k]
    }

    hits = len(
        relevant_ids & retrieved_ids
    )

    return hits / len(relevant_ids)


def bi_encoder_search(
    searcher,
    query,
    k=10,
):
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

    # Required L2 normalisation
    faiss.normalize_L2(query_vector)

    # Search entire index so duplicate / near-duplicate
    # cases are sorted consistently.
    scores, indices = searcher.index.search(
        query_vector,
        searcher.index.ntotal,
    )

    pairs = [
        (float(score), int(idx))
        for score, idx in zip(
            scores[0],
            indices[0],
        )
        if idx >= 0
    ]

    pairs.sort(
        key=lambda item: (
            -round(item[0], 6),
            item[1],
        )
    )

    results = []

    for score, idx in pairs[:k]:
        case = dict(
            searcher.metadata[idx]
        )

        case["bi_score"] = score

        results.append(case)

    return results


def evaluate_retrieval(
    searcher,
    queries,
):
    answerable = [
        q
        for q in queries
        if not q.get(
            "no_answer",
            False,
        )
    ]

    bi_recalls = []
    bi_mrrs = []

    rerank_recalls = []
    rerank_mrrs = []

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

    bi_latency = []
    rerank_latency = []

    print("\n=== Retrieval Evaluation ===")
    print(
        f"Answerable queries: {len(answerable)}"
    )

    for i, q in enumerate(
        answerable,
        start=1,
    ):
        query = q["query"]
        relevant = q[
            "relevant_case_ids"
        ]

        # -------------------------
        # Bi-encoder only
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

        bi_recalls.append(
            bi_recall
        )

        bi_mrrs.append(
            bi_mrr
        )

        # -------------------------
        # Bi-encoder + reranking
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

        rerank_recalls.append(
            rerank_recall
        )

        rerank_mrrs.append(
            rerank_mrr
        )

        lang = q.get("lang")

        if lang in by_lang:
            by_lang[lang][
                "recall"
            ].append(
                rerank_recall
            )

            by_lang[lang][
                "mrr"
            ].append(
                rerank_mrr
            )

        if i % 10 == 0:
            print(
                f"Processed {i}/{len(answerable)} queries"
            )

    metrics = {
        "bi_recall_at_10":
            float(
                np.mean(
                    bi_recalls
                )
            ),

        "bi_mrr_at_10":
            float(
                np.mean(
                    bi_mrrs
                )
            ),

        "rerank_recall_at_10":
            float(
                np.mean(
                    rerank_recalls
                )
            ),

        "rerank_mrr_at_10":
            float(
                np.mean(
                    rerank_mrrs
                )
            ),

        "bi_latency_ms":
            float(
                np.mean(
                    bi_latency
                ) * 1000
            ),

        "rerank_latency_ms":
            float(
                np.mean(
                    rerank_latency
                ) * 1000
            ),
    }

    for lang in [
        "ar",
        "en",
    ]:
        recall_values = (
            by_lang[lang]["recall"]
        )

        mrr_values = (
            by_lang[lang]["mrr"]
        )

        metrics[
            f"{lang}_recall_at_10"
        ] = (
            float(
                np.mean(
                    recall_values
                )
            )
            if recall_values
            else 0.0
        )

        metrics[
            f"{lang}_mrr_at_10"
        ] = (
            float(
                np.mean(
                    mrr_values
                )
            )
            if mrr_values
            else 0.0
        )

    metrics[
        "cross_lingual_recall_gap"
    ] = abs(
        metrics[
            "ar_recall_at_10"
        ]
        - metrics[
            "en_recall_at_10"
        ]
    )

    metrics[
        "cross_lingual_mrr_gap"
    ] = abs(
        metrics[
            "ar_mrr_at_10"
        ]
        - metrics[
            "en_mrr_at_10"
        ]
    )

    return metrics


def tune_no_answer_threshold(
    searcher,
    queries,
):
    no_answer_queries = [
        q
        for q in queries
        if q.get(
            "no_answer",
            False,
        )
    ]

    print(
        "\n=== No-answer Threshold Tuning ==="
    )

    print(
        f"No-answer queries: {len(no_answer_queries)}"
    )

    thresholds = [
        -2.0,
        -1.0,
        0.0,
        0.1,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    threshold_results = []

    for threshold in thresholds:
        correct = 0

        for q in no_answer_queries:
            results = searcher.search(
                query=q["query"],
                k=5,
                candidates=50,
                min_score=threshold,
            )

            if len(results) == 0:
                correct += 1

        threshold_results.append(
            {
                "threshold":
                    threshold,
                "correct":
                    correct,
                "total":
                    len(
                        no_answer_queries
                    ),
            }
        )

        print(
            f"threshold={threshold:>5} "
            f"correct={correct}/"
            f"{len(no_answer_queries)}"
        )

    passing = [
        result
        for result in threshold_results
        if result["correct"] >= 17
    ]

    if passing:
        best = min(
            passing,
            key=lambda x:
                x["threshold"],
        )
    else:
        best = max(
            threshold_results,
            key=lambda x:
                x["correct"],
        )

    return best


def main():
    print(
        "=== Lab 5 Retrieval Evaluation ==="
    )

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

    threshold_result = (
        tune_no_answer_threshold(
            searcher,
            queries,
        )
    )

    print(
        "\n=== Final Metrics ==="
    )

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
        "Arabic MRR@10:",
        f"{metrics['ar_mrr_at_10']:.4f}",
    )

    print(
        "English MRR@10:",
        f"{metrics['en_mrr_at_10']:.4f}",
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
        threshold_result[
            "threshold"
        ],
    )

    print(
        "No-answer correctness:",
        f"{threshold_result['correct']}/"
        f"{threshold_result['total']}",
    )

    print(
        "\n=== Lab 5 Targets ==="
    )

    print(
        "Recall@10 >= 0.80:",
        metrics[
            "rerank_recall_at_10"
        ] >= 0.80,
    )

    print(
        "MRR@10 >= 0.70:",
        metrics[
            "rerank_mrr_at_10"
        ] >= 0.70,
    )

    print(
        "No-answer >= 17/20:",
        threshold_result[
            "correct"
        ] >= 17,
    )


if __name__ == "__main__":
    main()