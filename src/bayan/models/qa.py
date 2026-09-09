"""Lab 3 starter: extractive QA post-processing."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    # Take the strongest start/end candidates
    start_indexes = np.argsort(start_logits)[-top_k:][::-1]
    end_indexes = np.argsort(end_logits)[-top_k:][::-1]

    best = None
    best_score = float("-inf")

    for start_idx in start_indexes:
        for end_idx in end_indexes:

            # Ignore special tokens / tokens without text offsets
            if offsets[start_idx] is None or offsets[end_idx] is None:
                continue

            # Reject inverted spans
            if end_idx < start_idx:
                continue

            # Reject answers that are too long
            if (end_idx - start_idx + 1) > max_answer_len:
                continue

            score = float(
                start_logits[start_idx] + end_logits[end_idx]
            )

            if score > best_score:
                best_score = score
                best = {
                    "answer": (
                        offsets[start_idx][0],
                        offsets[end_idx][1],
                    ),
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                    "score": score,
                }

    # Honest no-answer handling
    if best is None:
        return {
            "answer": None,
            "score": float(null_score),
        }

    if float(null_score) - best_score >= float(null_threshold):
        return {
            "answer": None,
            "score": float(null_score),
        }

    return best