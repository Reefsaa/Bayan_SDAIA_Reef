"""Lab 1 starter: sentence segmentation."""

import re
import spacy

from bayan.preprocessing.core import preprocess


def build_pipeline():
    """Build the spaCy sentence segmentation pipeline."""
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    """Preprocess then return non-empty sentence strings."""

    cleaned = preprocess(raw)
    doc = nlp(cleaned)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    merged = []
    i = 0

    while i < len(sentences):
        # Merge numbered-list marker with the following sentence
        if re.fullmatch(r"\d+\.", sentences[i]) and i + 1 < len(sentences):
            merged.append(f"{sentences[i]} {sentences[i + 1]}")
            i += 2
        else:
            merged.append(sentences[i])
            i += 1

    return merged