"""Lab 7/capstone: startup skew and behaviour canaries."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_DIR = Path("artifacts/topic_classifier")

INT8_MODEL_PATH = Path(
    "artifacts/onnx/topic_classifier/model_int8.onnx"
)


def run_startup_canaries() -> None:
    """Validate the winning classifier artefact before serving."""

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Classifier artefact missing: {MODEL_DIR}"
        )

    if not INT8_MODEL_PATH.exists():
        raise RuntimeError(
            f"INT8 ONNX artefact missing: {INT8_MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(INT8_MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    # Pinned bilingual behaviour canaries.
    samples = [
        "الطريق يحتاج إلى صيانة",
        "The road needs maintenance.",
    ]

    for text in samples:

        encoded = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,
        )

        inputs = {
            "input_ids": encoded[
                "input_ids"
            ].astype(np.int64),

            "attention_mask": encoded[
                "attention_mask"
            ].astype(np.int64),
        }

        outputs = session.run(
            None,
            inputs,
        )

        if not outputs:
            raise RuntimeError(
                "Classifier canary returned no output."
            )

        logits = outputs[0]

        if logits.ndim != 2:
            raise RuntimeError(
                "Unexpected classifier logits shape: "
                f"{logits.shape}"
            )

        if not np.isfinite(logits).all():
            raise RuntimeError(
                "Classifier canary produced "
                "non-finite logits."
            )

    print(
        "Startup canaries passed: "
        "classifier INT8 artefact is healthy."
    )