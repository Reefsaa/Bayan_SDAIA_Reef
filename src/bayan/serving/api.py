"""Lab 7 + capstone: Bayan FastAPI service."""

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

from bayan.serving.canaries import run_startup_canaries


MODEL_DIR = Path("artifacts/topic_classifier")

INT8_MODEL_PATH = Path(
    "artifacts/onnx/topic_classifier/model_int8.onnx"
)

MAX_LENGTH = 128
THREADS = 1


# ------------------------------------------------------------
# Request schema
# ------------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str


# ------------------------------------------------------------
# Model state
# ------------------------------------------------------------

tokenizer = None
session = None
id2label = None


def load_classifier():
    """Load the winning Lab 7 ONNX INT8 classifier."""

    global tokenizer
    global session
    global id2label

    if tokenizer is not None and session is not None:
        return

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Classifier artefact missing: {MODEL_DIR}"
        )

    if not INT8_MODEL_PATH.exists():
        raise RuntimeError(
            f"INT8 ONNX artefact missing: {INT8_MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
    )

    config = AutoConfig.from_pretrained(
        MODEL_DIR
    )

    id2label = {
        int(key): value
        for key, value in config.id2label.items()
    }

    options = ort.SessionOptions()

    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(INT8_MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


# ------------------------------------------------------------
# Startup
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    load_classifier()

    run_startup_canaries()

    yield


app = FastAPI(
    title=(
        "Bayan — Bilingual Citizen-Feedback "
        "Intelligence Service"
    ),
    lifespan=lifespan,
)


# ------------------------------------------------------------
# Health
# ------------------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "classifier": "onnx-int8",
    }


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------
@app.post("/v1/classify")
def classify(payload: ClassifyRequest):

    text = payload.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text must not be empty.",
        )

    if tokenizer is None or session is None:
        load_classifier()

    encoded = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # ONNX Runtime expects int64.
    if input_ids.dtype != np.int64:
        input_ids = input_ids.astype(
            np.int64,
            copy=False,
        )

    if attention_mask.dtype != np.int64:
        attention_mask = attention_mask.astype(
            np.int64,
            copy=False,
        )

    logits = session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )[0]

    prediction_id = int(
        logits[0].argmax()
    )

    label = id2label.get(
        prediction_id,
        str(prediction_id),
    )

    return {
        "label": label,
        "label_id": prediction_id,
    }


# ------------------------------------------------------------
# Capstone endpoints
# ------------------------------------------------------------

@app.post("/v1/entities")
async def entities(payload: dict):

    raise NotImplementedError(
        "Wire the NER artefact"
    )


@app.post("/v1/search")
async def search(payload: dict):

    raise NotImplementedError(
        "Wire the semantic-search component"
    )


@app.post("/v1/analyse")
async def analyse(payload: dict):

    raise NotImplementedError(
        "Assemble the Bayan capstone service"
    )