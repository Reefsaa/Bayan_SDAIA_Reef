"""Lab 7: ONNX export and dynamic INT8 quantisation."""

from pathlib import Path

import onnx
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from onnxruntime.quantization import quantize_dynamic, QuantType


# ============================================================
# Classifier paths
# ============================================================

CLASSIFIER_MODEL_DIR = Path("artifacts/topic_classifier")
CLASSIFIER_OUTPUT_DIR = Path(
    "artifacts/onnx/topic_classifier"
)

CLASSIFIER_FP32_PATH = (
    CLASSIFIER_OUTPUT_DIR / "model_fp32.onnx"
)

CLASSIFIER_QUANT_READY_PATH = (
    CLASSIFIER_OUTPUT_DIR
    / "model_fp32_quant_ready.onnx"
)

CLASSIFIER_INT8_PATH = (
    CLASSIFIER_OUTPUT_DIR / "model_int8.onnx"
)


# ============================================================
# NER paths
# ============================================================

NER_MODEL_DIR = Path("artifacts/ner")
NER_OUTPUT_DIR = Path(
    "artifacts/onnx/ner"
)

NER_FP32_PATH = (
    NER_OUTPUT_DIR / "model_fp32.onnx"
)

NER_QUANT_READY_PATH = (
    NER_OUTPUT_DIR
    / "model_fp32_quant_ready.onnx"
)

NER_INT8_PATH = (
    NER_OUTPUT_DIR / "model_int8.onnx"
)


# ============================================================
# Wrappers
# ============================================================

class ClassifierWrapper(torch.nn.Module):
    """Return classifier logits only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


class NERWrapper(torch.nn.Module):
    """Return NER token-classification logits only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


# ============================================================
# Classifier export
# ============================================================

def export_classifier():
    """Export classifier to FP32 ONNX."""

    if not CLASSIFIER_MODEL_DIR.exists():
        raise FileNotFoundError(
            "Classifier artefact not found: "
            f"{CLASSIFIER_MODEL_DIR}"
        )

    CLASSIFIER_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CLASSIFIER_FP32_PATH.exists():
        print(
            "Classifier FP32 ONNX already exists: "
            f"{CLASSIFIER_FP32_PATH}"
        )
        print(
            "Keeping existing classifier FP32 "
            "rollback artefact."
        )
        return

    print("Loading FP32 classifier...")

    tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_MODEL_DIR
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            CLASSIFIER_MODEL_DIR
        )
    )

    model.eval()
    model.to("cpu")

    wrapper = ClassifierWrapper(model)
    wrapper.eval()

    sample = tokenizer(
        "الخدمة ممتازة والطريق يحتاج إلى صيانة",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    print("Exporting classifier to ONNX...")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (
                sample["input_ids"],
                sample["attention_mask"],
            ),
            str(CLASSIFIER_FP32_PATH),
            input_names=[
                "input_ids",
                "attention_mask",
            ],
            output_names=[
                "logits",
            ],
            dynamic_axes={
                "input_ids": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "logits": {
                    0: "batch_size",
                },
            },
            opset_version=17,
            do_constant_folding=True,
        )

    tokenizer.save_pretrained(
        CLASSIFIER_OUTPUT_DIR
    )

    print(
        "Classifier FP32 ONNX exported: "
        f"{CLASSIFIER_FP32_PATH}"
    )


# ============================================================
# Generic quantisation preparation
# ============================================================

def prepare_for_quantisation(
    fp32_path,
    quant_ready_path,
):
    """Remove stale intermediate shape information."""

    if not fp32_path.exists():
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {fp32_path}"
        )

    print()
    print(
        "Preparing ONNX graph for "
        "INT8 quantisation..."
    )

    model = onnx.load(
        str(fp32_path),
        load_external_data=True,
    )

    # Avoid stale shape metadata problems during
    # ORT dynamic quantisation.
    del model.graph.value_info[:]

    data_filename = (
        quant_ready_path.name + ".data"
    )

    onnx.save_model(
        model,
        str(quant_ready_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
        convert_attribute=False,
    )

    print(
        "Quantisation-ready model: "
        f"{quant_ready_path}"
    )


def quantize_model(
    quant_ready_path,
    int8_path,
    model_name,
):
    """Create a dynamic INT8 ONNX model."""

    if not quant_ready_path.exists():
        raise FileNotFoundError(
            "Quantisation-ready model not found: "
            f"{quant_ready_path}"
        )

    print()
    print(
        f"Quantising {model_name} to INT8..."
    )

    quantize_dynamic(
        model_input=str(quant_ready_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        extra_options={
            "MatMulConstBOnly": True,
        },
    )

    print(
        f"{model_name} INT8 artefact: "
        f"{int8_path}"
    )


# ============================================================
# NER export
# ============================================================

def export_ner():
    """Export NER model to FP32 ONNX."""

    if not NER_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"NER artefact not found: {NER_MODEL_DIR}"
        )

    NER_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if NER_FP32_PATH.exists():
        print()
        print(
            "NER FP32 ONNX already exists: "
            f"{NER_FP32_PATH}"
        )
        print(
            "Keeping existing NER FP32 "
            "rollback artefact."
        )
        return

    print()
    print("Loading FP32 NER model...")

    tokenizer = AutoTokenizer.from_pretrained(
        NER_MODEL_DIR,
        use_fast=True,
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            NER_MODEL_DIR
        )
    )

    model.eval()
    model.to("cpu")

    wrapper = NERWrapper(model)
    wrapper.eval()

    sample = tokenizer(
        "الخدمة في الرياض تحتاج إلى تحسين",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    print("Exporting NER to ONNX...")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (
                sample["input_ids"],
                sample["attention_mask"],
            ),
            str(NER_FP32_PATH),
            input_names=[
                "input_ids",
                "attention_mask",
            ],
            output_names=[
                "logits",
            ],
            dynamic_axes={
                "input_ids": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "logits": {
                    0: "batch_size",
                    1: "sequence_length",
                },
            },
            opset_version=17,
            do_constant_folding=True,
        )

    tokenizer.save_pretrained(
        NER_OUTPUT_DIR
    )

    print(
        "NER FP32 ONNX exported: "
        f"{NER_FP32_PATH}"
    )


# ============================================================
# Artefact sizes
# ============================================================

def get_total_size(
    output_dir,
    prefix,
):
    """Return total bytes for model and external data."""

    files = list(
        output_dir.glob(
            f"{prefix}*"
        )
    )

    return sum(
        path.stat().st_size
        for path in files
        if path.is_file()
    )


def print_model_sizes(
    model_name,
    output_dir,
):
    """Print FP32 and INT8 model sizes."""

    fp32_bytes = get_total_size(
        output_dir,
        "model_fp32.onnx",
    )

    int8_bytes = get_total_size(
        output_dir,
        "model_int8.onnx",
    )

    print()
    print(
        f"=== Lab 7 {model_name} Artefacts ==="
    )

    print(
        "FP32 total size: "
        f"{fp32_bytes / (1024 ** 2):.2f} MB"
    )

    print(
        "INT8 total size: "
        f"{int8_bytes / (1024 ** 2):.2f} MB"
    )

    if int8_bytes > 0:
        reduction = (
            fp32_bytes / int8_bytes
        )

        print(
            "Size reduction: "
            f"{reduction:.2f}x"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========== CLASSIFIER =========="
    )

    export_classifier()

    # Do not rebuild classifier INT8 if it already exists.
    if CLASSIFIER_INT8_PATH.exists():
        print()
        print(
            "Classifier INT8 already exists."
        )
        print(
            "Keeping existing classifier "
            "INT8 artefact."
        )
    else:
        prepare_for_quantisation(
            CLASSIFIER_FP32_PATH,
            CLASSIFIER_QUANT_READY_PATH,
        )

        quantize_model(
            CLASSIFIER_QUANT_READY_PATH,
            CLASSIFIER_INT8_PATH,
            "Classifier",
        )

    print_model_sizes(
        "Classifier",
        CLASSIFIER_OUTPUT_DIR,
    )

    print()
    print(
        "============= NER ============="
    )

    export_ner()

    prepare_for_quantisation(
        NER_FP32_PATH,
        NER_QUANT_READY_PATH,
    )

    quantize_model(
        NER_QUANT_READY_PATH,
        NER_INT8_PATH,
        "NER",
    )

    print_model_sizes(
        "NER",
        NER_OUTPUT_DIR,
    )

    print()
    print(
        "Lab 7 ONNX export and "
        "INT8 quantisation complete."
    )


if __name__ == "__main__":
    main()