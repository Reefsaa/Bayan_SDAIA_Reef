"""Lab 3B: run the QA smoke set and report fixture consistency."""

import json
from pathlib import Path


DATA_PATH = Path("data/eval/qa_smoke_set.json")


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    qas = []

    for item in data.get("data", []):
        for paragraph in item.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                qas.append(qa)

    answerable = [qa for qa in qas if not qa.get("is_impossible", False)]
    unanswerable = [qa for qa in qas if qa.get("is_impossible", False)]

    print("=== Lab 3B: QA Smoke Set ===")
    print(f"Total questions: {len(qas)}")
    print(f"Answerable: {len(answerable)}")
    print(f"Unanswerable: {len(unanswerable)}")

    if len(answerable) == 9 and len(unanswerable) == 3:
        print("PASS: fixture contains 9 answerable + 3 unanswerable.")
    else:
        print(
            "FIXTURE MISMATCH: README target is 9 answerable + "
            f"3 unanswerable, but supplied fixture contains "
            f"{len(answerable)} answerable + {len(unanswerable)} unanswerable."
        )


if __name__ == "__main__":
    main()