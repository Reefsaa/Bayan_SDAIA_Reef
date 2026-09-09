"""Lab 6: behavioural test generators and runners."""


def run_behavioural_suite(
    predict_fn,
    *,
    invariance_tests=None,
    directional_tests=None,
    mft_tests=None,
):
    """
    Run behavioural tests for:
    - Invariance
    - Directional expectation
    - Minimum Functionality Tests (MFT)

    Each test is represented as a dictionary.

    Invariance example:
        {
            "original": "الخدمة ممتازة",
            "perturbed": "  الخدمة ممتازة  "
        }

    Directional example:
        {
            "text_a": "الخدمة جيدة",
            "text_b": "الخدمة ممتازة",
            "expected": "different"
        }

    MFT example:
        {
            "text": "حفرة في الطريق",
            "expected": "roads"
        }
    """

    invariance_tests = invariance_tests or []
    directional_tests = directional_tests or []
    mft_tests = mft_tests or []

    # -------------------------------------------------
    # Helper
    # -------------------------------------------------
    def predict(text):
        result = predict_fn(text)

        # Allow predictors returning dictionaries.
        if isinstance(result, dict):
            for key in (
                "label",
                "prediction",
                "predicted_label",
                "class",
            ):
                if key in result:
                    return result[key]

        return result

    # -------------------------------------------------
    # Invariance tests
    # -------------------------------------------------
    invariance_passed = 0

    for test in invariance_tests:
        original = test["original"]
        perturbed = test["perturbed"]

        original_prediction = predict(original)
        perturbed_prediction = predict(perturbed)

        if original_prediction == perturbed_prediction:
            invariance_passed += 1

    invariance_total = len(invariance_tests)

    invariance_rate = (
        invariance_passed / invariance_total
        if invariance_total
        else 0.0
    )

    # -------------------------------------------------
    # Directional tests
    # -------------------------------------------------
    directional_passed = 0

    for test in directional_tests:
        text_a = test["text_a"]
        text_b = test["text_b"]

        prediction_a = predict(text_a)
        prediction_b = predict(text_b)

        expected = test.get("expected", "different")

        if expected == "same":
            passed = prediction_a == prediction_b

        elif expected == "different":
            passed = prediction_a != prediction_b

        else:
            # If a specific expected label is supplied,
            # require the second example to reach it.
            passed = prediction_b == expected

        if passed:
            directional_passed += 1

    directional_total = len(directional_tests)

    directional_rate = (
        directional_passed / directional_total
        if directional_total
        else 0.0
    )

    # -------------------------------------------------
    # Minimum Functionality Tests
    # -------------------------------------------------
    mft_passed = 0

    for test in mft_tests:
        prediction = predict(test["text"])
        expected = test["expected"]

        if prediction == expected:
            mft_passed += 1

    mft_total = len(mft_tests)

    mft_rate = (
        mft_passed / mft_total
        if mft_total
        else 0.0
    )

    # -------------------------------------------------
    # Final report
    # -------------------------------------------------
    return {
        "invariance": {
            "passed": invariance_passed,
            "total": invariance_total,
            "rate": float(invariance_rate),
        },
        "directional": {
            "passed": directional_passed,
            "total": directional_total,
            "rate": float(directional_rate),
        },
        "mft": {
            "passed": mft_passed,
            "total": mft_total,
            "rate": float(mft_rate),
        },
    }