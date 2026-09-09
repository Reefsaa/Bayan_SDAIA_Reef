"""Lab 6: sliced evaluation report."""

from collections import defaultdict


def sliced_report(
    y_true,
    y_pred,
    *,
    language=None,
    dialect=None,
    lengths=None,
    min_slice_size=20,
):
    """
    Evaluate accuracy across language, dialect, class, and length slices.

    Small slices are flagged when their size is below min_slice_size.
    """

    y_true = list(y_true)
    y_pred = list(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    n = len(y_true)

    if language is not None and len(language) != n:
        raise ValueError("language must have the same length as y_true")

    if dialect is not None and len(dialect) != n:
        raise ValueError("dialect must have the same length as y_true")

    if lengths is not None and len(lengths) != n:
        raise ValueError("lengths must have the same length as y_true")

    def evaluate_slice(indices):
        indices = list(indices)

        size = len(indices)

        if size == 0:
            accuracy = 0.0
        else:
            correct = sum(
                y_true[i] == y_pred[i]
                for i in indices
            )
            accuracy = correct / size

        return {
            "n": size,
            "accuracy": float(accuracy),
            "small_slice": size < min_slice_size,
        }

    report = {}

    # Overall
    report["overall"] = evaluate_slice(range(n))

    # -------------------------
    # Language slices
    # -------------------------
    if language is not None:
        groups = defaultdict(list)

        for i, value in enumerate(language):
            groups[value].append(i)

        report["language"] = {
            str(value): evaluate_slice(indices)
            for value, indices in groups.items()
        }

    # -------------------------
    # Dialect slices
    # -------------------------
    if dialect is not None:
        groups = defaultdict(list)

        for i, value in enumerate(dialect):
            groups[value].append(i)

        report["dialect"] = {
            str(value): evaluate_slice(indices)
            for value, indices in groups.items()
        }

    # -------------------------
    # Class slices
    # -------------------------
    groups = defaultdict(list)

    for i, label in enumerate(y_true):
        groups[label].append(i)

    report["class"] = {
        str(label): evaluate_slice(indices)
        for label, indices in groups.items()
    }

    # -------------------------
    # Length slices
    # -------------------------
    if lengths is not None:
        length_groups = {
            "short": [],
            "medium": [],
            "long": [],
        }

        for i, length in enumerate(lengths):
            if length < 10:
                length_groups["short"].append(i)
            elif length < 30:
                length_groups["medium"].append(i)
            else:
                length_groups["long"].append(i)

        report["length"] = {
            name: evaluate_slice(indices)
            for name, indices in length_groups.items()
        }

    return report