"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """
    Return the sample mean and a percentile bootstrap confidence interval.

    Returns:
        point_estimate, lower_bound, upper_bound
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        values = values.ravel()

    if values.size == 0:
        raise ValueError("values must not be empty")

    rng = np.random.default_rng(seed)

    point_estimate = float(np.mean(values))

    boot_means = np.empty(n_boot, dtype=float)

    n = len(values)

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        boot_means[i] = np.mean(sample)

    lower = float(
        np.quantile(
            boot_means,
            alpha / 2,
        )
    )

    upper = float(
        np.quantile(
            boot_means,
            1 - alpha / 2,
        )
    )

    return point_estimate, lower, upper


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """
    Paired bootstrap for the mean difference between two matched samples.

    The returned delta is mean(a - b).

    Returns:
        delta, lower_bound, upper_bound
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 1:
        a = a.ravel()

    if b.ndim != 1:
        b = b.ravel()

    if a.size == 0 or b.size == 0:
        raise ValueError("a and b must not be empty")

    if len(a) != len(b):
        raise ValueError(
            "a and b must have the same length for paired bootstrap"
        )

    rng = np.random.default_rng(seed)

    differences = a - b

    delta = float(np.mean(differences))

    n = len(differences)

    boot_deltas = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        boot_deltas[i] = np.mean(
            differences[indices]
        )

    lower = float(
        np.quantile(
            boot_deltas,
            alpha / 2,
        )
    )

    upper = float(
        np.quantile(
            boot_deltas,
            1 - alpha / 2,
        )
    )

    return delta, lower, upper