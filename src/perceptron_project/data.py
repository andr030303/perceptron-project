"""Dataset generation and loading utilities."""

from __future__ import annotations

from typing import cast

from sklearn.datasets import load_iris
from sklearn.utils import Bunch


def generate_synthetic_data(
    samples: int = 200,
    seed: int = 42,
) -> tuple[list[list[float]], list[int]]:
    """Generate two linearly separable clusters in two dimensions.

    The number of samples is the total count, split as evenly as possible
    between class 0 and class 1.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2.")

    import numpy as np

    rng = np.random.default_rng(seed)
    class_0_count = samples // 2
    class_1_count = samples - class_0_count

    class_0 = rng.normal(
        loc=(2.0, 2.0),
        scale=0.45,
        size=(class_0_count, 2),
    )
    class_1 = rng.normal(
        loc=(6.0, 6.0),
        scale=0.45,
        size=(class_1_count, 2),
    )

    features = np.vstack((class_0, class_1))
    labels = [0] * class_0_count + [1] * class_1_count

    return features.tolist(), labels


def load_iris_binary() -> tuple[list[list[float]], list[int]]:
    """Load two Iris classes using petal length and petal width.

    The first two Iris classes are used, resulting in 100 samples.
    """
    iris = cast(Bunch, load_iris())
    features = iris.data[:100, 2:4].tolist()
    labels = iris.target[:100].astype(int).tolist()
    return features, labels
