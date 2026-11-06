"""Tests for data generation and loading."""

from perceptron_project.data import generate_synthetic_data, load_iris_binary


def test_synthetic_data_shapes() -> None:
    """Generate the requested number of two-dimensional samples."""
    X, y = generate_synthetic_data(samples=20, seed=1)

    assert len(X) == 20
    assert len(y) == 20
    assert all(len(sample) == 2 for sample in X)
    assert set(y) == {0, 1}


def test_synthetic_data_reproducible() -> None:
    """The same seed produces the same generated dataset."""
    X1, y1 = generate_synthetic_data(samples=20, seed=7)
    X2, y2 = generate_synthetic_data(samples=20, seed=7)

    assert X1 == X2
    assert y1 == y2


def test_iris_binary_data() -> None:
    """Load two Iris classes and two numeric features."""
    X, y = load_iris_binary()

    assert len(X) == 100
    assert len(y) == 100
    assert all(len(sample) == 2 for sample in X)
    assert set(y) == {0, 1}
