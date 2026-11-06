"""Tests for the Perceptron class."""

import pytest

from perceptron_project.perceptron import Perceptron


def test_activate_threshold() -> None:
    """Check the activation threshold, including zero."""
    assert Perceptron.activate(0) == 1
    assert Perceptron.activate(-1) == 0
    assert Perceptron.activate(1) == 1


def test_weight_initialization() -> None:
    """Check that the initial weight vector has the requested length."""
    model = Perceptron(n_features=3)
    assert len(model.w) == 3
    assert model.w == [0.0, 0.0, 0.0]


def test_predict_known_values() -> None:
    """Check predictions with manually configured weights."""
    model = Perceptron(n_features=2)
    model.w = [1.0, 1.0]
    model.bias = -2.0

    assert model.predict([3.0, 3.0]) == 1
    assert model.predict([0.0, 0.0]) == 0


def test_train_separable_data() -> None:
    """Check that training classifies a simple separable dataset."""
    X = [
        [1.0, 1.0],
        [2.0, 1.0],
        [4.0, 4.0],
        [5.0, 4.0],
    ]
    y = [0, 0, 1, 1]

    model = Perceptron(n_features=2, epochs=50)
    model.train(X, y)

    assert model.score(X, y) == pytest.approx(1.0)


def test_score_accuracy() -> None:
    """Check score calculation."""
    model = Perceptron(n_features=2)
    model.w = [1.0, 1.0]
    model.bias = -2.0

    X = [[3.0, 3.0], [0.0, 0.0], [1.0, 1.0]]
    y = [1, 0, 1]

    assert model.score(X, y) == 1


def test_predict_rejects_wrong_feature_count() -> None:
    """Check that input dimensionality is validated."""
    model = Perceptron(n_features=2)

    with pytest.raises(ValueError):
        model.predict([1.0])
