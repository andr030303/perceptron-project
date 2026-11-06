"""Plotting utilities for the perceptron project."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from perceptron_project.perceptron import Perceptron


def plot_decision_boundary(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    model: Perceptron,
) -> None:
    """Plot two classes and the model's learned decision boundary."""
    class_0 = [sample for sample, label in zip(X, y, strict=False) if label == 0]
    class_1 = [sample for sample, label in zip(X, y, strict=False) if label == 1]

    if class_0:
        plt.scatter(
            [sample[0] for sample in class_0],
            [sample[1] for sample in class_0],
            label="Class 0",
            alpha=0.8,
        )
    if class_1:
        plt.scatter(
            [sample[0] for sample in class_1],
            [sample[1] for sample in class_1],
            label="Class 1",
            alpha=0.8,
        )

    x_min = min(sample[0] for sample in X) - 0.5
    x_max = max(sample[0] for sample in X) + 0.5
    y_min = min(sample[1] for sample in X) - 0.5
    y_max = max(sample[1] for sample in X) + 0.5

    w1, w2 = model.w
    if abs(w2) > 1e-12:
        x_values = [x_min, x_max]
        y_values = [-(w1 * x_value + model.bias) / w2 for x_value in x_values]
        plt.plot(x_values, y_values, color="black", label="Decision boundary")
    elif abs(w1) > 1e-12:
        x_boundary = -model.bias / w1
        plt.axvline(x_boundary, color="black", label="Decision boundary")
    else:
        raise ValueError("Model has no valid decision boundary.")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron decision boundary")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_errors(model: Perceptron) -> None:
    """Plot the number of classification errors for each epoch."""
    epochs = range(1, len(model.errors_per_epoch) + 1)

    plt.plot(epochs, model.errors_per_epoch, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Number of errors")
    plt.title("Training errors per epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
