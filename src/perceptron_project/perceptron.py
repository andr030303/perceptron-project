"""Implementation and command-line interface for a binary perceptron."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from perceptron_project.data import generate_synthetic_data, load_iris_binary
from perceptron_project.visualization import (
    plot_decision_boundary,
    plot_training_errors,
)


class Perceptron:
    """A simple binary classifier trained with the perceptron learning rule."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Initialize model parameters and training settings."""
        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: list[float] = [0.0] * n_features
        self.bias: float = 0.0
        self.errors_per_epoch: list[int] = []

    @staticmethod
    def activate(z: float) -> int:
        """Return 1 when z >= 0, otherwise return 0."""
        return 1 if z >= 0 else 0

    def predict(self, x: Sequence[float]) -> int:
        """Predict the binary class of one feature vector."""
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}.")
        z = sum(weight * value for weight, value in zip(self.w, x, strict=False))
        z += self.bias
        return self.activate(z)

    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
    ) -> None:
        """Train weights using the perceptron update rule."""
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if not X:
            raise ValueError("Training data must not be empty.")
        if any(target not in (0, 1) for target in y):
            raise ValueError("Labels must be binary: 0 or 1.")

        self.errors_per_epoch = []

        for _ in range(self.epochs):
            errors = 0

            for x, target in zip(X, y, strict=False):
                prediction = self.predict(x)
                error = target - prediction

                if error != 0:
                    errors += 1
                    self.w = [
                        weight + self.learning_rate * error * value
                        for weight, value in zip(self.w, x, strict=False)
                    ]
                    self.bias += self.learning_rate * error

            self.errors_per_epoch.append(errors)

    def score(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
    ) -> float:
        """Return the fraction of correct predictions."""
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if not y:
            raise ValueError("Evaluation data must not be empty.")

        correct = sum(
            self.predict(sample) == target for sample, target in zip(X, y, strict=False)
        )
        return correct / len(y)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a binary perceptron and visualize its decision boundary."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Total number of synthetic samples (default: 200).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "iris"),
        default="synthetic",
        help="Dataset to use (default: synthetic).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not display visualization windows.",
    )
    return parser


def main() -> None:
    """Run data preparation, training, evaluation, and visualization."""
    parser = build_parser()
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least 2.")
    if args.lr <= 0:
        parser.error("--lr must be greater than 0.")
    if args.epochs <= 0:
        parser.error("--epochs must be greater than 0.")

    if args.dataset == "synthetic":
        X, y = generate_synthetic_data(args.samples)
    else:
        X, y = load_iris_binary()

    model = Perceptron(
        n_features=2,
        learning_rate=args.lr,
        epochs=args.epochs,
    )
    model.train(X, y)
    accuracy = model.score(X, y)

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(X)}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Training accuracy: {accuracy:.3f}")

    if not args.no_plots:
        plot_decision_boundary(X, y, model)
        plot_training_errors(model)


if __name__ == "__main__":
    main()
