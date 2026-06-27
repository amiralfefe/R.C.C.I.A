"""Evaluation metrics and report plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def build_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    frame = pd.DataFrame(matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(7, 6))
    sns.heatmap(frame, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Prediction")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
