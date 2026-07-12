"""Pure threshold-decision helpers kept separate from model inference."""

from __future__ import annotations

import math

from .schemas import PredictionResult, ThresholdDecision


THRESHOLD_EDUCATIONAL_WARNING = (
    "Seuil exploratoire uniquement : aucune recommandation medicale. Cette regle ne "
    "modifie ni les probabilites ni l'argmax original du modele."
)


def apply_binary_threshold(
    prediction: PredictionResult,
    positive_class: str,
    threshold: float,
    default_threshold: float = 0.50,
) -> ThresholdDecision:
    """Apply a binary rule to an existing prediction without mutating it."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Le seuil doit etre compris entre 0 et 1.")
    if not 0.0 <= default_threshold <= 1.0:
        raise ValueError("Le seuil par defaut doit etre compris entre 0 et 1.")
    if positive_class not in prediction.class_probabilities:
        raise ValueError(f"Classe positive absente des probabilites : {positive_class}.")

    negative_classes = [
        class_name
        for class_name in prediction.class_probabilities
        if class_name != positive_class
    ]
    if len(negative_classes) != 1:
        raise ValueError("La decision par seuil requiert exactement deux classes.")

    positive_probability = float(prediction.class_probabilities[positive_class])
    negative_class = negative_classes[0]
    thresholded_class = (
        positive_class if positive_probability >= threshold else negative_class
    )
    return ThresholdDecision(
        positive_class=positive_class,
        negative_class=negative_class,
        positive_probability=positive_probability,
        threshold=float(threshold),
        thresholded_class=thresholded_class,
        is_threshold_override=not math.isclose(
            threshold,
            default_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        default_threshold=float(default_threshold),
        educational_warning=THRESHOLD_EDUCATIONAL_WARNING,
    )
