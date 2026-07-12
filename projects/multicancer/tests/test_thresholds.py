from __future__ import annotations

from dataclasses import asdict

import pytest

from multicancer.schemas import PredictionResult
from multicancer.thresholds import apply_binary_threshold


def make_prediction(metastatic_probability: float = 0.47) -> PredictionResult:
    probabilities = {
        "non_metastatic": 1.0 - metastatic_probability,
        "metastatic": metastatic_probability,
    }
    predicted_class = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
    predicted_index = list(probabilities).index(predicted_class)
    return PredictionResult(
        project_id="metastasis",
        predicted_class=predicted_class,
        predicted_index=predicted_index,
        confidence=probabilities[predicted_class],
        class_probabilities=probabilities,
        model_name="efficientnet_b0",
        image_size=96,
    )


@pytest.mark.parametrize(
    ("threshold", "expected_class", "override"),
    [
        (0.30, "metastatic", True),
        (0.50, "non_metastatic", False),
        (0.70, "non_metastatic", True),
    ],
)
def test_threshold_decisions(
    threshold: float,
    expected_class: str,
    override: bool,
) -> None:
    decision = apply_binary_threshold(make_prediction(), "metastatic", threshold)

    assert decision.thresholded_class == expected_class
    assert decision.is_threshold_override is override
    assert decision.positive_probability == pytest.approx(0.47)


def test_probability_equal_to_threshold_is_positive() -> None:
    decision = apply_binary_threshold(make_prediction(0.50), "metastatic", 0.50)

    assert decision.thresholded_class == "metastatic"


def test_argmax_and_thresholded_class_can_differ() -> None:
    prediction = make_prediction(0.47)
    decision = apply_binary_threshold(prediction, "metastatic", 0.40)

    assert prediction.predicted_class == "non_metastatic"
    assert decision.thresholded_class == "metastatic"


def test_prediction_is_not_mutated() -> None:
    prediction = make_prediction(0.47)
    before = asdict(prediction)

    apply_binary_threshold(prediction, "metastatic", 0.30)

    assert asdict(prediction) == before


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="compris entre 0 et 1"):
        apply_binary_threshold(make_prediction(), "metastatic", threshold)


def test_missing_positive_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="Classe positive absente"):
        apply_binary_threshold(make_prediction(), "unknown", 0.50)
