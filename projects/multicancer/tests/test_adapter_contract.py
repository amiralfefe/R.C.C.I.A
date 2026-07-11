from __future__ import annotations

import inspect

from multicancer.adapters.base import BaseAdapter
from multicancer.adapters.leukemia_adapter import LeukemiaAdapter


def test_leukemia_adapter_implements_common_contract() -> None:
    assert issubclass(LeukemiaAdapter, BaseAdapter)
    assert not inspect.isabstract(LeukemiaAdapter)

    required_members = {
        "metadata",
        "checkpoint_status",
        "load",
        "predict",
        "explain",
        "unload",
        "is_loaded",
    }
    assert required_members.issubset(dir(LeukemiaAdapter))


def test_metadata_contains_adapter_contract_fields() -> None:
    metadata = LeukemiaAdapter().metadata()

    assert metadata.project_id == "leukemia"
    assert metadata.classes == ("normal", "leukemia_blast")
    assert metadata.model_name == "resnet18"
    assert metadata.image_size == "224x224"
    assert metadata.dataset
    assert metadata.supports_gradcam
    assert metadata.primary_metrics
    assert metadata.limitations
    assert metadata.disclaimer
