from __future__ import annotations

import inspect

from multicancer.adapters.base import BaseAdapter
from multicancer.adapters.breast_adapter import BreastAdapter
from multicancer.adapters.leukemia_adapter import LeukemiaAdapter
from multicancer.adapters.lung_colon_adapter import LungColonAdapter
from multicancer.adapters.metastasis_adapter import MetastasisAdapter


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


def test_breast_adapter_implements_common_contract() -> None:
    assert issubclass(BreastAdapter, BaseAdapter)
    assert not inspect.isabstract(BreastAdapter)

    required_members = {
        "metadata",
        "checkpoint_status",
        "load",
        "predict",
        "explain",
        "unload",
        "is_loaded",
    }
    assert required_members.issubset(dir(BreastAdapter))


def test_metastasis_adapter_implements_common_contract() -> None:
    assert issubclass(MetastasisAdapter, BaseAdapter)
    assert not inspect.isabstract(MetastasisAdapter)

    required_members = {
        "metadata",
        "checkpoint_status",
        "load",
        "predict",
        "explain",
        "unload",
        "is_loaded",
    }
    assert required_members.issubset(dir(MetastasisAdapter))


def test_lung_colon_adapter_implements_common_contract_and_modes() -> None:
    assert issubclass(LungColonAdapter, BaseAdapter)
    assert not inspect.isabstract(LungColonAdapter)
    required_members = {
        "metadata",
        "checkpoint_status",
        "load",
        "predict",
        "explain",
        "unload",
        "is_loaded",
        "available_modes",
        "current_mode",
        "set_mode",
    }
    assert required_members.issubset(dir(LungColonAdapter))

    adapter = LungColonAdapter()
    assert adapter.metadata().project_id == "lung_colon"
    assert adapter.mode_metadata("multiclass").model_name == "efficientnet_b0"
    assert adapter.mode_metadata("binary").model_name == "resnet18"


def test_metadata_contains_adapter_contract_fields() -> None:
    metadata = LeukemiaAdapter().metadata()

    assert metadata.project_id == "leukemia"
    assert metadata.classes == ("normal", "leukemia_blast")
    assert metadata.model_name == "resnet18"
    assert metadata.image_size == 224
    assert metadata.dataset
    assert metadata.supports_gradcam
    assert metadata.primary_metrics
    assert metadata.limitations
    assert metadata.disclaimer


def test_breast_metadata_preserves_patient_aware_context() -> None:
    metadata = BreastAdapter().metadata()

    assert metadata.project_id == "breast"
    assert metadata.classes == ("benign", "malignant")
    assert metadata.model_name == "efficientnet_b0"
    assert metadata.image_size == 224
    assert metadata.dataset == "BreakHis"
    assert metadata.supports_gradcam
    assert metadata.primary_metrics
    assert any("patient" in limitation.lower() for limitation in metadata.limitations)
    assert metadata.disclaimer


def test_metastasis_metadata_preserves_threshold_context() -> None:
    metadata = MetastasisAdapter().metadata()

    assert metadata.project_id == "metastasis"
    assert metadata.classes == ("non_metastatic", "metastatic")
    assert metadata.model_name == "efficientnet_b0"
    assert metadata.image_size == 96
    assert "PCam" in metadata.dataset
    assert metadata.supports_gradcam
    assert metadata.supports_threshold_exploration
    assert metadata.disclaimer
