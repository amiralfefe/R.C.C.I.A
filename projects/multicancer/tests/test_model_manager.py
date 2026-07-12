from __future__ import annotations

import pytest
from PIL import Image

from multicancer.adapters.base import BaseAdapter
from multicancer.adapters.breast_adapter import BreastAdapter
from multicancer.adapters.leukemia_adapter import LeukemiaAdapter
from multicancer.adapters.metastasis_adapter import MetastasisAdapter
from multicancer.exceptions import AdapterError
from multicancer.model_manager import ModelManager
from multicancer.registry import get_project
from multicancer.schemas import CheckpointStatus, ExplanationResult, PredictionResult


class FakeAdapter(BaseAdapter):
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.loaded = False
        self.unload_calls = 0

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def metadata(self):
        return get_project("leukemia")

    def checkpoint_status(self) -> CheckpointStatus:
        return CheckpointStatus("available", None, "test checkpoint")

    def load(self) -> None:
        self.loaded = True

    def predict(self, image: Image.Image) -> PredictionResult:
        raise NotImplementedError

    def explain(
        self,
        image: Image.Image,
        class_index: int | None = None,
    ) -> ExplanationResult:
        raise NotImplementedError

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False


def test_only_one_adapter_is_active_and_previous_is_unloaded() -> None:
    created: dict[str, FakeAdapter] = {}

    def factory(project_id: str):
        def create() -> FakeAdapter:
            adapter = FakeAdapter(project_id)
            created[project_id] = adapter
            return adapter

        return create

    manager = ModelManager(factories={"first": factory("first"), "second": factory("second")})
    first = manager.activate("first")
    first.load()

    second = manager.activate("second")
    second.load()

    assert manager.current_adapter is second
    assert manager.active_project_id == "second"
    assert created["first"].unload_calls == 1
    assert not created["first"].is_loaded

    first_again = manager.activate("first")

    assert manager.current_adapter is first_again
    assert manager.active_project_id == "first"
    assert created["second"].unload_calls == 1
    assert not created["second"].is_loaded


def test_reactivating_same_project_reuses_adapter() -> None:
    adapter = FakeAdapter("leukemia")
    manager = ModelManager(factories={"leukemia": lambda: adapter})

    first = manager.activate("leukemia")
    second = manager.activate("leukemia")

    assert first is second
    assert adapter.unload_calls == 0


def test_unknown_project_unloads_current_before_error() -> None:
    adapter = FakeAdapter("leukemia")
    manager = ModelManager(factories={"leukemia": lambda: adapter})
    manager.activate("leukemia")
    adapter.load()

    try:
        manager.activate("breast")
    except AdapterError as exc:
        assert "pas encore d'adaptateur" in str(exc)
    else:
        raise AssertionError("AdapterError was not raised")

    assert manager.current_adapter is None
    assert manager.active_project_id is None
    assert adapter.unload_calls == 1


def test_unload_current_is_idempotent() -> None:
    adapter = FakeAdapter("leukemia")
    manager = ModelManager(factories={"leukemia": lambda: adapter})
    manager.activate("leukemia")

    manager.unload_current()
    manager.unload_current()

    assert adapter.unload_calls == 1
    assert manager.current_adapter is None


def test_default_manager_activates_all_integrated_adapters() -> None:
    manager = ModelManager()

    leukemia = manager.activate("leukemia")
    breast = manager.activate("breast")
    metastasis = manager.activate("metastasis")

    assert isinstance(leukemia, LeukemiaAdapter)
    assert isinstance(breast, BreastAdapter)
    assert isinstance(metastasis, MetastasisAdapter)
    assert manager.current_adapter is metastasis
    assert manager.active_project_id == "metastasis"


@pytest.mark.parametrize(
    ("first_project", "second_project"),
    [
        ("breast", "metastasis"),
        ("leukemia", "metastasis"),
        ("metastasis", "breast"),
    ],
)
def test_integrated_project_switch_unloads_previous_model(
    first_project: str,
    second_project: str,
) -> None:
    adapters = {
        project_id: FakeAdapter(project_id)
        for project_id in ("leukemia", "breast", "metastasis")
    }
    manager = ModelManager(
        factories={
            project_id: (lambda project_id=project_id: adapters[project_id])
            for project_id in adapters
        }
    )

    first = manager.activate(first_project)
    first.load()
    second = manager.activate(second_project)

    assert adapters[first_project].unload_calls == 1
    assert not adapters[first_project].is_loaded
    assert sum(adapter.is_loaded for adapter in adapters.values()) == 0

    second.load()

    assert manager.active_project_id == second_project
    assert sum(adapter.is_loaded for adapter in adapters.values()) == 1
