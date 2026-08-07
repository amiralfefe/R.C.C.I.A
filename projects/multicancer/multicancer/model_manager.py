"""Session-level manager enforcing one active specialized adapter at a time."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .adapters.base import BaseAdapter
from .adapters.breast_adapter import BreastAdapter
from .adapters.leukemia_adapter import LeukemiaAdapter
from .adapters.lung_colon_adapter import LungColonAdapter
from .adapters.metastasis_adapter import MetastasisAdapter
from .exceptions import AdapterError


AdapterFactory = Callable[[], BaseAdapter]


class ModelManager:
    """Own one adapter and unload it before any project switch."""

    def __init__(self, factories: Mapping[str, AdapterFactory] | None = None) -> None:
        default_factories: Mapping[str, AdapterFactory] = {
            "leukemia": LeukemiaAdapter,
            "breast": BreastAdapter,
            "metastasis": MetastasisAdapter,
            "lung_colon": LungColonAdapter,
        }
        self._factories = dict(default_factories if factories is None else factories)
        self._current_adapter: BaseAdapter | None = None
        self._active_project_id: str | None = None

    @property
    def current_adapter(self) -> BaseAdapter | None:
        return self._current_adapter

    @property
    def active_project_id(self) -> str | None:
        return self._active_project_id

    def activate(self, project_id: str) -> BaseAdapter:
        if project_id == self._active_project_id and self._current_adapter is not None:
            return self._current_adapter

        self.unload_current()
        factory = self._factories.get(project_id)
        if factory is None:
            raise AdapterError(
                f"Le projet '{project_id}' n'a pas encore d'adaptateur MultiCancer integre."
            )

        adapter = factory()
        self._current_adapter = adapter
        self._active_project_id = project_id
        return adapter

    def set_active_mode(self, mode_id: str) -> None:
        if self._active_project_id != "lung_colon" or not isinstance(
            self._current_adapter,
            LungColonAdapter,
        ):
            raise AdapterError(
                "La selection de mode est disponible uniquement pour LungColon actif."
            )
        self._current_adapter.set_mode(mode_id)

    def unload_current(self) -> None:
        if self._current_adapter is not None:
            self._current_adapter.unload()
        self._current_adapter = None
        self._active_project_id = None
