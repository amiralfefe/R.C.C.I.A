"""Specialized adapters available to the MultiCancer hub."""

from .base import BaseAdapter
from .breast_adapter import BreastAdapter
from .leukemia_adapter import LeukemiaAdapter
from .lung_colon_adapter import LungColonAdapter
from .metastasis_adapter import MetastasisAdapter

__all__ = [
    "BaseAdapter",
    "BreastAdapter",
    "LeukemiaAdapter",
    "LungColonAdapter",
    "MetastasisAdapter",
]
