"""Specialized adapters available to the MultiCancer hub."""

from .base import BaseAdapter
from .breast_adapter import BreastAdapter
from .leukemia_adapter import LeukemiaAdapter

__all__ = ["BaseAdapter", "BreastAdapter", "LeukemiaAdapter"]
