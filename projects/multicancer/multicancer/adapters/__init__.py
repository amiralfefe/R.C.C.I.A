"""Specialized adapters available to the MultiCancer hub."""

from .base import BaseAdapter
from .leukemia_adapter import LeukemiaAdapter

__all__ = ["BaseAdapter", "LeukemiaAdapter"]
