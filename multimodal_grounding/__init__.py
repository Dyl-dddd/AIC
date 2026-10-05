"""Baseline components for AIC multimodal visual grounding."""

from .data import GroundingDataset, GroundingSample, load_annotations
from .model import TextGuidedBoxRegressor

__all__ = [
    "GroundingDataset",
    "GroundingSample",
    "TextGuidedBoxRegressor",
    "load_annotations",
]

__version__ = "0.1.0"
