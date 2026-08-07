"""Phase-field sintering model."""

from .model import ModelConfig
from .parity_runner import SinteringModel

__all__ = ["ModelConfig", "SinteringModel"]
