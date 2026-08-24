"""Phase-field sintering model.

The public high-level classes are imported lazily so low-level numerical and
geometry modules do not require optional HDF5/image-analysis dependencies at
import time.
"""

__all__ = ["ModelConfig", "SinteringModel"]


def __getattr__(name):
    if name == "ModelConfig":
        from .model import ModelConfig
        return ModelConfig
    if name == "SinteringModel":
        from .parity_runner import SinteringModel
        return SinteringModel
    raise AttributeError(name)
