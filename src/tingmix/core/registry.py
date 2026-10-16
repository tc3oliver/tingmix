"""Model name -> factory. Heavy imports happen inside the factories."""

from __future__ import annotations

import os


def _large(path: str | None = None, **kw):
    from ..models.large.model import LargeModel

    if path and path.endswith(".pt"):
        return LargeModel.from_checkpoint(path, **kw)
    return LargeModel.from_pretrained(*([path] if path else []), **kw)


def _small(**kw):
    from ..models.small import SmallModel

    return SmallModel(**kw)


MODELS = {"large": _large, "embed2asr-v1": _large, "small": _small}


def load_model(name: str = "large", **kw):
    if name.endswith(".pt") or os.path.isdir(name):  # training export or embed2asr release directory
        return _large(name, **kw)
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**kw)
