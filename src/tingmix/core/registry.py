"""Model name -> factory. Heavy imports happen inside the factories."""

from __future__ import annotations

import os


def _large(path: str | None = None, **kw):
    from ..models.large.model import LargeModel

    if path and path.endswith(".pt"):
        return LargeModel.from_checkpoint(path, **kw)
    return LargeModel.from_pretrained(*([path] if path else []), **kw)


def _small(path: str | None = None, **kw):
    from ..models.small import SmallModel

    keep = ("device", "threads", "lm", "lm_alpha", "lm_beta", "beam")
    return SmallModel.from_pretrained(*([path] if path else []), **{k: v for k, v in kw.items() if k in keep})


MODELS = {"large": _large, "embed2asr-v1": _large, "small": _small}


def load_model(name: str = "large", **kw):
    if os.path.isfile(os.path.join(name, "small.json")) or "tingmix-small" in name.lower():
        return _small(name, **kw)
    if name.endswith(".pt") or os.path.isdir(name) or "/" in name:  # training export, release directory or HF repo id
        return _large(name, **kw)
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**kw)
