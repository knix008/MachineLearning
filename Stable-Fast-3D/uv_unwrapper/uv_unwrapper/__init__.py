import torch  # noqa: F401

try:
    from . import _C  # noqa: F401
except ImportError:
    # Extension missing or failed to load; Unwrapper still imports but torch.ops will fail at runtime.
    pass

from .unwrap import Unwrapper

__all__ = ["Unwrapper"]
