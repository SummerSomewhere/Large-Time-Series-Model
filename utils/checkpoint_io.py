"""
Robust checkpoint loading for .pth files (CPU map_location, DDP key strip, common wrappers).
"""
from __future__ import annotations

import io
import os

import torch


def load_state_dict_from_pth(path: str) -> dict:
    """
    Load a PyTorch .pth file into a flat state_dict suitable for load_state_dict.

    - Resolves absolute path; checks file exists and non-empty.
    - Always loads tensors to CPU first (avoids GPU OOM / some load quirks).
    - Unwraps ``state_dict``, ``model``, or ``model_state_dict`` if present.
    - Strips ``module.`` prefix from DDP / DataParallel saves.
    - If direct torch.load fails, retries from an in-memory buffer (helps some FS clients).
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint path is not a file: {path}")
    size = os.path.getsize(path)
    if size < 32:
        raise ValueError(f"Checkpoint file too small to be valid ({size} bytes): {path}")

    def _load_from_file_obj(fobj):
        kw = {"map_location": "cpu"}
        try:
            return torch.load(fobj, **kw, weights_only=False)
        except TypeError:
            # Older PyTorch without weights_only
            return torch.load(fobj, **kw)

    try:
        with open(path, "rb") as f:
            obj = _load_from_file_obj(f)
    except Exception as e1:
        try:
            with open(path, "rb") as f:
                blob = f.read()
            if len(blob) != size:
                raise RuntimeError(f"Read {len(blob)} bytes but file size is {size}")
            obj = _load_from_file_obj(io.BytesIO(blob))
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load checkpoint {path} ({size} bytes). "
                f"Direct load: {e1!r}. Buffer load: {e2!r}."
            ) from e2

    if isinstance(obj, dict):
        if "state_dict" in obj:
            obj = obj["state_dict"]
        elif "model_state_dict" in obj:
            obj = obj["model_state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]

    if not isinstance(obj, dict):
        raise TypeError(f"Expected state_dict (dict), got {type(obj)} from {path}")

    out = {}
    for k, v in obj.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out
