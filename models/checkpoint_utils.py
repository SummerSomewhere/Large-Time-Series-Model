"""
Load Timer/TrmEncoder backbone weights from disk.

Handles: torch.save(state_dict) flat files, nested {"state_dict": ...}, DDP "module.*",
full-Model checkpoints with "backbone.*", Lightning-style "model.*".
"""
from __future__ import annotations

import glob
import os
import re
import warnings

import torch


class CheckpointZipCorruptError(RuntimeError):
    """Raised when torch.load fails with a zip/PyTorchStreamReader style error."""

    def __init__(self, path: str, size_b: int, original: BaseException):
        self.path = path
        self.size_b = size_b
        self.original = original
        super().__init__(str(original))


def _normalize_param_key_for_backbone(key: str) -> str:
    """
    Strip common prefixes so keys match TimerBackbone / TrmEncoderBackbone state_dict.

    Timer.Model sets enc_embedding = backbone.patch_embedding; full-model checkpoints then
    contain enc_embedding.* while TimerBackbone only registers patch_embedding.*.
    """
    k = key
    while True:
        old = k
        if k.startswith("module."):
            k = k[7:]
        elif k.startswith("backbone."):
            k = k[9:]
        elif k.startswith("model."):
            k = k[6:]
        if k == old:
            break
    if k.startswith("enc_embedding."):
        k = "patch_embedding." + k[len("enc_embedding.") :]
    return k


def _is_zip_style_load_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "zip" in msg or "central directory" in msg or "pytorchstream" in msg


def _iter_corrupt_fallback_paths(primary: str) -> list[str]:
    """
    Other checkpoint files in the same directory to try if primary is zip-truncated.
    Order: paired name (e.g. .ckpt <-> .pth), then checkpoint.pth / checkpoint.ckpt,
    then checkpoint_<epoch>.{pth,ckpt} (newest epoch first, .pth before .ckpt).
    """
    primary_a = os.path.abspath(os.path.expanduser(primary))
    directory = os.path.dirname(primary_a)
    seen: set[str] = {primary_a}
    out: list[str] = []

    def add(p: str) -> None:
        p = os.path.abspath(p)
        if p in seen or not os.path.isfile(p):
            return
        seen.add(p)
        out.append(p)

    bn = os.path.basename(primary_a)
    if bn == "checkpoint.pth":
        add(os.path.join(directory, "checkpoint.ckpt"))
    elif bn == "checkpoint.ckpt":
        add(os.path.join(directory, "checkpoint.pth"))
    else:
        m = re.match(r"checkpoint_(\d+)\.pth$", bn)
        if m:
            add(os.path.join(directory, f"checkpoint_{m.group(1)}.ckpt"))
        else:
            m = re.match(r"checkpoint_(\d+)\.ckpt$", bn)
            if m:
                add(os.path.join(directory, f"checkpoint_{m.group(1)}.pth"))

    add(os.path.join(directory, "checkpoint.pth"))
    add(os.path.join(directory, "checkpoint.ckpt"))

    epoch_paths: list[tuple[int, str]] = []
    for pattern in ("checkpoint_*.pth", "checkpoint_*.ckpt"):
        for p in glob.glob(os.path.join(directory, pattern)):
            m = re.match(r"checkpoint_(\d+)\.(pth|ckpt)$", os.path.basename(p))
            if m:
                epoch_paths.append((int(m.group(1)), p))
    epoch_paths.sort(key=lambda t: (-t[0], 0 if t[1].endswith(".pth") else 1))
    for _ep, p in epoch_paths:
        add(p)

    return out


def _load_raw_checkpoint_once(path: str):
    """Single-file torch.load; raises CheckpointZipCorruptError on zip truncation."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found or not a regular file: {path}\n"
            f"(Use absolute path or run from repo root; file must be .pth/.ckpt, not a directory.)"
        )
    size_b = os.path.getsize(path)
    if size_b < 64:
        raise RuntimeError(
            f"Checkpoint file is too small ({size_b} bytes), likely empty or not a real save: {path}"
        )
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except RuntimeError as e:
        if _is_zip_style_load_error(e):
            raise CheckpointZipCorruptError(path, size_b, e) from e
        raise


def load_raw_checkpoint(path: str):
    """
    Load file from CPU. If the requested file is zip-corrupt, tries other checkpoints
    in the same directory (paired .pth/.ckpt, then per-epoch files).
    """
    path = os.path.abspath(os.path.expanduser(path))
    try:
        return _load_raw_checkpoint_once(path)
    except CheckpointZipCorruptError as e:
        fallbacks = _iter_corrupt_fallback_paths(path)
        for alt in fallbacks:
            try:
                out = _load_raw_checkpoint_once(alt)
            except CheckpointZipCorruptError:
                continue
            warnings.warn(
                f"{os.path.basename(path)} is unreadable; loaded from {os.path.basename(alt)} instead.",
                UserWarning,
                stacklevel=2,
            )
            return out
        tried = ", ".join(os.path.basename(p) for p in fallbacks) or "(no other files in directory)"
        raise RuntimeError(
            f"Cannot read checkpoint (truncated/corrupt zip): {e.path}\n"
            f"Size on disk: {e.size_b} bytes.\n"
            f"Tried fallbacks in the same directory: {tried}\n"
            f"Fix: re-train with latest code (rank-0-only save + atomic write), or restore from backup; "
            f"avoid copying while training is still writing.\n"
            f"Original: {e.original}"
        ) from e


def extract_state_dict(raw, *, from_lightning_ckpt: bool) -> dict:
    """Get the parameter dict from torch.load result."""
    if from_lightning_ckpt:
        if isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
        elif isinstance(raw, dict):
            # Same-dir fallback may load flat checkpoint.pth while the requested path was .ckpt.
            sd = raw
        else:
            raise KeyError(
                ".ckpt path expected a dict with 'state_dict' or a flat state_dict; "
                f"got {type(raw)}"
            )
    else:
        if isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
        else:
            sd = raw
    if not isinstance(sd, dict):
        raise TypeError(f"Expected state_dict to be dict, got {type(sd)}")
    return sd


def load_backbone_state_dict(path: str, *, from_lightning_ckpt: bool) -> dict:
    """
    Load backbone-compatible state_dict from .pth (from_lightning_ckpt=False)
    or Lightning .ckpt (from_lightning_ckpt=True).
    """
    raw = load_raw_checkpoint(path)
    sd = extract_state_dict(raw, from_lightning_ckpt=from_lightning_ckpt)
    return {_normalize_param_key_for_backbone(k): v for k, v in sd.items()}
