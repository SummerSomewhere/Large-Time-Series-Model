#!/usr/bin/env python3
"""Print GeoHPE learnable a (patch_curv_scale_a) and b (pe_curv_scale_b) from checkpoint.pth or .ckpt."""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=str, help="Path to checkpoint.pth or Lightning .ckpt")
    args = ap.parse_args()
    obj = torch.load(args.checkpoint, map_location="cpu")
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    if not isinstance(sd, dict):
        print("Unexpected checkpoint format.", file=sys.stderr)
        sys.exit(1)
    keys_of_interest = ("patch_curv_scale_a", "pe_curv_scale_b")
    found = False
    for k, v in sorted(sd.items()):
        if any(s in k for s in keys_of_interest):
            print(f"{k}: {v.detach().cpu().flatten().tolist()}")
            found = True
    if not found:
        print(
            "No patch_curv_scale_a / pe_curv_scale_b in checkpoint.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
