"""Occlusion experiment: key-part mask vs. equal-area control mask on the bird.

Subset rule (fixed before any test-set prediction, uses annotations only):
  per species, the test images whose key part (species_notes.yaml `key_part`) is
  annotated visible; take 10 per species with a fixed seed -> 200 images.
Mask: gray (128) square of side `0.30 * sqrt(bbox_w * bbox_h)`.
  key  : centred on the key part ("beak"/"crown"/... ; eye-level parts use the
         CUB landmark of the same name).
  ctrl : centred on another visible bird part from CONTROL_PARTS, chosen as the
         NEAREST part to the key part whose square does not overlap the key square
         (keeps the control on the bird body and at a comparable location; if none
         qualifies the image is dropped before sampling).
Ground-truth part locations are used ONLY here, offline; never shown to the model.

Usage:  python -m kgbr.occlusion   (writes data/splits/occlusion_subset.csv)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .cub import load_cub
from .knowledge import load_notes

CONTROL_PARTS = ["back", "belly", "left wing", "right wing", "tail", "left leg", "right leg",
                 "nape", "breast", "crown", "throat"]
SIDE_FRAC = 0.30
PER_CLASS = 10
OUT = Path("data/splits/occlusion_subset.csv")


def _square(cx, cy, s):
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def plan_masks(cub, image_id: int, key_part: str):
    parts = cub.part_xyv(image_id)
    kx, ky, kv = parts[key_part]
    if not kv:
        return None
    r = cub.images.loc[image_id]
    s = SIDE_FRAC * math.sqrt(r.bw * r.bh)
    kbox = _square(kx, ky, s)
    best = None
    for p in CONTROL_PARTS:
        if p == key_part:
            continue
        x, y, v = parts[p]
        if not v:
            continue
        cbox = _square(x, y, s)
        if _overlap(kbox, cbox):
            continue
        d = math.hypot(x - kx, y - ky)
        if best is None or d < best[0]:
            best = (d, p, cbox)
    if best is None:
        return None
    return {"key_part": key_part, "key_box": kbox, "ctrl_part": best[1], "ctrl_box": best[2], "side": s}


def apply_mask(img: Image.Image, box) -> Image.Image:
    img = img.convert("RGB").copy()
    ImageDraw.Draw(img).rectangle([round(v) for v in box], fill=(128, 128, 128))
    return img


def build_subset() -> pd.DataFrame:
    cub = load_cub()
    notes = load_notes()
    man = pd.read_csv("data/splits/manifest.csv")
    test = man[man.split == "test"]
    rng = np.random.default_rng(442)
    rows = []
    for sid, g in test.groupby("sid"):
        kp = notes[sid]["key_part"]
        ok = []
        for iid in g.image_id:
            m = plan_masks(cub, int(iid), kp)
            if m:
                ok.append((int(iid), m))
        pick = rng.choice(len(ok), size=min(PER_CLASS, len(ok)), replace=False)
        for i in sorted(pick):
            iid, m = ok[i]
            rows.append({"image_id": iid, "sid": sid, "key_part": m["key_part"], "ctrl_part": m["ctrl_part"],
                         **{f"key_{k}": v for k, v in zip("x0 y0 x1 y1".split(), m["key_box"])},
                         **{f"ctrl_{k}": v for k, v in zip("x0 y0 x1 y1".split(), m["ctrl_box"])}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"occlusion subset: {len(df)} images; per class:\n{df.groupby('sid').size().to_string()}")
    print(df.ctrl_part.value_counts().to_string())
    return df


def load_subset() -> pd.DataFrame:
    return pd.read_csv(OUT)


def masked_image(cub, row: dict, variant: str) -> Image.Image:
    img = Image.open(cub.image_path(int(row["image_id"])))
    if variant == "orig":
        return img.convert("RGB")
    p = "key" if variant == "mask_key" else "ctrl"
    return apply_mask(img, (row[f"{p}_x0"], row[f"{p}_y0"], row[f"{p}_x1"], row[f"{p}_y1"]))


if __name__ == "__main__":
    build_subset()
