"""Loading CUB-200-2011 metadata (labels, splits, boxes, parts, attributes)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PART_NAMES = [
    "back", "beak", "belly", "breast", "crown", "forehead", "left eye", "left leg",
    "left wing", "nape", "right eye", "right leg", "right wing", "tail", "throat",
]


def cub_root() -> Path:
    """CUB root, from $KGBR_CUB_ROOT or ./data/CUB_200_2011."""
    root = Path(os.environ.get("KGBR_CUB_ROOT", "data/CUB_200_2011"))
    if not (root / "images.txt").exists():
        raise FileNotFoundError(
            f"CUB-200-2011 not found at {root}. Download CUB_200_2011.tgz from "
            "https://www.vision.caltech.edu/datasets/cub_200_2011/ and set KGBR_CUB_ROOT."
        )
    return root


@dataclass
class CUB:
    root: Path
    images: pd.DataFrame          # image_id -> path, class_id, is_train, bbox
    classes: pd.DataFrame         # class_id -> name
    parts: pd.DataFrame           # (image_id, part_id) -> x, y, visible
    attr_names: pd.DataFrame      # attr_id -> name
    attr_labels: pd.DataFrame     # (image_id, attr_id) -> present, certainty

    def image_path(self, image_id: int) -> Path:
        return self.root / "images" / self.images.loc[image_id, "path"]

    def part_xyv(self, image_id: int) -> dict[str, tuple[float, float, int]]:
        sub = self.parts.loc[image_id]
        return {PART_NAMES[pid - 1]: (r.x, r.y, int(r.visible)) for pid, r in sub.iterrows()}

    def attr_matrix(self, image_ids, min_certainty: int = 1) -> np.ndarray:
        """Binary (n_images, 312) matrix; entries with certainty < min_certainty set to NaN."""
        al = self.attr_labels.loc[list(image_ids)]
        present = al["present"].to_numpy().reshape(len(image_ids), 312).astype(float)
        cert = al["certainty"].to_numpy().reshape(len(image_ids), 312)
        present[cert < min_certainty] = np.nan
        return present


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="python")


@lru_cache(maxsize=1)
def load_cub(root: str | None = None) -> CUB:
    r = Path(root) if root else cub_root()
    images = _read(r / "images.txt", ["image_id", "path"])
    labels = _read(r / "image_class_labels.txt", ["image_id", "class_id"])
    split = _read(r / "train_test_split.txt", ["image_id", "is_train"])
    bbox = _read(r / "bounding_boxes.txt", ["image_id", "bx", "by", "bw", "bh"])
    df = images.merge(labels, on="image_id").merge(split, on="image_id").merge(bbox, on="image_id")
    df = df.set_index("image_id")

    classes = _read(r / "classes.txt", ["class_id", "folder"]).set_index("class_id")
    classes["name"] = classes["folder"].str.split(".", n=1).str[1].str.replace("_", " ")

    parts = _read(r / "parts" / "part_locs.txt", ["image_id", "part_id", "x", "y", "visible"])
    parts = parts.set_index(["image_id", "part_id"]).sort_index()

    attr_file = r / "attributes.txt"
    if not attr_file.exists():
        attr_file = r.parent / "attributes.txt"
    attr_names = pd.read_csv(attr_file, sep=" ", header=None, names=["attr_id", "name"]).set_index("attr_id")

    # image_attribute_labels.txt has a few malformed lines with extra columns; keep first 4 fields.
    rows = []
    with open(r / "attributes" / "image_attribute_labels.txt") as f:
        for line in f:
            t = line.split()
            rows.append((int(t[0]), int(t[1]), int(t[2]), int(t[3])))
    attr_labels = pd.DataFrame(rows, columns=["image_id", "attr_id", "present", "certainty"])
    attr_labels = attr_labels.set_index(["image_id", "attr_id"]).sort_index()

    return CUB(r, df, classes, parts, attr_names, attr_labels)
