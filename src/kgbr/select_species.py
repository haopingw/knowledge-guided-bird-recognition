"""Choose the 20 similar species and build dev / val / test splits.

Selection rule (uses official TRAINING images only, decided before any model run):
  1. For each of the 200 classes, average the image-level attribute vectors
     (only annotations with certainty >= 3, i.e. "probably"/"definitely").
  2. Start from the three Figure-1 seed species.
  3. Greedily add the class with the highest mean cosine similarity to the
     classes already selected, until 20 classes are chosen.
Splits: official train images of each class -> 80% dev / 20% val (seed 0);
official test images -> test (final evaluation only).

Usage:  python -m kgbr.select_species
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cub import load_cub

SEED_CLASSES = [116, 132, 129]  # Chipping, White-crowned, Song Sparrow (Figure 1)
N_SPECIES = 20
OUT = Path("data/splits")


def class_attribute_means(cub) -> np.ndarray:
    tr = cub.images[cub.images.is_train == 1]
    means = np.zeros((200, 312))
    for c in range(1, 201):
        ids = tr.index[tr.class_id == c]
        m = cub.attr_matrix(ids, min_certainty=3)
        means[c - 1] = np.nan_to_num(np.nanmean(m, axis=0), nan=0.0)
    return means


def greedy_select(means: np.ndarray, seeds: list[int], k: int) -> list[tuple[int, float]]:
    unit = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-9)
    sim = unit @ unit.T
    chosen = list(seeds)
    log = [(c, float("nan")) for c in seeds]
    while len(chosen) < k:
        idx = np.array(chosen) - 1
        score = sim[:, idx].mean(axis=1)
        score[idx] = -np.inf
        best = int(np.argmax(score)) + 1
        chosen.append(best)
        log.append((best, float(score[best - 1])))
    return log


def main() -> None:
    cub = load_cub()
    with np.errstate(invalid="ignore"):
        means = class_attribute_means(cub)
    log = greedy_select(means, SEED_CLASSES, N_SPECIES)
    OUT.mkdir(parents=True, exist_ok=True)

    species = []
    for i, (cid, score) in enumerate(sorted(log, key=lambda t: t[0])):
        species.append({"sid": f"S{i + 1:02d}", "class_id": cid,
                        "name": cub.classes.loc[cid, "name"]})
    sel_order = [{"class_id": c, "name": cub.classes.loc[c, "name"], "mean_cos_to_set": s}
                 for c, s in log]
    (OUT / "species.json").write_text(json.dumps(
        {"rule": __doc__.split("Usage")[0].strip(), "selection_order": sel_order,
         "species": species}, indent=2))

    rng = np.random.default_rng(0)
    rows = []
    sid_of = {s["class_id"]: s["sid"] for s in species}
    for s in species:
        imgs = cub.images[cub.images.class_id == s["class_id"]]
        tr = imgs.index[imgs.is_train == 1].to_numpy().copy()
        rng.shuffle(tr)
        n_val = int(round(0.2 * len(tr)))
        for iid in tr[:n_val]:
            rows.append((iid, "val"))
        for iid in tr[n_val:]:
            rows.append((iid, "dev"))
        for iid in imgs.index[imgs.is_train == 0]:
            rows.append((iid, "test"))
    df = pd.DataFrame(rows, columns=["image_id", "split"])
    df["class_id"] = cub.images.loc[df.image_id, "class_id"].to_numpy()
    df["sid"] = df.class_id.map(sid_of)
    df["path"] = cub.images.loc[df.image_id, "path"].to_numpy()
    df = df.sort_values(["split", "class_id", "image_id"])
    df.to_csv(OUT / "manifest.csv", index=False)

    print("Selected species:")
    for c, s in log:
        print(f"  {c:3d} {cub.classes.loc[c, 'name']:<32s} {s:.3f}")
    print(df.groupby("split").size())


if __name__ == "__main__":
    main()
