"""Supervised reference: linear probe on frozen Qwen3-VL vision features.

Uses TRAINING LABELS (dev split), so it is reported separately and is not an
equal-footing comparison with the zero-shot prompting conditions.
  features : mean-pooled visual tokens from the frozen Qwen3-VL vision encoder
             (same pixel budget as the main experiment), L2-normalised
  classifier: multinomial logistic regression; C chosen on val, refit on dev, test once.

Usage: python -m kgbr.probe
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression

from .cub import load_cub
from .model import QwenVL, resize_to_budget

OUT = Path("results/probe")


@torch.inference_mode()
def extract(model: QwenVL, paths: list[Path]) -> np.ndarray:
    feats = []
    proc = model.processor.image_processor
    for p in paths:
        img = resize_to_budget(Image.open(p).convert("RGB"), model.max_pixels)
        x = proc(images=[img], return_tensors="pt")
        pv = x["pixel_values"].to(model.model.device, dtype=model.model.dtype)
        grid = x["image_grid_thw"].to(model.model.device)
        emb = model.model.model.get_image_features(pixel_values=pv, image_grid_thw=grid)
        emb = emb[0] if isinstance(emb, (tuple, list)) else emb
        emb = emb[0] if isinstance(emb, (tuple, list)) else emb
        if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
            emb = emb.pooler_output
            emb = emb[0] if isinstance(emb, (tuple, list)) else emb
        f = emb.float().mean(0)
        feats.append((f / f.norm()).cpu().numpy())
    return np.stack(feats)


def mpca(y, p) -> float:
    df = pd.DataFrame({"y": y, "c": (np.asarray(y) == np.asarray(p)).astype(int)})
    return float(df.groupby("y").c.mean().mean())


def main() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    cub = load_cub()
    man = pd.read_csv("data/splits/manifest.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "features.npz"
    if cache.exists():
        z = np.load(cache)
        X, ids = z["X"], z["ids"]
    else:
        model = QwenVL(cfg["model_id"], cfg["dtype"], cfg["max_pixels"], 1)
        ids = man.image_id.to_numpy()
        X = extract(model, [cub.image_path(int(i)) for i in ids])
        np.savez(cache, X=X, ids=ids)
    idx = {int(i): k for k, i in enumerate(ids)}
    get = lambda split: (X[[idx[i] for i in man[man.split == split].image_id]],  # noqa: E731
                         man[man.split == split].sid.to_numpy())
    Xd, yd = get("dev"); Xv, yv = get("val"); Xt, yt = get("test")
    grid = [0.1, 0.3, 1, 3, 10, 30, 100]
    val = {}
    for C in grid:
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xd, yd)
        val[C] = mpca(yv, clf.predict(Xv))
    best = max(grid, key=lambda c: (val[c], -c))
    clf = LogisticRegression(C=best, max_iter=5000).fit(Xd, yd)
    res = {"val_acc_by_C": val, "best_C": best, "test_mean_per_class_acc": mpca(yt, clf.predict(Xt)),
           "n_dev": len(yd), "n_val": len(yv), "n_test": len(yt), "feature_dim": int(X.shape[1])}
    (OUT / "probe_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
