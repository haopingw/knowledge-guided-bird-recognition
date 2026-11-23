"""Evaluation metrics (rules fixed before the test run).

Classification  mean per-class accuracy; invalid / unparsable answers count as wrong.
Attributes      evaluated on (image, attribute) pairs whose CUB annotation has
                certainty >= 3 ("probably"/"definitely"). Prediction "yes" = positive;
                "no", "unclear" and missing = negative. Macro F1 (over attributes) and
                micro F1. The "unclear" rate is reported separately.
Landmarks       for each of the 6 parts annotated visible: normalised error =
                ||pred - gt|| / bbox diagonal. Reported: coverage (fraction predicted),
                mean error over predicted parts, and PCK@0.1 (correct = predicted AND
                error <= 0.1; missing counts as a miss, so answering less cannot help).
Uncertainty     paired bootstrap, resampling images within each class (stratified),
                same resample for every condition being compared.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .cub import load_cub
from .knowledge import attribute_set
from .prompts import PARTS

PCK_T = 0.1


def load_raw(tag: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(f"results/raw/{tag}.jsonl").read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(rows)


def per_image_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Adds per-image metric columns (counts so that ratios can be re-aggregated)."""
    cub = load_cub()
    attrs = attribute_set()
    aids = [a["aid"] for a in attrs]
    cub_ids = [a["cub_id"] for a in attrs]
    out = []
    for r in df.itertuples():
        p = r.parsed
        iid = int(r.image_id)
        rec = {"correct": int(p["species"] == r.sid_true), "valid": int(p["species"] is not None),
               "json_ok": int(p["json_ok"]), "pred": p["species"]}
        # attributes
        al = cub.attr_labels.loc[iid]
        tp = np.zeros(len(aids)); fp = np.zeros(len(aids)); fn = np.zeros(len(aids)); n = np.zeros(len(aids))
        unclear = 0
        for k, (aid, cid) in enumerate(zip(aids, cub_ids)):
            pv = p["attributes"].get(aid)
            unclear += pv == "unclear"
            row = al.loc[cid]
            if row.certainty < 3:
                continue
            gt = int(row.present)
            pr = int(pv == "yes")
            n[k] = 1
            tp[k] += gt and pr
            fp[k] += (not gt) and pr
            fn[k] += gt and (not pr)
        rec.update(attr_tp=tp, attr_fp=fp, attr_fn=fn, attr_n=n, attr_unclear=unclear / len(aids))
        # landmarks
        img = cub.images.loc[iid]
        W, H = _image_size(cub, iid)
        diag = math.hypot(img.bw, img.bh)
        gtp = cub.part_xyv(iid)
        n_vis = n_pred = n_hit = 0
        errs = []
        for part in PARTS:
            gx, gy, gv = gtp[part]
            if not gv:
                continue
            n_vis += 1
            pt = p["parts"].get(part)
            if pt is None:
                continue
            n_pred += 1
            e = math.hypot(pt[0] / 1000 * W - gx, pt[1] / 1000 * H - gy) / diag
            errs.append(e)
            n_hit += e <= PCK_T
        rec.update(lm_vis=n_vis, lm_pred=n_pred, lm_hit=n_hit, lm_err_sum=float(sum(errs)))
        out.append(rec)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


_SIZE_CACHE: dict[int, tuple[int, int]] = {}


def _image_size(cub, iid):
    if iid not in _SIZE_CACHE:
        from PIL import Image
        with Image.open(cub.image_path(iid)) as im:
            _SIZE_CACHE[iid] = im.size
    return _SIZE_CACHE[iid]


# ------------------------------------------------------------------ aggregates
def mean_per_class_acc(s: pd.DataFrame) -> float:
    return float(s.groupby("sid_true")["correct"].mean().mean())


def attr_f1(s: pd.DataFrame) -> tuple[float, float]:
    tp = np.stack(s.attr_tp).sum(0); fp = np.stack(s.attr_fp).sum(0); fn = np.stack(s.attr_fn).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / (2 * tp + fp + fn)
    macro = float(np.nanmean(f1))
    micro = float(2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum()))
    return macro, micro


def landmark(s: pd.DataFrame) -> dict:
    vis, pred, hit = s.lm_vis.sum(), s.lm_pred.sum(), s.lm_hit.sum()
    return {"pck": hit / max(vis, 1), "coverage": pred / max(vis, 1),
            "mean_err": s.lm_err_sum.sum() / max(pred, 1)}


def summarize(s: pd.DataFrame) -> dict:
    macro, micro = attr_f1(s)
    lm = landmark(s)
    return {"n": len(s), "acc": mean_per_class_acc(s), "attr_macro_f1": macro, "attr_micro_f1": micro,
            "attr_unclear": s.attr_unclear.mean(), "pck@0.1": lm["pck"], "lm_coverage": lm["coverage"],
            "lm_mean_err": lm["mean_err"], "json_fail": 1 - s.json_ok.mean(),
            "invalid_species": 1 - s.valid.mean(), "time_s": s.time_s.mean(),
            "in_tokens": s.n_in_tokens.mean(), "out_tokens": s.n_out_tokens.mean(),
            "truncated": s.truncated.mean()}


METRICS = {
    "acc": mean_per_class_acc,
    "attr_macro_f1": lambda s: attr_f1(s)[0],
    "pck@0.1": lambda s: landmark(s)["pck"],
}


def stratified_indices(sids: np.ndarray, rng) -> np.ndarray:
    idx = []
    for c in np.unique(sids):
        members = np.flatnonzero(sids == c)
        idx.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(idx)


def paired_bootstrap(scored: pd.DataFrame, cond_a, cond_b, metric: str, key="condition",
                     B: int = 2000, seed: int = 0) -> dict:
    """Difference metric(b) - metric(a) on the same images; returns point, 95% CI, p(two-sided)."""
    fn = METRICS[metric]
    A = scored[scored[key] == cond_a].sort_values("image_id").reset_index(drop=True)
    Bd = scored[scored[key] == cond_b].sort_values("image_id").reset_index(drop=True)
    common = np.intersect1d(A.image_id, Bd.image_id)
    A = A[A.image_id.isin(common)].reset_index(drop=True)
    Bd = Bd[Bd.image_id.isin(common)].reset_index(drop=True)
    point = fn(Bd) - fn(A)
    rng = np.random.default_rng(seed)
    sids = A.sid_true.to_numpy()
    diffs = np.empty(B)
    for b in range(B):
        ix = stratified_indices(sids, rng)
        diffs[b] = fn(Bd.iloc[ix]) - fn(A.iloc[ix])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": point, "lo": lo, "hi": hi, "p": min(1.0, p), "n": len(common)}


def bootstrap_ci(s: pd.DataFrame, metric: str, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    fn = METRICS[metric]
    s = s.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sids = s.sid_true.to_numpy()
    vals = [fn(s.iloc[stratified_indices(sids, rng)]) for _ in range(B)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
