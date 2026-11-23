"""Tables and figures.

  python -m kgbr.analyze quick --tag pilot          # per-condition summary for one run
  python -m kgbr.analyze full                       # main + occlusion report -> results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .cub import load_cub  # noqa: E402
from .knowledge import load_controls, load_species  # noqa: E402
from .metrics import bootstrap_ci, load_raw, paired_bootstrap, per_image_scores, summarize  # noqa: E402
from .prompts import PARTS  # noqa: E402

RES = Path("results")
FIG = RES / "figures"
COND_NAMES = {"A": "A no knowledge", "B": "B generic checklist", "C": "C species notes",
              "D": "D irrelevant notes", "E": "E swapped notes"}
COLORS = {"A": "#8c8c8c", "B": "#5b8db8", "C": "#2a9d55", "D": "#c9a227", "E": "#c4473a"}


def fmt(x, pct=True):
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def table(rows: list[dict], cols: list[tuple[str, str, bool]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * len(cols)
    body = []
    for r in rows:
        body.append("| " + " | ".join(
            (fmt(r[k], p) if isinstance(r[k], (float, np.floating)) else str(r[k])) for k, _, p in cols) + " |")
    return "\n".join([head, sep, *body])


def quick(tag: str) -> None:
    s = per_image_scores(load_raw(tag))
    rows = []
    for c, g in s.groupby("condition"):
        rows.append({"condition": c, **summarize(g)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


# ------------------------------------------------------------------ full report
def _sid_names():
    return {s["sid"]: s["name"] for s in load_species()}


def main_results(s: pd.DataFrame, out: list[str]) -> pd.DataFrame:
    conds = sorted(s.condition.unique())
    rows = []
    for c in conds:
        g = s[s.condition == c]
        r = {"condition": COND_NAMES[c], **summarize(g)}
        lo, hi = bootstrap_ci(g, "acc")
        r["acc_ci"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
        rows.append(r)
    df = pd.DataFrame(rows)
    out += ["## 1. Main results (test split)", "",
            table(rows, [("condition", "Condition", False), ("n", "n", False), ("acc", "Mean per-class acc %", True),
                         ("acc_ci", "95% CI", False), ("attr_macro_f1", "Attr macro-F1 %", True),
                         ("attr_micro_f1", "Attr micro-F1 %", True), ("attr_unclear", "Attr 'unclear' %", True),
                         ("pck@0.1", "PCK@0.1 %", True), ("lm_coverage", "Part coverage %", True),
                         ("lm_mean_err", "Mean part err (x diag)", False)]), "",
            "Output reliability and cost:", "",
            table(rows, [("condition", "Condition", False), ("json_fail", "JSON fail %", True),
                         ("invalid_species", "Invalid species %", True), ("truncated", "Truncated %", True),
                         ("in_tokens", "Input tokens", False), ("out_tokens", "Output tokens", False),
                         ("time_s", "Sec / image (batch 8)", False)]), ""]
    # paired differences
    comps = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("D", "C"), ("E", "C")]
    drows = []
    for a, b in comps:
        if a not in conds or b not in conds:
            continue
        for m in ["acc", "attr_macro_f1", "pck@0.1"]:
            d = paired_bootstrap(s, a, b, m)
            drows.append({"comparison": f"{b} - {a}", "metric": m, "diff": d["diff"],
                          "ci": f"[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]", "p": f"{d['p']:.3f}"})
    out += ["## 2. Paired differences (stratified paired bootstrap, 2000 resamples)", "",
            table(drows, [("comparison", "Comparison", False), ("metric", "Metric", False),
                          ("diff", "Diff (pp)", True), ("ci", "95% CI (pp)", False), ("p", "p", False)]), ""]

    # figure: main bars
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (m, lab) in zip(axes, [("acc", "Mean per-class accuracy"), ("attr_macro_f1", "Attribute macro-F1"),
                                   ("pck@0.1", "Part PCK@0.1")]):
        vals, errs = [], []
        for c in conds:
            g = s[s.condition == c]
            v = summarize(g)[m if m != "pck@0.1" else "pck@0.1"]
            lo, hi = bootstrap_ci(g, m, B=1000)
            vals.append(v); errs.append([v - lo, hi - v])
        ax.bar(conds, vals, color=[COLORS[c] for c in conds], yerr=np.array(errs).T, capsize=4)
        ax.set_title(lab); ax.set_ylim(0, max(vals) * 1.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "main_results.png", dpi=150); plt.close(fig)
    out += ["![main](figures/main_results.png)", ""]
    return df


def per_class(s: pd.DataFrame, out: list[str]) -> None:
    names = _sid_names()
    acc = s.groupby(["sid_true", "condition"]).correct.mean().unstack()
    acc.index = [f"{i} {names[i]}" for i in acc.index]
    out += ["## 3. Per-class accuracy (%)", "", (acc * 100).round(1).to_markdown(), ""]
    # confusion matrices
    sids = sorted(names)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, c in zip(axes, ["A", "C"]):
        g = s[s.condition == c]
        M = np.zeros((len(sids), len(sids) + 1))
        for t, p in zip(g.sid_true, g.pred):
            M[sids.index(t), sids.index(p) if p in sids else len(sids)] += 1
        M = M / M.sum(1, keepdims=True)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(sids) + 1), sids + ["inv"], rotation=90, fontsize=7)
        ax.set_yticks(range(len(sids)), [f"{i} {names[i][:14]}" for i in sids], fontsize=7)
        ax.set_title(f"Confusion, {COND_NAMES[c]}"); ax.set_xlabel("predicted")
    fig.tight_layout(); fig.savefig(FIG / "confusion_A_C.png", dpi=150); plt.close(fig)
    out += ["![confusion](figures/confusion_A_C.png)", ""]

    # does classification gain go with evidence gain? per class
    from .metrics import attr_f1, landmark
    rows = []
    for sid in sids:
        a = s[(s.condition == "A") & (s.sid_true == sid)]
        c = s[(s.condition == "C") & (s.sid_true == sid)]
        rows.append({"sid": sid, "d_acc": c.correct.mean() - a.correct.mean(),
                     "d_f1": attr_f1(c)[0] - attr_f1(a)[0], "d_pck": landmark(c)["pck"] - landmark(a)["pck"]})
    r = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, k, lab in [(axes[0], "d_f1", "Δ attribute macro-F1 (C−A)"), (axes[1], "d_pck", "Δ PCK@0.1 (C−A)")]:
        ax.scatter(r[k] * 100, r.d_acc * 100, color=COLORS["C"])
        for _, q in r.iterrows():
            ax.annotate(q.sid, (q[k] * 100, q.d_acc * 100), fontsize=7)
        ax.axhline(0, color="#bbb", lw=.8); ax.axvline(0, color="#bbb", lw=.8)
        ax.set_xlabel(lab + " (pp)"); ax.set_ylabel("Δ accuracy (pp)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "gain_vs_evidence.png", dpi=150); plt.close(fig)
    rho_f1 = r[["d_acc", "d_f1"]].corr(method="spearman").iloc[0, 1]
    rho_pck = r[["d_acc", "d_pck"]].corr(method="spearman").iloc[0, 1]
    out += ["## 4. Does the classification gain go with better evidence? (per class, C vs A)", "",
            f"Spearman ρ(Δacc, Δattr-F1) = {rho_f1:.2f}; ρ(Δacc, ΔPCK) = {rho_pck:.2f} (20 classes).", "",
            "![gain](figures/gain_vs_evidence.png)", ""]


def swap_analysis(s: pd.DataFrame, out: list[str]) -> None:
    ctl = load_controls()
    partner = {}
    for a, b in ctl["swap_pairs"]:
        partner[a], partner[b] = b, a
    rows = []
    for c in ["A", "C", "E"]:
        g = s[s.condition == c]
        sw = g[g.sid_true.isin(partner)]
        un = g[~g.sid_true.isin(partner)]
        to_partner = (sw.pred == sw.sid_true.map(partner)).mean()
        rows.append({"condition": COND_NAMES[c], "acc_swapped": sw.correct.mean(),
                     "acc_unswapped": un.correct.mean(), "pred_partner": to_partner})
    out += ["## 5. Misleading notes (condition E)", "",
            "Swapped species = the 10 species in the 5 swapped pairs. `→ partner` = share of their images "
            "predicted as the pair partner whose features they were given.", "",
            table(rows, [("condition", "Condition", False), ("acc_swapped", "Acc swapped species %", True),
                         ("acc_unswapped", "Acc other species %", True), ("pred_partner", "→ partner %", True)]), ""]


def occlusion(main: pd.DataFrame, occ: pd.DataFrame, out: list[str]) -> None:
    from .occlusion import load_subset
    sub = load_subset()
    orig = main[main.image_id.isin(sub.image_id) & main.condition.isin(["A", "C"])].copy()
    orig["variant"] = "orig"
    s = pd.concat([orig, occ], ignore_index=True)
    s["cv"] = s.condition + "|" + s.variant
    rows = []
    for v in ["orig", "mask_ctrl", "mask_key"]:
        r = {"variant": v}
        for c in ["A", "C"]:
            r[c] = s[(s.cv == f"{c}|{v}")].correct.mean()
        d = paired_bootstrap(s, f"A|{v}", f"C|{v}", "acc", key="cv")
        r["gain"] = d["diff"]; r["gain_ci"] = f"[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
        rows.append(r)
    # difference of differences: (C-A)|ctrl - (C-A)|key, paired bootstrap over images
    rng = np.random.default_rng(0)
    piv = s.pivot_table(index=["image_id", "sid_true"], columns="cv", values="correct").reset_index()
    sids = piv.sid_true.to_numpy()

    def dd(df):
        acc = lambda col: df.groupby("sid_true")[col].mean().mean()  # noqa: E731
        return (acc("C|mask_ctrl") - acc("A|mask_ctrl")) - (acc("C|mask_key") - acc("A|mask_key"))
    from .metrics import stratified_indices
    point = dd(piv)
    boots = np.array([dd(piv.iloc[stratified_indices(sids, rng)]) for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out += ["## 6. Occlusion experiment (200 test images, 10 per class)", "",
            "Knowledge gain Δ = acc(C) − acc(A) under each image condition. Masks are gray squares "
            "(side 0.30·√bbox area) on the species' key part vs. the nearest non-overlapping visible body part.", "",
            table(rows, [("variant", "Image", False), ("A", "Acc A %", True), ("C", "Acc C %", True),
                         ("gain", "Δ = C−A (pp)", True), ("gain_ci", "95% CI", False)]), "",
            f"**Δ_ctrl − Δ_key = {100 * point:+.1f} pp, 95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]** "
            "(positive = masking the key part removes more of the knowledge gain than masking a control region).", ""]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(3)
    for k, c in enumerate(["A", "C"]):
        ax.bar(x + (k - .5) * .38, [r[c] for r in rows], .38, color=COLORS[c], label=COND_NAMES[c])
    ax.set_xticks(x, ["original", "control mask", "key-part mask"]); ax.set_ylabel("accuracy")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "occlusion.png", dpi=150); plt.close(fig)
    out += ["![occlusion](figures/occlusion.png)", ""]
    occlusion_examples(sub)


def occlusion_examples(sub: pd.DataFrame) -> None:
    from .occlusion import masked_image
    cub = load_cub()
    rows = [sub[sub.sid == sid].iloc[0] for sid in ["S04", "S15", "S19"]]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    for i, r in enumerate(rows):
        for j, v in enumerate(["orig", "mask_key", "mask_ctrl"]):
            axes[i, j].imshow(masked_image(cub, r.to_dict(), v)); axes[i, j].axis("off")
            axes[i, j].set_title(f"{r.sid} {v}" + (f" ({r.key_part})" if v == "mask_key" else
                                                    f" ({r.ctrl_part})" if v == "mask_ctrl" else ""), fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "occlusion_examples.png", dpi=120); plt.close(fig)


def cases(s: pd.DataFrame, out: list[str]) -> None:
    """Pre-defined case categories, first two images (by id) in each."""
    cub = load_cub()
    names = _sid_names()
    piv = s.pivot_table(index="image_id", columns="condition", values="correct")
    ctl = load_controls()
    swapped = {x for p in ctl["swap_pairs"] for x in p}
    true = s.drop_duplicates("image_id").set_index("image_id").sid_true
    cats = {
        "A wrong → C right": piv.index[(piv.A == 0) & (piv.C == 1)],
        "A right → C wrong": piv.index[(piv.A == 1) & (piv.C == 0)],
        "C right → E wrong (swapped species)": piv.index[(piv.C == 1) & (piv.E == 0) & true.loc[piv.index].isin(swapped)],
    }
    fig, axes = plt.subplots(len(cats), 2, figsize=(9, 4 * len(cats)))
    lines = ["## 7. Cases (pre-defined categories; first two image ids in each)", ""]
    for i, (cat, ids) in enumerate(cats.items()):
        lines.append(f"- **{cat}**: {len(ids)} images")
        for j in range(2):
            ax = axes[i, j]; ax.axis("off")
            if j >= len(ids):
                continue
            iid = int(sorted(ids)[j])
            img = Image.open(cub.image_path(iid)).convert("RGB")
            W, H = img.size
            d = ImageDraw.Draw(img)
            for c, col in [("A", (140, 140, 140)), ("C", (42, 157, 85))]:
                p = s[(s.image_id == iid) & (s.condition == c)].iloc[0].parsed
                for part in PARTS:
                    pt = p["parts"].get(part)
                    if pt:
                        x, y = pt[0] / 1000 * W, pt[1] / 1000 * H
                        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=col, width=3)
            gt = cub.part_xyv(iid)
            for part in PARTS:
                x, y, v = gt[part]
                if v:
                    d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=(230, 60, 60))
            preds = {c: s[(s.image_id == iid) & (s.condition == c)].iloc[0].pred for c in "ACE"}
            ax.imshow(img)
            ax.set_title(f"#{iid} true {true[iid]} {names[true[iid]][:18]}\n"
                         + " ".join(f"{c}:{preds[c]}" for c in "ACE"), fontsize=8)
    fig.suptitle("red squares = CUB part annotations; gray circles = A parts; green circles = C parts", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "cases.png", dpi=120); plt.close(fig)
    out += lines + ["", "![cases](figures/cases.png)", ""]


def full(main_tag="main", occ_tag="occlusion") -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out = ["# Results", "", "Generated by `python -m kgbr.analyze full`. Model: Qwen3-VL-4B-Instruct (frozen, "
           "greedy, bf16). Test split of the 20 selected species.", ""]
    s = per_image_scores(load_raw(main_tag))
    main_results(s, out)
    per_class(s, out)
    swap_analysis(s, out)
    if Path(f"results/raw/{occ_tag}.jsonl").exists():
        occ = per_image_scores(load_raw(occ_tag))
        occlusion(s, occ, out)
        out += ["![occlusion examples](figures/occlusion_examples.png)", ""]
    cases(s, out)
    meta = json.loads(Path(f"results/raw/{main_tag}.meta.json").read_text())
    out += ["## Run metadata", "", "```json", json.dumps(meta, indent=2), "```"]
    (RES / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote results/RESULTS.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quick", "full"])
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    quick(a.tag) if a.cmd == "quick" else full()
