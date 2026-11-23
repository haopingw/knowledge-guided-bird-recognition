"""Knowledge texts for the five conditions and the fixed attribute set.

Conditions
  A  no added knowledge
  B  generic observation checklist
  C  correct species identification notes (all 20 candidates)
  D  length-matched irrelevant notes (same structure as C)
  E  C with the first features of selected similar-species pairs swapped
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

KNOW_DIR = Path(__file__).resolve().parents[2] / "knowledge"
SPLIT_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"
CONDITIONS = ["A", "B", "C", "D", "E"]


@lru_cache(maxsize=1)
def load_notes() -> dict:
    return yaml.safe_load((KNOW_DIR / "species_notes.yaml").read_text(encoding="utf-8"))["species"]


@lru_cache(maxsize=1)
def load_controls() -> dict:
    return yaml.safe_load((KNOW_DIR / "controls.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_species() -> list[dict]:
    return json.loads((SPLIT_DIR / "species.json").read_text())["species"]


def feature_texts(condition: str) -> dict[str, list[str]]:
    """sid -> list of feature sentences for knowledge conditions C, D, E."""
    notes = load_notes()
    correct = {sid: [f["text"] for f in v["features"]] for sid, v in notes.items()}
    if condition == "C":
        return correct
    if condition == "E":
        ctl = load_controls()
        out = {sid: list(t) for sid, t in correct.items()}
        k = ctl["n_swap"]
        for a, b in ctl["swap_pairs"]:
            out[a][:k], out[b][:k] = correct[b][:k], correct[a][:k]
        return out
    if condition == "D":
        pool = load_controls()["irrelevant_pool"]
        out = {}
        for i, (sid, texts) in enumerate(sorted(correct.items())):
            used: set[int] = set()
            repl = []
            for j, t in enumerate(texts):
                # deterministic: closest length, ties broken by a species-specific rotation
                order = sorted(range(len(pool)),
                               key=lambda p: (abs(len(pool[p]) - len(t)), (p + 7 * i + j) % len(pool)))
                p = next(p for p in order if p not in used)
                used.add(p)
                repl.append(_trim_to(pool[p], len(t)))
            out[sid] = repl
        return out
    raise ValueError(condition)


def _trim_to(sentence: str, n: int) -> str:
    """Cut a pool sentence at a word boundary so its length is close to n characters."""
    if len(sentence) <= n + 3:
        return sentence
    cut = sentence.rfind(" ", 0, n + 3)
    return sentence[:cut if cut > 0 else n]


def knowledge_block(condition: str, sid_order: list[str], names: dict[str, str]) -> str:
    """Text inserted into the prompt for a condition (empty for A)."""
    if condition == "A":
        return ""
    if condition == "B":
        return load_controls()["generic_checklist"].strip()
    feats = feature_texts(condition)
    lines = ["Notes for each candidate species:"]
    for sid in sid_order:
        lines.append(f"{sid} {names[sid]}: " + "; ".join(feats[sid]) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------- attributes
@lru_cache(maxsize=1)
def attribute_set() -> list[dict]:
    """Fixed attribute list asked of the model: [{aid, cub_id, name, text}]."""
    return json.loads((SPLIT_DIR / "attributes.json").read_text())["attributes"]


def attr_text(cub_name: str) -> str:
    """'has_breast_pattern::striped' -> 'breast pattern: striped'."""
    group, value = cub_name.split("::")
    group = group.removeprefix("has_").replace("_", " ").replace("upperparts", "upper parts")
    value = value.replace("_", " ").replace("-", " ")
    return f"{group}: {value}"


def build_attribute_set(max_attrs: int = 24) -> None:
    """Select attributes from DEV annotations only, and write a knowledge check report.

    Rule: candidates = CUB attributes referenced by the species notes. Rank them by the
    between-class standard deviation of their positive rate on dev images (certainty >= 3)
    among the 20 species and keep the top `max_attrs`.
    """
    import numpy as np
    import pandas as pd

    from .cub import load_cub

    cub = load_cub()
    man = pd.read_csv(SPLIT_DIR / "manifest.csv")
    dev = man[man.split == "dev"]
    notes = load_notes()
    sids = sorted(notes)
    rate = {}
    for sid in sids:
        ids = dev.image_id[dev.sid == sid].tolist()
        with np.errstate(invalid="ignore"):
            rate[sid] = np.nanmean(cub.attr_matrix(ids, min_certainty=3), axis=0)
    R = np.nan_to_num(np.stack([rate[s] for s in sids]))  # (20, 312)

    cands = sorted({a for v in notes.values() for f in v["features"] for a in f["cub_attrs"]})
    score = {a: float(R[:, a - 1].std()) for a in cands}
    keep = sorted(sorted(cands, key=lambda a: -score[a])[:max_attrs])
    attrs = [{"aid": f"A{i + 1:02d}", "cub_id": a, "name": cub.attr_names.loc[a, "name"],
              "text": attr_text(cub.attr_names.loc[a, "name"]), "between_class_std": round(score[a], 4)}
             for i, a in enumerate(keep)]
    (SPLIT_DIR / "attributes.json").write_text(json.dumps(
        {"rule": build_attribute_set.__doc__.strip(), "attributes": attrs}, indent=2))

    # Knowledge check: does the dev data support each mapped attribute?
    lines = ["# Knowledge check against dev annotations", "",
             "For every feature -> CUB attribute mapping: positive rate (certainty >= 3) in the "
             "species' own dev images vs. the mean over the other 19 species. "
             "`ok` = own rate higher than others.", "",
             "| sid | species | feature | CUB attribute | own | others | check |",
             "|---|---|---|---|---|---|---|"]
    n_ok = n_all = 0
    for i, sid in enumerate(sids):
        for f in notes[sid]["features"]:
            for a in f["cub_attrs"]:
                own = R[i, a - 1]
                oth = np.delete(R[:, a - 1], i).mean()
                ok = own > oth
                n_ok += ok
                n_all += 1
                lines.append(f"| {sid} | {notes[sid]['name']} | {f['text'][:50]} | "
                             f"{cub.attr_names.loc[a, 'name']} | {own:.2f} | {oth:.2f} | "
                             f"{'ok' if ok else '**check**'} |")
    lines.insert(4, f"Supported: {n_ok}/{n_all} mappings.\n")
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "knowledge_check.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(attrs)} attributes kept; knowledge check {n_ok}/{n_all} supported")
    for a in attrs:
        print(f"  {a['aid']} {a['text']:<35s} std={a['between_class_std']}")


if __name__ == "__main__":
    build_attribute_set()
