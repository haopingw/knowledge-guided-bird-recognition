"""Batch inference with resume. One JSON line per (image, condition, variant).

Examples
  python -m kgbr.run --split dev --pilot --conditions A,C --tag pilot
  python -m kgbr.run --split val --conditions A,C --max-pixels 262144 --tag val_512
  python -m kgbr.run --split test --conditions A,B,C,D,E --tag main
  python -m kgbr.run --split test --occlusion --conditions A,C --tag occlusion
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from .cub import load_cub
from .knowledge import attribute_set, load_species
from .parse import parse_reply
from .prompts import SYSTEM, build_prompt, candidate_order, prompt_hash

RAW = Path("results/raw")


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def file_hash(*paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:10]


def select_images(args) -> pd.DataFrame:
    man = pd.read_csv("data/splits/manifest.csv")
    df = man[man.split == args.split]
    if args.pilot:  # 5 similar species x 10 dev images
        sids = ["S04", "S05", "S03", "S07", "S16"]  # Chipping, Clay-colored, Brewer, Field, Tree
        df = df[df.sid.isin(sids)].groupby("sid").head(10)
    if args.limit:
        df = df.groupby("sid").head(args.limit)
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test", choices=["dev", "val", "test"])
    ap.add_argument("--conditions", default="A,B,C,D,E")
    ap.add_argument("--occlusion", action="store_true", help="run mask_key / mask_ctrl variants on the occlusion subset")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max images per species")
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.max_pixels:
        cfg["max_pixels"] = args.max_pixels
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    conds = args.conditions.split(",")
    cub = load_cub()
    names = {s["sid"]: s["name"] for s in load_species()}
    attr_ids = [a["aid"] for a in attribute_set()]

    # work items
    if args.occlusion:
        from .occlusion import load_subset, masked_image
        sub = load_subset()
        items = [(row, v, c) for v in ["mask_key", "mask_ctrl"] for c in conds for row in sub.itertuples()]
        get_img = lambda row, v: masked_image(cub, row._asdict(), v)  # noqa: E731
    else:
        df = select_images(args)
        items = [(row, "orig", c) for c in conds for row in df.itertuples()]
        get_img = lambda row, v: Image.open(cub.image_path(int(row.image_id))).convert("RGB")  # noqa: E731

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{args.tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["image_id"], r["condition"], r["variant"]))
    todo = [it for it in items if (int(it[0].image_id), it[2], it[1]) not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to run -> {out}")
    if not todo:
        return

    from .model import QwenVL
    model = QwenVL(cfg["model_id"], cfg["dtype"], cfg["max_pixels"], cfg["max_new_tokens"],
                   cfg.get("revision"), cfg.get("load_in_4bit", False))
    meta = {"config": cfg, "config_hash": file_hash(args.config),
            "knowledge_hash": file_hash("knowledge/species_notes.yaml", "knowledge/controls.yaml"),
            "model_revision": model.revision, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
            "platform": platform.platform()}
    (RAW / f"{args.tag}.meta.json").write_text(json.dumps(meta, indent=2))

    bs = cfg["batch_size"]
    torch.cuda.reset_peak_memory_stats()
    with out.open("a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(todo), bs), desc=args.tag):
            batch = todo[i:i + bs]
            imgs = [get_img(row, v) for row, v, c in batch]
            prompts = [build_prompt(int(row.image_id), c) for row, v, c in batch]
            res = model.generate(imgs, prompts, SYSTEM)
            for (row, v, c), p, r in zip(batch, prompts, res):
                iid = int(row.image_id)
                rec = {"image_id": iid, "sid_true": row.sid, "condition": c, "variant": v,
                       "split": args.split, "candidate_order": candidate_order(iid),
                       "prompt_hash": prompt_hash(p), "config_hash": meta["config_hash"],
                       "knowledge_hash": meta["knowledge_hash"], **r,
                       "parsed": parse_reply(r["raw"], names, attr_ids)}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            # Release cached blocks each batch: on Windows (WDDM) an over-full VRAM pool
            # spills into shared memory and generation can stall indefinitely.
            torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 1e9
    meta["peak_gpu_mem_gb"] = round(peak, 2)
    (RAW / f"{args.tag}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done; peak GPU memory {peak:.2f} GB")


if __name__ == "__main__":
    main()
