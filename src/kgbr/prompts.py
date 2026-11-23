"""Prompt construction. Everything except the knowledge block is identical across conditions."""
from __future__ import annotations

import hashlib
import random

from .knowledge import attribute_set, knowledge_block, load_species

PARTS = ["beak", "crown", "nape", "throat", "breast", "tail"]

SYSTEM = "You are a careful expert in North American bird identification."


def candidate_order(image_id: int) -> list[str]:
    """Per-image candidate order from a fixed seed; identical for all conditions."""
    sids = [s["sid"] for s in load_species()]
    random.Random(1000 + int(image_id)).shuffle(sids)
    return sids


def build_prompt(image_id: int, condition: str) -> str:
    names = {s["sid"]: s["name"] for s in load_species()}
    order = candidate_order(image_id)
    attrs = attribute_set()
    cand = "\n".join(f"{sid}: {names[sid]}" for sid in order)
    attr_lines = "\n".join(f"{a['aid']}: {a['text']}" for a in attrs)
    know = knowledge_block(condition, order, names)
    parts_json = ", ".join(f'"{p}": [x, y]' for p in PARTS)
    attr_json = ", ".join(f'"{a["aid"]}": "yes"' for a in attrs[:2]) + ", ..."

    sections = [
        "The photo shows one bird. It belongs to exactly one of these candidate species:",
        cand,
    ]
    if know:
        sections.append(know)
    sections += [
        "First record what is visible, then choose the species.",
        "1) Part locations: give the pixel location of each part as [x, y] in relative "
        "coordinates from 0 to 1000 (0,0 = top-left corner, 1000,1000 = bottom-right). "
        "Use null if the part is not visible.",
        "2) Attributes: for each attribute below answer \"yes\" (clearly visible on this bird), "
        "\"no\" (clearly not), or \"unclear\".",
        attr_lines,
        "3) Species: the ID of the single most likely candidate.",
        "Reply with ONLY one compact JSON object on a single line (no spaces or line breaks, "
        "no other text), in this format:",
        f'{{"parts": {{{parts_json}}}, "attributes": {{{attr_json}}}, "species": "S00"}}',
    ]
    return "\n\n".join(sections)


def prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]
