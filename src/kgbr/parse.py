"""Robust parsing of model output into {species, parts, attributes}.

Rules (fixed before testing):
  * The first balanced {...} object in the reply is parsed (code fences allowed).
  * species must be a candidate ID (e.g. "S07"); an exact candidate *name* is accepted
    and flagged `species_by_name`. Anything else -> species = None (scored as wrong).
  * part = [x, y] with both numbers in [0, 1000]; otherwise None (= not predicted).
  * attribute values normalised to yes / no / unclear; anything else -> None.
"""
from __future__ import annotations

import json
import re

from .prompts import PARTS

_YES = {"yes", "y", "true", "present", "1"}
_NO = {"no", "n", "false", "absent", "0"}
_UNC = {"unclear", "unknown", "not visible", "uncertain", "n/a", "null", "none"}


def _first_json_object(text: str) -> str | None:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _point(v):
    if isinstance(v, dict):
        v = [v.get("x"), v.get("y")]
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            x, y = float(v[0]), float(v[1])
        except (TypeError, ValueError):
            return None
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return [x, y]
    return None


def _attr(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "unclear"
    s = str(v).strip().lower()
    if s in _YES:
        return "yes"
    if s in _NO:
        return "no"
    if s in _UNC:
        return "unclear"
    return None


def parse_reply(text: str, candidates: dict[str, str], attr_ids: list[str]) -> dict:
    """candidates: sid -> species name."""
    out = {"json_ok": False, "species": None, "species_by_name": False,
           "parts": {p: None for p in PARTS}, "attributes": {a: None for a in attr_ids}}
    blob = _first_json_object(text)
    if blob is None:
        return out
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        try:  # tolerate trailing commas / single quotes
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", blob).replace("'", '"'))
        except json.JSONDecodeError:
            return out
    if not isinstance(obj, dict):
        return out
    out["json_ok"] = True

    sp = obj.get("species")
    if isinstance(sp, str):
        s = sp.strip()
        m = re.fullmatch(r"[Ss]\s*0*(\d{1,2})", s)
        if m and f"S{int(m.group(1)):02d}" in candidates:
            out["species"] = f"S{int(m.group(1)):02d}"
        else:
            norm = lambda t: re.sub(r"[^a-z]", "", t.lower())  # noqa: E731
            hits = [sid for sid, n in candidates.items() if norm(n) == norm(s)]
            if len(hits) == 1:
                out["species"], out["species_by_name"] = hits[0], True

    parts = obj.get("parts") or {}
    if isinstance(parts, dict):
        for p in PARTS:
            out["parts"][p] = _point(parts.get(p))

    attrs = obj.get("attributes") or {}
    if isinstance(attrs, dict):
        for a in attr_ids:
            if a in attrs:
                out["attributes"][a] = _attr(attrs[a])
    return out
