"""Unit tests that need neither the GPU nor the CUB images."""
import numpy as np
import pandas as pd
from PIL import Image

from kgbr.knowledge import CONDITIONS, feature_texts, knowledge_block, load_controls, load_notes
from kgbr.metrics import mean_per_class_acc, stratified_indices
from kgbr.model import resize_to_budget
from kgbr.occlusion import _overlap, apply_mask
from kgbr.parse import parse_reply
from kgbr.prompts import PARTS, build_prompt, candidate_order

CANDS = {"S01": "American Pipit", "S04": "Chipping Sparrow", "S15": "Song Sparrow"}
AIDS = ["A01", "A02"]


def test_parse_clean():
    r = parse_reply('{"parts":{"beak":[10,20],"crown":null},"attributes":{"A01":"yes","A02":"unclear"},'
                    '"species":"S04"}', CANDS, AIDS)
    assert r["json_ok"] and r["species"] == "S04"
    assert r["parts"]["beak"] == [10, 20] and r["parts"]["crown"] is None
    assert r["attributes"] == {"A01": "yes", "A02": "unclear"}


def test_parse_fenced_and_name_fallback():
    r = parse_reply('Sure!\n```json\n{"species": "song sparrow", "parts": {}, "attributes": {}}\n```', CANDS, AIDS)
    assert r["species"] == "S15" and r["species_by_name"]


def test_parse_invalid():
    assert parse_reply("I think it is a sparrow", CANDS, AIDS)["species"] is None
    r = parse_reply('{"species": "S99", "parts": {"beak": [2000, 5]}}', CANDS, AIDS)
    assert r["json_ok"] and r["species"] is None and r["parts"]["beak"] is None


def test_parse_trailing_comma_and_short_id():
    r = parse_reply('{"species": "S4", "attributes": {"A01": true,},}', CANDS, AIDS)
    assert r["species"] == "S04" and r["attributes"]["A01"] == "yes"


def test_conditions_cover_all_species_and_lengths():
    notes = load_notes()
    assert len(notes) == 20
    for v in notes.values():
        assert 3 <= len(v["features"]) <= 5
        assert v["key_part"] in PARTS
    C, D = feature_texts("C"), feature_texts("D")
    lc = sum(len(t) for ts in C.values() for t in ts)
    ld = sum(len(t) for ts in D.values() for t in ts)
    assert abs(ld - lc) / lc < 0.10, (lc, ld)
    for sid in C:
        assert len(C[sid]) == len(D[sid])


def test_swap_is_exact():
    C, E = feature_texts("C"), feature_texts("E")
    ctl = load_controls()
    k = ctl["n_swap"]
    swapped = set()
    for a, b in ctl["swap_pairs"]:
        assert E[a][:k] == C[b][:k] and E[b][:k] == C[a][:k]
        assert E[a][k:] == C[a][k:]
        swapped |= {a, b}
    for sid in C:
        if sid not in swapped:
            assert E[sid] == C[sid]


def test_prompts_identical_except_knowledge():
    iid = 6743
    order = candidate_order(iid)
    assert order == candidate_order(iid) and sorted(order) == sorted(load_notes())
    names = {s: load_notes()[s]["name"] for s in order}
    pa = build_prompt(iid, "A")
    for c in CONDITIONS[1:]:
        pc = build_prompt(iid, c)
        assert pc.replace(knowledge_block(c, order, names) + "\n\n", "") == pa


def test_mean_per_class_acc():
    s = pd.DataFrame({"sid_true": ["a", "a", "a", "b"], "correct": [1, 1, 0, 0]})
    assert mean_per_class_acc(s) == (2 / 3 + 0) / 2


def test_stratified_indices_keep_class_sizes():
    sids = np.array(["a"] * 5 + ["b"] * 3)
    ix = stratified_indices(sids, np.random.default_rng(0))
    assert (sids[ix] == "a").sum() == 5 and (sids[ix] == "b").sum() == 3


def test_resize_budget_and_mask():
    img = Image.new("RGB", (1000, 600), (255, 0, 0))
    r = resize_to_budget(img, 512 * 512)
    assert r.size[0] * r.size[1] <= 512 * 512 and r.size[0] % 32 == 0 and r.size[1] % 32 == 0
    m = apply_mask(img, (10, 10, 20, 20))
    assert m.getpixel((15, 15)) == (128, 128, 128) and m.getpixel((30, 30)) == (255, 0, 0)
    assert _overlap((0, 0, 10, 10), (5, 5, 15, 15)) and not _overlap((0, 0, 10, 10), (10, 0, 20, 10))
