"""Late, gated split of Pathway-Layout-Scorer-style cofactor hubs on a finished Escher map.

A hub is a currency metabolite drawn as one glyph linked to ≥2 reactions.  Each de-novo
reaction gets its own clone; the clone keeps the same midmarker distance (edge length) and
is rotated around the midmarker until official layout-benchmark metrics do not worsen.
Curated reaction geometry is left untouched.
"""
from __future__ import annotations

import copy
import math

from cofactor_hubs import cofactor_hubs
from core import currency

# Official columns used for the main-pair backbone gate (and reported in prior baselines).
_GATE_COLUMNS = (
    "crossings",
    "overlapping_pairs",
    "edge_length_deviation",
    "edge_orthogonality",
    "participant_consistency",
    "overlap_area_share",
    "crossing_density",
)

# Short deterministic sweep; prefer separations that worked on e_coli_core hubs.
_ANGLE_CANDIDATES = (-40, 40, -32, 32, -48, 48, -24, 24, -16, 16, -56, 56, 8, -8, 64, -64)


def _midmarker(nodes, rx):
    for seg in (rx.get("segments") or {}).values():
        for nid in (seg.get("from_node_id"), seg.get("to_node_id")):
            if nodes.get(nid, {}).get("node_type") == "midmarker":
                return nid
    return None


def _next_id(nodes):
    nums = []
    for k in nodes:
        try:
            nums.append(int(k))
        except ValueError:
            continue
    return str(max(nums) + 1 if nums else 1)


def _official(map_json, pathway):
    from layout_benchmark.external import escher_drawing
    from layout_benchmark import metrics as m

    return m.all_metrics(escher_drawing(pathway, map_json), pathway)


def _no_regression(before, after, tol=1e-4):
    from layout_benchmark import metrics as m

    dirs = {met.column: met.better for met in m.METRICS}
    for key in _GATE_COLUMNS:
        value = before.get(key)
        if value is None or after.get(key) is None:
            continue
        better = dirs.get(key)
        if better == "lower" and float(after[key]) > float(value) + tol:
            return False
        if better == "higher" and float(after[key]) < float(value) - tol:
            return False
    return True


def _split_one_metabolite(map_json, model, metabolite, *, curated, cur, angle_deg):
    """Clone movers for one hub metabolite, rotating around each reaction midmarker."""
    data = copy.deepcopy(map_json)
    nodes = data[1]["nodes"]
    reactions = data[1]["reactions"]
    rid_to_key = {rx.get("bigg_id"): key for key, rx in reactions.items() if rx.get("bigg_id")}
    hubs = [h for h in cofactor_hubs(data, model, cur=cur)["hubs"] if h["metabolite"] == metabolite]
    if not hubs:
        return data, []
    hub = hubs[0]
    glyph = hub["glyph"]
    rids = hub["reactions"]
    denovo = [rid for rid in rids if rid not in curated]
    keep = [rid for rid in rids if rid in curated]
    if not denovo:
        return data, []
    holders = keep or denovo[:1]
    movers = [rid for rid in denovo if rid not in holders] or denovo[1:]
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    split = []
    for rid in movers:
        key = rid_to_key.get(rid)
        if key is None:
            continue
        rx = reactions[key]
        mid = _midmarker(nodes, rx)
        if mid is None:
            continue
        mx, my = float(nodes[mid]["x"]), float(nodes[mid]["y"])
        hx, hy = float(nodes[glyph]["x"]), float(nodes[glyph]["y"])
        dx, dy = hx - mx, hy - my
        rx_, ry_ = ca * dx - sa * dy, sa * dx + ca * dy
        nid = _next_id(nodes)
        nodes[nid] = dict(nodes[glyph])
        nodes[nid]["x"] = round(mx + rx_, 2)
        nodes[nid]["y"] = round(my + ry_, 2)
        if "label_x" in nodes[glyph]:
            nodes[nid]["label_x"] = round(nodes[nid]["x"] + (float(nodes[glyph]["label_x"]) - hx), 2)
            nodes[nid]["label_y"] = round(nodes[nid]["y"] + (float(nodes[glyph]["label_y"]) - hy), 2)
        nodes[nid]["node_is_primary"] = False
        for seg in (rx.get("segments") or {}).values():
            if seg.get("from_node_id") == glyph:
                seg["from_node_id"] = nid
                seg["b1"] = seg["b2"] = None
            if seg.get("to_node_id") == glyph:
                seg["to_node_id"] = nid
                seg["b1"] = seg["b2"] = None
        split.append(dict(metabolite=metabolite, reaction=rid, from_glyph=glyph, to_glyph=nid, angle_deg=angle_deg))
    return data, split


def apply_gated(map_json, model, pathway=None, curated=(), cur=None, log=None):
    """Split hubs one metabolite at a time when official layout-benchmark metrics do not worsen."""
    if pathway is None:
        return map_json, dict(accepted=False, reason="no_pathway", hubs_before=None, hubs_after=None, split=[])

    cur = currency(model) if cur is None else set(cur)
    curated = set(curated)
    current = map_json
    baseline = cofactor_hubs(current, model, cur=cur)
    if baseline["hub_count"] == 0:
        return map_json, dict(accepted=False, reason="no_hubs", hubs_before=0, hubs_after=0, split=[])

    official_before = _official(current, pathway)
    accepted_splits = []
    rejected = 0
    for hub in list(baseline["hubs"]):
        met = hub["metabolite"]
        placed = False
        for angle in _ANGLE_CANDIDATES:
            trial, splits = _split_one_metabolite(
                current, model, met, curated=curated, cur=cur, angle_deg=angle
            )
            if not splits:
                break
            after = _official(trial, pathway)
            if not _no_regression(official_before, after):
                continue
            current = trial
            official_before = after
            accepted_splits.extend(splits)
            placed = True
            break
        if not placed:
            rejected += 1

    final = cofactor_hubs(current, model, cur=cur)
    ok = bool(accepted_splits) and final["hub_count"] < baseline["hub_count"]
    if log:
        log(
            f"  cofactor hub split: {baseline['hub_count']} -> {final['hub_count']} "
            f"accepted={len(accepted_splits)} rejected_hubs={rejected}"
        )
    return current, dict(
        accepted=ok,
        reason="accepted" if ok else ("lb_regression" if rejected else "no_split"),
        hubs_before=baseline["hub_count"],
        hubs_after=final["hub_count"],
        split=accepted_splits,
        rejected=rejected,
    )
