"""Late, gated split of Pathway-Scorer-style cofactor hubs on a finished Escher map.

A hub is a currency metabolite drawn as one glyph linked to ≥2 reactions.
Each de-novo reaction gets its own clone; the clone keeps the midmarker distance
and is rotated onto the bisector of a free angular gap around the midmarker when
every directed gate metric stays at least as good as the pre-split baseline.
Curated reaction geometry is left untouched.

The gate is ``layout_gate`` (compose-path code only).
"""
from __future__ import annotations

import copy
import math

from core import currency
from cofactor_hubs import cofactor_hubs
from hub_drawing import drawing_graph, stub_endpoints
from layout_gate import CrossingIndex, score_map, score_trial
from layout_quality_evaluator import HUB_DIRS, METRIC_EPS, metrics_not_worse


def no_regression(before, after, tol=None):
    """Hub-split no-regression via the shared evaluator tolerance and directions."""
    return metrics_not_worse(before, after, HUB_DIRS, METRIC_EPS if tol is None else tol)


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


def _clone_stub(nodes, glyph, mid, angle_deg):
    """Place a length-preserving clone of ``glyph`` rotated about ``mid``."""
    mx, my = float(nodes[mid]["x"]), float(nodes[mid]["y"])
    hx, hy = float(nodes[glyph]["x"]), float(nodes[glyph]["y"])
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    dx, dy = hx - mx, hy - my
    rx_, ry_ = ca * dx - sa * dy, sa * dx + ca * dy
    nid = _next_id(nodes)
    nodes[nid] = dict(nodes[glyph])
    nodes[nid]["x"] = round(mx + rx_, 2)
    nodes[nid]["y"] = round(my + ry_, 2)
    nodes[nid]["node_is_primary"] = False
    if "label_x" in nodes[glyph]:
        nodes[nid]["label_x"] = round(
            nodes[nid]["x"] + (float(nodes[glyph]["label_x"]) - hx), 2
        )
        nodes[nid]["label_y"] = round(
            nodes[nid]["y"] + (float(nodes[glyph]["label_y"]) - hy), 2
        )
    return nid


def _retarget(rx, glyph, nid):
    """Point segments that ended on ``glyph`` at the new clone; keep multimarker chains."""
    for seg in (rx.get("segments") or {}).values():
        if seg.get("from_node_id") == glyph:
            seg["from_node_id"] = nid
            seg["b1"] = seg["b2"] = None
        if seg.get("to_node_id") == glyph:
            seg["to_node_id"] = nid
            seg["b1"] = seg["b2"] = None


def _hub_movers(hub, curated):
    rids = hub["reactions"]
    denovo = [rid for rid in rids if rid not in curated]
    keep = [rid for rid in rids if rid in curated]
    if not denovo:
        return []
    holders = keep or denovo[:1]
    return [rid for rid in denovo if rid not in holders]


def _neighbour_angles(nodes, rx, mid):
    """Polar angles of metabolite endpoints around a reaction midmarker."""
    mx, my = float(nodes[mid]["x"]), float(nodes[mid]["y"])
    angs = []
    for seg in (rx.get("segments") or {}).values():
        for nid in (seg.get("from_node_id"), seg.get("to_node_id")):
            n = nodes.get(nid) or {}
            if n.get("node_type") != "metabolite":
                continue
            angs.append(math.atan2(float(n["y"]) - my, float(n["x"]) - mx))
    return angs


def _normalize_deg(rot: float) -> float:
    while rot <= -180:
        rot += 360
    while rot > 180:
        rot -= 360
    return rot


def _gap_rotation_degrees(nodes, rx, glyph) -> list[float]:
    """Rotations (degrees) onto the bisector of each free gap, largest gap first."""
    mid = _midmarker(nodes, rx)
    if mid is None or glyph not in nodes:
        return []
    mx, my = float(nodes[mid]["x"]), float(nodes[mid]["y"])
    gx, gy = float(nodes[glyph]["x"]), float(nodes[glyph]["y"])
    glyph_ang = math.atan2(gy - my, gx - mx)
    angs = sorted(_neighbour_angles(nodes, rx, mid))
    uniq = []
    for a in angs:
        if not uniq or abs(a - uniq[-1]) > 1e-6:
            uniq.append(a)
    if not uniq:
        return []
    if len(uniq) == 1:
        rot = _normalize_deg(math.degrees(uniq[0] + math.pi - glyph_ang))
        return [round(rot, 3)] if abs(rot) >= 1e-3 else []

    gaps = []
    for i, a0 in enumerate(uniq):
        a1 = uniq[(i + 1) % len(uniq)]
        if i + 1 == len(uniq):
            a1 += 2 * math.pi
        gap = a1 - a0
        mid_ang = a0 + 0.5 * gap
        gaps.append((gap, mid_ang))
    gaps.sort(key=lambda item: (-item[0], item[1]))

    candidates = []
    found = set()
    for _gap, mid_ang in gaps:
        rot = round(_normalize_deg(math.degrees(mid_ang - glyph_ang)), 3)
        if abs(rot) < 1e-3 or rot in found:
            continue
        found.add(rot)
        candidates.append(rot)
    return candidates


def _angle_candidates(map_json, movers, glyph, rid_to_key) -> list[float]:
    """Union of gap-bisector rotations from each mover, largest gap first."""
    nodes = map_json[1]["nodes"]
    reactions = map_json[1]["reactions"]
    seen = []
    found = set()
    for rid in movers:
        key = rid_to_key.get(rid)
        if key is None:
            continue
        for ang in _gap_rotation_degrees(nodes, reactions[key], glyph):
            if ang in found:
                continue
            found.add(ang)
            seen.append(ang)
    return seen


def _record_split_clones(data, clone_ids):
    meta = data[0].setdefault("metadata", {})
    prior = [str(x) for x in (meta.get("hub_split_clones") or [])]
    for nid in clone_ids:
        s = str(nid)
        if s not in prior:
            prior.append(s)
    meta["hub_split_clones"] = prior


def _split_movers(map_json, metabolite, movers, glyph, angle_deg):
    data = copy.deepcopy(map_json)
    nodes = data[1]["nodes"]
    reactions = data[1]["reactions"]
    rid_to_key = {rx.get("bigg_id"): key for key, rx in reactions.items() if rx.get("bigg_id")}
    split = []
    stub_pairs = []
    clone_ids = []
    for rid in movers:
        key = rid_to_key.get(rid)
        if key is None:
            continue
        rx = reactions[key]
        mid = _midmarker(nodes, rx)
        if mid is None:
            continue
        nid = _clone_stub(nodes, glyph, mid, angle_deg)
        _retarget(rx, glyph, nid)
        clone_ids.append(nid)
        split.append(
            dict(
                metabolite=metabolite,
                reaction=rid,
                from_glyph=glyph,
                to_glyph=nid,
                angle_deg=angle_deg,
            )
        )
        stub_pairs.append((mid, nid))
    if clone_ids:
        _record_split_clones(data, clone_ids)
    return data, split, stub_pairs


def _old_stub_pairs(map_json, movers, glyph):
    nodes = map_json[1]["nodes"]
    reactions = map_json[1]["reactions"]
    rid_to_key = {rx.get("bigg_id"): key for key, rx in reactions.items() if rx.get("bigg_id")}
    pairs = []
    for rid in movers:
        key = rid_to_key.get(rid)
        if key is None:
            continue
        mid = _midmarker(nodes, reactions[key])
        if mid is None:
            continue
        pairs.append((mid, glyph))
    return pairs


def apply_gated(map_json, model, pathway, curated=(), cur=None, log=None):
    """Split hubs one metabolite at a time when gate metrics do not worsen vs baseline.

    Each trial must stay non-worse than both the original pre-split score and the
    score of the current (already-accepted) map.
    """
    cur = currency(model) if cur is None else set(cur)
    curated = set(curated)
    current = map_json
    baseline = cofactor_hubs(current, model, cur=cur)
    if baseline["hub_count"] == 0:
        return map_json, dict(
            accepted=False,
            reason="no_hubs",
            hubs_before=0,
            hubs_after=0,
            split=[],
        )

    original_score = score_map(pathway, current)
    current_score = original_score
    g0, pos0, routes0 = drawing_graph(pathway, current)
    before_index = CrossingIndex(g0.to_undirected(), pos0, routes0)
    accepted_splits = []
    rejected = 0

    for hub in baseline["hubs"]:
        met = hub["metabolite"]
        glyph = hub["glyph"]
        movers = _hub_movers(hub, curated)
        if not movers:
            rejected += 1
            continue
        reactions = current[1]["reactions"]
        rid_to_key = {rx.get("bigg_id"): key for key, rx in reactions.items() if rx.get("bigg_id")}
        angles = _angle_candidates(current, movers, glyph, rid_to_key)
        old_stubs = _old_stub_pairs(current, movers, glyph)
        old_graph_stubs = stub_endpoints(pathway, current, old_stubs)
        old_stub_x = before_index.stub_crossing_count(old_graph_stubs)
        placed = False
        for angle in angles:
            trial, splits, new_stubs = _split_movers(current, met, movers, glyph, angle)
            if not splits:
                break
            after = score_trial(
                pathway,
                current,
                trial,
                old_stubs,
                new_stubs,
                before_index=before_index,
                old_stub_crossings=old_stub_x,
            )
            if (
                after is None
                or not no_regression(original_score, after)
                or not no_regression(current_score, after)
            ):
                continue
            current = trial
            current_score = after
            g0, pos0, routes0 = drawing_graph(pathway, current)
            before_index = CrossingIndex(g0.to_undirected(), pos0, routes0)
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
        reason="accepted" if ok else ("metric_regression" if rejected else "no_split"),
        hubs_before=baseline["hub_count"],
        hubs_after=final["hub_count"],
        split=accepted_splits,
        rejected=rejected,
    )
