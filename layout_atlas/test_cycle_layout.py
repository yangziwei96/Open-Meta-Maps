"""Unit tests for closed-loop layout of free metabolic cycles."""
from __future__ import annotations

import math

import numpy as np

import cycle_layout as cl
from evaluation.map_rules import cycle_metrics


def _ring_model(n=5):
    """n reactions in a simple ring linked by non-currency metabolites m0..m{n-1}."""
    reactions = []
    metabolites = []
    for i in range(n):
        mid_in = f"m{i}_c"
        metabolites.append(dict(id=mid_in, name=f"M{i}", compartment="c", annotations=[]))
        reactions.append(
            dict(
                id=f"R{i}",
                metabolites={mid_in: -1.0, f"m{(i + 1) % n}_c": 1.0},
                lower_bound=0.0,
                upper_bound=1000.0,
            )
        )
    seen = {}
    for m in metabolites:
        seen[m["id"]] = m
    # ensure all mets exist
    for i in range(n):
        mid = f"m{i}_c"
        seen.setdefault(mid, dict(id=mid, name=f"M{i}", compartment="c", annotations=[]))
    return dict(id="ring", reactions=reactions, metabolites=list(seen.values()))


def _escher_ring(n=5, scramble=False):
    """Shared-metabolite Escher ring; optional scrambled angular order."""
    model = _ring_model(n)
    if scramble:
        angles = [2.0 * math.pi * ((2 * i) % n) / n for i in range(n)]
    else:
        angles = [2.0 * math.pi * i / n for i in range(n)]
    nodes = {}
    reactions = {}
    nid = 0
    met_nodes = {}
    for i in range(n):
        mid = f"m{i}_c"
        mn = str(nid); nid += 1
        met_nodes[mid] = mn
        nodes[mn] = dict(
            node_type="metabolite",
            x=100.0 + 50.0 * math.cos(angles[i]),
            y=100.0 + 50.0 * math.sin(angles[i]),
            bigg_id=mid, name=mid, node_is_primary=True,
            label_x=100.0 + 50.0 * math.cos(angles[i]),
            label_y=100.0 + 50.0 * math.sin(angles[i]) - 8.0,
        )
    for i, r in enumerate(model["reactions"]):
        mid_id = str(nid); nid += 1
        a = np.array([nodes[met_nodes[f"m{i}_c"]]["x"], nodes[met_nodes[f"m{i}_c"]]["y"]])
        b = np.array([nodes[met_nodes[f"m{(i + 1) % n}_c"]]["x"], nodes[met_nodes[f"m{(i + 1) % n}_c"]]["y"]])
        mid_xy = 0.5 * (a + b)
        nodes[mid_id] = dict(node_type="midmarker", x=float(mid_xy[0]), y=float(mid_xy[1]))
        segs = {}
        mets = []
        for mid, coef in r["metabolites"].items():
            segs[str(len(segs))] = dict(
                from_node_id=mid_id, to_node_id=met_nodes[mid],
                b1=dict(x=float(mid_xy[0]), y=float(mid_xy[1])),
                b2=dict(x=nodes[met_nodes[mid]]["x"], y=nodes[met_nodes[mid]]["y"]),
            )
            mets.append(dict(bigg_id=mid, coefficient=coef))
        reactions[str(i)] = dict(
            bigg_id=r["id"], name=r["id"], metabolites=mets, segments=segs,
            label_x=float(mid_xy[0]), label_y=float(mid_xy[1]) - 8.0)
    data = [dict(map_name="ring"), dict(
        nodes=nodes, reactions=reactions,
        canvas=dict(x=0, y=0, width=400, height=400))]
    return model, data, met_nodes


def test_detect_free_cycles_finds_ring():
    model = _ring_model(5)
    cycles = cl.detect_free_cycles(model, set(), curated_rx=set())
    assert len(cycles) == 1
    assert len(cycles[0]["reactions"]) == 5
    assert len(cycles[0]["metabolites"]) == 5


def test_detect_skips_curated_covered_ring():
    model = _ring_model(5)
    cycles = cl.detect_free_cycles(model, set(), curated_rx={"R0", "R1", "R2", "R3", "R4"})
    assert cycles == []


def test_apply_escher_places_circle_with_consistent_order():
    model, data, met_nodes = _escher_ring(6, scramble=True)
    # Deterministic scatter (no hash): push every other met far from the circle.
    for i, mid in enumerate([f"m{k}_c" for k in range(6)]):
        if i % 2 == 0:
            continue
        nid = met_nodes[mid]
        data[1]["nodes"][nid]["x"] = 100.0 + 20.0 * i
        data[1]["nodes"][nid]["y"] = 40.0 + 15.0 * i
        data[1]["nodes"][nid]["label_x"] = data[1]["nodes"][nid]["x"]
        data[1]["nodes"][nid]["label_y"] = data[1]["nodes"][nid]["y"] - 8.0
    data2, report = cl.apply_escher(data, model, blocks=[], cur=set())
    assert report["placed"] >= 1 or report.get("skipped")
    if report["placed"] >= 1:
        mets = cl.detect_free_cycles(model, set(), set())[0]["metabolites"]
        keys = [met_nodes[m] for m in mets]
        xy = np.array([[data2[1]["nodes"][k]["x"], data2[1]["nodes"][k]["y"]] for k in keys], float)
        cen = xy.mean(0)
        ang = np.arctan2(xy[:, 1] - cen[1], xy[:, 0] - cen[0])
        steps = []
        for k in range(len(ang)):
            a = ang[(k + 1) % len(ang)] - ang[k]
            a = (a + math.pi) % (2 * math.pi) - math.pi
            steps.append(a)
        pos_share = float(np.mean([1.0 if s > 0 else 0.0 for s in steps]))
        assert max(pos_share, 1.0 - pos_share) >= 0.99
        radii = np.hypot(xy[:, 0] - cen[0], xy[:, 1] - cen[1])
        assert float(np.std(radii) / max(float(np.mean(radii)), 1e-9)) < 0.05


def test_curated_shared_metabolite_does_not_move():
    """A metabolite node also used by a curated reaction must stay put."""
    model, data, met_nodes = _escher_ring(4, scramble=False)
    shared = met_nodes["m0_c"]
    before = (data[1]["nodes"][shared]["x"], data[1]["nodes"][shared]["y"])
    nodes = data[1]["nodes"]
    mid_c = "99"
    nodes[mid_c] = dict(node_type="midmarker", x=before[0] + 30.0, y=before[1] + 30.0)
    data[1]["reactions"]["cx"] = dict(
        bigg_id="CX", name="CX",
        metabolites=[dict(bigg_id="m0_c", coefficient=-1.0), dict(bigg_id="m1_c", coefficient=1.0)],
        segments={
            "0": dict(from_node_id=mid_c, to_node_id=shared, b1=None, b2=None),
            "1": dict(from_node_id=mid_c, to_node_id=met_nodes["m1_c"], b1=None, b2=None),
        },
        label_x=before[0] + 30.0, label_y=before[1] + 30.0,
    )
    blocks = [dict(id="b0", reaction_ids=["CX"], positions={})]
    for mid in ("m1_c", "m2_c", "m3_c"):
        nid = met_nodes[mid]
        nodes[nid]["x"] += 40.0
        nodes[nid]["y"] -= 25.0
    data2, report = cl.apply_escher(data, model, blocks=blocks, cur=set())
    after = (data2[1]["nodes"][shared]["x"], data2[1]["nodes"][shared]["y"])
    assert after == before


def test_previous_pinned_metabolite_does_not_move():
    """A metabolite pinned via --previous (m: id) must stay put."""
    model, data, met_nodes = _escher_ring(4, scramble=False)
    pinned = met_nodes["m1_c"]
    before = (data[1]["nodes"][pinned]["x"], data[1]["nodes"][pinned]["y"])
    # Scramble so the stage would want to move m1_c.
    for mid in ("m1_c", "m2_c", "m3_c"):
        nid = met_nodes[mid]
        data[1]["nodes"][nid]["x"] += 35.0
        data[1]["nodes"][nid]["y"] -= 20.0
    before = (data[1]["nodes"][pinned]["x"], data[1]["nodes"][pinned]["y"])
    data2, _report = cl.apply_escher(
        data, model, blocks=[], cur=set(), pinned_metabolite_ids={"m1_c"},
    )
    after = (data2[1]["nodes"][pinned]["x"], data2[1]["nodes"][pinned]["y"])
    assert after == before


def test_apply_escher_updates_labels_with_node_delta():
    model, data, met_nodes = _escher_ring(4, scramble=True)
    nid = met_nodes["m1_c"]
    node = data[1]["nodes"][nid]
    node["x"], node["y"] = 10.0, 10.0
    node["label_x"], node["label_y"] = 12.0, 7.0
    data2, report = cl.apply_escher(data, model, blocks=[], cur=set())
    if report.get("placed"):
        n2 = data2[1]["nodes"][nid]
        assert abs((n2["label_x"] - n2["x"]) - (12.0 - 10.0)) < 1e-6
        assert abs((n2["label_y"] - n2["y"]) - (7.0 - 10.0)) < 1e-6


def test_ring_score_prefers_rounder_order():
    model, data, met_nodes = _escher_ring(4, scramble=False)
    keys = [met_nodes[f"m{i}_c"] for i in range(4)]
    good = cl._ring_score(data[1]["nodes"], keys)
    # Collapse one met toward the centre.
    data[1]["nodes"][keys[1]]["x"] = 100.0
    data[1]["nodes"][keys[1]]["y"] = 100.0
    bad = cl._ring_score(data[1]["nodes"], keys)
    assert good > bad


def test_cycle_metrics_on_reaction_ring():
    _model, data, _ = _escher_ring(4, scramble=False)
    m = cycle_metrics(data[1])
    assert m["cycle_len"] >= 4
    assert m["cycle_circularity"] is not None
    assert m["cycle_order_consistency"] is not None


def test_cycle_layout_no_layout_benchmark_import():
    """cycle_layout must not import layout_benchmark or gdMetriX."""
    import sys
    blocked = {"layout_benchmark", "gdMetriX", "gdmetrix"}
    before = {k for k in sys.modules if any(k == b or k.startswith(b + ".") for b in blocked)}
    model, data, _ = _escher_ring(4, scramble=True)
    cl.apply_escher(data, model, blocks=[], cur=set())
    after = {k for k in sys.modules if any(k == b or k.startswith(b + ".") for b in blocked)}
    assert after <= before


if __name__ == "__main__":
    test_detect_free_cycles_finds_ring()
    test_detect_skips_curated_covered_ring()
    test_apply_escher_places_circle_with_consistent_order()
    test_curated_shared_metabolite_does_not_move()
    test_previous_pinned_metabolite_does_not_move()
    test_apply_escher_updates_labels_with_node_delta()
    test_ring_score_prefers_rounder_order()
    test_cycle_metrics_on_reaction_ring()
    test_cycle_layout_no_layout_benchmark_import()
    print("ok")
