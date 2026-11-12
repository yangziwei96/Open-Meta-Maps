"""Closed-loop layout for metabolic cycles on the drawn map.

Builds a reaction–reaction graph from shared non-currency metabolite glyphs,
finds short simple rings, and moves free-only metabolite glyphs onto each
ring's circle when that improves the ring's own circularity × order score.
Nodes that belong to any curated or pinned reaction, and metabolites pinned
via ``--previous`` (``m:`` anchors), never move. A move is kept only when
in-repo crossings, crossing angles, edge-length evenness, participant
placement and node-overlap counts do not get worse. Bezier controls and
labels move with their nodes; no new multimarker nodes are created.
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict

import numpy as np

from units_library import _bip, short_cycles

# At least 4 reactions so the metabolite projection has a closed loop.
MIN_RX = 4


def curated_reaction_ids(blocks) -> set[str]:
    return {rid for b in (blocks or []) for rid in (b.get("reaction_ids") or [])}


def detect_free_cycles(model, cur, curated_rx, min_rx=MIN_RX, max_rx=None):
    """Main metabolic cycles among reactions not covered by curated structures.

    Uses the same bipartite short-cycle search as Atlas unit cutting (model
    currency already removed). Returns greedily selected, reaction-disjoint
    cycles, longest first. Each item: dict(reactions, metabolites).
    """
    curated_rx = set(curated_rx or ())
    free = [r["id"] for r in model["reactions"] if r["id"] not in curated_rx]
    if len(free) < min_rx:
        return []
    if max_rx is None:
        # Bound by the free subgraph size (no fixed magic length).
        max_rx = max(min_rx, len(free))
    r2m = {}
    for r in model["reactions"]:
        if r["id"] not in free:
            continue
        r2m["r:" + r["id"]] = sorted("m:" + m for m in r["metabolites"] if m not in cur)
    if len(r2m) < min_rx:
        return []
    g = _bip(sorted(r2m), r2m)
    cycles = short_cycles(g, max_rx)
    cycles = sorted(
        (rs for rs in cycles if len(rs) >= min_rx),
        key=lambda rs: (-len(rs), rs),
    )
    used = set()
    out = []
    reactions = {r["id"]: r for r in model["reactions"]}
    for rs in cycles:
        rids = [r[2:] for r in rs]
        if any(rid in used for rid in rids):
            continue
        ordered = _order_ring(reactions, rids, cur)
        if ordered is None or len(ordered["metabolites"]) < min_rx:
            continue
        used.update(rids)
        out.append(ordered)
    return out


def _order_ring(reactions, rids, cur):
    """Order reactions and linking metabolites around a simple bipartite ring."""
    users = defaultdict(list)
    for rid in rids:
        for mid in reactions[rid]["metabolites"]:
            if mid not in cur:
                users[mid].append(rid)
    links = {m: tuple(sorted(rs)) for m, rs in users.items() if len(rs) == 2}
    if len(links) < len(rids):
        return None
    adj = defaultdict(dict)
    for mid, (a, b) in links.items():
        adj[a][b] = mid
        adj[b][a] = mid
    if any(len(adj[r]) != 2 for r in rids):
        return None
    start = min(rids)
    nbrs = sorted(adj[start])
    seq = [start, nbrs[0]]
    while len(seq) < len(rids):
        prev, cur_r = seq[-2], seq[-1]
        nxts = [n for n in adj[cur_r] if n != prev]
        if len(nxts) != 1:
            return None
        seq.append(nxts[0])
    if seq[-1] not in adj[start]:
        return None
    mets = []
    for i, rid in enumerate(seq):
        nxt = seq[(i + 1) % len(seq)]
        mets.append(adj[rid][nxt])
    return dict(reactions=seq, metabolites=mets)


def detect_drawn_cycles(body, cur, min_rx=MIN_RX):
    """Short cycles in the drawn reaction–reaction graph.

    Two reactions are linked when they share a non-currency metabolite glyph.
    Returns dict(reactions, met_nids, mid_nids, extent) sorted by midmarker extent.
    """
    nodes = body["nodes"]
    met_rx = defaultdict(set)
    mid_of = {}
    for rx in body["reactions"].values():
        rid = rx["bigg_id"]
        mid = None
        for s in (rx.get("segments") or {}).values():
            for nid in (s.get("from_node_id"), s.get("to_node_id")):
                n = nodes.get(nid) or {}
                if n.get("node_type") == "midmarker":
                    mid = nid
                elif n.get("node_type") == "metabolite" and n.get("bigg_id"):
                    if n["bigg_id"] not in cur:
                        met_rx[nid].add(rid)
        if mid is not None:
            mid_of[rid] = mid

    pair_mets = defaultdict(list)
    import networkx as nx

    g = nx.Graph()
    for nid, rids in met_rx.items():
        rs = sorted(rids)
        for i, a in enumerate(rs):
            for b in rs[i + 1 :]:
                g.add_edge(a, b)
                pair_mets[(a, b)].append(nid)

    out = []
    if g.number_of_nodes() < min_rx:
        return out
    for cyc in nx.cycle_basis(g):
        if len(cyc) < min_rx:
            continue
        adj = defaultdict(set)
        for a, b in g.subgraph(cyc).edges():
            adj[a].add(b)
            adj[b].add(a)
        if any(len(adj[r]) != 2 for r in cyc):
            continue
        start = min(cyc)
        seq = [start, sorted(adj[start])[0]]
        ok = True
        while len(seq) < len(cyc):
            prev, cur_r = seq[-2], seq[-1]
            nxts = [x for x in adj[cur_r] if x != prev]
            if len(nxts) != 1:
                ok = False
                break
            seq.append(nxts[0])
        if not ok or seq[-1] not in adj[start]:
            continue
        met_nids = []
        for i, rid in enumerate(seq):
            a, b = sorted((rid, seq[(i + 1) % len(seq)]))
            cands = pair_mets.get((a, b)) or []
            exclusive = [nid for nid in cands if met_rx[nid] == {a, b}]
            pool = exclusive or cands
            if not pool:
                met_nids.append(None)
                continue
            # Deterministic pick among candidates.
            met_nids.append(sorted(pool)[0])
        if any(nid is None for nid in met_nids):
            continue
        if any(rid not in mid_of for rid in seq):
            continue
        xy = np.array([[nodes[mid_of[r]]["x"], nodes[mid_of[r]]["y"]] for r in seq], float)
        extent = float(np.max(np.hypot(xy[:, 0] - xy.mean(0)[0], xy[:, 1] - xy.mean(0)[1])))
        out.append(dict(reactions=seq, met_nids=met_nids, mid_nids=[mid_of[r] for r in seq], extent=extent))

    seen = set()
    uniq = []
    for c in sorted(out, key=lambda c: (c["extent"], len(c["reactions"]), c["reactions"])):
        key = frozenset(c["reactions"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _escher_reaction_index(body):
    """bigg_id -> (reaction dict, midmarker node id)."""
    nodes = body["nodes"]
    out = {}
    for rx in body["reactions"].values():
        mid = None
        for s in (rx.get("segments") or {}).values():
            for nid in (s.get("from_node_id"), s.get("to_node_id")):
                if (nodes.get(nid) or {}).get("node_type") == "midmarker":
                    mid = nid
                    break
            if mid:
                break
        if mid is not None:
            out[rx["bigg_id"]] = (rx, mid)
    return out


def _met_nodes_of(rx, body):
    """bigg_id -> node id for metabolites linked by this reaction's segments."""
    nodes = body["nodes"]
    found = {}
    for s in (rx.get("segments") or {}).values():
        for nid in (s.get("from_node_id"), s.get("to_node_id")):
            n = nodes.get(nid) or {}
            if n.get("node_type") == "metabolite" and n.get("bigg_id"):
                found.setdefault(n["bigg_id"], nid)
    return found


def _node_owners(body):
    """metabolite/midmarker node id -> set of reaction bigg_ids that touch it."""
    nodes = body["nodes"]
    owner = defaultdict(set)
    for rx in body["reactions"].values():
        rid = rx["bigg_id"]
        for s in (rx.get("segments") or {}).values():
            for nid in (s.get("from_node_id"), s.get("to_node_id")):
                n = nodes.get(nid) or {}
                if n.get("node_type") in ("metabolite", "midmarker"):
                    owner[nid].add(rid)
    return owner


def _characteristic_spacing(body):
    """Median midmarker→metabolite chord length on the drawn map.

    Returns None when the map has no such chords (caller should skip placement).
    """
    nodes = body["nodes"]
    lens = []
    for rx in body["reactions"].values():
        mid = None
        mets = []
        for s in (rx.get("segments") or {}).values():
            for nid in (s.get("from_node_id"), s.get("to_node_id")):
                n = nodes.get(nid) or {}
                if n.get("node_type") == "midmarker":
                    mid = nid
                elif n.get("node_type") == "metabolite":
                    mets.append(nid)
        if mid is None or not mets:
            continue
        mx, my = nodes[mid]["x"], nodes[mid]["y"]
        for nid in mets:
            lens.append(math.hypot(nodes[nid]["x"] - mx, nodes[nid]["y"] - my))
    if not lens:
        return None
    return float(np.median(lens))


def _fit_circle(xy):
    if len(xy) < 3:
        return None
    m = np.zeros((3, 3), float)
    v = np.zeros(3, float)
    for x, y in xy:
        row = np.array([2.0 * x, 2.0 * y, 1.0])
        b = x * x + y * y
        m += np.outer(row, row)
        v += row * b
    try:
        p = np.linalg.solve(m, v)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = float(p[0]), float(p[1]), float(p[2])
    r = math.sqrt(max(c + cx * cx + cy * cy, 1e-9))
    return cx, cy, r


def _circle_from_mids(nodes, mid_nids):
    xy = np.array([[nodes[nid]["x"], nodes[nid]["y"]] for nid in mid_nids], float)
    fitted = _fit_circle(xy)
    if fitted is not None:
        return fitted
    cx, cy = float(xy.mean(0)[0]), float(xy.mean(0)[1])
    radius = float(np.median(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)))
    return cx, cy, max(radius, 1.0)


def _ring_score(nodes, met_nids):
    """Circularity × order-consistency on a ring's metabolite glyphs (in-repo).

    Circularity is 1 − std(radius)/mean(radius) after a circle fit (floor at 0).
    Order consistency is the larger of the clockwise / counter-clockwise shares
    of consecutive angular steps around the fitted centre.
    """
    if len(met_nids) < 3:
        return 0.0
    xy = np.array([[nodes[nid]["x"], nodes[nid]["y"]] for nid in met_nids], float)
    fitted = _fit_circle(xy)
    if fitted is None:
        cx, cy = float(xy.mean(0)[0]), float(xy.mean(0)[1])
        radii = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        r = float(np.mean(radii))
    else:
        cx, cy, r = fitted
        radii = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    if r <= 1e-9:
        circ = 0.0
    else:
        circ = max(0.0, 1.0 - float(np.std(radii)) / r)
    ang = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    steps = []
    for i in range(len(ang)):
        d = ang[(i + 1) % len(ang)] - ang[i]
        d = (d + math.pi) % (2.0 * math.pi) - math.pi
        steps.append(d)
    pos = float(np.mean([1.0 if s > 0 else 0.0 for s in steps]))
    order = max(pos, 1.0 - pos)
    return circ * order


def _restyle_edge(rx, mid_id, met_nid, nodes):
    """Refresh Bezier controls on mid→metabolite (and reverse) segments."""
    P = np.array([nodes[mid_id]["x"], nodes[mid_id]["y"]], float)
    Q = np.array([nodes[met_nid]["x"], nodes[met_nid]["y"]], float)
    c1 = P + (Q - P) / 3.0
    c2 = P + 2.0 * (Q - P) / 3.0
    for s in (rx.get("segments") or {}).values():
        a, b = s.get("from_node_id"), s.get("to_node_id")
        if {a, b} != {mid_id, met_nid}:
            continue
        if a == mid_id:
            s["b1"] = dict(x=round(float(c1[0]), 2), y=round(float(c1[1]), 2))
            s["b2"] = dict(x=round(float(c2[0]), 2), y=round(float(c2[1]), 2))
        else:
            s["b1"] = dict(x=round(float(c2[0]), 2), y=round(float(c2[1]), 2))
            s["b2"] = dict(x=round(float(c1[0]), 2), y=round(float(c1[1]), 2))


def _translate_label(node, dx, dy):
    if "label_x" in node:
        node["label_x"] = float(node["label_x"]) + dx
        node["label_y"] = float(node["label_y"]) + dy


def _quality_bundle(data, skip_edges=(), spacing=None):
    """In-repo gates: crossings, crossing angles, edge evenness, participants, overlaps."""
    import layout_quality

    summary, _issues = layout_quality.evaluate(data, skip_edges)
    cross = (summary.get("core") or {}).get("crossings") or {}
    crossings = int(cross.get("algorithm_algorithm", 0)) + int(
        cross.get("algorithm_curated", 0)
    )
    events = (summary.get("details") or {}).get("crossing_events") or []
    if not events:
        angle_score = 1.0 / (1.0 + float(cross.get("per_100_edges") or 0.0))
    else:
        vals = []
        for ev in events:
            ang = ev.get("angle_deg")
            if ang is None:
                continue
            vals.append(abs(math.sin(math.radians(2.0 * float(ang)))))
        angle_score = float(np.mean(vals)) if vals else 1.0
    body = data[1]
    nodes = body["nodes"]
    lens = []
    for rx in body["reactions"].values():
        mid = None
        mets = []
        for s in (rx.get("segments") or {}).values():
            for nid in (s.get("from_node_id"), s.get("to_node_id")):
                n = nodes.get(nid) or {}
                if n.get("node_type") == "midmarker":
                    mid = nid
                elif n.get("node_type") == "metabolite":
                    mets.append(nid)
        if mid is None:
            continue
        mx, my = nodes[mid]["x"], nodes[mid]["y"]
        for nid in mets:
            lens.append(math.hypot(nodes[nid]["x"] - mx, nodes[nid]["y"] - my))
    if len(lens) < 2:
        evenness = 1.0
    else:
        arr = np.asarray(lens, float)
        mean = float(np.mean(arr))
        evenness = 1.0 / (1.0 + (float(np.std(arr)) / mean if mean > 1e-9 else 0.0))
    return dict(
        crossings=crossings,
        crossing_angle=angle_score,
        edge_evenness=evenness,
        participant_share=_participant_share(body),
        overlap_pairs=_node_overlap_pairs(body, spacing),
    )


def _participant_share(body):
    """Share of (reactant, product) pairs on opposite sides of their midmarker (>90°).

    Same pair-counting definition as the layout-benchmark / evaluate participant
    consistency measure, computed from Escher midmarker and metabolite glyphs.
    """
    nodes = body["nodes"]
    opposite = 0
    pairs = 0
    for rx in body["reactions"].values():
        mid = None
        for s in (rx.get("segments") or {}).values():
            for nid in (s.get("from_node_id"), s.get("to_node_id")):
                if (nodes.get(nid) or {}).get("node_type") == "midmarker":
                    mid = nid
                    break
            if mid is not None:
                break
        if mid is None:
            continue
        coeff = {
            m["bigg_id"]: m["coefficient"]
            for m in (rx.get("metabolites") or [])
            if isinstance(m, dict) and m.get("bigg_id")
        }
        subs, prods = [], []
        mx, my = nodes[mid]["x"], nodes[mid]["y"]
        for s in (rx.get("segments") or {}).values():
            for nid in (s.get("from_node_id"), s.get("to_node_id")):
                n = nodes.get(nid) or {}
                if n.get("node_type") != "metabolite" or not n.get("bigg_id"):
                    continue
                v = np.array([n["x"] - mx, n["y"] - my], float)
                if float(np.hypot(*v)) < 1e-9:
                    continue
                if coeff.get(n["bigg_id"], 0) < 0:
                    subs.append(v)
                elif coeff.get(n["bigg_id"], 0) > 0:
                    prods.append(v)
        for a in subs:
            for b in prods:
                pairs += 1
                opposite += float(np.dot(a, b)) < 0
    if pairs == 0:
        return 1.0
    return opposite / pairs


def _node_overlap_pairs(body, spacing):
    """Count axis-aligned glyph overlaps; half-size derived from map spacing."""
    if spacing is None or spacing <= 0:
        return 0
    half = 0.5 * spacing
    nodes = body["nodes"]
    boxes = []
    for n in nodes.values():
        if n.get("node_type") not in ("metabolite", "midmarker") or "x" not in n:
            continue
        boxes.append((n["x"] - half, n["y"] - half, n["x"] + half, n["y"] + half))
    count = 0
    for i in range(len(boxes)):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                count += 1
    return count


def _quality_worse(before, after, tol=1e-4):
    if after["crossings"] > before["crossings"]:
        return True, "crossings"
    if after["crossing_angle"] < before["crossing_angle"] - tol:
        return True, "crossing_angle"
    if after["edge_evenness"] < before["edge_evenness"] - tol:
        return True, "edge_evenness"
    if after["participant_share"] < before["participant_share"] - tol:
        return True, "participant_share"
    if after["overlap_pairs"] > before["overlap_pairs"]:
        return True, "overlap_pairs"
    return False, None


def _half_angle(ang_a, ang_b):
    """Angle halfway along the short arc from ang_a to ang_b."""
    diff = (ang_b - ang_a + math.pi) % (2.0 * math.pi) - math.pi
    return ang_a + 0.5 * diff


def _move_eps(spacing):
    """Ignore sub-pixel noise relative to the map's typical spoke length."""
    if spacing is None or spacing <= 0:
        return 1e-9
    return spacing * 1e-4


def _move_node(nodes, nid, nx, ny, eps):
    dx, dy = nx - nodes[nid]["x"], ny - nodes[nid]["y"]
    if abs(dx) <= eps and abs(dy) <= eps:
        return False
    nodes[nid]["x"], nodes[nid]["y"] = float(nx), float(ny)
    _translate_label(nodes[nid], dx, dy)
    return True


def _place_drawn_ring(body, cyc, frozen_rx, frozen_mets, owner, spacing, lo, hi):
    """Place free linking mets on the ring circle defined by current midmarkers."""
    nodes = body["nodes"]
    rx_index = _escher_reaction_index(body)
    seq = cyc["reactions"]
    met_nids = cyc["met_nids"]
    mid_nids = cyc["mid_nids"]
    n = len(seq)
    cx, cy, radius = _circle_from_mids(nodes, mid_nids)
    # Expand only when chords would be tighter than the map's typical spoke length.
    min_r = spacing / (2.0 * math.sin(math.pi / n)) if n >= 3 else spacing
    radius = max(radius, min_r)
    # Keep the circle inside the content bbox by one typical spoke length.
    pad = spacing
    room = min(cx - lo[0], hi[0] - cx, cy - lo[1], hi[1] - cy) - pad
    if room <= 0:
        return []
    radius = min(radius, room)
    eps = _move_eps(spacing)

    mid_angles = [
        math.atan2(nodes[mid_id]["y"] - cy, nodes[mid_id]["x"] - cx) for mid_id in mid_nids
    ]
    moved = []
    for i, nid in enumerate(met_nids):
        n = nodes.get(nid) or {}
        bigg = n.get("bigg_id")
        if bigg in frozen_mets:
            continue
        if (owner.get(nid) or set()) & frozen_rx:
            continue
        ang = _half_angle(mid_angles[i], mid_angles[(i + 1) % len(seq)])
        nx = cx + radius * math.cos(ang)
        ny = cy + radius * math.sin(ang)
        if _move_node(nodes, nid, nx, ny, eps):
            moved.append(nid)
            for rid, mid_id in ((seq[i], mid_nids[i]), (seq[(i + 1) % len(seq)], mid_nids[(i + 1) % len(seq)])):
                if rid in frozen_rx:
                    continue
                rx = rx_index.get(rid)
                if rx is not None:
                    _restyle_edge(rx[0], mid_id, nid, nodes)
    for i, mid_id in enumerate(mid_nids):
        rid = seq[i]
        if rid in frozen_rx:
            continue
        rx = rx_index.get(rid)
        if rx is None:
            continue
        for met_nid in (met_nids[i], met_nids[(i - 1) % len(seq)]):
            _restyle_edge(rx[0], mid_id, met_nid, nodes)
    return moved


def _drawn_from_free_model_cycle(body, cyc):
    """Project a model free-cycle onto drawn mid/met node ids when possible."""
    rx_index = _escher_reaction_index(body)
    rids = cyc["reactions"]
    mets = cyc["metabolites"]
    mid_nids = []
    met_nids = []
    for rid in rids:
        if rid not in rx_index:
            return None
        mid_nids.append(rx_index[rid][1])
    nodes = body["nodes"]
    for i, mid in enumerate(mets):
        left, right = rids[i], rids[(i + 1) % len(rids)]
        left_mets = _met_nodes_of(rx_index[left][0], body)
        right_mets = _met_nodes_of(rx_index[right][0], body)
        shared = sorted(
            nid
            for nid in set(left_mets.values()) & set(right_mets.values())
            if (nodes.get(nid) or {}).get("bigg_id") == mid
        )
        if shared:
            met_nids.append(shared[0])
            continue
        cand = left_mets.get(mid) or right_mets.get(mid)
        if cand is None:
            return None
        met_nids.append(cand)
    xy = np.array([[nodes[m]["x"], nodes[m]["y"]] for m in mid_nids], float)
    extent = float(np.max(np.hypot(xy[:, 0] - xy.mean(0)[0], xy[:, 1] - xy.mean(0)[1])))
    return dict(reactions=list(rids), met_nids=met_nids, mid_nids=mid_nids, extent=extent)


def apply_escher(
    data,
    model,
    blocks,
    cur,
    pinned_reaction_ids=None,
    pinned_metabolite_ids=None,
    skip_edges=(),
):
    """Place free glyphs of compact metabolic cycles onto closed loops.

    Prefers short rings in the drawn reaction–reaction graph; also considers
    spatially local free model cycles from detect_free_cycles. Curated/pinned
    reaction glyphs and ``--previous`` metabolite pins stay put. A move is kept
    only when that ring's circularity × order improves and in-repo crossings,
    crossing angles, edge-length evenness, participant placement and node
    overlaps do not get worse.
    """
    body = data[1]
    curated_rx = curated_reaction_ids(blocks)
    pinned_reaction_ids = set(pinned_reaction_ids or ())
    pinned_metabolite_ids = set(pinned_metabolite_ids or ())
    frozen_rx = curated_rx | pinned_reaction_ids

    drawn = detect_drawn_cycles(body, cur)
    drawn_max = max((len(c["reactions"]) for c in drawn), default=MIN_RX)
    free_model = detect_free_cycles(model, cur, curated_rx, max_rx=max(MIN_RX, drawn_max))
    for fc in free_model:
        proj = _drawn_from_free_model_cycle(body, fc)
        if proj is not None:
            drawn.append(proj)

    seen = set()
    cycles = []
    for c in sorted(drawn, key=lambda c: (c["extent"], len(c["reactions"]), c["reactions"])):
        key = frozenset(c["reactions"])
        if key in seen:
            continue
        seen.add(key)
        cycles.append(c)

    report = dict(
        cycles=len(cycles),
        lengths=[len(c["reactions"]) for c in cycles],
        placed=0,
        moved=0,
        skipped=[],
    )
    if not cycles:
        return data, report

    xy_all = np.array([[n["x"], n["y"]] for n in body["nodes"].values() if "x" in n and "y" in n], float)
    if len(xy_all) < 2:
        return data, report
    lo, hi = xy_all.min(0), xy_all.max(0)
    spacing = _characteristic_spacing(body)
    if spacing is None:
        report["skipped"] = [dict(length=len(c["reactions"]), reason="no_spacing") for c in cycles]
        return data, report

    quality_before = _quality_bundle(data, skip_edges, spacing)
    quality = quality_before
    owner = _node_owners(body)
    placed = 0
    moved = 0
    skipped = []
    used_rx = set()

    for cyc in cycles:
        n = len(cyc["reactions"])
        if any(rid in used_rx for rid in cyc["reactions"]):
            skipped.append(dict(length=n, reason="overlap"))
            continue

        score_before = _ring_score(body["nodes"], cyc["met_nids"])
        snap = copy.deepcopy(body)
        multi_before = {nid for nid, n in body["nodes"].items() if n.get("node_type") == "multimarker"}
        moved_ids = _place_drawn_ring(
            body, cyc, frozen_rx, pinned_metabolite_ids, owner, spacing, lo, hi
        )
        if not moved_ids:
            data[1] = snap
            body = data[1]
            skipped.append(dict(length=n, reason="no_move", extent=cyc["extent"]))
            continue
        multi_after = {nid for nid, n in body["nodes"].items() if n.get("node_type") == "multimarker"}
        if multi_after - multi_before:
            data[1] = snap
            body = data[1]
            skipped.append(dict(length=n, reason="new_multimarker", extent=cyc["extent"]))
            continue
        xy2 = np.array([[n["x"], n["y"]] for n in body["nodes"].values() if "x" in n and "y" in n], float)
        lo2, hi2 = xy2.min(0), xy2.max(0)
        if (lo2[0] < lo[0] - 1e-6 or lo2[1] < lo[1] - 1e-6
                or hi2[0] > hi[0] + 1e-6 or hi2[1] > hi[1] + 1e-6):
            data[1] = snap
            body = data[1]
            skipped.append(dict(length=n, reason="bbox_growth", extent=cyc["extent"]))
            continue
        score_after = _ring_score(body["nodes"], cyc["met_nids"])
        if score_after <= score_before:
            data[1] = snap
            body = data[1]
            skipped.append(dict(
                length=n, reason="ring_score",
                score_before=score_before, score_after=score_after, extent=cyc["extent"],
            ))
            continue
        after = _quality_bundle(data, skip_edges, spacing)
        worse, reason = _quality_worse(quality, after)
        if worse:
            data[1] = snap
            body = data[1]
            skipped.append(dict(length=n, reason=reason, extent=cyc["extent"]))
            continue
        quality = after
        owner = _node_owners(body)
        placed += 1
        moved += len(moved_ids)
        used_rx.update(cyc["reactions"])

    report.update(
        placed=placed,
        moved=moved,
        skipped=skipped,
        quality_before=quality_before,
        quality_after=quality,
    )
    return data, report
