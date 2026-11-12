"""Layout-rule metrics on a final Escher map (transfer-check field names).

Reimplemented from the BioHackathon layout rules / layout-transfer-check
field names. Never copies GPL code. Thresholds that involve angles or relative
lengths are the group's first guesses (documented in docs/layout_evaluation.md).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

try:
    from core import CURRENCY, chem
except ImportError:  # pragma: no cover
    CURRENCY, chem = set(), lambda mid: mid

# First-guess relative thresholds (scale-free; not model-specific).
HUB_SPACING_FACTOR = 4.0
CURRENCY_LONG_SPACING_FACTOR = 3.0
OPPOSITE_DEG = 120.0  # transfer-check / OpenMetaMaps orientation default
PARTICIPANT_CONSISTENCY_DEG = 90.0  # layout-benchmark participant_consistency
PERIPHERY_BAND = 0.2  # outer 20% of component hull radius
ASPECT_RATIO_TARGET = "4:3–16:9"  # packing band; miss only outside ASPECT_MIN–ASPECT_MAX
ASPECT_MIN = 4.0 / 3.0
ASPECT_MAX = 16.0 / 9.0
BEND_TURN_DEG = 20.0
AXIS_ALIGN_DEG = 10.0  # ±10° of horizontal or vertical
# Small fraction of median nearest-neighbour main-metabolite distance (not the full
# NN length — that makes almost every point "aligned"). Calibrated so the curated
# BiGG e_coli_core Escher map scores clearly above our compose (~0.85 vs ~0.16).
GRID_TOL_NN_FRAC = 0.02
GRID_TOL_FLOOR_PX = 1.0  # fixed floor so tiny maps still snap

# Goal reference: curated BiGG e_coli_core ("Core metabolism") under the definitions
# above. Gate axis share is per-edge (count); length-weighted is reported alongside.
# Source: data/source_inputs/raw/escher/e_coli_core/e_coli_core.Core metabolism.json
ORDERLINESS_GOALS = dict(
    main_segment_axis_share=0.5468,  # count-weighted per main edge (gate)
    main_segment_axis_share_length=0.5732,  # length-weighted (report only)
    reaction_axis_share=0.7097,
    chain_turn_median_deg=0.0,
    metabolite_grid_share=0.8462,
)


def _is_currency_id(mid: str | None) -> bool:
    if not mid:
        return False
    return chem(mid) in CURRENCY


def _midmarkers(body: dict) -> dict[str, tuple[float, float]]:
    nodes = body["nodes"]
    out = {}
    for rx in body["reactions"].values():
        for s in rx["segments"].values():
            for nid in (s["from_node_id"], s["to_node_id"]):
                n = nodes.get(nid) or {}
                if n.get("node_type") == "midmarker":
                    out[rx["bigg_id"]] = (float(n["x"]), float(n["y"]))
                    break
            if rx["bigg_id"] in out:
                break
    return out


def _reaction_spacing(mids: dict[str, tuple[float, float]]) -> float:
    if len(mids) < 2:
        return 60.0
    xy = np.array(list(mids.values()), float)
    d = cKDTree(xy).query(xy, k=2)[0][:, 1]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 60.0


def currency_clone_metrics(edges: list[dict], mids: dict[str, tuple[float, float]], spacing: float) -> dict[str, Any]:
    """currency_hub_violations + currency_long_arc_share (relative to median reaction spacing).

    A hub violation is a currency *instance* (glyph) linked to two or more reactions
    whose farthest pair is farther than ``HUB_SPACING_FACTOR ×`` median reaction spacing.
    """
    by_glyph: dict[str, list[str]] = defaultdict(list)
    currency_lens = []
    currency_glyphs = set()
    for e in edges:
        if not e.get("currency"):
            continue
        currency_glyphs.add(e["ends"][1])
        by_glyph[e["ends"][1]].append(e["rid"])
        pts = e["pts"]
        currency_lens.append(float(np.sum(np.hypot(*np.diff(pts, axis=0).T))))
    thr_hub = HUB_SPACING_FACTOR * max(spacing, 1.0)
    thr_long = CURRENCY_LONG_SPACING_FACTOR * max(spacing, 1.0)
    violations = 0
    for nid, rids in by_glyph.items():
        uniq = sorted(set(rids))
        if len(uniq) < 2:
            continue
        pts = [mids[r] for r in uniq if r in mids]
        if len(pts) < 2:
            continue
        arr = np.array(pts, float)
        # diameter via pairwise distances
        dmax = float(np.max(np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)))
        if dmax > thr_hub:
            violations += 1
    long_share = float(np.mean(np.array(currency_lens) > thr_long)) if currency_lens else 0.0
    return dict(
        currency_hub_violations=int(violations),
        currency_long_arc_share=round(long_share, 4),
        currency_glyphs=len(currency_glyphs),
        currency_arcs=len(currency_lens),
        hub_threshold_spacings=HUB_SPACING_FACTOR,
        long_threshold_spacings=CURRENCY_LONG_SPACING_FACTOR,
        reaction_spacing_px=round(spacing, 1),
    )


def bends_per_arc(edges: list[dict], turn_deg: float = BEND_TURN_DEG) -> dict[str, float | None]:
    """Mean number of real direction changes per connection (all lines)."""
    if not edges:
        return dict(bends_per_arc=None, bent_arc_share=None, arcs=0.0)
    bends = []
    for e in edges:
        pts = e["pts"]
        if len(pts) < 3:
            bends.append(0)
            continue
        v = np.diff(pts, axis=0)
        n = np.hypot(v[:, 0], v[:, 1])
        ok = n > 1e-6
        v = v[ok] / n[ok][:, None]
        if len(v) < 2:
            bends.append(0)
            continue
        ang = np.degrees(np.arccos(np.clip((v[:-1] * v[1:]).sum(1), -1, 1)))
        bends.append(int((ang > turn_deg).sum()))
    arr = np.array(bends, float)
    return dict(
        bends_per_arc=round(float(arr.mean()), 3),
        bent_arc_share=round(float(np.mean(arr > 0)), 4),
        arcs=float(len(arr)),
    )


def edge_len_max_rel(lengths: np.ndarray) -> float | None:
    if lengths is None or len(lengths) == 0:
        return None
    med = float(np.median(lengths))
    if med <= 1e-9:
        return None
    return round(float(np.max(lengths) / med), 3)


def _axis_aligned(dx: float, dy: float, tol_deg: float = AXIS_ALIGN_DEG) -> bool:
    """True when the vector is within ``tol_deg`` of horizontal or vertical."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return False
    ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
    return ang <= tol_deg or ang >= (180.0 - tol_deg) or abs(ang - 90.0) <= tol_deg


def _reaction_axis(rx: dict, nodes: dict) -> np.ndarray | None:
    """Substrate→product axis of non-currency metabolites around the midmarker."""
    mid = None
    for s in rx["segments"].values():
        for nid in (s["from_node_id"], s["to_node_id"]):
            if nodes.get(nid, {}).get("node_type") == "midmarker":
                mid = nodes[nid]
                break
        if mid is not None:
            break
    if mid is None:
        return None
    coeff = {m["bigg_id"]: m["coefficient"] for m in rx.get("metabolites") or [] if isinstance(m, dict)}
    subs, prods = [], []
    for s in rx["segments"].values():
        for nid in (s["from_node_id"], s["to_node_id"]):
            n = nodes.get(nid) or {}
            if n.get("node_type") != "metabolite":
                continue
            mid_id = n.get("bigg_id")
            if _is_currency_id(mid_id) or n.get("node_is_primary") is False:
                continue
            v = np.array([n["x"] - mid["x"], n["y"] - mid["y"]], float)
            (subs if coeff.get(mid_id, 0) < 0 else prods).append(v)
    if not subs or not prods:
        return None
    axis = np.mean(prods, axis=0) - np.mean(subs, axis=0)
    if float(np.hypot(*axis)) < 1e-9:
        return None
    return axis


def orderliness_metrics(body: dict, edges: list[dict], spacing: float) -> dict[str, Any]:
    """Axis alignment, chain turns and metabolite grid share (cofactors excluded).

    ``main_segment_axis_share`` (gate) counts each main reaction–metabolite edge once:
    the edge is axis-aligned when ≥50% of its drawn path length is within ±10° of H/V
    (weight independent of total length and of sampling density).
    ``main_segment_axis_share_length`` is the older length-weighted share (report only).
    Grid tolerance uses median nearest-neighbour distance among main metabolite nodes.

    Not yet optimised by the pipeline; exposed so orderliness can become a gate.
    """
    del spacing  # kept for call-site compatibility; grid tol is node-spacing based
    nodes = body["nodes"]
    # --- segment axis share: count-weighted (gate) + length-weighted (report)
    total_len = 0.0
    axis_len = 0.0
    main_edges = 0
    edge_axis_hits = 0
    for e in edges:
        if e.get("currency"):
            continue
        pts = e["pts"]
        edge_len = 0.0
        edge_axis = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            edge_len += L
            total_len += L
            if _axis_aligned(dx, dy):
                axis_len += L
                edge_axis += L
        if edge_len > 0:
            main_edges += 1
            if edge_axis / edge_len >= 0.5:
                edge_axis_hits += 1
    seg_axis = round(edge_axis_hits / main_edges, 4) if main_edges > 0 else None
    seg_axis_length = round(axis_len / total_len, 4) if total_len > 0 else None

    # --- reaction axis share
    scored = aligned = 0
    for rx in body["reactions"].values():
        axis = _reaction_axis(rx, nodes)
        if axis is None:
            continue
        scored += 1
        if _axis_aligned(float(axis[0]), float(axis[1])):
            aligned += 1
    rx_axis = round(aligned / scored, 4) if scored else None

    # --- median turn along linear chains (metabolite shared by exactly 2 reactions)
    # Travel R1 → metabolite → R2: 0° means the two midmarkers sit on opposite sides (collinear).
    mids = _midmarkers(body)
    met_rx: dict[str, list[str]] = defaultdict(list)
    for rx in body["reactions"].values():
        rid = rx["bigg_id"]
        for s in rx["segments"].values():
            for nid in (s["from_node_id"], s["to_node_id"]):
                n = nodes.get(nid) or {}
                if n.get("node_type") != "metabolite":
                    continue
                if _is_currency_id(n.get("bigg_id")) or n.get("node_is_primary") is False:
                    continue
                met_rx[nid].append(rid)
    turns = []
    for nid, rids in met_rx.items():
        uniq = [r for r in sorted(set(rids)) if r in mids]
        if len(uniq) != 2:
            continue
        mx, my = float(nodes[nid]["x"]), float(nodes[nid]["y"])
        a = np.array([mids[uniq[0]][0] - mx, mids[uniq[0]][1] - my], float)
        b = np.array([mids[uniq[1]][0] - mx, mids[uniq[1]][1] - my], float)
        na, nb = float(np.hypot(*a)), float(np.hypot(*b))
        if na < 1e-9 or nb < 1e-9:
            continue
        u, v = -a / na, b / nb
        cos = float(np.clip(u @ v, -1.0, 1.0))
        turns.append(math.degrees(math.acos(cos)))
    chain_turn = round(float(np.median(turns)), 2) if turns else None

    # --- grid alignment of main metabolite glyphs (tol from node spacing, not curves)
    mains = [
        (float(n["x"]), float(n["y"]))
        for n in nodes.values()
        if n.get("node_type") == "metabolite"
        and not _is_currency_id(n.get("bigg_id"))
        and n.get("node_is_primary") is not False
    ]
    if len(mains) >= 2:
        pts = np.array(mains, float)
        nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
        med_nn = float(np.median(nn))
    else:
        med_nn = 0.0
    tol = max(GRID_TOL_NN_FRAC * med_nn, GRID_TOL_FLOOR_PX)
    grid_hits = 0
    if len(mains) >= 2:
        xs = np.array([p[0] for p in mains], float)
        ys = np.array([p[1] for p in mains], float)
        for i, (x, y) in enumerate(mains):
            share_x = np.any(np.abs(xs[np.arange(len(mains)) != i] - x) <= tol)
            share_y = np.any(np.abs(ys[np.arange(len(mains)) != i] - y) <= tol)
            if share_x or share_y:
                grid_hits += 1
    grid = round(grid_hits / len(mains), 4) if mains else None

    return dict(
        main_segment_axis_share=seg_axis,
        main_segment_axis_share_length=seg_axis_length,
        reaction_axis_share=rx_axis,
        chain_turn_median_deg=chain_turn,
        metabolite_grid_share=grid,
        orderliness_segments=main_edges,
        orderliness_reactions=scored,
        orderliness_chain_turns=len(turns),
        orderliness_metabolites=len(mains),
        orderliness_grid_tol_px=round(tol, 2),
    )


def flow_direction_metrics(body: dict) -> dict[str, Any]:
    """Dominant reading direction among reactant→product axes; share aligned with it."""
    nodes = body["nodes"]
    dirs = []
    for rx in body["reactions"].values():
        mid = None
        for s in rx["segments"].values():
            for nid in (s["from_node_id"], s["to_node_id"]):
                if nodes.get(nid, {}).get("node_type") == "midmarker":
                    mid = nodes[nid]
                    break
            if mid is not None:
                break
        if mid is None:
            continue
        coeff = {m["bigg_id"]: m["coefficient"] for m in rx.get("metabolites") or [] if isinstance(m, dict)}
        subs, prods = [], []
        for s in rx["segments"].values():
            for nid in (s["from_node_id"], s["to_node_id"]):
                n = nodes.get(nid) or {}
                if n.get("node_type") != "metabolite":
                    continue
                mid_id = n.get("bigg_id")
                if _is_currency_id(mid_id):
                    continue
                v = np.array([n["x"] - mid["x"], n["y"] - mid["y"]], float)
                (subs if coeff.get(mid_id, 0) < 0 else prods).append(v)
        if not subs or not prods:
            continue
        axis = np.mean(prods, axis=0) - np.mean(subs, axis=0)
        if float(np.hypot(*axis)) < 1e-9:
            continue
        ang = math.degrees(math.atan2(axis[1], axis[0]))  # -180..180, +y down in Escher
        # 4-way bins: right / down / left / up
        if -45 <= ang < 45:
            dirs.append("right")
        elif 45 <= ang < 135:
            dirs.append("down")
        elif ang >= 135 or ang < -135:
            dirs.append("left")
        else:
            dirs.append("up")
    if not dirs:
        return dict(flow_main_dir=None, flow_main_dir_share=None, reactions_scored=0)
    counts = defaultdict(int)
    for d in dirs:
        counts[d] += 1
    main = max(counts, key=counts.get)
    return dict(
        flow_main_dir=main,
        flow_main_dir_share=round(counts[main] / len(dirs), 4),
        reactions_scored=len(dirs),
    )


def _polygon_area_perimeter(pts: np.ndarray) -> tuple[float, float]:
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    per = float(np.sum(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))))
    return area, per


def _winding(pts: np.ndarray) -> str:
    x, y = pts[:, 0], pts[:, 1]
    cross = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return "ccw" if cross > 0 else "cw"


def cycle_metrics(body: dict, max_cycles: int = 40) -> dict[str, Any]:
    """cycle_circularity / cycle_direction / cycle_order_consistency on simple cycles.

    Graph: reactions linked when they share a non-currency metabolite instance path
    (same drawn metabolite node). Circularity is the isoperimetric quotient 4πA/P² of
    the reaction-midmarker polygon (1 = circle). Order consistency: share of
    consecutive topological neighbours that stay angular neighbours around the centroid.
    """
    import networkx as nx

    nodes = body["nodes"]
    mids = _midmarkers(body)
    # metabolite node id -> reactions touching it (non-currency)
    met_rx: dict[str, set[str]] = defaultdict(set)
    for rx in body["reactions"].values():
        rid = rx["bigg_id"]
        for s in rx["segments"].values():
            for nid in (s["from_node_id"], s["to_node_id"]):
                n = nodes.get(nid) or {}
                if n.get("node_type") != "metabolite":
                    continue
                if _is_currency_id(n.get("bigg_id")):
                    continue
                met_rx[nid].add(rid)
    g = nx.Graph()
    for rids in met_rx.values():
        rs = sorted(r for r in rids if r in mids)
        for i, a in enumerate(rs):
            for b in rs[i + 1 :]:
                g.add_edge(a, b)
    if g.number_of_nodes() < 3:
        return dict(
            cycle_circularity=None,
            cycle_direction=None,
            cycle_order_consistency=None,
            cycles=0,
            cycle_len=None,
        )
    # Deterministic root order so cycle_basis is stable across runs.
    bases = nx.cycle_basis(g, root=sorted(g.nodes())[0])
    bases = sorted((c for c in bases if len(c) >= 3), key=lambda c: (len(c), tuple(sorted(c))))[:max_cycles]
    if not bases:
        return dict(
            cycle_circularity=None,
            cycle_direction=None,
            cycle_order_consistency=None,
            cycles=0,
            cycle_len=None,
        )
    circs, orders, winds = [], [], []
    for cyc in bases:
        pts = np.array([mids[r] for r in cyc if r in mids], float)
        if len(pts) < 3:
            continue
        # order points by topology along the cycle
        area, per = _polygon_area_perimeter(pts)
        circs.append((4.0 * math.pi * area / (per * per)) if per > 1e-9 else 0.0)
        winds.append(_winding(pts))
        # angular neighbour consistency
        cen = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
        order_ang = list(np.argsort(ang))
        n = len(pts)
        # map topological i -> angular rank neighbours
        ok = 0
        for i in range(n):
            topo_nb = {(i - 1) % n, (i + 1) % n}
            pos = order_ang.index(i)
            ang_nb = {order_ang[(pos - 1) % n], order_ang[(pos + 1) % n]}
            ok += len(topo_nb & ang_nb) / 2.0
        orders.append(ok / n)
    if not circs:
        return dict(cycle_circularity=None, cycle_direction=None, cycle_order_consistency=None, cycles=0, cycle_len=None)
    # pick the most circular cycle for the headline direction / length
    best = int(np.argmax(circs))
    dir_counts = defaultdict(int)
    for w in winds:
        dir_counts[w] += 1
    main_dir = max(dir_counts, key=dir_counts.get)
    return dict(
        cycle_circularity=round(float(np.mean(circs)), 4),
        cycle_direction=main_dir,
        cycle_direction_consistency=round(dir_counts[main_dir] / len(winds), 4),
        cycle_order_consistency=round(float(np.mean(orders)), 4),
        cycles=len(circs),
        cycle_len=len(bases[best]),
        note="isoperimetric quotient over reaction midmarkers of simple non-currency cycles",
    )


def exchange_periphery_share(body: dict, band: float = PERIPHERY_BAND) -> dict[str, Any]:
    """Share of exchange/demand reactions that sit in the outer band of their component hull."""
    import networkx as nx

    mids = _midmarkers(body)
    nodes = body["nodes"]
    # component graph via non-currency metabolites
    met_rx: dict[str, set[str]] = defaultdict(set)
    for rx in body["reactions"].values():
        rid = rx["bigg_id"]
        for s in rx["segments"].values():
            for nid in (s["from_node_id"], s["to_node_id"]):
                n = nodes.get(nid) or {}
                if n.get("node_type") == "metabolite" and not _is_currency_id(n.get("bigg_id")):
                    met_rx[nid].add(rid)
    g = nx.Graph()
    g.add_nodes_from(mids)
    for rids in met_rx.values():
        rs = [r for r in rids if r in mids]
        for i, a in enumerate(rs):
            for b in rs[i + 1 :]:
                g.add_edge(a, b)

    def is_exchange(rid: str) -> bool:
        u = rid.upper()
        return u.startswith("EX_") or u.startswith("DM_") or u.startswith("SINK_") or u.startswith("R_EX_")

    exchanges = [r for r in mids if is_exchange(r)]
    if not exchanges:
        return dict(exchange_periphery_share=None, exchange_reactions=0, note="no exchange/demand reactions detected")
    in_band = 0
    scored = 0
    for comp in nx.connected_components(g):
        members = [r for r in comp if r in mids]
        ex = [r for r in members if r in set(exchanges)]
        if not ex or len(members) < 3:
            continue
        pts = np.array([mids[r] for r in members], float)
        cen = pts.mean(0)
        radii = np.linalg.norm(pts - cen, axis=1)
        rmax = float(radii.max()) if len(radii) else 0.0
        if rmax < 1e-9:
            continue
        thr = (1.0 - band) * rmax
        for r in ex:
            scored += 1
            d = float(np.hypot(*(np.array(mids[r]) - cen)))
            if d >= thr:
                in_band += 1
    if scored == 0:
        # fall back: whole-map hull
        pts = np.array([mids[r] for r in mids], float)
        cen = pts.mean(0)
        rmax = float(np.max(np.linalg.norm(pts - cen, axis=1)))
        thr = (1.0 - band) * rmax
        for r in exchanges:
            scored += 1
            if float(np.hypot(*(np.array(mids[r]) - cen))) >= thr:
                in_band += 1
    return dict(
        exchange_periphery_share=round(in_band / scored, 4) if scored else None,
        exchange_reactions=len(exchanges),
        scored=scored,
        periphery_band=band,
    )


def blank_space_from_escher(data, edges: list[dict], spacing: float) -> dict[str, Any]:
    """Call the single ``evaluation.metrics.blank_space`` implementation on final map geometry."""
    from evaluation.metrics import blank_space

    body = data[1]
    nodes = body["nodes"]
    points = [
        (float(n["x"]), float(n["y"]))
        for n in nodes.values()
        if n.get("node_type") in ("metabolite", "midmarker") and "x" in n
    ]
    segs = []
    for e in edges:
        pts = e["pts"]
        for a, b in zip(pts[:-1], pts[1:]):
            segs.append((a, b))
    # Label boxes: metabolite + reaction labels when present.
    boxes = []
    try:
        import drawing

        font = drawing.FONT
        hidden = set((data[0].get("metadata") or {}).get("hidden_metabolite_labels") or [])
        for nid, n in nodes.items():
            if n.get("node_type") != "metabolite" or "label_x" not in n or n.get("label_hidden") or nid in hidden:
                continue
            w = float(font.getlength(n.get("bigg_id") or ""))
            boxes.append((n["label_x"] - 2, n["label_y"] - 14, n["label_x"] + w + 2, n["label_y"] + 4))
        for rx in body["reactions"].values():
            if "label_x" not in rx or rx.get("label_hidden"):
                continue
            w = float(font.getlength(rx.get("bigg_id") or ""))
            boxes.append((rx["label_x"] - 2, rx["label_y"] - 14, rx["label_x"] + w + 2, rx["label_y"] + 4))
    except Exception:
        boxes = []
    return blank_space(points, segs, boxes, spacing=spacing)


def unrelated_overlaps_from_escher(data, edges: list[dict]) -> dict[str, Any]:
    """Unrelated reaction overlaps on the final Escher map (via ``metrics.unrelated_overlaps``).

    Collinear checks use each connection's chord (endpoints only) so cost stays O(E²) on
    connections, not on Bézier sample pieces. Edge-through-node still walks sampled pieces.
    """
    from evaluation.metrics import unrelated_overlaps

    body = data[1]
    nodes = body["nodes"]
    # reaction → non-currency metabolite ids
    rx_mets: dict[str, set[str]] = defaultdict(set)
    for rx in body["reactions"].values():
        rid = rx["bigg_id"]
        for m in rx.get("metabolites") or []:
            if isinstance(m, dict) and m.get("bigg_id") and not _is_currency_id(m["bigg_id"]):
                rx_mets[rid].add(m["bigg_id"])
    related: set[tuple[str, str]] = set()
    rids = sorted(rx_mets)
    by_met: dict[str, list[str]] = defaultdict(list)
    for rid, mets in rx_mets.items():
        for m in mets:
            by_met[m].append(rid)
    for rs in by_met.values():
        rs = sorted(rs)
        for i, a in enumerate(rs):
            for b in rs[i + 1 :]:
                related.add((a, b))

    # Own nodes per reaction: midmarker + its metabolite glyphs on drawn edges.
    node_xy: dict[str, tuple[float, float]] = {}
    node_reaction: dict[str, str] = {}
    for e in edges:
        rid = e["rid"]
        for nid in e["ends"]:
            n = nodes.get(nid) or {}
            if "x" not in n:
                continue
            # A glyph may serve several reactions; keep first owner for node-on-node (conservative).
            if nid not in node_reaction:
                node_xy[nid] = (float(n["x"]), float(n["y"]))
                node_reaction[nid] = rid
    # Sampled pieces for edge-through-node; one chord per connection for collinear overlaps.
    chords = []
    pieces = []
    for e in edges:
        pts = e["pts"]
        if len(pts) < 2:
            continue
        chords.append((e["rid"], e["ends"][0], np.asarray(pts[0], float), np.asarray(pts[-1], float)))
        for a, b in zip(pts[:-1], pts[1:]):
            pieces.append((e["rid"], e["ends"][0], np.asarray(a, float), np.asarray(b, float)))
    return unrelated_overlaps(node_xy, node_reaction, pieces, related, chords=chords)


def content_bbox(body: dict, canvas: dict | None = None) -> dict[str, float | None]:
    """Axis-aligned bbox of metabolite + midmarker nodes (one definition for size / aspect / packing).

    Map size, aspect ratio (W/H) and blank-space utilisation all use this content box.
    Falls back to the Escher canvas only when fewer than two content nodes exist.
    """
    nodes = body["nodes"]
    xy = np.array(
        [[n["x"], n["y"]] for n in nodes.values() if n.get("node_type") in ("metabolite", "midmarker") and "x" in n],
        float,
    )
    if len(xy) < 2:
        if canvas and canvas.get("width") and canvas.get("height"):
            w, h = float(canvas["width"]), float(canvas["height"])
            aspect = w / max(h, 1e-9)
            in_band = ASPECT_MIN - 1e-9 <= aspect <= ASPECT_MAX + 1e-9
            return dict(
                width=round(w, 1),
                height=round(h, 1),
                aspect_ratio=round(aspect, 4),
                aspect_ratio_target=ASPECT_RATIO_TARGET,
                aspect_ratio_vs_target=0.0 if in_band else round(min(abs(aspect - ASPECT_MIN), abs(aspect - ASPECT_MAX)), 4),
                aspect_in_band=in_band,
                source="canvas",
            )
        return dict(width=None, height=None, aspect_ratio=None, aspect_ratio_target=ASPECT_RATIO_TARGET,
                    aspect_ratio_vs_target=None, aspect_in_band=None, source=None)
    w = float(xy[:, 0].max() - xy[:, 0].min())
    h = float(xy[:, 1].max() - xy[:, 1].min())
    aspect = w / max(h, 1e-9)
    in_band = ASPECT_MIN - 1e-9 <= aspect <= ASPECT_MAX + 1e-9
    return dict(
        width=round(w, 1),
        height=round(h, 1),
        aspect_ratio=round(aspect, 4),
        aspect_ratio_target=ASPECT_RATIO_TARGET,
        aspect_ratio_vs_target=0.0 if in_band else round(min(abs(aspect - ASPECT_MIN), abs(aspect - ASPECT_MAX)), 4),
        aspect_in_band=in_band,
        source="content_nodes",
    )


def aspect_ratio_wh(body: dict, canvas: dict | None = None) -> dict[str, float | None]:
    """Width / height of the content bbox (one definition everywhere)."""
    return content_bbox(body, canvas)


def species_in_other_compartment(body: dict) -> float | None:
    """Share of metabolite glyphs inside another compartment's hull (map-native ids).

    Delegates to ``official_metrics.species_in_other_compartment`` (ConvexHull +
    ray casting; no optional shapely).
    """
    import official_metrics as om

    return om.species_in_other_compartment([{}, body])


def collect_map_rule_details(
    data,
    edges: list[dict],
    lengths: np.ndarray,
    spacing: float,
    subsystems: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bundle transfer-check / layout-rule detail metrics for the shared evaluator.

    ``subsystems`` is accepted for call-site compatibility but unused (subsystem
    convention rows were dropped from the Layout quality panel).
    """
    del subsystems  # kept in the signature so callers need not change
    body = data[1]
    canvas = body.get("canvas") or {}
    mids = _midmarkers(body)
    out = {}
    out.update(currency_clone_metrics(edges, mids, spacing))
    out.update(bends_per_arc(edges))
    out["edge_len_max_rel"] = edge_len_max_rel(lengths)
    out.update(flow_direction_metrics(body))
    out.update(cycle_metrics(body))
    out.update(orderliness_metrics(body, edges, spacing))
    out.update(exchange_periphery_share(body))
    blank = blank_space_from_escher(data, edges, spacing)
    out.update(blank)
    # Space utilisation is the complement of blank_space_share on the same raster.
    bs = blank.get("blank_space_share")
    out["space_utilisation"] = round(1.0 - bs, 4) if bs is not None else None
    out.update(unrelated_overlaps_from_escher(data, edges))
    out.update(content_bbox(body, canvas))
    out["species_in_other_compartment"] = (
        round(v, 4) if (v := species_in_other_compartment(body)) is not None else None
    )
    out["thresholds_note"] = (
        f"official participant_consistency >{PARTICIPANT_CONSISTENCY_DEG}° on the panel; "
        f"orientation diagnostic still reports ≥{OPPOSITE_DEG}° under details.orientation; "
        "hub/long currency thresholds are relative to median reaction spacing; "
        "blank_space / unrelated_overlaps follow MetaCarto ideas (CC BY 4.0; reimplemented, not copied); "
        "map size, aspect W/H (4:3–16:9 band) and blank/utilisation share one content-node bbox; "
        f"orderliness axis tolerance ±{AXIS_ALIGN_DEG:g}° (count-weighted gate; length-weighted reported); "
        f"grid tolerance {GRID_TOL_NN_FRAC:g} × median nearest-neighbour "
        f"main-metabolite distance (floor {GRID_TOL_FLOOR_PX:g} px)"
    )
    return out
