"""Official layout-benchmark metric definitions, computed in this repository.

Clean-room reimplementation from the published definitions in
https://github.com/elixir2026-project22/layout-benchmark (MIT, pin 837067d).
No gdMetriX code is copied or vendored (GPL-3.0). General measures follow
Purchase / gdMetriX *definitions* only; pathway measures follow the benchmark's
documented rules for Escher drawings.

One shared implementation for the quality gate, the Layout quality panel, and
``evaluation.score`` / ``evaluation.score_map``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

# Escher 1.8 default radii (layout-benchmark ESCHER_RADIUS).
ESCHER_RADIUS = {"primary": 20.0, "secondary": 10.0, "marker": 5.0}

# Content aspect band used by packing / the Layout quality panel (W/H).
ASPECT_MIN = 4.0 / 3.0
ASPECT_MAX = 16.0 / 9.0

PARTICIPANT_DEG = 90.0  # opposite sides: directions more than 90° apart (dot < 0)


def _body(data) -> dict:
    return data[1] if isinstance(data, list) else data


def _point_key(p: Sequence[float]) -> str:
    return f"@{float(p[0]):.6g},{float(p[1]):.6g}"


def _reaction_mid(nodes: dict, reaction: dict) -> str | None:
    import networkx as nx

    g = nx.Graph()
    for seg in (reaction.get("segments") or {}).values():
        g.add_edge(seg["from_node_id"], seg["to_node_id"])
    return next((n for n in g if nodes.get(n, {}).get("node_type") == "midmarker"), None)


def _xy(node: dict) -> tuple[float, float]:
    return float(node["x"]), float(node["y"])


def escher_drawn_edges(data) -> list[dict[str, Any]]:
    """One drawn participant edge per mid↔metabolite path (markers as route bends).

    Matches layout-benchmark ``escher_drawing`` geometry: polyline through
    multimarkers, Bézier controls ignored. Shared geometric pieces are later
    deduplicated by ``piece_segments``.
    """
    import networkx as nx

    body = _body(data)
    nodes = body["nodes"]
    edges = []
    for rx in body.get("reactions", {}).values():
        mid = _reaction_mid(nodes, rx)
        if mid is None:
            continue
        g = nx.Graph()
        for seg in (rx.get("segments") or {}).values():
            g.add_edge(seg["from_node_id"], seg["to_node_id"])
        coefficient = {m["bigg_id"]: m["coefficient"] for m in rx.get("metabolites") or []}
        for nid in g:
            n = nodes.get(nid) or {}
            if n.get("node_type") != "metabolite":
                continue
            if not nx.has_path(g, mid, nid):
                continue
            path = nx.shortest_path(g, mid, nid)
            pts = [_xy(nodes[k]) for k in path if k in nodes]
            if len(pts) < 2:
                continue
            bigg = n.get("bigg_id")
            coef = coefficient.get(bigg, 0)
            role = "product" if coef > 0 else "reactant"
            edges.append(
                dict(
                    rid=rx.get("bigg_id"),
                    met=bigg,
                    role=role,
                    mid=mid,
                    met_nid=nid,
                    pts=pts,
                    primary=bool(n.get("node_is_primary", True)),
                )
            )
    return edges


def piece_segments(edges: Sequence[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Unique straight pieces of drawn edges (shared pieces once)."""
    seen: set[tuple[str, str]] = set()
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for e in edges:
        pts = e["pts"]
        for a, b in zip(pts[:-1], pts[1:]):
            if a == b:
                continue
            ka, kb = _point_key(a), _point_key(b)
            key = (ka, kb) if ka <= kb else (kb, ka)
            if key in seen:
                continue
            seen.add(key)
            segs.append((a, b))
    return segs


def _proper_intersect(a, b, c, d) -> tuple[bool, tuple[float, float] | None]:
    """Proper open-segment intersection (endpoints excluded)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    dx, dy = float(d[0]), float(d[1])
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return False, None
    qx, qy = cx - ax, cy - ay
    t = (qx * sy - qy * sx) / den
    u = (qx * ry - qy * rx) / den
    if 1e-9 < t < 1.0 - 1e-9 and 1e-9 < u < 1.0 - 1e-9:
        return True, (ax + t * rx, ay + t * ry)
    return False, None


def count_piece_crossings(segs: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> int:
    """Number of crossings among unique pieces; multi-way point → each pair once.

    Adjacent pieces that share an endpoint never cross. Matches the layout-benchmark
    / gdMetriX *definition* of ``number_of_crossings`` on the segment graph.
    """
    n = len(segs)
    if n < 2:
        return 0
    # Spatial hash to keep mid/large maps interactive.
    cell = 80.0
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(segs):
        lo_x, hi_x = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
        lo_y, hi_y = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
        for gx in range(int(math.floor(lo_x / cell)), int(math.floor(hi_x / cell)) + 1):
            for gy in range(int(math.floor(lo_y / cell)), int(math.floor(hi_y / cell)) + 1):
                buckets[gx, gy].append(i)
    count = 0
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        m = len(members)
        if m < 2:
            continue
        for ii in range(m):
            i = members[ii]
            a, b = segs[i]
            ends_i = {_point_key(a), _point_key(b)}
            for jj in range(ii + 1, m):
                j = members[jj]
                key = (i, j) if i < j else (j, i)
                if key in checked:
                    continue
                checked.add(key)
                c, d = segs[j]
                if ends_i & {_point_key(c), _point_key(d)}:
                    continue
                hit, _ = _proper_intersect(a, b, c, d)
                if hit:
                    count += 1
    return count


def edge_length_deviation(edges: Sequence[dict]) -> float | None:
    """Purchase edge-length deviation on drawn polyline lengths (benchmark definition)."""
    lengths = []
    for e in edges:
        pts = np.asarray(e["pts"], float)
        if len(pts) < 2:
            continue
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if L > 0:
            lengths.append(L)
    if not lengths:
        return None
    mean = float(np.mean(lengths))
    if mean <= 0:
        return None
    return float(np.mean([abs(mean - x) / mean for x in lengths]))


def edge_orthogonality(segs: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> float | None:
    """Purchase edge orthogonality on unique pieces (1 = all axis-parallel)."""
    if not segs:
        return None
    total = 0.0
    for a, b in segs:
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        if abs(dx) < 1e-15 and abs(dy) < 1e-15:
            continue
        angle = abs(math.atan2(dy, dx))  # [0, π]
        # Deviation from nearest H/V, normalised by π/4 so max deviation → 1.
        deg = angle % (math.pi / 2)
        deg = min(deg, math.pi / 2 - deg)
        total += deg * 4.0 / math.pi
    n = len(segs)
    return float(1.0 - total / n) if n else None


def _glyph_radius(node: dict) -> float | None:
    t = node.get("node_type")
    if t == "metabolite":
        return ESCHER_RADIUS["primary" if node.get("node_is_primary", True) else "secondary"]
    if t == "midmarker":
        return ESCHER_RADIUS["marker"]
    return None


def escher_glyphs(data) -> list[dict[str, Any]]:
    """Drawn metabolite + midmarker glyphs with Escher radii."""
    body = _body(data)
    out = []
    for nid, n in body.get("nodes", {}).items():
        r = _glyph_radius(n)
        if r is None or "x" not in n:
            continue
        out.append(dict(id=nid, x=float(n["x"]), y=float(n["y"]), r=float(r), kind=n.get("node_type"),
                        bigg=n.get("bigg_id"), primary=n.get("node_is_primary")))
    return out


def _label_boxes(data) -> list[dict[str, Any]]:
    """Axis-aligned label boxes with owning node ids (exclude self glyph–label pairs)."""
    import drawing

    font = drawing.FONT
    body = _body(data)
    meta = (data[0].get("metadata") if isinstance(data, list) else None) or {}
    hidden = set(meta.get("hidden_metabolite_labels") or [])
    boxes = []
    for nid, n in body.get("nodes", {}).items():
        if n.get("node_type") != "metabolite":
            continue
        if "label_x" not in n or n.get("label_hidden") or nid in hidden:
            continue
        w = font.getlength(n.get("bigg_id", ""))
        boxes.append(dict(
            box=(n["label_x"] - 2, n["label_y"] - 14, n["label_x"] + w + 2, n["label_y"] + 4),
            owners={nid},
        ))
    for rx in body.get("reactions", {}).values():
        if "label_x" not in rx or rx.get("label_hidden"):
            continue
        mid = _reaction_mid(body["nodes"], rx)
        w = font.getlength(rx.get("bigg_id", ""))
        owners = {mid} if mid else set()
        boxes.append(dict(
            box=(rx["label_x"] - 2, rx["label_y"] - 14, rx["label_x"] + w + 2, rx["label_y"] + 4),
            owners=owners,
        ))
    return boxes


def _circle_overlap_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2 or d <= 0:
        return 0.0 if d >= r1 + r2 else math.pi * min(r1, r2) ** 2
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    r1_sq, r2_sq = r1 * r1, r2 * r2
    alpha = math.acos(np.clip((d * d + r1_sq - r2_sq) / (2 * d * r1), -1, 1))
    beta = math.acos(np.clip((d * d + r2_sq - r1_sq) / (2 * d * r2), -1, 1))
    return r1_sq * alpha + r2_sq * beta - 0.5 * (
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    ) ** 0.5


def _box_overlap_area(a, b) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy


def _circle_box_overlap_area(x: float, y: float, r: float, box) -> float:
    """Approximate circle∩AABB area via clipped sample grid (deterministic)."""
    x0, y0, x1, y1 = box
    if x + r <= x0 or x - r >= x1 or y + r <= y0 or y - r >= y1:
        return 0.0
    # Exact enough for gate inequality: positive area iff shapes intersect with area > 0.
    # Use analytic AABB–circle intersection test first.
    cx = min(max(x, x0), x1)
    cy = min(max(y, y0), y1)
    if (cx - x) ** 2 + (cy - y) ** 2 >= r * r - 1e-12:
        return 0.0
    # Sample a small grid for area share (gate only needs pair counts; area is optional).
    n = 8
    hits = 0
    for i in range(n):
        for j in range(n):
            px = x0 + (i + 0.5) / n * (x1 - x0)
            py = y0 + (j + 0.5) / n * (y1 - y0)
            if (px - x) ** 2 + (py - y) ** 2 < r * r:
                hits += 1
    return (hits / (n * n)) * max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def glyph_overlaps(data, *, include_labels: bool = False) -> dict[str, float | int | None]:
    """Overlapping pairs at Escher glyph sizes (touching is not overlapping).

    ``include_labels=True`` also counts label–label and glyph–label pairs (quality gate).
    Official scorer uses glyphs only.
    """
    glyphs = escher_glyphs(data)
    if len(glyphs) < 2 and not include_labels:
        return dict(overlapping_pairs=0, overlap_area_share=0.0, glyph_pairs=0, label_pairs=0, glyph_label_pairs=0)
    pairs = 0
    area = 0.0
    # Glyph–glyph via KD-tree on centres with max radius query.
    max_r = max((g["r"] for g in glyphs), default=0.0)
    xy = np.array([[g["x"], g["y"]] for g in glyphs], float) if glyphs else np.zeros((0, 2))
    if len(glyphs) >= 2:
        tree = cKDTree(xy)
        for i, j in tree.query_pairs(2.0 * max_r + 1e-6):
            d = float(np.hypot(*(xy[i] - xy[j])))
            r1, r2 = glyphs[i]["r"], glyphs[j]["r"]
            if d < r1 + r2 - 1e-9:  # positive area
                pairs += 1
                area += _circle_overlap_area(r1, r2, d)
    glyph_pairs = pairs
    label_pairs = 0
    glyph_label_pairs = 0
    boxes = _label_boxes(data) if include_labels else []
    if include_labels and boxes:
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                inter = _box_overlap_area(boxes[i]["box"], boxes[j]["box"])
                if inter > 0:
                    label_pairs += 1
                    area += inter
                    pairs += 1
        for g in glyphs:
            for lb in boxes:
                if g["id"] in lb["owners"]:
                    continue  # a label may sit on its own glyph
                inter = _circle_box_overlap_area(g["x"], g["y"], g["r"], lb["box"])
                if inter > 0:
                    glyph_label_pairs += 1
                    area += inter
                    pairs += 1
    total_area = sum(math.pi * g["r"] ** 2 for g in glyphs)
    if include_labels:
        total_area += sum(
            max(0.0, b["box"][2] - b["box"][0]) * max(0.0, b["box"][3] - b["box"][1]) for b in boxes
        )
    return dict(
        overlapping_pairs=int(pairs),
        overlap_area_share=(area / total_area) if total_area else 0.0,
        glyph_pairs=int(glyph_pairs),
        label_pairs=int(label_pairs),
        glyph_label_pairs=int(glyph_label_pairs),
    )


def participant_consistency(data) -> float | None:
    """Share of (reactant, product) pairs whose first pieces at the reaction are >90° apart."""
    edges = escher_drawn_edges(data)
    by_rx: dict[str, list] = defaultdict(list)
    for e in edges:
        by_rx[e["rid"]].append(e)
    opposite = pairs = 0
    body = _body(data)
    nodes = body["nodes"]
    for rid, elist in by_rx.items():
        mid = elist[0]["mid"]
        c = np.array(_xy(nodes[mid]), float)
        ins = []
        outs = []
        for e in elist:
            pts = e["pts"]
            if len(pts) < 2:
                continue
            # Direction of the first piece leaving the reaction node.
            v = np.array(pts[1], float) - c
            if e["role"] == "reactant":
                # Reactant edge is stored mid→met; first piece still leaves the mid toward the met.
                # Official uses the piece at the reaction along the consumption edge: for
                # reactant the polyline is met→…→mid, last piece into mid = mid←pts[-2].
                # Our pts are mid→…→met, so the first piece mid→pts[1] is the same direction
                # the reaction "sees" toward that participant.
                ins.append(v)
            else:
                outs.append(v)
        for a in ins:
            for b in outs:
                pairs += 1
                opposite += float(np.dot(a, b)) < 0
    return opposite / pairs if pairs else None


def _point_in_hull(hull_pts: np.ndarray, x: float, y: float) -> bool:
    """Strict containment in a convex hull (scipy vertices, no shapely)."""
    # Ray casting on the hull polygon.
    n = len(hull_pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = hull_pts[i]
        xj, yj = hull_pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def species_in_other_compartment(data) -> float | None:
    """Share of metabolite glyphs whose centre lies inside another compartment's hull.

    Compartment id is taken from the BiGG suffix ``_c``, ``_e``, … . Implemented with
    scipy ConvexHull + ray casting (no optional shapely import).
    """
    import re

    body = _body(data)
    by_comp: dict[str, list[tuple[float, float]]] = defaultdict(list)
    members: list[tuple[str, float, float]] = []
    for n in body.get("nodes", {}).values():
        if n.get("node_type") != "metabolite":
            continue
        mid = n.get("bigg_id") or ""
        m = re.search(r"_([a-z][0-9]?)$", mid)
        if not m:
            continue
        c = m.group(1)
        xy = (float(n["x"]), float(n["y"]))
        by_comp[c].append(xy)
        members.append((c, xy[0], xy[1]))
    if len(by_comp) < 2 or not members:
        return None
    hulls: dict[str, np.ndarray] = {}
    for c, pts in by_comp.items():
        arr = np.asarray(pts, float)
        if len(arr) < 3:
            continue
        try:
            hulls[c] = arr[ConvexHull(arr).vertices]
        except Exception:
            continue
    if not hulls:
        return None
    intruders = sum(
        1
        for c, x, y in members
        if any(oc != c and _point_in_hull(h, x, y) for oc, h in hulls.items())
    )
    return intruders / len(members)


def content_aspect(data) -> dict[str, Any]:
    """W/H of metabolite+midmarker bbox, with 4:3–16:9 band pass/fail."""
    body = _body(data)
    xy = np.array(
        [[n["x"], n["y"]] for n in body.get("nodes", {}).values()
         if n.get("node_type") in ("metabolite", "midmarker") and "x" in n],
        float,
    )
    if len(xy) < 2:
        canvas = body.get("canvas") or {}
        w = float(canvas.get("width") or 0)
        h = float(canvas.get("height") or 0)
        if w <= 0 or h <= 0:
            return dict(aspect_ratio=None, aspect_in_band=None, aspect_min=ASPECT_MIN, aspect_max=ASPECT_MAX)
    else:
        w = float(xy[:, 0].max() - xy[:, 0].min())
        h = float(xy[:, 1].max() - xy[:, 1].min())
    aspect = w / max(h, 1e-9)
    in_band = ASPECT_MIN - 1e-9 <= aspect <= ASPECT_MAX + 1e-9
    return dict(
        width=round(w, 1),
        height=round(h, 1),
        aspect_ratio=round(aspect, 4),
        aspect_min=ASPECT_MIN,
        aspect_max=ASPECT_MAX,
        aspect_in_band=bool(in_band),
        aspect_band_label="4:3–16:9",
    )


def score_escher_map(data) -> dict[str, Any]:
    """All official figures derived from one Escher map (no SBML required)."""
    edges = escher_drawn_edges(data)
    segs = piece_segments(edges)
    ov = glyph_overlaps(data, include_labels=False)
    ov_gate = glyph_overlaps(data, include_labels=True)
    aspect = content_aspect(data)
    return dict(
        crossings=count_piece_crossings(segs),
        pieces=len(segs),
        drawn_edges=len(edges),
        overlapping_pairs=ov["overlapping_pairs"],
        overlap_area_share=ov["overlap_area_share"],
        gate_overlapping_pairs=ov_gate["overlapping_pairs"],
        gate_glyph_pairs=ov_gate["glyph_pairs"],
        gate_label_pairs=ov_gate["label_pairs"],
        gate_glyph_label_pairs=ov_gate["glyph_label_pairs"],
        participant_consistency=participant_consistency(data),
        edge_length_deviation=edge_length_deviation(edges),
        edge_orthogonality=edge_orthogonality(segs),
        species_in_other_compartment=species_in_other_compartment(data),
        **aspect,
        note="official layout-benchmark definitions (MIT); gdMetriX not used",
    )


__all__ = [
    "ASPECT_MAX",
    "ASPECT_MIN",
    "ESCHER_RADIUS",
    "PARTICIPANT_DEG",
    "content_aspect",
    "count_piece_crossings",
    "edge_length_deviation",
    "edge_orthogonality",
    "escher_drawn_edges",
    "escher_glyphs",
    "glyph_overlaps",
    "participant_consistency",
    "piece_segments",
    "score_escher_map",
    "species_in_other_compartment",
]
