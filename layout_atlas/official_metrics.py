"""Official layout quality figures for Escher maps, computed in this repository.

Implements the published *definitions* from the layout-benchmark project
(https://github.com/elixir2026-project22/layout-benchmark, MIT, pin 837067d)
and from Purchase's graph-drawing measures. This file is an independent
implementation: no functions are copied from layout-benchmark or gdMetriX
(GPL-3.0 is never vendored).

Used by the quality gate, the Layout quality panel, and ``evaluation.score``.
Optional cross-checks against an installed ``layout_benchmark`` package live
only in ``evaluation.score_map``.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

# Escher 1.8 default radii (primary / secondary / midmarker).
ESCHER_RADIUS = {"primary": 20.0, "secondary": 10.0, "marker": 5.0}

# Packing / panel content aspect band (width / height).
ASPECT_MIN = 4.0 / 3.0
ASPECT_MAX = 16.0 / 9.0

# Opposite-side participants: directions more than 90° apart.
PARTICIPANT_DEG = 90.0


def _map_body(data) -> dict:
    return data[1] if isinstance(data, list) else data


def _xy(node: dict) -> tuple[float, float]:
    return float(node["x"]), float(node["y"])


def _endpoint_id(p: Sequence[float]) -> tuple[float, float]:
    """Stable geometric identity for a polyline vertex (independent rounding)."""
    return (round(float(p[0]), 6), round(float(p[1]), 6))


def _midmarker_id(nodes: dict, reaction: dict) -> str | None:
    import networkx as nx

    g = nx.Graph()
    for seg in (reaction.get("segments") or {}).values():
        g.add_edge(seg["from_node_id"], seg["to_node_id"])
    for nid in g:
        if (nodes.get(nid) or {}).get("node_type") == "midmarker":
            return nid
    return None


def escher_drawn_edges(data) -> list[dict[str, Any]]:
    """Participant polylines on an Escher map (reaction mid → metabolite).

    Each metabolite attached to a reaction contributes one polyline that walks
    through multimarker bends. Bézier control points are ignored (straight
    pieces only). Shared geometry is deduplicated later by ``piece_segments``.
    """
    import networkx as nx

    body = _map_body(data)
    nodes = body["nodes"]
    out: list[dict[str, Any]] = []
    for reaction in body.get("reactions", {}).values():
        mid = _midmarker_id(nodes, reaction)
        if mid is None:
            continue
        graph = nx.Graph(
            (seg["from_node_id"], seg["to_node_id"])
            for seg in (reaction.get("segments") or {}).values()
        )
        coef = {m["bigg_id"]: m["coefficient"] for m in reaction.get("metabolites") or []}
        for nid, node in ((n, nodes[n]) for n in graph if n in nodes):
            if node.get("node_type") != "metabolite":
                continue
            if not nx.has_path(graph, mid, nid):
                continue
            chain = nx.shortest_path(graph, mid, nid)
            pts = [_xy(nodes[k]) for k in chain if k in nodes]
            if len(pts) < 2:
                continue
            bigg = node.get("bigg_id")
            out.append(
                dict(
                    rid=reaction.get("bigg_id"),
                    met=bigg,
                    role="product" if coef.get(bigg, 0) > 0 else "reactant",
                    mid=mid,
                    met_nid=nid,
                    pts=pts,
                    primary=bool(node.get("node_is_primary", True)),
                )
            )
    return out


def piece_segments(edges: Sequence[dict]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Unique straight pieces across drawn edges (a shared piece counts once)."""
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for edge in edges:
        pts = edge["pts"]
        for a, b in zip(pts[:-1], pts[1:]):
            if a == b:
                continue
            ia, ib = _endpoint_id(a), _endpoint_id(b)
            key = (ia, ib) if ia <= ib else (ib, ia)
            if key in seen:
                continue
            seen.add(key)
            segs.append((a, b))
    return segs


def _segments_cross(a, b, c, d) -> bool:
    """True when open segments ab and cd properly intersect (shared ends excluded)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    dx, dy = float(d[0]), float(d[1])
    abx, aby = bx - ax, by - ay
    cdx, cdy = dx - cx, dy - cy
    den = abx * cdy - aby * cdx
    if abs(den) < 1e-12:
        return False
    acx, acy = cx - ax, cy - ay
    t = (acx * cdy - acy * cdx) / den
    u = (acx * aby - acy * abx) / den
    return 1e-9 < t < 1.0 - 1e-9 and 1e-9 < u < 1.0 - 1e-9


def count_piece_crossings(segs: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> int:
    """Count proper crossings among unique pieces (each crossing pair once).

    Pieces that share an endpoint are adjacent route bends, not crossings.
    """
    n = len(segs)
    if n < 2:
        return 0
    cell = 80.0
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(segs):
        x0, x1 = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
        y0, y1 = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
        for gx in range(int(math.floor(x0 / cell)), int(math.floor(x1 / cell)) + 1):
            for gy in range(int(math.floor(y0 / cell)), int(math.floor(y1 / cell)) + 1):
                buckets[gx, gy].append(i)
    total = 0
    seen_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        m = len(members)
        for ii in range(m):
            i = members[ii]
            a, b = segs[i]
            ends_i = {_endpoint_id(a), _endpoint_id(b)}
            for jj in range(ii + 1, m):
                j = members[jj]
                pair = (i, j) if i < j else (j, i)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                c, d = segs[j]
                if ends_i & {_endpoint_id(c), _endpoint_id(d)}:
                    continue
                if _segments_cross(a, b, c, d):
                    total += 1
    return total


def edge_length_deviation(edges: Sequence[dict]) -> float | None:
    """Mean relative deviation of drawn polyline lengths from their mean."""
    lengths = []
    for edge in edges:
        pts = np.asarray(edge["pts"], float)
        if len(pts) < 2:
            continue
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if length > 0:
            lengths.append(length)
    if not lengths:
        return None
    mean = float(np.mean(lengths))
    if mean <= 0:
        return None
    return float(np.mean([abs(mean - length) / mean for length in lengths]))


def edge_orthogonality(segs: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> float | None:
    """1 when every unique piece is axis-aligned; lower when angles drift from H/V."""
    if not segs:
        return None
    penalty = 0.0
    for a, b in segs:
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        if abs(dx) < 1e-15 and abs(dy) < 1e-15:
            continue
        angle = abs(math.atan2(dy, dx)) % (math.pi / 2)
        angle = min(angle, math.pi / 2 - angle)
        penalty += angle * 4.0 / math.pi
    return float(1.0 - penalty / len(segs))


def _glyph_radius(node: dict) -> float | None:
    kind = node.get("node_type")
    if kind == "metabolite":
        return ESCHER_RADIUS["primary" if node.get("node_is_primary", True) else "secondary"]
    if kind == "midmarker":
        return ESCHER_RADIUS["marker"]
    return None


def escher_glyphs(data) -> list[dict[str, Any]]:
    """Metabolite and midmarker circles at Escher default radii."""
    body = _map_body(data)
    glyphs = []
    for nid, node in body.get("nodes", {}).items():
        radius = _glyph_radius(node)
        if radius is None or "x" not in node:
            continue
        glyphs.append(
            dict(
                id=nid,
                x=float(node["x"]),
                y=float(node["y"]),
                r=float(radius),
                kind=node.get("node_type"),
                bigg=node.get("bigg_id"),
                primary=node.get("node_is_primary"),
            )
        )
    return glyphs


def _label_boxes(data) -> list[dict[str, Any]]:
    """Axis-aligned label boxes; ``owners`` names glyphs that may cover their own text."""
    import drawing

    font = drawing.FONT
    body = _map_body(data)
    meta = (data[0].get("metadata") if isinstance(data, list) else None) or {}
    hidden = set(meta.get("hidden_metabolite_labels") or [])
    boxes = []
    for nid, node in body.get("nodes", {}).items():
        if node.get("node_type") != "metabolite":
            continue
        if "label_x" not in node or node.get("label_hidden") or nid in hidden:
            continue
        width = font.getlength(node.get("bigg_id", ""))
        boxes.append(
            dict(
                box=(node["label_x"] - 2, node["label_y"] - 14, node["label_x"] + width + 2, node["label_y"] + 4),
                owners={nid},
            )
        )
    for reaction in body.get("reactions", {}).values():
        if "label_x" not in reaction or reaction.get("label_hidden"):
            continue
        mid = _midmarker_id(body["nodes"], reaction)
        width = font.getlength(reaction.get("bigg_id", ""))
        boxes.append(
            dict(
                box=(reaction["label_x"] - 2, reaction["label_y"] - 14,
                     reaction["label_x"] + width + 2, reaction["label_y"] + 4),
                owners={mid} if mid else set(),
            )
        )
    return boxes


def _circle_overlap_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= 0 or d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    r1_sq, r2_sq = r1 * r1, r2 * r2
    alpha = math.acos(np.clip((d * d + r1_sq - r2_sq) / (2 * d * r1), -1, 1))
    beta = math.acos(np.clip((d * d + r2_sq - r1_sq) / (2 * d * r2), -1, 1))
    return r1_sq * alpha + r2_sq * beta - 0.5 * math.sqrt(
        max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    )


def _box_overlap_area(a, b) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return 0.0 if dx <= 0 or dy <= 0 else dx * dy


def _circle_box_overlap_area(x: float, y: float, r: float, box) -> float:
    x0, y0, x1, y1 = box
    if x + r <= x0 or x - r >= x1 or y + r <= y0 or y - r >= y1:
        return 0.0
    nearest_x = min(max(x, x0), x1)
    nearest_y = min(max(y, y0), y1)
    if (nearest_x - x) ** 2 + (nearest_y - y) ** 2 >= r * r - 1e-12:
        return 0.0
    # Deterministic grid estimate for area-share; pair counts only need positivity.
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
    """Count overlapping Escher glyphs (touching is allowed; positive area counts).

    With ``include_labels=True``, also count label–label and foreign glyph–label
    pairs for the quality gate. Official headline overlaps use glyphs only.
    """
    glyphs = escher_glyphs(data)
    empty = dict(overlapping_pairs=0, overlap_area_share=0.0, glyph_pairs=0, label_pairs=0, glyph_label_pairs=0)
    if len(glyphs) < 2 and not include_labels:
        return empty
    pairs = 0
    area = 0.0
    max_r = max((g["r"] for g in glyphs), default=0.0)
    centres = np.array([[g["x"], g["y"]] for g in glyphs], float) if glyphs else np.zeros((0, 2))
    if len(glyphs) >= 2:
        for i, j in cKDTree(centres).query_pairs(2.0 * max_r + 1e-6):
            dist = float(np.hypot(*(centres[i] - centres[j])))
            r1, r2 = glyphs[i]["r"], glyphs[j]["r"]
            if dist < r1 + r2 - 1e-9:
                pairs += 1
                area += _circle_overlap_area(r1, r2, dist)
    glyph_pairs = pairs
    label_pairs = glyph_label_pairs = 0
    boxes = _label_boxes(data) if include_labels else []
    if include_labels and boxes:
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                inter = _box_overlap_area(boxes[i]["box"], boxes[j]["box"])
                if inter > 0:
                    label_pairs += 1
                    area += inter
                    pairs += 1
        for glyph in glyphs:
            for box in boxes:
                if glyph["id"] in box["owners"]:
                    continue
                inter = _circle_box_overlap_area(glyph["x"], glyph["y"], glyph["r"], box["box"])
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
    """Share of reactant–product pairs leaving a reaction more than 90° apart."""
    by_reaction: dict[str, list] = defaultdict(list)
    for edge in escher_drawn_edges(data):
        by_reaction[edge["rid"]].append(edge)
    opposite = pairs = 0
    nodes = _map_body(data)["nodes"]
    cos_limit = math.cos(math.radians(PARTICIPANT_DEG))  # 90° → 0
    for edges in by_reaction.values():
        mid = edges[0]["mid"]
        origin = np.array(_xy(nodes[mid]), float)
        reactants = []
        products = []
        for edge in edges:
            if len(edge["pts"]) < 2:
                continue
            direction = np.array(edge["pts"][1], float) - origin
            (products if edge["role"] == "product" else reactants).append(direction)
        for u in reactants:
            nu = float(np.linalg.norm(u))
            if nu < 1e-12:
                continue
            u_hat = u / nu
            for v in products:
                nv = float(np.linalg.norm(v))
                if nv < 1e-12:
                    continue
                pairs += 1
                # Angle > 90° ⇔ cos < 0 for unit vectors.
                opposite += float(np.dot(u_hat, v / nv)) < cos_limit
    return opposite / pairs if pairs else None


def _point_in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    """Ray-cast containment test for a convex polygon."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def species_in_other_compartment(data) -> float | None:
    """Share of metabolite centres that fall inside another compartment's hull.

    Compartment ids come from BiGG suffixes (``_c``, ``_e``, …). Uses scipy
    ``ConvexHull`` plus ray casting — no shapely dependency.
    """
    body = _map_body(data)
    by_comp: dict[str, list[tuple[float, float]]] = defaultdict(list)
    members: list[tuple[str, float, float]] = []
    for node in body.get("nodes", {}).values():
        if node.get("node_type") != "metabolite":
            continue
        match = re.search(r"_([a-z][0-9]?)$", node.get("bigg_id") or "")
        if not match:
            continue
        comp = match.group(1)
        x, y = float(node["x"]), float(node["y"])
        by_comp[comp].append((x, y))
        members.append((comp, x, y))
    if len(by_comp) < 2 or not members:
        return None
    hulls: dict[str, np.ndarray] = {}
    for comp, pts in by_comp.items():
        arr = np.asarray(pts, float)
        if len(arr) < 3:
            continue
        try:
            hulls[comp] = arr[ConvexHull(arr).vertices]
        except Exception:
            continue
    if not hulls:
        return None
    intruders = sum(
        1
        for comp, x, y in members
        if any(other != comp and _point_in_polygon(poly, x, y) for other, poly in hulls.items())
    )
    return intruders / len(members)


def content_aspect(data) -> dict[str, Any]:
    """Content width/height ratio with a 4:3–16:9 pass band."""
    body = _map_body(data)
    xy = np.array(
        [
            [n["x"], n["y"]]
            for n in body.get("nodes", {}).values()
            if n.get("node_type") in ("metabolite", "midmarker") and "x" in n
        ],
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
    return dict(
        width=round(w, 1),
        height=round(h, 1),
        aspect_ratio=round(aspect, 4),
        aspect_min=ASPECT_MIN,
        aspect_max=ASPECT_MAX,
        aspect_in_band=bool(ASPECT_MIN - 1e-9 <= aspect <= ASPECT_MAX + 1e-9),
        aspect_band_label="4:3–16:9",
    )


def score_escher_map(data) -> dict[str, Any]:
    """All official figures for one Escher map (no SBML required)."""
    edges = escher_drawn_edges(data)
    segs = piece_segments(edges)
    glyphs_only = glyph_overlaps(data, include_labels=False)
    with_labels = glyph_overlaps(data, include_labels=True)
    aspect = content_aspect(data)
    return dict(
        crossings=count_piece_crossings(segs),
        pieces=len(segs),
        drawn_edges=len(edges),
        overlapping_pairs=glyphs_only["overlapping_pairs"],
        overlap_area_share=glyphs_only["overlap_area_share"],
        gate_overlapping_pairs=with_labels["overlapping_pairs"],
        gate_glyph_pairs=with_labels["glyph_pairs"],
        gate_label_pairs=with_labels["label_pairs"],
        gate_glyph_label_pairs=with_labels["glyph_label_pairs"],
        participant_consistency=participant_consistency(data),
        edge_length_deviation=edge_length_deviation(edges),
        edge_orthogonality=edge_orthogonality(segs),
        species_in_other_compartment=species_in_other_compartment(data),
        **aspect,
        note="independent implementation of published definitions; gdMetriX not used",
    )


# Internal aliases kept for in-repo call sites (evaluation helpers / tests).
_body = _map_body
_reaction_mid = _midmarker_id
_point_in_hull = _point_in_polygon


def _proper_intersect(a, b, c, d):
    return _segments_cross(a, b, c, d), None


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
