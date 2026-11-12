"""Layout quality metrics aligned with Project 22 layout-benchmark rules.

Pathway-specific metrics are adapted from layout-benchmark `metrics.py` (MIT).
General graph-drawing metrics come from gdMetriX when installed (GPL-3.0; optional
import only — never vendored).

``blank_space`` and ``unrelated_overlaps`` are clean-room reimplementations of ideas
from MetaCarto (`src/layout/metrics.py`, CC BY 4.0) — see docs/layout_evaluation.md.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import ConvexHull, cKDTree, procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .pathway import PRODUCT, REACTANT, SPECIES, Pathway

Positions = dict[str, tuple[float, float]]
SYMMETRY_MAX_NODES = 300
CROSSING_MAX_EDGES = 1200  # number_of_crossings is quadratic; skip on large GEMs
BLANK_CELL_FACTOR = 0.6  # cell size ≈ 0.6 × median reaction spacing (MetaCarto idea)
BLANK_CLEAR_SPACINGS = 1.0  # a cell is blank if no ink within ~1× spacing

try:
    from shapely import STRtree, box
    from shapely.geometry import Point, Polygon

    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover - optional for core installs
    _HAS_SHAPELY = False

try:
    import gdMetriX

    _HAS_GDMETRIX = True
except ImportError:  # pragma: no cover
    gdMetriX = None  # type: ignore
    _HAS_GDMETRIX = False


def gdmetrix_available() -> bool:
    return _HAS_GDMETRIX


def shapely_available() -> bool:
    return _HAS_SHAPELY


def general_metrics(pathway: Pathway, pos: Positions) -> dict[str, float | None]:
    """gdMetriX metrics on the undirected graph. Values are None when gdMetriX is missing."""
    if not _HAS_GDMETRIX:
        return {
            "crossings": None,
            "crossing_density": None,
            "crossing_angular_resolution": None,
            "angular_resolution": None,
            "edge_length_deviation": None,
            "edge_orthogonality": None,
            "stress": None,
            "aspect_ratio": None,
            "area_per_node": None,
            "homogeneity": None,
            "coherence_to_average_flow": None,
            "visual_symmetry": None,
            "gdmetrix": 0.0,
        }
    g = pathway.undirected().subgraph([n for n in pathway.undirected() if n in pos]).copy()
    # gdMetriX needs every graph node in pos
    missing = [n for n in g if n not in pos]
    g.remove_nodes_from(missing)
    n = g.number_of_nodes()
    if n < 2 or g.number_of_edges() < 1:
        return {k: None for k in (
            "crossings", "crossing_density", "crossing_angular_resolution", "angular_resolution",
            "edge_length_deviation", "edge_orthogonality", "stress", "aspect_ratio", "area_per_node",
            "homogeneity", "coherence_to_average_flow", "visual_symmetry",
        )} | {"gdmetrix": 1.0}
    directed = pathway.graph.subgraph([n for n in pathway.graph if n in pos]).copy()
    e = g.number_of_edges()
    do_cross = e <= CROSSING_MAX_EDGES

    def _f(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            return None

    m = {
        "crossings": _f(gdMetriX.number_of_crossings, g, pos=pos) if do_cross else None,
        "crossing_density": _f(gdMetriX.crossing_density, g, pos=pos) if do_cross else None,
        "crossing_angular_resolution": _f(gdMetriX.crossing_angular_resolution, g, pos=pos) if do_cross else None,
        "angular_resolution": _f(gdMetriX.angular_resolution, g, pos=pos),
        "edge_length_deviation": _f(gdMetriX.edge_length_deviation, g, pos=pos),
        "edge_orthogonality": _f(gdMetriX.edge_orthogonality, g, pos=pos),
        "stress": _f(gdMetriX.stress, g, pos=pos),
        # gdMetriX aspect is kept under a separate key; public aspect_ratio is always W/H.
        "gdmetrix_aspect_ratio": _f(gdMetriX.aspect_ratio, g, pos=pos),
        "aspect_ratio": None,
        "area_per_node": (lambda a: (a / max(n, 1)) if a is not None else None)(_f(gdMetriX.area, g, pos=pos)),
        "homogeneity": _f(gdMetriX.homogeneity, g, pos=pos),
        "coherence_to_average_flow": _f(gdMetriX.coherence_to_average_flow, directed, pos=pos),
        # visual_symmetry opens a matplotlib figure; keep it opt-in via env OPM_EVAL_SYMMETRY=1
        "visual_symmetry": (
            _f(gdMetriX.visual_symmetry, g, pos=pos)
            if n <= SYMMETRY_MAX_NODES and os.environ.get("OPM_EVAL_SYMMETRY") == "1"
            else None
        ),
        "gdmetrix": 1.0,
        "gdmetrix_crossings_computed": 1.0 if do_cross else 0.0,
    }
    return {k: (float(v) if v is not None else None) for k, v in m.items()}


def _boxes(pathway: Pathway, pos: Positions):
    nodes = [n for n in pathway.graph if n in pos]
    boxes = []
    for n in nodes:
        d = pathway.graph.nodes[n]
        x, y = pos[n]
        w, h = d["width"] / 2, d["height"] / 2
        boxes.append(box(x - w, y - h, x + w, y + h))
    return nodes, boxes


def overlaps(pathway: Pathway, pos: Positions) -> dict[str, float | None]:
    """Pairs of node boxes that overlap (touching is no overlap), and overlap area share."""
    if not _HAS_SHAPELY:
        return {"overlapping_pairs": None, "overlap_area_share": None}
    nodes, boxes_ = _boxes(pathway, pos)
    if not boxes_:
        return {"overlapping_pairs": 0.0, "overlap_area_share": 0.0}
    tree = STRtree(boxes_)
    pairs, area = 0, 0.0
    for i, b in enumerate(boxes_):
        for j in tree.query(b):
            if j > i:
                inter = b.intersection(boxes_[j]).area
                if inter > 0:
                    pairs += 1
                    area += inter
    total = sum(b.area for b in boxes_)
    return {"overlapping_pairs": float(pairs), "overlap_area_share": area / total if total else 0.0}


def participant_consistency(pathway: Pathway, pos: Positions) -> float | None:
    """Share of (reactant, product) pairs on opposite sides of their reaction (>90°)."""
    g = pathway.graph
    opposite, pairs = 0, 0
    for r in pathway.reactions:
        if r not in pos:
            continue
        c = np.array(pos[r], dtype=float)
        reactants = [np.array(pos[u]) - c for u, _, d in g.in_edges(r, data=True) if d.get("role") == REACTANT and u in pos]
        products = [np.array(pos[v]) - c for _, v, d in g.out_edges(r, data=True) if d.get("role") == PRODUCT and v in pos]
        for a in reactants:
            for b in products:
                pairs += 1
                opposite += float(np.dot(a, b)) < 0
    return opposite / pairs if pairs else None


def participant_consistency_details(pathway: Pathway, pos: Positions, limit: int = 40) -> list[dict[str, Any]]:
    """Worst reactions by opposite-side share (for localisation in the report)."""
    g = pathway.graph
    rows = []
    for r in pathway.reactions:
        if r not in pos:
            continue
        c = np.array(pos[r], dtype=float)
        reactants = [(u, np.array(pos[u]) - c) for u, _, d in g.in_edges(r, data=True) if d.get("role") == REACTANT and u in pos]
        products = [(v, np.array(pos[v]) - c) for _, v, d in g.out_edges(r, data=True) if d.get("role") == PRODUCT and v in pos]
        if not reactants or not products:
            continue
        pairs = opposite = 0
        for _, a in reactants:
            for _, b in products:
                pairs += 1
                opposite += float(np.dot(a, b)) < 0
        share = opposite / pairs
        rows.append(
            dict(
                reaction=r,
                label=g.nodes[r].get("label") or r,
                opposite_share=round(share, 3),
                pairs=pairs,
                x=round(float(pos[r][0]), 1),
                y=round(float(pos[r][1]), 1),
            )
        )
    rows.sort(key=lambda r: (r["opposite_share"], -r["pairs"], r["reaction"]))
    return rows[:limit]


def compartment_metrics(pathway: Pathway, pos: Positions) -> dict[str, float | None]:
    """Share of species inside another compartment's hull (0 = well separated)."""
    if not _HAS_SHAPELY:
        return {"species_in_other_compartment": None}
    by_comp: dict[str, list[str]] = {}
    for n, d in pathway.graph.nodes(data=True):
        if d["kind"] == SPECIES and d.get("compartment") and n in pos:
            by_comp.setdefault(d["compartment"], []).append(n)
    if len(by_comp) < 2:
        return {"species_in_other_compartment": None}
    hulls = {}
    for c, members in by_comp.items():
        pts = np.array([pos[n] for n in members])
        if len(pts) >= 3:
            try:
                hulls[c] = Polygon(pts[ConvexHull(pts).vertices])
            except (ValueError, RuntimeError):
                continue
    species = [n for members in by_comp.values() for n in members]
    if not species:
        return {"species_in_other_compartment": None}
    intruders = sum(
        1
        for n in species
        if any(
            c != pathway.graph.nodes[n]["compartment"] and h.contains(Point(pos[n]))
            for c, h in hulls.items()
        )
    )
    return {"species_in_other_compartment": intruders / len(species)}


def reference_similarity(pos: Positions, reference: Positions, k: int = 5) -> dict[str, float | None]:
    """Similarity after Procrustes alignment: disparity, distance Spearman, kNN preservation."""
    common = sorted(set(pos) & set(reference))
    if len(common) < 3:
        return {"procrustes": None, "distance_correlation": None, "knn_preservation": None}
    a = np.array([pos[n] for n in common], dtype=float)
    b = np.array([reference[n] for n in common], dtype=float)
    _, _, disparity = procrustes(b, a)
    da, db = pdist(a), pdist(b)
    corr = spearmanr(da, db).statistic
    k = min(k, len(common) - 1)

    def knn(xy):
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        np.fill_diagonal(d, math.inf)
        return [set(np.argsort(row)[:k]) for row in d]

    ka, kb = knn(a), knn(b)
    preservation = float(np.mean([len(x & y) / k for x, y in zip(ka, kb)]))
    return {
        "procrustes": float(disparity),
        "distance_correlation": float(corr),
        "knn_preservation": preservation,
        "reference_nodes": float(len(common)),
    }


def long_edge_metrics(pathway: Pathway, pos: Positions, long_factor: float = 3.0) -> dict[str, float | None]:
    """Edge-length statistics on placed undirected edges. ``long`` = length > long_factor × median."""
    lengths = []
    for u, v in pathway.undirected().edges():
        if u in pos and v in pos:
            lengths.append(float(np.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1])))
    if not lengths:
        return {
            "edge_length_mean": None,
            "edge_length_median": None,
            "edge_length_p90": None,
            "long_edge_share": None,
            "long_edge_factor": float(long_factor),
        }
    arr = np.array(lengths, float)
    med = float(np.median(arr))
    return {
        "edge_length_mean": float(arr.mean()),
        "edge_length_median": med,
        "edge_length_p90": float(np.percentile(arr, 90)),
        "long_edge_share": float(np.mean(arr > long_factor * max(med, 1e-9))),
        "long_edge_factor": float(long_factor),
        "edge_count": float(len(arr)),
    }


def long_edge_details(pathway: Pathway, pos: Positions, long_factor: float = 3.0, limit: int = 30) -> list[dict]:
    med_lengths = []
    edges = []
    for u, v in pathway.undirected().edges():
        if u in pos and v in pos:
            L = float(np.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]))
            med_lengths.append(L)
            edges.append((u, v, L))
    if not med_lengths:
        return []
    med = float(np.median(med_lengths))
    thr = long_factor * max(med, 1e-9)
    rows = []
    for u, v, L in sorted(edges, key=lambda t: -t[2]):
        if L < thr:
            break
        rows.append(
            dict(
                u=u,
                v=v,
                length=round(L, 1),
                relative_to_median=round(L / max(med, 1e-9), 2),
                mid_x=round(0.5 * (pos[u][0] + pos[v][0]), 1),
                mid_y=round(0.5 * (pos[u][1] + pos[v][1]), 1),
            )
        )
        if len(rows) >= limit:
            break
    return rows


def compactness_fallback(pos: Positions) -> dict[str, float | None]:
    """Aspect ratio (W/H) and area per node without gdMetriX (axis-aligned bbox).

    ``aspect_ratio`` is always width÷height (Project 22 / transfer-check definition),
    not max/min. gdMetriX may still report its own aspect under ``gdmetrix_aspect_ratio``.
    """
    if len(pos) < 2:
        return {"aspect_ratio": None, "aspect_ratio_bbox": None, "area_per_node_bbox": None}
    xy = np.array(list(pos.values()), float)
    w = float(xy[:, 0].max() - xy[:, 0].min())
    h = float(xy[:, 1].max() - xy[:, 1].min())
    wh = w / max(h, 1e-9)
    return {
        "aspect_ratio": wh,
        "aspect_ratio_bbox": wh,
        "area_per_node_bbox": (w * h) / len(pos),
    }


def _largest_empty_rectangle(empty: np.ndarray) -> int:
    """Largest-area axis-aligned rectangle of True cells (histogram stack method)."""
    if empty.size == 0:
        return 0
    rows, cols = empty.shape
    height = np.zeros(cols, dtype=int)
    best = 0
    for r in range(rows):
        for c in range(cols):
            height[c] = height[c] + 1 if empty[r, c] else 0
        stack: list[int] = []
        for c in range(cols + 1):
            h = int(height[c]) if c < cols else 0
            while stack and height[stack[-1]] > h:
                top = stack.pop()
                width = c if not stack else c - stack[-1] - 1
                best = max(best, int(height[top]) * width)
            stack.append(c)
    return best


def blank_space(
    points: np.ndarray | Sequence[Sequence[float]] | None = None,
    segments: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
    label_boxes: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    spacing: float | None = None,
    cell_factor: float = BLANK_CELL_FACTOR,
    clear_spacings: float = BLANK_CLEAR_SPACINGS,
    pad_spacings: float = 0.5,
) -> dict[str, float | None]:
    """Raster blank-space share and largest empty rectangle (MetaCarto idea; clean-room).

    Grid cell size is ``cell_factor ×`` median reaction spacing (default 0.6). Nodes,
    polyline segments and label boxes are stamped as ink. A cell counts as blank only
    if no ink lies within ``clear_spacings ×`` spacing (default 1). Returns:

    - ``blank_space_share`` — blank cells / cells
    - ``largest_empty_rect_share`` — largest empty axis-aligned rectangle / canvas cells
    - ``cell_px``, ``spacing_px``, ``grid`` shape metadata

    Single implementation for the evaluation CLI, the Layout quality panel, and any
    component-packing work that needs the same whitespace definition.
    """
    pts = np.asarray(points if points is not None else np.zeros((0, 2)), float).reshape(-1, 2)
    segs = [] if segments is None else list(segments)
    boxes = [] if label_boxes is None else [tuple(map(float, b)) for b in label_boxes if len(b) >= 4]

    if spacing is None or spacing <= 0:
        if len(pts) >= 2:
            d = cKDTree(pts).query(pts, k=2)[0][:, 1]
            d = d[d > 0]
            spacing = float(np.median(d)) if len(d) else 60.0
        else:
            spacing = 60.0
    cell = max(float(cell_factor) * float(spacing), 1.0)
    clear_r = max(float(clear_spacings) * float(spacing), cell)

    coords = [pts] if len(pts) else []
    for s in segs:
        a = np.asarray(s[0], float)
        b = np.asarray(s[1], float)
        coords.append(np.vstack([a, b]))
    for b in boxes:
        coords.append(np.array([[b[0], b[1]], [b[2], b[3]]], float))
    if not coords:
        return {
            "blank_space_share": None,
            "largest_empty_rect_share": None,
            "cell_px": cell,
            "spacing_px": float(spacing),
            "cells": 0.0,
        }
    all_xy = np.vstack(coords)
    pad = float(pad_spacings) * float(spacing)
    x0 = float(all_xy[:, 0].min()) - pad
    y0 = float(all_xy[:, 1].min()) - pad
    x1 = float(all_xy[:, 0].max()) + pad
    y1 = float(all_xy[:, 1].max()) + pad
    ncols = max(int(math.ceil((x1 - x0) / cell)), 1)
    nrows = max(int(math.ceil((y1 - y0) / cell)), 1)
    # Cap very large grids (huge GEMs) to keep evaluation interactive.
    max_cells = 400_000
    if ncols * nrows > max_cells:
        scale = math.sqrt((ncols * nrows) / max_cells)
        cell *= scale
        ncols = max(int(math.ceil((x1 - x0) / cell)), 1)
        nrows = max(int(math.ceil((y1 - y0) / cell)), 1)

    ink = np.zeros((nrows, ncols), dtype=bool)

    def stamp_xy(x: float, y: float) -> None:
        c = int((x - x0) // cell)
        r = int((y - y0) // cell)
        if 0 <= r < nrows and 0 <= c < ncols:
            ink[r, c] = True

    def stamp_segment(a: np.ndarray, b: np.ndarray) -> None:
        length = float(np.hypot(*(b - a)))
        steps = max(int(math.ceil(length / (0.5 * cell))), 1)
        for t in np.linspace(0.0, 1.0, steps + 1):
            p = a + t * (b - a)
            stamp_xy(float(p[0]), float(p[1]))

    for p in pts:
        stamp_xy(float(p[0]), float(p[1]))
    for s in segs:
        stamp_segment(np.asarray(s[0], float), np.asarray(s[1], float))
    for x_a, y_a, x_b, y_b in boxes:
        c0 = max(int((min(x_a, x_b) - x0) // cell), 0)
        c1 = min(int((max(x_a, x_b) - x0) // cell), ncols - 1)
        r0 = max(int((min(y_a, y_b) - y0) // cell), 0)
        r1 = min(int((max(y_a, y_b) - y0) // cell), nrows - 1)
        ink[r0 : r1 + 1, c0 : c1 + 1] = True

    # Dilate ink: cell is non-blank if any ink is within clear_r.
    rad = int(math.ceil(clear_r / cell))
    occupied = ink.copy()
    if rad > 0 and ink.any():
        ys, xs = np.nonzero(ink)
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                if dx * dx + dy * dy > (clear_r / cell) ** 2 + 1e-9:
                    continue
                yy = ys + dy
                xx = xs + dx
                m = (yy >= 0) & (yy < nrows) & (xx >= 0) & (xx < ncols)
                occupied[yy[m], xx[m]] = True

    total = float(nrows * ncols)
    blank = ~occupied
    blank_n = float(blank.sum())
    largest = _largest_empty_rectangle(blank)
    return {
        "blank_space_share": round(blank_n / total, 4) if total else None,
        "largest_empty_rect_share": round(largest / total, 4) if total else None,
        "cell_px": round(cell, 2),
        "spacing_px": round(float(spacing), 1),
        "cells": total,
        "grid_rows": float(nrows),
        "grid_cols": float(ncols),
        "cell_factor": float(cell_factor),
        "clear_spacings": float(clear_spacings),
    }


def blank_space_from_positions(
    pos: Positions,
    edges: Iterable[tuple[str, str]] | None = None,
    label_boxes: Sequence[Sequence[float]] | None = None,
    spacing: float | None = None,
    **kwargs: Any,
) -> dict[str, float | None]:
    """``blank_space`` from node id→(x,y) and undirected edges (component-packing friendly)."""
    points = np.array(list(pos.values()), float).reshape(-1, 2) if pos else np.zeros((0, 2))
    segs = []
    if edges is not None:
        for u, v in edges:
            if u in pos and v in pos:
                segs.append((pos[u], pos[v]))
    return blank_space(points, segs, label_boxes, spacing=spacing, **kwargs)


def _collinear_edge_overlaps(
    seg_list: Sequence[tuple[str, str, np.ndarray, np.ndarray]],
    related_sets: dict[str, set[str]],
    collinear_tol: float,
) -> int:
    """Count nearly collinear overlapping segment pairs of unrelated reactions.

    Same predicate as the previous all-pairs loop. Candidates are generated by a
    uniform spatial grid over axis-aligned boxes grown by ``collinear_tol``: any
    pair whose grown boxes are disjoint cannot satisfy the geometric tests, so the
    result matches the all-pairs count exactly.
    """
    n = len(seg_list)
    if n < 2:
        return 0
    A = np.empty((n, 2), float)
    B = np.empty((n, 2), float)
    rid = [""] * n
    for i, (ra, _, a, b) in enumerate(seg_list):
        rid[i] = ra
        A[i] = a
        B[i] = b
    D = B - A
    N = np.hypot(D[:, 0], D[:, 1])
    ok = N >= 1e-6
    U = np.zeros_like(D)
    U[ok] = D[ok] / N[ok, None]
    # Grown AABB of each segment (pad = collinear_tol).
    lo = np.minimum(A, B) - collinear_tol
    hi = np.maximum(A, B) + collinear_tol
    # Cell size ≈ typical segment extent so buckets stay small but cover the pad.
    span = hi - lo
    cell = float(max(4.0 * collinear_tol, np.median(np.maximum(span[:, 0], span[:, 1])) if n else 4.0 * collinear_tol))
    if cell < 1e-6:
        cell = 4.0 * collinear_tol
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        if not ok[i]:
            continue
        ix0 = int(math.floor(lo[i, 0] / cell))
        iy0 = int(math.floor(lo[i, 1] / cell))
        ix1 = int(math.floor(hi[i, 0] / cell))
        iy1 = int(math.floor(hi[i, 1] / cell))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                buckets.setdefault((ix, iy), []).append(i)

    collinear = 0
    seen: set[tuple[int, int]] = set()
    for members in buckets.values():
        m = len(members)
        if m < 2:
            continue
        for a in range(m):
            i = members[a]
            for b in range(a + 1, m):
                j = members[b]
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                seen.add(key)
                # Grown-AABB rejection (exact filter for the distance/overlap tests).
                if hi[i, 0] < lo[j, 0] or hi[j, 0] < lo[i, 0] or hi[i, 1] < lo[j, 1] or hi[j, 1] < lo[i, 1]:
                    continue
                ra, rb = rid[i], rid[j]
                if ra == rb or rb in related_sets.get(ra, ()) or ra in related_sets.get(rb, ()):
                    continue
                u1, a1, n1 = U[i], A[i], N[i]
                if abs(float(u1 @ U[j])) < 0.98:  # not nearly parallel
                    continue
                off = A[j] - a1
                if abs(float(off[0] * u1[1] - off[1] * u1[0])) > collinear_tol:
                    continue
                pa = float((A[j] - a1) @ u1)
                pb = float((B[j] - a1) @ u1)
                lo_p = max(0.0, min(pa, pb))
                hi_p = min(n1, max(pa, pb))
                if hi_p - lo_p > collinear_tol:  # overlapping run
                    collinear += 1
    return collinear


def unrelated_overlaps(
    node_xy: dict[str, tuple[float, float]],
    node_reaction: dict[str, str],
    segments: Sequence[tuple[str, str, np.ndarray, np.ndarray]],
    related: set[tuple[str, str]],
    *,
    chords: Sequence[tuple[str, str, np.ndarray, np.ndarray]] | None = None,
    node_radius: float = 7.0,
    collinear_tol: float = 3.0,
) -> dict[str, float | None]:
    """Overlaps between reactions that share no non-currency metabolite (MetaCarto idea).

    Counts:
    - ``unrelated_node_overlaps`` — node centres of unrelated reactions closer than 2×radius
    - ``unrelated_edge_through_node`` — a segment of reaction A passes near a node of unrelated B
    - ``unrelated_collinear_edge_overlaps`` — nearly collinear overlapping segment pairs of unrelated reactions,
      measured on ``chords`` when given (one straight piece per connection), else on ``segments``
    """
    ids = [n for n in node_xy if n in node_reaction]
    if len(ids) < 2:
        return {
            "unrelated_node_overlaps": 0.0,
            "unrelated_edge_through_node": 0.0,
            "unrelated_collinear_edge_overlaps": 0.0,
        }

    related_sets: dict[str, set[str]] = {}
    for a, b in related:
        related_sets.setdefault(a, set()).add(b)
        related_sets.setdefault(b, set()).add(a)

    def related_pair(a: str, b: str) -> bool:
        if a == b:
            return True
        return b in related_sets.get(a, ()) or a in related_sets.get(b, ())

    # node-on-node
    xy = np.array([node_xy[n] for n in ids], float)
    tree = cKDTree(xy)
    node_hits = 0
    for i, j in tree.query_pairs(2.0 * node_radius):
        if not related_pair(node_reaction[ids[i]], node_reaction[ids[j]]):
            node_hits += 1

    # edge through node (query nearby node centres only)
    through = 0
    for rid_a, _nid_u, a, b in segments:
        ab = b - a
        lab2 = float(ab @ ab)
        if lab2 < 1e-12:
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * float(np.hypot(*ab)) + node_radius
        for j in tree.query_ball_point(mid, half):
            nid = ids[j]
            rid_b = node_reaction[nid]
            if related_pair(rid_a, rid_b):
                continue
            p = xy[j]
            t = float((p - a) @ ab) / lab2
            if t <= 0.05 or t >= 0.95:
                continue
            proj = a + t * ab
            if float(np.hypot(*(p - proj))) <= node_radius:
                through += 1

    # collinear edge-on-edge (unrelated): spatial grid, same count as all-pairs
    collinear = _collinear_edge_overlaps(
        list(segments if chords is None else chords), related_sets, collinear_tol
    )

    return {
        "unrelated_node_overlaps": float(node_hits),
        "unrelated_edge_through_node": float(through),
        "unrelated_collinear_edge_overlaps": float(collinear),
        "unrelated_overlaps_total": float(node_hits + through + collinear),
    }


def all_benchmark_metrics(pathway: Pathway, pos: Positions, reference: Positions | None = None) -> dict[str, float | None]:
    m = general_metrics(pathway, pos)
    m.update(overlaps(pathway, pos))
    m["participant_consistency"] = participant_consistency(pathway, pos)
    m.update(compartment_metrics(pathway, pos))
    m.update(long_edge_metrics(pathway, pos))
    compact = compactness_fallback(pos)
    m.update(compact)
    # Prefer W/H bbox aspect; leave gdMetriX value only under gdmetrix_aspect_ratio.
    if compact.get("aspect_ratio") is not None:
        m["aspect_ratio"] = compact["aspect_ratio"]
    if m.get("area_per_node") is None and compact.get("area_per_node_bbox") is not None:
        m["area_per_node"] = compact["area_per_node_bbox"]
    # Raster blank space on the placed undirected graph (same function the panel uses).
    m.update(
        {
            k: v
            for k, v in blank_space_from_positions(pos, pathway.undirected().edges()).items()
            if k in ("blank_space_share", "largest_empty_rect_share", "cell_px", "spacing_px")
        }
    )
    if reference is not None:
        m.update(reference_similarity(pos, reference))
    return m
