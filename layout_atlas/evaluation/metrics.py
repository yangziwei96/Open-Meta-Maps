"""Layout metrics used by ``evaluate`` / packing helpers.

Official layout-benchmark figures live in ``official_metrics`` (one shared
implementation). This module keeps the MetaCarto-idea helpers
(``blank_space``, ``unrelated_overlaps``) and thin wrappers that feed
``evaluation.score`` without duplicating the official definitions.

gdMetriX is never vendored. Optional external scoring against the installed
``layout_benchmark`` package is ``evaluation.score_map``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from .pathway import PRODUCT, REACTANT, SPECIES, Pathway
import official_metrics as om

Positions = dict[str, tuple[float, float]]
BLANK_CELL_FACTOR = 0.6
BLANK_CLEAR_SPACINGS = 1.0


def gdmetrix_available() -> bool:
    """Always False here: gdMetriX is not imported in this path (GPL-3.0)."""
    return False


def shapely_available() -> bool:
    """Compartment / overlap geometry no longer needs shapely."""
    return True


def participant_consistency(pathway: Pathway, pos: Positions) -> float | None:
    """Share of (reactant, product) pairs on opposite sides (>90°, layout-benchmark)."""
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


def overlaps(pathway: Pathway, pos: Positions) -> dict[str, float | None]:
    """Box overlaps from pathway node width/height (evaluate CLI fallback).

    Prefer ``official_metrics.glyph_overlaps`` on the Escher map when available.
    """
    nodes = [n for n in pathway.graph if n in pos]
    if len(nodes) < 2:
        return {"overlapping_pairs": 0.0, "overlap_area_share": 0.0}
    pairs, area, total = 0, 0.0, 0.0
    boxes = []
    for n in nodes:
        d = pathway.graph.nodes[n]
        x, y = pos[n]
        w, h = float(d.get("width", 10)) / 2, float(d.get("height", 10)) / 2
        boxes.append((x - w, y - h, x + w, y + h))
        total += (2 * w) * (2 * h)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            dx = min(a[2], b[2]) - max(a[0], b[0])
            dy = min(a[3], b[3]) - max(a[1], b[1])
            if dx > 0 and dy > 0:
                pairs += 1
                area += dx * dy
    return {"overlapping_pairs": float(pairs), "overlap_area_share": area / total if total else 0.0}


def compartment_metrics(pathway: Pathway, pos: Positions) -> dict[str, float | None]:
    """Share of species inside another compartment hull (ConvexHull; no shapely)."""
    by_comp: dict[str, list[str]] = {}
    for n, d in pathway.graph.nodes(data=True):
        if d["kind"] == SPECIES and d.get("compartment") and n in pos:
            by_comp.setdefault(d["compartment"], []).append(n)
    if len(by_comp) < 2:
        return {"species_in_other_compartment": None}
    from scipy.spatial import ConvexHull

    hulls = {}
    for c, members in by_comp.items():
        pts = np.array([pos[n] for n in members], float)
        if len(pts) >= 3:
            try:
                hulls[c] = pts[ConvexHull(pts).vertices]
            except (ValueError, RuntimeError):
                continue
    species = [n for members in by_comp.values() for n in members]
    if not species or not hulls:
        return {"species_in_other_compartment": None}
    intruders = sum(
        1
        for n in species
        if any(
            c != pathway.graph.nodes[n]["compartment"] and om._point_in_hull(h, pos[n][0], pos[n][1])
            for c, h in hulls.items()
        )
    )
    return {"species_in_other_compartment": intruders / len(species)}


def reference_similarity(pos: Positions, reference: Positions, k: int = 5) -> dict[str, float | None]:
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
    """Raster blank-space share and largest empty rectangle (MetaCarto idea; clean-room)."""
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
    lo = np.minimum(A, B) - collinear_tol
    hi = np.maximum(A, B) + collinear_tol
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
                if hi[i, 0] < lo[j, 0] or hi[j, 0] < lo[i, 0] or hi[i, 1] < lo[j, 1] or hi[j, 1] < lo[i, 1]:
                    continue
                ra, rb = rid[i], rid[j]
                if ra == rb or rb in related_sets.get(ra, ()) or ra in related_sets.get(rb, ()):
                    continue
                u1, a1, n1 = U[i], A[i], N[i]
                if abs(float(u1 @ U[j])) < 0.98:
                    continue
                off = A[j] - a1
                if abs(float(off[0] * u1[1] - off[1] * u1[0])) > collinear_tol:
                    continue
                pa = float((A[j] - a1) @ u1)
                pb = float((B[j] - a1) @ u1)
                lo_p = max(0.0, min(pa, pb))
                hi_p = min(n1, max(pa, pb))
                if hi_p - lo_p > collinear_tol:
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

    xy = np.array([node_xy[n] for n in ids], float)
    tree = cKDTree(xy)
    node_hits = 0
    for i, j in tree.query_pairs(2.0 * node_radius):
        if not related_pair(node_reaction[ids[i]], node_reaction[ids[j]]):
            node_hits += 1

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

    collinear = _collinear_edge_overlaps(
        list(segments if chords is None else chords), related_sets, collinear_tol
    )

    return {
        "unrelated_node_overlaps": float(node_hits),
        "unrelated_edge_through_node": float(through),
        "unrelated_collinear_edge_overlaps": float(collinear),
        "unrelated_overlaps_total": float(node_hits + through + collinear),
    }


def general_metrics(pathway: Pathway, pos: Positions) -> dict[str, float | None]:
    """Purchase-style general metrics without importing gdMetriX.

    ``edge_length_deviation`` / ``edge_orthogonality`` use straight undirected
    edges of the placed pathway graph (evaluate CLI). Final-map figures use
    ``official_metrics`` on Escher polylines.
    """
    g = pathway.undirected().subgraph([n for n in pathway.undirected() if n in pos]).copy()
    missing = [n for n in g if n not in pos]
    g.remove_nodes_from(missing)
    if g.number_of_nodes() < 2 or g.number_of_edges() < 1:
        return {
            "crossings": None,
            "edge_length_deviation": None,
            "edge_orthogonality": None,
            "aspect_ratio": None,
            "gdmetrix": 0.0,
        }
    segs = []
    lengths = []
    for u, v in g.edges():
        a, b = pos[u], pos[v]
        segs.append((a, b))
        lengths.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
    mean = float(np.mean(lengths)) if lengths else 0.0
    eld = float(np.mean([abs(mean - x) / mean for x in lengths])) if mean else None
    eorth = om.edge_orthogonality(segs)
    compact = compactness_fallback(pos)
    return {
        "crossings": None,  # use official_metrics on the Escher map
        "edge_length_deviation": eld,
        "edge_orthogonality": eorth,
        "aspect_ratio": compact.get("aspect_ratio"),
        "gdmetrix": 0.0,
    }


def all_benchmark_metrics(pathway: Pathway, pos: Positions, reference: Positions | None = None) -> dict[str, float | None]:
    m = general_metrics(pathway, pos)
    m.update(overlaps(pathway, pos))
    m["participant_consistency"] = participant_consistency(pathway, pos)
    m.update(compartment_metrics(pathway, pos))
    m.update(long_edge_metrics(pathway, pos))
    compact = compactness_fallback(pos)
    m.update(compact)
    if compact.get("aspect_ratio") is not None:
        m["aspect_ratio"] = compact["aspect_ratio"]
    if m.get("area_per_node") is None and compact.get("area_per_node_bbox") is not None:
        m["area_per_node"] = compact["area_per_node_bbox"]
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
