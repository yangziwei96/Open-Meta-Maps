"""Accept-gate metrics for hub-split trials (compose path).

Numpy / networkx only. Scores a clone-aware Escher drawing: Purchase-style
angular resolution and edge orthogonality on drawn polyline pieces, straight-line
crossings, glyph overlaps at Escher radii, reactant/product opposite-side share,
and bounding-box aspect / quadrant balance.

Edge orthogonality uses every straight step of each participant polyline
(midmarker → multimarker bends → metabolite), counting a geometrically identical
step once when several edges share it — the same piece set the Pathway Scorer
uses for that metric.

Aspect ratio (higher) and homogeneity (lower) are gated deliberately even though
some scorer tables leave them undirected.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from hub_drawing import REACTION, SPECIES, drawing_graph, stub_endpoints

# Default Escher glyph radii (primary metabolite, secondary, reaction midmarker).
_RADIUS = {SPECIES: (20.0, 10.0), REACTION: 5.0}

# Gate directions live in layout_quality_evaluator.HUB_DIRS (single copy).


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_intersect(a, b, c, d, eps=1e-12) -> bool:
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if o1 * o2 >= -eps or o3 * o4 >= -eps:
        return False
    return True


def _crossing_point(a, b, c, d):
    den = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    if abs(den) < 1e-18:
        return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / den
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _edge_polyline(u, v, pos, routes) -> list[tuple[float, float]]:
    a = (float(pos[u][0]), float(pos[u][1]))
    b = (float(pos[v][0]), float(pos[v][1]))
    inner = routes.get((u, v))
    if inner is None:
        inner = routes.get((v, u))
        if inner is not None:
            inner = list(reversed(inner))
    if not inner:
        return [a, b]
    return [a, *[(float(p[0]), float(p[1])) for p in inner], b]


def _bend_key(p):
    """Coincide bends that land on the same drawn point (six significant figures)."""
    return (float(f"{float(p[0]):.6g}"), float(f"{float(p[1]):.6g}"))


def _straight_pieces(g: nx.Graph, pos: dict, routes: dict) -> list[dict]:
    """Unique straight steps of drawn participant polylines.

    Walk each drawing-graph edge's polyline (species glyph → multimarker bends →
    reaction midmarker). A step that several parents share is kept once. Bend
    coincidence uses six significant figures so the piece set matches the
    Pathway Scorer segment graph used for official ``edge_orthogonality``.
    """
    pieces = []
    seen: set[tuple] = set()
    for u, v in g.edges():
        if u not in pos or v not in pos:
            continue
        pts = _edge_polyline(u, v, pos, routes)
        # Endpoint identities: graph nodes for ends, bend keys for intermediate points.
        ids = [u] + [_bend_key(p) for p in pts[1:-1]] + [v]
        for (ia, ib), (a, b) in zip(zip(ids, ids[1:]), zip(pts, pts[1:])):
            if a == b or ia == ib:
                continue
            key = (ia, ib) if repr(ia) <= repr(ib) else (ib, ia)
            if key in seen:
                continue
            seen.add(key)
            ra, rb = _bend_key(a), _bend_key(b)
            pieces.append(
                dict(
                    a=a,
                    b=b,
                    ends=frozenset((ra, rb)),
                    parent=(u, v),
                    xmin=min(a[0], b[0]),
                    xmax=max(a[0], b[0]),
                    ymin=min(a[1], b[1]),
                    ymax=max(a[1], b[1]),
                )
            )
    return pieces


def _grid_cell_size(pieces: list[dict]) -> float:
    """Cell size from median piece length (problem-derived, not a fixed constant)."""
    lengths = [
        float(math.hypot(p["b"][0] - p["a"][0], p["b"][1] - p["a"][1])) for p in pieces
    ]
    if not lengths:
        return 1.0
    return max(float(np.median(lengths)), 1.0)


def _cells_along_segment(a, b, cell: float):
    """Yield grid cells a segment passes through (visit cells along the line, not its bbox)."""
    x0, y0 = a[0] / cell, a[1] / cell
    x1, y1 = b[0] / cell, b[1] / cell
    ix0, iy0 = int(math.floor(x0)), int(math.floor(y0))
    ix1, iy1 = int(math.floor(x1)), int(math.floor(y1))
    yield (ix0, iy0)
    if ix0 == ix1 and iy0 == iy1:
        return
    dx = x1 - x0
    dy = y1 - y0
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    # Distance along the ray to the next vertical / horizontal grid line.
    if step_x == 0:
        t_max_x = math.inf
        t_delta_x = math.inf
    else:
        next_vx = ix0 + (1 if step_x > 0 else 0)
        t_max_x = (next_vx - x0) / dx
        t_delta_x = abs(1.0 / dx)
    if step_y == 0:
        t_max_y = math.inf
        t_delta_y = math.inf
    else:
        next_vy = iy0 + (1 if step_y > 0 else 0)
        t_max_y = (next_vy - y0) / dy
        t_delta_y = abs(1.0 / dy)
    ix, iy = ix0, iy0
    # Bound steps by Manhattan cell distance + 2.
    limit = abs(ix1 - ix0) + abs(iy1 - iy0) + 2
    for _ in range(limit):
        if t_max_x < t_max_y:
            ix += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            iy += step_y
            t_max_y += t_delta_y
        else:
            ix += step_x
            iy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        yield (ix, iy)
        if ix == ix1 and iy == iy1:
            return


def _build_piece_index(pieces: list[dict], cell: float):
    """Map grid cells → piece indices by walking each segment."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, p in enumerate(pieces):
        for cell_key in _cells_along_segment(p["a"], p["b"], cell):
            buckets[cell_key].append(i)
    return buckets


def _pieces_share_vertex(p, q) -> bool:
    return bool(p["ends"] & q["ends"])


def _count_crossings_among(pieces: list[dict], cell: float, buckets=None) -> tuple[int, list[dict]]:
    """Count proper crossings with a segment-walk grid; return (count, hit records)."""
    if len(pieces) < 2:
        return 0, []
    if buckets is None:
        buckets = _build_piece_index(pieces, cell)
    found = []
    seen: set[tuple[int, int]] = set()
    for idxs in buckets.values():
        for ai in range(len(idxs)):
            i = idxs[ai]
            pi = pieces[i]
            for bi in range(ai + 1, len(idxs)):
                j = idxs[bi]
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                pj = pieces[j]
                if pi["xmax"] < pj["xmin"] or pj["xmax"] < pi["xmin"]:
                    continue
                if pi["ymax"] < pj["ymin"] or pj["ymax"] < pi["ymin"]:
                    continue
                if _pieces_share_vertex(pi, pj):
                    continue
                if _proper_intersect(pi["a"], pi["b"], pj["a"], pj["b"]):
                    found.append(
                        dict(
                            edges=((pi["a"], pi["b"]), (pj["a"], pj["b"])),
                            point=_crossing_point(pi["a"], pi["b"], pj["a"], pj["b"]),
                            pair=pair,
                        )
                    )
    return len(found), found


def _hit_record(a_seg, b_seg) -> dict:
    return dict(
        edges=((a_seg["a"], a_seg["b"]), (b_seg["a"], b_seg["b"])),
        point=_crossing_point(a_seg["a"], a_seg["b"], b_seg["a"], b_seg["b"]),
    )


def _crossings_between(
    query: list[dict],
    base_pieces: list[dict],
    cell: float,
    buckets,
    skip_ids: set[int] | None = None,
) -> tuple[int, list[dict]]:
    """Crossings of query pieces against an indexed base set (and among query)."""
    skip_ids = skip_ids or set()
    hits: list[dict] = []
    # Query vs base
    for q in query:
        cand = set()
        for cell_key in _cells_along_segment(q["a"], q["b"], cell):
            for j in buckets.get(cell_key, ()):
                if j in skip_ids:
                    continue
                cand.add(j)
        for j in cand:
            p = base_pieces[j]
            if q["xmax"] < p["xmin"] or p["xmax"] < q["xmin"]:
                continue
            if q["ymax"] < p["ymin"] or p["ymax"] < q["ymin"]:
                continue
            if _pieces_share_vertex(q, p):
                continue
            if _proper_intersect(q["a"], q["b"], p["a"], p["b"]):
                hits.append(_hit_record(q, p))
    # Query vs query
    for i in range(len(query)):
        for j in range(i + 1, len(query)):
            qi, qj = query[i], query[j]
            if qi["xmax"] < qj["xmin"] or qj["xmax"] < qi["xmin"]:
                continue
            if qi["ymax"] < qj["ymin"] or qj["ymax"] < qi["ymin"]:
                continue
            if _pieces_share_vertex(qi, qj):
                continue
            if _proper_intersect(qi["a"], qi["b"], qj["a"], qj["b"]):
                hits.append(_hit_record(qi, qj))
    return len(hits), hits


def _parent_edge_set(stubs: list[tuple[Any, Any]]) -> set[frozenset]:
    return {frozenset((u, v)) for u, v in stubs}


def _stub_piece_indices(pieces: list[dict], stubs: list[tuple[Any, Any]]) -> list[int]:
    """Indices of pieces whose parent drawing edge is one of the stub edges."""
    wanted = _parent_edge_set(stubs)
    return [
        i
        for i, p in enumerate(pieces)
        if frozenset(p["parent"]) in wanted
    ]


def _ortho_deviation(a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    ang = abs(math.atan2(dy, dx)) % (math.pi / 2)
    ang = min(ang, math.pi / 2 - ang)
    return ang * 4 / math.pi


def _edge_orthogonality_from_pieces(pieces: list[dict]) -> float:
    if not pieces:
        return 0.0
    total = sum(_ortho_deviation(p["a"], p["b"]) for p in pieces)
    return 1.0 - total / len(pieces)


def _polyline_length(u, v, pos, routes) -> float:
    pts = np.asarray(_edge_polyline(u, v, pos, routes), float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _edge_length_deviation(g: nx.Graph, pos: dict, routes: dict) -> float:
    lengths = []
    for u, v in g.edges():
        if u not in pos or v not in pos:
            continue
        lengths.append(_polyline_length(u, v, pos, routes))
    if not lengths:
        return 0.0
    mean = float(np.mean(lengths))
    if mean <= 0:
        return 0.0
    return float(np.mean([abs(mean - L) / mean for L in lengths]))


def _first_piece_angles(g: nx.Graph, node, pos: dict, routes: dict) -> list[float]:
    x0, y0 = pos[node]
    angs = []
    for nb in g.neighbors(node):
        pts = _edge_polyline(node, nb, pos, routes)
        if len(pts) < 2:
            continue
        x1, y1 = pts[1]
        angs.append(math.atan2(y1 - y0, x1 - x0))
    angs.sort()
    return angs


def _angular_resolution_over(g: nx.Graph, pos: dict, nodes, routes: dict) -> float | None:
    total = 0.0
    counted = 0
    for node in nodes:
        if node not in pos or node not in g:
            continue
        deg = g.degree(node)
        if deg < 2:
            continue
        angs = _first_piece_angles(g, node, pos, routes)
        if len(angs) < 2:
            continue
        gaps = [angs[i + 1] - angs[i] for i in range(len(angs) - 1)]
        gaps.append(angs[0] + 2 * math.pi - angs[-1])
        tightest = min(gaps)
        ideal = 2 * math.pi / deg
        total += abs((ideal - tightest) / ideal)
        counted += 1
    if not counted:
        return None
    return total / counted


def _angular_resolution(g: nx.Graph, pos: dict, routes: dict) -> float:
    value = _angular_resolution_over(g, pos, g.nodes(), routes)
    return 0.0 if value is None else value


def local_angular_resolution(g: nx.Graph, pos: dict, nodes: set, routes: dict | None = None) -> float | None:
    return _angular_resolution_over(g, pos, nodes, routes or {})


def _aspect_ratio(pos: dict) -> float | None:
    if len(pos) < 2:
        return 1.0 if pos else None
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    bigger = max(w, h)
    smaller = min(w, h)
    if bigger == 0 and smaller == 0:
        return 1.0
    return smaller / bigger


def _homogeneity(pos: dict) -> float:
    if len(pos) <= 1:
        return 0.0
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    counts = [0, 0, 0, 0]
    for x, y in zip(xs, ys):
        counts[(0 if x < cx else 1) + (0 if y < cy else 2)] += 1
    n = float(len(pos))
    target = n / 4.0
    return sum(abs(c - target) for c in counts) / (1.5 * n)


def _circle_overlap_area(r1: float, r2: float, dist: float) -> float:
    if dist >= r1 + r2 or dist <= 0:
        return 0.0
    if dist <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    r1sq, r2sq = r1 * r1, r2 * r2
    alpha = math.acos(np.clip((dist * dist + r1sq - r2sq) / (2 * dist * r1), -1.0, 1.0))
    beta = math.acos(np.clip((dist * dist + r2sq - r1sq) / (2 * dist * r2), -1.0, 1.0))
    return (
        r1sq * alpha
        + r2sq * beta
        - 0.5
        * math.sqrt(
            max(
                0.0,
                (-dist + r1 + r2)
                * (dist + r1 - r2)
                * (dist - r1 + r2)
                * (dist + r1 + r2),
            )
        )
    )


def _glyph_radius(kind, primary: bool) -> float:
    if kind == REACTION:
        return float(_RADIUS[REACTION])
    primary_r, secondary_r = _RADIUS[SPECIES]
    return primary_r if primary else secondary_r


def _node_overlaps(g: nx.Graph, pos: dict) -> dict[str, float]:
    nodes = []
    for n, d in g.nodes(data=True):
        kind = d.get("kind")
        if kind not in (SPECIES, REACTION) or n not in pos:
            continue
        r = _glyph_radius(kind, bool(d.get("primary")))
        x, y = pos[n]
        nodes.append((n, x, y, r))
    if not nodes:
        return {"overlapping_pairs": 0.0, "overlap_area_share": 0.0}
    cell = max(max(r for *_, r in nodes) * 2.0, 1.0)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (_, x, y, _) in enumerate(nodes):
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        buckets.setdefault(key, []).append(i)
    pairs = 0
    overlap_area = 0.0
    seen: set[tuple[int, int]] = set()
    for (cx, cy), idxs in buckets.items():
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(buckets.get((cx + dx, cy + dy), ()))
        for a in idxs:
            _, ax, ay, ar = nodes[a]
            for b in candidates:
                if b <= a:
                    continue
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                _, bx, by, br = nodes[b]
                dist = math.hypot(ax - bx, ay - by)
                area = _circle_overlap_area(ar, br, dist)
                if area > 0.0:
                    pairs += 1
                    overlap_area += area
    total = sum(math.pi * r * r for *_, r in nodes)
    return {
        "overlapping_pairs": float(pairs),
        "overlap_area_share": (overlap_area / total) if total else 0.0,
    }


def _through_reaction_share(g: nx.DiGraph, pos: dict) -> float | None:
    good = 0
    total = 0
    for rid, data in g.nodes(data=True):
        if data.get("kind") != REACTION or rid not in pos:
            continue
        origin = np.asarray(pos[rid], float)
        inward = []
        outward = []
        for src, _, edge in g.in_edges(rid, data=True):
            if edge.get("role") == "reactant" and src in pos:
                inward.append(np.asarray(pos[src], float) - origin)
        for _, dst, edge in g.out_edges(rid, data=True):
            if edge.get("role") == "product" and dst in pos:
                outward.append(np.asarray(pos[dst], float) - origin)
        for vin in inward:
            for vout in outward:
                total += 1
                if float(np.dot(vin, vout)) < 0.0:
                    good += 1
    if total == 0:
        return None
    return good / total


def multimarker_counts(map_json) -> dict[str, int]:
    body = map_json[1]
    nodes = body["nodes"]
    out = {}
    for rx in body["reactions"].values():
        rid = rx.get("bigg_id")
        if not rid:
            continue
        mids = set()
        for seg in (rx.get("segments") or {}).values():
            for nid in (seg.get("from_node_id"), seg.get("to_node_id")):
                if nodes.get(nid, {}).get("node_type") == "multimarker":
                    mids.add(nid)
        out[rid] = len(mids)
    return out


def multimarkers_dropped(before_map, after_map) -> bool:
    b = multimarker_counts(before_map)
    a = multimarker_counts(after_map)
    for rid, n0 in b.items():
        if a.get(rid, 0) < n0:
            return True
    return False


def _crossing_density(n_cross: int, pieces: list[dict]) -> float:
    """1 − crossings / (unordered piece pairs that do not share a vertex)."""
    m = len(pieces)
    deg = defaultdict(int)
    for p in pieces:
        for end in p["ends"]:
            deg[end] += 1
    share_vertex = sum(math.comb(d, 2) for d in deg.values() if d >= 2)
    capacity = math.comb(m, 2) - share_vertex if m >= 2 else 0
    if capacity <= 0:
        return 0.0
    return 1.0 - n_cross / capacity


def _crossing_angular_resolution(hits: list[dict]) -> float | None:
    """Mean Purchase-style angular deviation of crossing rays (higher is better)."""
    if not hits:
        return None
    deviations = []
    for hit in hits:
        px, py = hit["point"]
        angles = []
        for a, b in hit["edges"]:
            for ex, ey in (a, b):
                angles.append(math.atan2(ey - py, ex - px))
        angles.sort()
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(angles[0] + 2 * math.pi - angles[-1])
        smallest = min(gaps)
        ideal = math.pi / max(len(hit["edges"]), 1)
        deviations.append(abs((ideal - smallest) / ideal))
    return 1.0 - float(np.mean(deviations)) if deviations else None


def _crossing_bundle_from_hits(n_cross: int, pieces: list[dict], hits: list[dict]) -> dict:
    return {
        "crossings": float(n_cross),
        "crossing_density": float(_crossing_density(n_cross, pieces)),
        "crossing_angular_resolution": _crossing_angular_resolution(hits),
    }


class CrossingIndex:
    """Baseline piece set + segment-walk grid for incremental stub updates."""

    def __init__(self, g: nx.Graph, pos: dict, routes: dict):
        self.g = g
        self.pos = pos
        self.routes = routes
        self.pieces = _straight_pieces(g, pos, routes)
        self.cell = _grid_cell_size(self.pieces)
        self.buckets = _build_piece_index(self.pieces, self.cell)
        self.n_cross, self.hits = _count_crossings_among(
            self.pieces, self.cell, self.buckets
        )

    def stub_crossing_count(self, stubs: list[tuple[Any, Any]]) -> int:
        idxs = _stub_piece_indices(self.pieces, stubs)
        if not idxs:
            return 0
        stub_pieces = [self.pieces[i] for i in idxs]
        skip = set(idxs)
        n, _hits = _crossings_between(
            stub_pieces, self.pieces, self.cell, self.buckets, skip
        )
        return n

    def hits_without_stubs(self, stub_idxs: set[int]) -> list[dict]:
        """Baseline hits whose pieces are not among the replaced stubs."""
        if not stub_idxs:
            return list(self.hits)
        return [
            hit
            for hit in self.hits
            if hit.get("pair") is not None
            and hit["pair"][0] not in stub_idxs
            and hit["pair"][1] not in stub_idxs
        ]


def score_map(pathway, map_json) -> dict[str, Any]:
    """Directed gate metrics on a clone-aware Escher drawing."""
    import layout_quality as lq

    g, pos, routes = drawing_graph(pathway, map_json)
    u = g.to_undirected()
    pieces = _straight_pieces(u, pos, routes)
    cell = _grid_cell_size(pieces)
    n_cross, hits = _count_crossings_among(pieces, cell)
    out: dict[str, Any] = {}
    out.update(_crossing_bundle_from_hits(n_cross, pieces, hits))
    out["angular_resolution"] = _angular_resolution(u, pos, routes)
    out["edge_length_deviation"] = _edge_length_deviation(u, pos, routes)
    out["edge_orthogonality"] = _edge_orthogonality_from_pieces(pieces)
    out["aspect_ratio"] = _aspect_ratio(pos)
    out["homogeneity"] = _homogeneity(pos)
    out.update(_node_overlaps(u, pos))
    out["participant_consistency"] = _through_reaction_share(g, pos)
    # Same geometry as the panel / LayoutQualityEvaluator no-new hard gate.
    out["curves_through_nodes"] = float(lq.curves_through_nodes_count(map_json))
    return out


def score_trial(
    pathway,
    before_map,
    trial_map,
    old_stubs: list[tuple],
    new_stubs: list[tuple],
    before_index: CrossingIndex | None = None,
    old_stub_crossings: int | None = None,
) -> dict[str, Any] | None:
    """Trial metrics, or ``None`` when the trial is rejected before scoring.

    Crossings and CAR update from the cached baseline index: old stub hits are
    dropped and new stub hits are added against the segment-walk grid.
    """
    if multimarkers_dropped(before_map, trial_map):
        return None

    if before_index is None:
        g0, pos0, routes0 = drawing_graph(pathway, before_map)
        before_index = CrossingIndex(g0.to_undirected(), pos0, routes0)
    if old_stub_crossings is None:
        old_stub_crossings = before_index.stub_crossing_count(
            stub_endpoints(pathway, before_map, old_stubs)
        )

    g1, pos1, routes1 = drawing_graph(pathway, trial_map)
    u1 = g1.to_undirected()
    new_graph_stubs = stub_endpoints(pathway, trial_map, new_stubs)
    trial_pieces_all = _straight_pieces(u1, pos1, routes1)
    wanted = _parent_edge_set(new_graph_stubs)
    new_stub_pieces = [p for p in trial_pieces_all if frozenset(p["parent"]) in wanted]

    old_graph_stubs = stub_endpoints(pathway, before_map, old_stubs)
    old_idxs = set(_stub_piece_indices(before_index.pieces, old_graph_stubs))
    new_stub_x, new_hits = _crossings_between(
        new_stub_pieces,
        before_index.pieces,
        before_index.cell,
        before_index.buckets,
        old_idxs,
    )
    if new_stub_x > old_stub_crossings:
        return None

    n_cross = before_index.n_cross - old_stub_crossings + new_stub_x
    kept_hits = before_index.hits_without_stubs(old_idxs)
    trial_hits = kept_hits + new_hits

    focus = {rid for rid, _gid in old_graph_stubs + new_graph_stubs}
    ar0 = local_angular_resolution(
        before_index.g, before_index.pos, focus, before_index.routes
    )
    ar1 = local_angular_resolution(u1, pos1, focus, routes1)
    if ar0 is not None and ar1 is not None and ar1 > ar0 + 1e-4:
        return None

    import layout_quality as lq

    after = {
        "crossings": float(max(n_cross, 0)),
        "crossing_density": float(_crossing_density(max(n_cross, 0), trial_pieces_all)),
        "crossing_angular_resolution": _crossing_angular_resolution(trial_hits),
        "edge_orthogonality": _edge_orthogonality_from_pieces(trial_pieces_all),
        "edge_length_deviation": _edge_length_deviation(u1, pos1, routes1),
        "angular_resolution": _angular_resolution(u1, pos1, routes1),
        "aspect_ratio": _aspect_ratio(pos1),
        "homogeneity": _homogeneity(pos1),
        "participant_consistency": _through_reaction_share(g1, pos1),
        "curves_through_nodes": float(lq.curves_through_nodes_count(trial_map)),
    }
    after.update(_node_overlaps(u1, pos1))
    return after


