"""Orthogonal polylines for algorithm-placed edges (official edge_orthogonality).

The layout-benchmark Escher reader ignores Bezier handles ``b1``/``b2``: every
drawn piece is the straight chord between end markers.  Curves that only look
orthogonal therefore do not raise ``edge_orthogonality``.  This module writes
real multimarker bend points so each piece is axis-parallel.

Scope:
  - de-novo (algorithm) reactions: non-stub edges become orthogonal polylines
    with at most two bends
  - long algorithm edges on otherwise curated reactions (inter-block links)
  - short cofactor stubs and curated / source-path edges are left untouched

Route choice is deterministic with a fixed candidate budget.  Obstacle clearance
and bend/length cost come first; crossings are only a final tie-break (never
crossing-first).

Accept gates run entirely in the project venv (no layout-benchmark / gdMetriX):
a bent route is kept only when crossings, angular resolution, participant
consistency and edge length deviation do not worsen.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

import edge_routing as er
import line_render
from core import CURRENCY

# Local accept gates: layout_quality_evaluator.ORTHO_DIRS + METRIC_EPS.


STUB_MAX = 90.0          # px: short cofactor spokes stay as stubs
LONG_EDGE = 160.0        # px: long algorithm edges always get orthogonal routes
SNAP_TOL = 18.0          # px: treat as already axis-aligned / snap band
ORTHO_CANDIDATES = 12    # hard budget of path candidates per edge
FIT_EXTEND_TRIALS = 24   # hard budget of post-prefix keep trials (large maps)
LENGTH_CEILING_TOL = 0.05  # px: stop when the lo/hi length window is this tight
# Chord must lie within 30° of H/V (min/max ≤ tan30°) so an L-leave stays in the
# metabolite half-plane and participant_consistency does not fall.
AXIS_ALIGN_RATIO = 3 ** -0.5
CLEAR = 5.0
SEG_PREFIX = 'es_s'      # same prefix as edge_style corridors so quality_gate collapses them


def _is_cur(bigg):
    return re.sub(r'_[a-z][0-9]?$', '', bigg or '') in CURRENCY


def _curated_reactions(data):
    """Reaction ids of curated Atlas modules recorded in metadata ``expert_structures``."""
    out = set()
    for st in (data[0].get('metadata') or {}).get('expert_structures') or []:
        out.update(st.get('reactions') or st.get('reaction_ids') or [])
    return out


def _algo_key(key):
    s = str(key)
    return s.startswith(SEG_PREFIX) or s.startswith('route') or s.startswith('hr') or s.startswith('qg') or s.startswith('or_s')


def _load_pathway(gem_path):
    """Load a Pathway from an SBML GEM via the in-repo evaluation port (no layout-benchmark)."""
    if not gem_path:
        return None
    path = Path(gem_path)
    if not path.is_file():
        return None
    try:
        from evaluation.pathway import read_sbml
        return read_sbml(path)
    except Exception:
        return None


def load_pathway(gem_path):
    """Public: load a GEM Pathway once for reuse across tip / orthogonal gates."""
    return _load_pathway(gem_path)


def gate_metrics(data, pathway=None):
    """Public: crossings, ELD, angular resolution, participant consistency."""
    return _gate_metrics(data, pathway=pathway)


def piece_axis_alignment(data, pathway=None):
    """Mean axis-alignment of drawn marker pieces (Bezier handles ignored).

    Official ``edge_orthogonality`` scores the same pieces. For each piece the
    acute angle to the nearer axis is mapped linearly onto [0, 1] (aligned → 1,
    45° → 0).
    """
    ends, pos, routes = _drawing_parts(data, pathway=pathway)
    segs, spos = _segment_graph(ends, pos, routes)
    vals = []
    quarter = 0.25 * math.pi
    half = 0.5 * math.pi
    for a, b in segs.edges():
        dx = float(spos[b][0] - spos[a][0])
        dy = float(spos[b][1] - spos[a][1])
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            continue
        # Acute angle to the nearer H/V axis, in [0, π/4].
        to_near = min(abs(math.atan2(dy, dx)) % half, half - (abs(math.atan2(dy, dx)) % half))
        vals.append(1.0 - to_near / quarter)
    return float(np.mean(vals)) if vals else 1.0


def _id_lookup(pathway):
    if pathway is None:
        return None
    try:
        from evaluation.escher_bridge import _id_lookup as _lookup
        return _lookup(pathway)
    except Exception:
        return {n: n for n in getattr(pathway, 'graph', {})}


def _drawing_parts(data, pathway=None):
    """Marker polylines as drawn: ends digraph, positions (y up), bend routes.

    Same structure as the official Escher reader so local gates track the same
    geometry layout-benchmark scores.  Optional ``pathway`` filters metabolites
    / reactions to GEM ids.
    """
    body = data[1]
    nodes = body['nodes']
    lookup = _id_lookup(pathway) if pathway is not None else None

    def xy(node):
        return float(node['x']), -float(node['y'])

    drawn, pos, routes = {}, {}, {}
    ends = nx.DiGraph()
    for nid, node in nodes.items():
        if node.get('node_type') != 'metabolite':
            continue
        if lookup is not None and node.get('bigg_id') not in lookup:
            continue
        did = f"m@{nid}"
        drawn[nid] = did
        pos[did] = xy(node)
        ends.add_node(did, kind='species')

    for key, reaction in body['reactions'].items():
        if lookup is not None and reaction.get('bigg_id') not in lookup:
            continue
        seg = nx.Graph()
        for s in reaction['segments'].values():
            seg.add_edge(s['from_node_id'], s['to_node_id'])
        mid = next((n for n in seg if nodes.get(n, {}).get('node_type') == 'midmarker'), None)
        if mid is None:
            continue
        rid = reaction.get('bigg_id') or key
        did = rid if rid not in ends else f"{rid}@{key}"
        ends.add_node(did, kind='reaction')
        pos[did] = xy(nodes[mid])
        coefficient = {m['bigg_id']: m['coefficient'] for m in reaction.get('metabolites') or []}
        for nid in seg:
            if nid not in drawn:
                continue
            path = nx.shortest_path(seg, nid, mid)
            inner = [xy(nodes[n]) for n in path[1:-1] if n in nodes]
            bigg = nodes[nid].get('bigg_id')
            if coefficient.get(bigg, 0) > 0:
                ends.add_edge(did, drawn[nid])
                routes[(did, drawn[nid])] = inner[::-1]
            else:
                ends.add_edge(drawn[nid], did)
                routes[(drawn[nid], did)] = inner
    return ends, pos, routes


def _polyline(pos, routes, u, v):
    return [pos[u], *routes.get((u, v), []), pos[v]]


def _segment_graph(ends, pos, routes):
    """Shared-chord undirected pieces (one edge per unique marker segment)."""
    g = nx.Graph()
    g.add_nodes_from(pos)
    spos = dict(pos)

    def pkey(p):
        return f"@{p[0]:.6g},{p[1]:.6g}"

    for u, v in ends.edges():
        points = _polyline(pos, routes, u, v)
        keys = [u] + [pkey(p) for p in points[1:-1]] + [v]
        for k, p in zip(keys, points):
            spos.setdefault(k, (float(p[0]), float(p[1])))
            g.add_node(k)
        for a, b in zip(keys, keys[1:]):
            if a != b and spos[a] != spos[b]:
                g.add_edge(a, b)
    return g, spos


def _stub_graph(ends, pos, routes):
    """First/last piece stubs so angles at drawn nodes match the picture."""
    g = nx.Graph()
    stub_pos = dict(pos)
    g.add_nodes_from(pos)
    for i, (u, v) in enumerate(ends.edges()):
        points = _polyline(pos, routes, u, v)
        for end, towards, tag in ((u, points[1], 'a'), (v, points[-2], 'b')):
            stub = f'#{i}{tag}'
            stub_pos[stub] = (float(towards[0]), float(towards[1]))
            g.add_edge(end, stub)
    return g, stub_pos


def _edge_angles_at(g, node, pos):
    """Clockwise consecutive angles (radians) between edges at ``node``."""
    nbrs = list(g.neighbors(node))
    if len(nbrs) <= 1:
        return [2 * math.pi]
    cx, cy = pos[node]

    def ang(n):
        x, y = pos[n]
        # Clockwise order from the upward vector (0, 1), matching Purchase AR.
        return math.atan2(x - cx, y - cy) % (2 * math.pi)

    ordered = sorted(nbrs, key=ang)
    out = []
    for i, n in enumerate(ordered):
        a = np.array(pos[n], float) - (cx, cy)
        b = np.array(pos[ordered[(i + 1) % len(ordered)]], float) - (cx, cy)
        na, nb = float(np.hypot(*a)), float(np.hypot(*b))
        if na < 1e-12 or nb < 1e-12:
            out.append(0.0)
            continue
        # Signed clockwise angle from a to b in [0, 2π).
        cross = a[0] * b[1] - a[1] * b[0]
        dot = float(np.clip((a @ b) / (na * nb), -1.0, 1.0))
        theta = math.acos(dot)
        if cross > 0:
            # a→b is counter-clockwise; clockwise supplement
            theta = 2 * math.pi - theta
        out.append(theta)
    return out


def _angular_resolution(g, pos):
    """Purchase angular-resolution deviation on a stub graph (lower is better)."""
    node_count = 0
    deviation_sum = 0.0
    for node in g.nodes():
        neighbours = sum(1 for _ in g.neighbors(node))
        if neighbours <= 1:
            continue
        node_count += 1
        optimal = 2 * math.pi / neighbours
        min_angle = min(_edge_angles_at(g, node, pos))
        deviation_sum += abs((optimal - min_angle) / optimal)
    return deviation_sum / node_count if node_count else 0.0


def _count_crossings(g, spos):
    """Number of proper pairwise segment crossings (shared chords once)."""
    if g.number_of_edges() < 2:
        return 0
    segs = []
    for a, b in g.edges():
        segs.append((spos[a][0], spos[a][1], spos[b][0], spos[b][1]))
    S = np.array(segs, float)
    total = 0
    for i in range(len(segs)):
        other = S[i + 1:]
        if len(other):
            total += er._seg_cross((S[i, 0], S[i, 1]), (S[i, 2], S[i, 3]), other)
    return int(total)


def _participant_consistency(ends, pos, routes):
    """Share of (reactant, product) leave pairs on opposite sides of a reaction."""
    opposite = pairs = 0
    for r, d in ends.nodes(data=True):
        if d.get('kind') != 'reaction':
            continue
        c = np.array(pos[r], float)
        ins = [np.array(_polyline(pos, routes, u, r)[-2]) - c for u, _ in ends.in_edges(r)]
        outs = [np.array(_polyline(pos, routes, r, v)[1]) - c for _, v in ends.out_edges(r)]
        for a in ins:
            for b in outs:
                pairs += 1
                opposite += float(np.dot(a, b)) < 0
    return (opposite / pairs) if pairs else None


def _edge_lengths(ends, pos, routes):
    lengths = []
    for u, v in ends.edges():
        pts = np.array(_polyline(pos, routes, u, v), float)
        lengths.append(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))
    return [x for x in lengths if x > 0]


def _gate_metrics(data, pathway=None):
    """In-venv accept gates: crossings, AR, PC, edge length deviation."""
    ends, pos, routes = _drawing_parts(data, pathway=pathway)
    segs, spos = _segment_graph(ends, pos, routes)
    stubs, stub_pos = _stub_graph(ends, pos, routes)
    lengths = _edge_lengths(ends, pos, routes)
    mean = float(np.mean(lengths)) if lengths else 0.0
    return {
        'crossings': float(_count_crossings(segs, spos)),
        'edge_length_deviation': (
            float(np.mean([abs(mean - x) / mean for x in lengths])) if mean else None),
        'angular_resolution': float(_angular_resolution(stubs, stub_pos)),
        'participant_consistency': _participant_consistency(ends, pos, routes),
    }


def _gate_ok(pre, cur, eps=None):
    """True when no gated metric is worse than ``pre`` beyond ``eps``."""
    from layout_quality_evaluator import METRIC_EPS, ORTHO_DIRS, metrics_not_worse

    if not pre or not cur:
        return False
    return metrics_not_worse(pre, cur, ORTHO_DIRS, METRIC_EPS if eps is None else eps)


def _path_hits(pts, rects, pad=CLEAR):
    if len(pts) < 2 or not len(rects):
        return 0
    P = np.asarray(pts[:-1], float)
    Q = np.asarray(pts[1:], float)
    return int(er._blocked_many(P, Q, rects, pad).any(axis=1).sum())


def _path_length(pts):
    P = np.asarray(pts, float)
    return float(np.sum(np.hypot(*(np.diff(P, axis=0).T))))


def _path_cross(pts, S):
    if not len(S) or len(pts) < 2:
        return 0
    return er._seg_cross(pts[:-1], pts[1:], S)


def ortho_candidates(p, q, budget=ORTHO_CANDIDATES):
    """Deterministic axis-parallel paths from p to q with ≤2 bends (waypoints include ends)."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    out = []

    def add(pts):
        if len(out) >= budget:
            return
        arr = [np.asarray(x, float) for x in pts]
        # drop zero-length pieces
        cleaned = [arr[0]]
        for a in arr[1:]:
            if float(np.hypot(*(a - cleaned[-1]))) > 1e-6:
                cleaned.append(a)
        if len(cleaned) < 2:
            return
        if len(cleaned) - 2 > 2:
            return
        # every piece must be axis-parallel
        for a, b in zip(cleaned[:-1], cleaned[1:]):
            if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6:
                return
        key = tuple((round(float(x[0]), 2), round(float(x[1]), 2)) for x in cleaned)
        if key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    seen = set()
    # 0-bend (already aligned, or snapped)
    if abs(p[0] - q[0]) <= SNAP_TOL:
        add([p, np.array([p[0], q[1]]), q] if abs(p[1] - q[1]) > 1e-6 else [p, q])
        add([p, np.array([p[0], q[1]])])
    if abs(p[1] - q[1]) <= SNAP_TOL:
        add([p, np.array([q[0], p[1]]), q] if abs(p[0] - q[0]) > 1e-6 else [p, q])
        add([p, np.array([q[0], p[1]])])
    if abs(p[0] - q[0]) < 1e-6 or abs(p[1] - q[1]) < 1e-6:
        add([p, q])

    # 1-bend L routes
    add([p, np.array([q[0], p[1]]), q])
    add([p, np.array([p[0], q[1]]), q])

    # 2-bend Z routes at fixed fractions (deterministic budget; no overshoot
    # U-detours — those inflate edge_length_deviation).
    dx, dy = float(q[0] - p[0]), float(q[1] - p[1])
    for f in (0.5, 1.0 / 3.0, 2.0 / 3.0):
        x = p[0] + f * dx
        y = p[1] + f * dy
        add([p, np.array([x, p[1]]), np.array([x, q[1]]), q])
        add([p, np.array([p[0], y]), np.array([q[0], y]), q])
        if len(out) >= budget:
            break
    return out[:budget]


def _length_deviation(lengths):
    """Purchase edge_length_deviation on a list of polyline lengths."""
    xs = [float(x) for x in lengths if x and x > 0]
    if not xs:
        return 0.0
    mean = float(sum(xs) / len(xs))
    if mean <= 0:
        return 0.0
    return float(sum(abs(mean - x) / mean for x in xs) / len(xs))


def _collect_edge_lengths(data):
    """Mid→metabolite polyline lengths keyed by (reaction, metabolite)."""
    body = data[1]
    nodes = body['nodes']
    out = {}
    for rx in body['reactions'].values():
        rid = rx['bigg_id']
        for path in line_render._terminals(rx, nodes):
            if nodes.get(path[-1], {}).get('node_type') != 'metabolite':
                continue
            met = nodes[path[-1]].get('bigg_id')
            pts = [(nodes[n]['x'], nodes[n]['y']) for n in path]
            out[(rid, met)] = _path_length(pts)
    return out


def choose_ortho(p, q, rects, S_other, prefer_first_dir=None, max_cross=None,
                 max_length=None, require_leave_toward=True):
    """Pick an orthogonal path.

    Score order: obstacle hits, crossings (capped by ``max_cross`` when set),
    length, bends, leave-dir mismatch.  Crossings are never the primary objective
    (hits come first); ``max_length`` rejects detours that would stretch the edge
    beyond the pre-pass length budget.  ``require_leave_toward`` keeps the first
    step in the metabolite half-plane (participant_consistency).
    """
    best = None
    chord = np.asarray(q, float) - np.asarray(p, float)
    for path in ortho_candidates(p, q):
        length = _path_length(path)
        if max_length is not None and length > max_length + 1e-6:
            continue
        leave = path[1] - path[0]
        if require_leave_toward and float(leave @ chord) <= 0:
            continue
        hits = _path_hits(path, rects)
        cross = _path_cross(path, S_other)
        if max_cross is not None and cross > max_cross:
            continue
        bends = len(path) - 2
        leave_n = float(np.hypot(*leave))
        mismatch = 0.0
        if prefer_first_dir is not None and leave_n > 1e-9:
            u = prefer_first_dir / max(float(np.hypot(*prefer_first_dir)), 1e-9)
            v = leave / leave_n
            mismatch = 1.0 - float(u @ v)
        score = (hits, cross, round(length, 2), bends, round(mismatch, 3))
        if best is None or score < best[0]:
            best = (score, path)
    if best is None:
        return None, None
    return best[1], best[0]


def _next_es_id(nodes, counter):
    """Allocate a fresh multimarker id that does not collide with existing nodes."""
    while True:
        counter[0] += 1
        nid = f'es{counter[0]}'
        if nid not in nodes:
            return nid


def write_polyline(rx, mid_id, met_id, path, nodes, counter):
    """Replace mid→met connection with an orthogonal multimarker chain (b1=b2=None)."""
    # Drop existing segments that form the mid→met path (algo or direct).
    out = defaultdict(list)
    for key, s in rx['segments'].items():
        out[s['from_node_id']].append((key, s))
    drop_keys = set()
    drop_nodes = set()
    for key, s in list(rx['segments'].items()):
        if s['from_node_id'] != mid_id:
            continue
        if s['to_node_id'] == met_id:
            drop_keys.add(key)
            continue
        cur = s['to_node_id']
        chain = []
        ok = False
        while nodes.get(cur, {}).get('node_type') == 'multimarker':
            chain.append(cur)
            nxt = out.get(cur)
            if not nxt:
                break
            cur = nxt[0][1]['to_node_id']
        if cur == met_id:
            ok = True
        if ok:
            drop_keys.add(key)
            drop_nodes.update(chain)
            for k2, s2 in list(rx['segments'].items()):
                if s2['from_node_id'] in chain or s2['to_node_id'] in chain:
                    drop_keys.add(k2)
    for k in drop_keys:
        rx['segments'].pop(k, None)
    for nid in drop_nodes:
        nodes.pop(nid, None)

    prev = mid_id
    # path includes p and q; intermediate points become multimarkers
    for i, pt in enumerate(path[1:], start=1):
        last = i == len(path) - 1
        if last:
            to = met_id
        else:
            to = _next_es_id(nodes, counter)
            nodes[to] = dict(node_type='multimarker',
                             x=round(float(pt[0]), 2), y=round(float(pt[1]), 2))
        counter[0] += 1
        sid = f'{SEG_PREFIX}{counter[0]}'
        while sid in rx['segments']:
            counter[0] += 1
            sid = f'{SEG_PREFIX}{counter[0]}'
        rx['segments'][sid] = dict(from_node_id=prev, to_node_id=to, b1=None, b2=None)
        prev = to


def should_ortho(rid, met_bigg, length, curated):
    """Whether this algorithm edge should become an orthogonal polyline."""
    stub = _is_cur(met_bigg) and length <= STUB_MAX
    if stub:
        return False
    if rid not in curated:
        return True
    # curated reaction: only long algorithm (inter-block) edges
    return length >= LONG_EDGE


def _chord_near_axis(p, q, max_ratio=AXIS_ALIGN_RATIO):
    """True when p→q is within arctan(max_ratio) of horizontal or vertical."""
    dx, dy = abs(float(q[0] - p[0])), abs(float(q[1] - p[1]))
    m = max(dx, dy)
    if m < 1e-9:
        return True
    return (min(dx, dy) / m) <= max_ratio + 1e-12


def apply_orthogonal(data, skip_edges=(), only=None, bezier_stub_fn=None, gem_path=None,
                     pathway=None):
    """Route eligible algorithm edges as orthogonal polylines in place.

    ``bezier_stub_fn(rx, mid_id, seg_key, seg, t, side, u, …)`` may style short
    cofactor stubs; when None, stubs stay unchanged.  ``gem_path`` optionally
    filters the local accept gate to GEM species/reactions.  Pass a preloaded
    ``pathway`` to avoid re-parsing the SBML.
    """
    import layout_quality as lq
    import edge_style

    body = data[1]
    nodes, reactions = body['nodes'], body['reactions']
    # Shared-axis placement of mid/main nodes stays in earlier layout stages
    # (moving them here raised official overlapping_pairs). Orthogonal candidates
    # already snap nearly-aligned endpoints (SNAP_TOL).

    all_rects, all_owner, all_kind, _ = lq._final_obstacles(data)
    is_node = np.array([k == 'node' for k in all_kind])
    rects = all_rects[is_node] if len(all_rects) else np.zeros((0, 4))
    owner_arr = np.array(all_owner, dtype=object)[is_node] if len(all_owner) else np.zeros(0, dtype=object)
    block_rects, block_owners = edge_style._curated_block_rects(data)
    if len(block_rects):
        rects = np.vstack([rects, block_rects]) if len(rects) else block_rects
        owner_arr = np.concatenate([owner_arr, block_owners]) if len(owner_arr) else block_owners

    skip = {tuple(e) for e in skip_edges}
    only_set = set(only) if only is not None else None
    curated = _curated_reactions(data)
    if pathway is None:
        pathway = _load_pathway(gem_path)
    import copy
    pre_snap = copy.deepcopy(data)
    # Compose always passes gem_path.  If it was set but the GEM cannot be read,
    # refuse to write bent orthos (ungated changes failed the merge gate).
    if gem_path and pathway is None:
        return dict(
            edges=0, curated_or_source=0, stubs=0, orthogonal=0, bends_added=0,
            unresolved=0, stub_bezier=0, crossing_cap=0, skipped='gem_pathway_unreadable')

    # Pre-pass length distribution (Purchase deviation on marker polylines).
    length_map = _collect_edge_lengths(data)
    length_keys = sorted(length_map)
    lengths = [length_map[k] for k in length_keys]
    key_index = {k: i for i, k in enumerate(length_keys)}
    pre_dev = _length_deviation(lengths)

    # Segment index for soft crossing tie-break (list growth; materialize when needed).
    seg_rows, seg_owner = [], []
    for rx in reactions.values():
        for path in line_render._terminals(rx, nodes):
            if nodes[path[-1]].get('node_type') != 'metabolite':
                continue
            pts = [(nodes[n]['x'], nodes[n]['y']) for n in path]
            for a, b in zip(pts[:-1], pts[1:]):
                seg_rows.append((a[0], a[1], b[0], b[1]))
                seg_owner.append(rx['bigg_id'])
    S_all = np.array(seg_rows, float).reshape(-1, 4) if seg_rows else np.zeros((0, 4))
    S_owner = np.array(seg_owner, dtype=object) if seg_owner else np.zeros(0, dtype=object)
    new_segs, new_owners = [], []

    pre_gate = _gate_metrics(data, pathway=pathway)
    crossing_cap = int(pre_gate['crossings'])
    report = dict(
        edges=0, curated_or_source=0, stubs=0, orthogonal=0, bends_added=0,
        unresolved=0, stub_bezier=0, crossing_cap=crossing_cap,
        length_deviation_cap=round(pre_dev, 6))
    report['gate_before'] = {k: (None if v is None else round(v, 6)) for k, v in pre_gate.items()}
    routed = []
    counter = [0]

    def max_length_for(key, old_L):
        """Largest replacement length that keeps Purchase deviation ≤ pre_dev."""
        idx = key_index.get(key)
        if idx is None:
            return old_L
        # Ceiling: Manhattan upper bound for ≤2-bend ortho paths (√2 × chord).
        lo, hi = float(old_L), float(old_L) * math.sqrt(2)
        best = float(old_L)
        while hi - lo >= LENGTH_CEILING_TOL:
            mid = 0.5 * (lo + hi)
            trial = list(lengths)
            trial[idx] = mid
            if _length_deviation(trial) <= pre_dev + 1e-12:
                best = mid
                lo = mid
            else:
                hi = mid
        return best

    for rx in sorted(reactions.values(), key=lambda r: r['bigg_id']):
        rid = rx['bigg_id']
        if only_set is not None and rid not in only_set:
            continue
        mid_id = next((s['from_node_id'] for s in rx['segments'].values()
                       if nodes.get(s['from_node_id'], {}).get('node_type') == 'midmarker'), None)
        if mid_id is None:
            continue
        # Do not collapse other algorithm chains here: flattening edges we later
        # leave alone changes leave angles and can drop participant_consistency.
        # write_polyline replaces only the mid→met chain being orthogonalised.

        coeff = {}
        for m in rx.get('metabolites') or []:
            coeff.setdefault(m['bigg_id'], m['coefficient'])
        P = np.array([nodes[mid_id]['x'], nodes[mid_id]['y']], float)

        # Direct mid→metabolite segments plus one-hop mid→multimarker→… terminals.
        edges = []
        seen_met = set()
        for key, seg in list(rx['segments'].items()):
            if seg['from_node_id'] != mid_id:
                continue
            t = nodes.get(seg['to_node_id'])
            if t and t.get('node_type') == 'metabolite':
                edges.append((key, seg, t, seg['to_node_id']))
                seen_met.add(seg['to_node_id'])
        for path in line_render._terminals(rx, nodes):
            if path[0] != mid_id or nodes.get(path[-1], {}).get('node_type') != 'metabolite':
                continue
            met_nid = path[-1]
            if met_nid in seen_met:
                continue
            t = nodes[met_nid]
            # Synthetic head segment for chain routing (write_polyline walks the chain).
            edges.append((f'_chain_{met_nid}', dict(from_node_id=mid_id, to_node_id=met_nid,
                                                    b1=None, b2=None), t, met_nid))
            seen_met.add(met_nid)
        if not edges:
            continue
        subs = [np.array([t['x'], t['y']]) - P for _, _, t, _ in edges if coeff.get(t['bigg_id'], 0) < 0 and not _is_cur(t['bigg_id'])]
        prods = [np.array([t['x'], t['y']]) - P for _, _, t, _ in edges if coeff.get(t['bigg_id'], 0) > 0 and not _is_cur(t['bigg_id'])]
        u = (np.mean(prods, axis=0) if prods else np.zeros(2)) - (np.mean(subs, axis=0) if subs else np.zeros(2))
        u = u / np.hypot(*u) if np.hypot(*u) > 1e-6 else np.array([1.0, 0.0])

        for key, seg, t, met_nid in edges:
            report['edges'] += 1
            bg = t.get('bigg_id')
            if (rid, bg) in skip or str(key).startswith('spseg'):
                report['curated_or_source'] += 1
                continue
            # Leave curated Bezier / multimarker source geometry alone when the
            # segment already carries controls and belongs to a curated reaction.
            if rid in curated and (seg.get('b1') or seg.get('b2')) and not _algo_key(key):
                report['curated_or_source'] += 1
                continue
            q = np.array([t['x'], t['y']], float)
            L = float(np.hypot(*(q - P)))
            if L < 1e-6:
                continue
            edge_key = (rid, bg)
            if not should_ortho(rid, bg, L, curated):
                report['stubs'] += 1
                # Short cofactor stub: optional Bezier styling; official metric
                # ignores handles, so leave geometry alone when no stub styler.
                if str(key).startswith('_chain_'):
                    continue
                if bezier_stub_fn is not None:
                    bezier_stub_fn(rx, mid_id, key, seg, t, P, q, u)
                    report['stub_bezier'] += 1
                continue
            if not _chord_near_axis(P, q):
                report['unresolved'] += 1
                continue

            mine = (owner_arr == mid_id) | (owner_arr == met_nid) if len(owner_arr) else np.zeros(0, bool)
            others = rects[~mine] if len(rects) else rects
            if new_segs:
                S_use = np.vstack([S_all, np.asarray(new_segs, float)]) if len(S_all) else np.asarray(new_segs, float)
                O_use = np.concatenate([S_owner, np.asarray(new_owners, dtype=object)]) if len(S_owner) else np.asarray(new_owners, dtype=object)
            else:
                S_use, O_use = S_all, S_owner
            S_other = S_use[O_use != rid] if len(S_use) else S_use
            role = 'product' if coeff.get(bg, 0) > 0 else 'substrate'
            prefer = u if role == 'product' else -u
            straight_cross = _path_cross([P, q], S_other)
            max_len = max_length_for(edge_key, length_map.get(edge_key, L))
            path, score = choose_ortho(
                P, q, others, S_other, prefer_first_dir=prefer, max_cross=0,
                max_length=max_len)
            if path is None and straight_cross > 0:
                path, score = choose_ortho(
                    P, q, others, S_other, prefer_first_dir=prefer,
                    max_cross=straight_cross, max_length=max_len)
            if path is None:
                # Keep existing geometry rather than raise crossings / stretch /
                # flip leave directions.
                report['unresolved'] += 1
                continue
            bends = max(0, len(path) - 2)
            # Incremental ship: only real multimarker bends.  0-bend handle clears
            # drop Bézier coverage without helping official orthogonality.
            if bends < 1:
                report['unresolved'] += 1
                continue
            path = _quantize_ortho_path(path)
            plen = _path_length(path)
            if plen > max_len + 1e-6:
                report['unresolved'] += 1
                continue
            # Re-check Purchase deviation with the quantized length before write.
            if edge_key in key_index:
                trial = list(lengths)
                trial[key_index[edge_key]] = plen
                from layout_quality_evaluator import METRIC_EPS
                if _length_deviation(trial) > pre_dev + METRIC_EPS:
                    report['unresolved'] += 1
                    continue
            write_polyline(rx, mid_id, met_nid, path, nodes, counter)
            report['orthogonal'] += 1
            report['bends_added'] += bends
            if edge_key in key_index:
                lengths[key_index[edge_key]] = plen
            routed.append(dict(
                reaction=rid, metabolite=bg, kind='orthogonal',
                bends=bends, hits=int(score[0]), crossings=int(score[1]),
                length=round(plen, 1), chord=round(L, 1),
                long_edge=L >= LONG_EDGE, curated_reaction=rid in curated,
                path=[(round(float(p[0]), 2), round(float(p[1]), 2)) for p in path],
                mid=mid_id, met_nid=met_nid))
            for a, b in zip(path[:-1], path[1:]):
                new_segs.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
                new_owners.append(rid)

    # Refit from the pre-orthogonal snapshot under the local accept gate.
    kept_n = _install_gated_orthogonals(
        data, pre_snap=pre_snap, pre_gate=pre_gate, routed=routed, pathway=pathway)
    reverted = max(0, len(routed) - kept_n)
    report['reverted_crossing'] = reverted
    report['orthogonal'] = kept_n
    report['length_deviation_after'] = round(_length_deviation(_collect_edge_lengths(data).values()), 6)
    surviving = [r for r in routed if not _edge_was_reverted(data, r)]
    after_gate = _gate_metrics(data, pathway=pathway)
    report['gate_after'] = {k: (None if v is None else round(v, 6)) for k, v in after_gate.items()}
    report['gate_ok'] = _gate_ok(pre_gate, after_gate)
    # Hard length-cap check on the final map (independent of gate metric rounding).
    from layout_quality_evaluator import METRIC_EPS
    if report['length_deviation_after'] > pre_dev + METRIC_EPS:
        report['gate_ok'] = False
    if not report['gate_ok']:
        data[0], data[1] = copy.deepcopy(pre_snap)[0], copy.deepcopy(pre_snap)[1]
        surviving = []
        report['orthogonal'] = 0
        report['reverted_crossing'] = len(routed)
        report['gate_ok'] = True
        report['gate_after'] = {k: (None if v is None else round(v, 6)) for k, v in pre_gate.items()}
        report['length_deviation_after'] = round(pre_dev, 6)
    data[0]['metadata']['routed_edges'] = surviving
    data[0].setdefault('metadata', {})['orthogonal_routing'] = dict(
        orthogonal=report['orthogonal'], stubs=report['stubs'], bends_added=report['bends_added'],
        reverted_crossing=reverted, length_deviation_cap=report['length_deviation_cap'],
        length_deviation_after=report['length_deviation_after'])
    return report


def _quantize_ortho_path(path):
    """Snap bend coordinates so every piece is exactly axis-parallel."""
    pts = [np.asarray(p, float).copy() for p in path]
    for i in range(1, len(pts) - 1):
        prev, cur, nxt = pts[i - 1], pts[i], pts[i + 1]
        # Prefer sharing x with the longer adjacent axis leg.
        if abs(cur[0] - prev[0]) <= abs(cur[1] - prev[1]):
            cur[0] = prev[0]
        else:
            cur[1] = prev[1]
        if abs(nxt[0] - cur[0]) <= abs(nxt[1] - cur[1]):
            cur[0] = nxt[0]
        else:
            cur[1] = nxt[1]
        pts[i] = cur
    # Final pass: force each piece onto H or V by copying from the end that is closer.
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if abs(a[0] - b[0]) <= abs(a[1] - b[1]):
            b[0] = a[0]
        else:
            b[1] = a[1]
        pts[i + 1] = b
    return pts


def _edge_was_reverted(data, row):
    """True when a bent ortho was collapsed back to a chord."""
    body = data[1]
    nodes = body['nodes']
    for rx in body['reactions'].values():
        if rx['bigg_id'] != row['reaction']:
            continue
        for path in line_render._terminals(rx, nodes):
            if nodes.get(path[-1], {}).get('bigg_id') != row['metabolite']:
                continue
            return not any(nodes.get(n, {}).get('node_type') == 'multimarker' for n in path[1:-1])
    return True


def _piece_crossing_count(data, pathway=None):
    """Local shared-chord piece crossings (same drawing as the accept gate)."""
    return int(_gate_metrics(data, pathway=pathway)['crossings'])


def _write_saved_ortho(data, rid, mid, met_nid, path_pts, counter):
    nodes = data[1]['nodes']
    rx = next(r for r in data[1]['reactions'].values() if r['bigg_id'] == rid)
    path = [np.asarray(p, float) for p in path_pts]
    # Snap ends to current node positions (alignment-stable).
    path[0] = np.array([nodes[mid]['x'], nodes[mid]['y']], float)
    path[-1] = np.array([nodes[met_nid]['x'], nodes[met_nid]['y']], float)
    write_polyline(rx, mid, met_nid, path, nodes, counter)


def _keep_priority(row, length):
    """Deterministic keep order: short detours and clean local routes first."""
    cross = int(row.get('crossings', 99)) if row else 99
    hits = int(row.get('hits', 99)) if row else 99
    bends = int(row.get('bends', 99)) if row else 99
    chord = float(row.get('chord') or 0.0) if row else 0.0
    stretch = round(length / chord, 4) if chord > 1e-9 else 99.0
    long_edge = bool(row.get('long_edge')) if row else length >= LONG_EDGE
    return (
        0 if cross == 0 and hits == 0 else 1,
        stretch,
        bends,
        0 if long_edge else 1,
        cross, hits,
        round(length, 2),
    )


def _bent_candidates(routed):
    """Priority-sorted bent orthogonal candidates from this pass."""
    saved = []
    for row in routed:
        if row.get('kind') != 'orthogonal' or int(row.get('bends') or 0) <= 0 or not row.get('path'):
            continue
        length = float(row.get('length') or _path_length(row['path']))
        saved.append((
            _keep_priority(row, length),
            row['reaction'], row['metabolite'], row['mid'], row['met_nid'],
            row['path'], length))
    saved.sort(key=lambda t: (t[0], t[1], t[2]))
    return saved


def _install_gated_orthogonals(data, pre_snap, pre_gate, routed, pathway):
    """Reset to ``pre_snap`` and re-install orthos that pass the local gate.

    Each orthogonalisation is accepted only when angular resolution, crossings,
    participant consistency and edge length deviation stay within ``pre_gate``.
    """
    import copy
    saved = _bent_candidates(routed)
    data[0], data[1] = copy.deepcopy(pre_snap)[0], copy.deepcopy(pre_snap)[1]
    if not saved:
        return 0
    # Fast path: try all bent routes; keep if the gate holds.
    counter = [10 ** 6]
    probe = copy.deepcopy(data)
    for _prio, rid, _met, mid, met_nid, path_pts, _L in saved:
        _write_saved_ortho(probe, rid, mid, met_nid, path_pts, counter)
    if _gate_ok(pre_gate, _gate_metrics(probe, pathway)):
        data[0], data[1] = probe[0], probe[1]
        return len(saved)

    n = len(saved)
    snap0 = copy.deepcopy(data)
    counter = [10 ** 6]

    def install_prefix(k):
        data[0], data[1] = copy.deepcopy(snap0)[0], copy.deepcopy(snap0)[1]
        for _prio, rid, _met, mid, met_nid, path_pts, _L in saved[:k]:
            _write_saved_ortho(data, rid, mid, met_nid, path_pts, counter)

    lo, hi, best_k = 0, n, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        install_prefix(mid)
        if _gate_ok(pre_gate, _gate_metrics(data, pathway)):
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1
    install_prefix(best_k)
    kept = best_k
    budget = FIT_EXTEND_TRIALS
    for item in saved[best_k:]:
        if budget <= 0:
            break
        _prio, rid, _met, mid, met_nid, path_pts, _L = item
        snap = copy.deepcopy(data)
        _write_saved_ortho(data, rid, mid, met_nid, path_pts, counter)
        budget -= 1
        if _gate_ok(pre_gate, _gate_metrics(data, pathway)):
            kept += 1
        else:
            data[0], data[1] = snap[0], snap[1]
    return kept
