"""Translation-only curated placement helpers.

Curated structures may only be translated (and uniformly scaled); rotation and reflection
are forbidden.  Blocks that share a source map keep one common scale and their original
relative offsets from that map.  When absolute source-map coordinates are recoverable
(via ``source_sha256``), same-source fragments are placed as one rigid group so a large
shared map is reused instead of packing each cut independently.
"""
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from core import INPUTS, ROOT, read, unify_map_scale
from map_edit import set_node_xy, translate_node, translate_reaction_label

# Angle (degrees) below which a Kabsch rotation is treated as "axes kept".
AXES_ANGLE_DEG = 12.0
# Same-source rigid backbone: group must cover this share of model reactions, or this
# share of curated reactions with at least RIGID_MIN_BLOCKS members.
RIGID_MODEL_SHARE = 0.35
RIGID_CURATED_SHARE = 0.50
RIGID_MIN_BLOCKS = 3
# Absolute same-map rebuild uses source coordinates 1:1 (one uniform scale for the whole
# group).  Do not compress mid↔met lengths — that tangles the curated geometry.
RIGID_ABS_SCALE = 1.0


def _angle_deg(rot):
    """Rotation angle in degrees for a 2×2 matrix (assumes det ≈ +1)."""
    r = np.asarray(rot, float)
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


@lru_cache(maxsize=1)
def _registry_by_sha():
    path = INPUTS / 'source_registry.json'
    if not path.exists():
        return {}
    reg = read(path)
    return {f['sha256']: f for f in reg.get('files', []) if f.get('sha256')}


@lru_cache(maxsize=32)
def absolute_source_coords(source_sha256):
    """Reaction midmarker + primary metabolite coordinates from the original source map.

    Returns dict(reactions={id:[x,y]}, metabolites={id:[x,y]}, path=...) or None.
    """
    if not source_sha256:
        return None
    meta = _registry_by_sha().get(source_sha256)
    path = None
    if meta and meta.get('path'):
        path = ROOT / meta['path']
        if not path.is_file():
            path = Path(meta['path'])
    if path is None or not path.is_file():
        # Fall back to map_observations (absolute xy already extracted).
        return _absolute_from_observations(source_sha256)
    try:
        payload = read(path)
    except Exception:
        return _absolute_from_observations(source_sha256)
    # Escher: [header, {nodes, reactions}]
    if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], dict):
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            rel = str(path)
        return _absolute_from_escher(payload[1], rel)
    return _absolute_from_observations(source_sha256)


def _absolute_from_escher(body, path, skip_nids=None):
    nodes = body.get('nodes') or {}
    reactions = body.get('reactions') or {}
    rx, mets = {}, defaultdict(list)
    edge_lengths = []
    # Midmarker↔metabolite pairs for corridor padding compress (r:id, m:id).
    edges = []
    skip = {str(x) for x in (skip_nids or ())}
    for nid, n in nodes.items():
        if skip and str(nid) in skip:
            continue
        if n.get('node_type') == 'metabolite' and n.get('bigg_id'):
            mets[n['bigg_id']].append((n.get('node_is_primary', True), n['x'], n['y']))
    for _rid, r in reactions.items():
        bigg = r.get('bigg_id')
        if not bigg:
            continue
        mid = None
        met_nodes = []
        for seg in (r.get('segments') or {}).values():
            for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                n = nodes.get(nid) or {}
                if n.get('node_type') == 'midmarker':
                    mid = [n['x'], n['y']]
                elif n.get('node_type') == 'metabolite' and n.get('bigg_id'):
                    met_nodes.append((n['bigg_id'], [n['x'], n['y']]))
        if mid is None and 'label_x' in r:
            mid = [r['label_x'], r['label_y']]
        if mid is not None:
            rx[bigg] = mid
            for met_id, mp in met_nodes:
                d = float(np.hypot(mid[0] - mp[0], mid[1] - mp[1]))
                if d > 1e-6:
                    edge_lengths.append(d)
                    edges.append(('r:' + bigg, 'm:' + met_id))
    met_xy = {}
    for mid, opts in mets.items():
        opts = sorted(opts, key=lambda t: (not t[0], t[1], t[2]))
        met_xy[mid] = [opts[0][1], opts[0][2]]
    return dict(reactions=rx, metabolites=met_xy, path=path, edge_lengths=edge_lengths, edges=edges)


@lru_cache(maxsize=1)
def _observed_maps():
    """Source-map observations by source_sha256 (first observation wins), read once per process."""
    from core import source_maps
    out = {}
    for m in source_maps():
        out.setdefault(m.get('source_sha256'), m)
    return out


@lru_cache(maxsize=8)
def _absolute_from_observations(source_sha256):
    m = _observed_maps().get(source_sha256)
    if m is None:
        return None
    rx, met_xy = {}, {}
    edge_lengths = []
    glyphs = m.get('glyphs') or {}
    if isinstance(glyphs, list):
        glyphs = {g['key']: g for g in glyphs if isinstance(g, dict) and 'key' in g}
    for g in glyphs.values() if isinstance(glyphs, dict) else []:
        if g.get('kind') == 'metabolite' and g.get('entity') and g.get('xy'):
            # Prefer primary when present.
            ent = g['entity']
            if ent not in met_xy or g.get('primary', True):
                met_xy[ent] = list(g['xy'])
    for r in m.get('reactions') or []:
        if isinstance(r, dict) and r.get('id') and r.get('xy'):
            rx[r['id']] = list(r['xy'])
            for gid in r.get('glyphs') or []:
                g = glyphs.get(gid) if isinstance(glyphs, dict) else None
                if not g or g.get('kind') != 'metabolite' or not g.get('xy'):
                    continue
                d = float(np.hypot(r['xy'][0] - g['xy'][0], r['xy'][1] - g['xy'][1]))
                if d > 1e-6:
                    edge_lengths.append(d)
    return dict(reactions=rx, metabolites=met_xy, path=m.get('name') or 'map_observations',
                edge_lengths=edge_lengths)


def enrich_absolute_source_xy(items):
    """Replace collapsed module-local ``source_xy`` with absolute source-map coords when possible."""
    report = dict(enriched=0, skipped=0, maps={})
    for p in items or []:
        sha = p.get('source_sha256')
        abs_map = absolute_source_coords(sha) if sha else None
        if not abs_map:
            report['skipped'] += 1
            continue
        new_xy = {}
        for rid in p.get('reaction_ids') or []:
            if rid in abs_map['reactions']:
                new_xy['r:' + rid] = list(abs_map['reactions'][rid])
        # Metabolite keys already present on the proposal / block.
        for k in list((p.get('source_xy') or p.get('positions') or {}).keys()):
            if k.startswith('m:') and k[2:] in abs_map['metabolites']:
                new_xy[k] = list(abs_map['metabolites'][k[2:]])
        if len(new_xy) < 2:
            report['skipped'] += 1
            continue
        p['source_xy'] = new_xy
        p['source_xy_absolute'] = True
        report['enriched'] += 1
        report['maps'][sha[:12]] = abs_map.get('path')
    return items, report


def _group_key(p):
    """Prefer source_sha256 (exact map bytes); fall back to map_id."""
    return p.get('source_sha256') or p.get('map_id')


def reapply_same_map_layout(selected):
    """Rewrite ``positions`` of selected modules: one scale per source map, source relative offsets.

    Absolute source-map coordinates are attached for later rigid packing, but positions are
    not rewritten from them here — fit-frame scales are for module-local coords and would
    explode absolute map units.  Rigid absolute placement runs after ``pack_modules``.
    """
    _, enrich_rep = enrich_absolute_source_xy(selected)
    by_map = defaultdict(list)
    for p in selected or []:
        if p.get('source_xy') and p.get('positions') and _group_key(p):
            by_map[_group_key(p)].append(p)
    report = dict(maps=0, modules=0, common_scales={}, absolute=enrich_rep.get('enriched', 0),
                  deferred_absolute=0)
    for map_id, group in by_map.items():
        # Absolute groups: keep module-local positions until bake_same_map_positions.
        if any(p.get('source_xy_absolute') for p in group):
            report['maps'] += 1
            report['modules'] += len(group)
            report['deferred_absolute'] += len(group)
            continue
        frames = [p.get('source_to_initial_frame') for p in group]
        scale = unify_map_scale(frames)
        report['common_scales'][map_id if isinstance(map_id, str) and len(map_id) < 40 else (map_id[:24] if map_id else '?')] = round(scale, 4)
        report['maps'] += 1
        report['modules'] += len(group)
        src_c, dst_c = [], []
        for p in group:
            sxy = np.array(list(p['source_xy'].values()), float)
            dxy = np.array([p['positions'][k] for k in p['source_xy'] if k in p['positions']], float)
            if len(sxy) == 0 or len(dxy) == 0:
                continue
            src_c.append(sxy.mean(0))
            dst_c.append(dxy.mean(0))
        if not src_c:
            continue
        src_c, dst_c = np.array(src_c), np.array(dst_c)
        # Collapsed module-local centroids (~0): cannot restore relatives here.
        if float(np.max(np.ptp(src_c, axis=0))) < 1e-3:
            continue
        group_delta = dst_c.mean(0) - scale * src_c.mean(0)
        for p in group:
            new_pos = {}
            for k, xy in p['source_xy'].items():
                new_pos[k] = (scale * np.asarray(xy, float) + group_delta).tolist()
            for k, xy in p['positions'].items():
                if k not in new_pos:
                    new_pos[k] = xy
            p['positions'] = new_pos
            frame = dict(p.get('source_to_initial_frame') or {})
            src_vals = np.array(list(p['source_xy'].values()), float)
            frame.update(rotation=np.eye(2).tolist(), scale=scale, reflect=False, rotate=False,
                         source_center=src_vals.mean(0).tolist(),
                         target_center=(np.array(list(new_pos.values()), float).mean(0)).tolist())
            p['source_to_initial_frame'] = frame
    return selected, report


def _rigid_eligible(idxs, blocks, model_n_reactions=None):
    n_rx = sum(len(blocks[i].get('reaction_ids') or []) for i in idxs)
    curated_n = sum(len(b.get('reaction_ids') or []) for b in blocks)
    if len(idxs) < 2:
        return False, n_rx
    model_n = max(int(model_n_reactions or 0), curated_n, 1)
    share_model = n_rx / model_n
    share_curated = n_rx / max(curated_n, 1)
    ok = share_model >= RIGID_MODEL_SHARE or (len(idxs) >= RIGID_MIN_BLOCKS and share_curated >= RIGID_CURATED_SHARE)
    return ok, n_rx


def _abs_lookup_for_block(block, abs_map):
    """Map block position keys → absolute source xy (reactions + metabolites)."""
    out = {}
    rx, mets = abs_map['reactions'], abs_map['metabolites']
    for rid in block.get('reaction_ids') or []:
        if rid in rx:
            out['r:' + rid] = list(rx[rid])
    for k in block.get('positions') or {}:
        if k.startswith('m:') and k[2:] in mets:
            out[k] = list(mets[k[2:]])
        elif k.startswith('r:') and k[2:] in rx:
            out[k] = list(rx[k[2:]])
    return out


def _common_abs_scale(blocks, idxs, abs_lookups, abs_map=None):
    """One uniform scale for the absolute same-map group (default 1:1 source units)."""
    return float(RIGID_ABS_SCALE)


# Gaps longer than this × median segment length are treated as alignment padding.
# Keep H/V alignment / order; collapse empty runs longer than ~one typical mid↔met span.
# Keep ≥1.0 so global remap does not eat BiGG axis/orderliness; empty corridors are
# shortened separately by ``compress_empty_axis_edges``.
PADDING_GAP_FACTOR = 1.0


def _remap_from_breaks(breaks):
    """Callable from serialised ``[(threshold, accumulated_shift), ...]`` break list."""
    if not breaks:
        return None

    def remap(value):
        total = 0.0
        for position, accumulated in breaks:
            if value >= position:
                total = accumulated
            else:
                break
        return value - total

    return remap


def build_axis_remap(values, max_gap):
    """Monotone map that shrinks empty runs longer than ``max_gap`` (order preserved).

    Returns ``(callable_or_None, stats)`` where stats includes serialisable ``breaks``.
    """
    empty = dict(gaps=0, removed=0.0, breaks=[])
    if not values:
        return None, empty
    points = sorted(set(round(float(v), 3) for v in values))
    if len(points) < 2:
        return None, empty
    shift, breaks = 0.0, []
    gaps = 0
    for prev, cur in zip(points, points[1:]):
        gap = cur - prev
        if gap > max_gap:
            shift += gap - max_gap
            breaks.append((float(cur), float(shift)))
            gaps += 1
    if not breaks:
        return None, empty
    return _remap_from_breaks(breaks), dict(gaps=gaps, removed=round(float(shift), 2), breaks=breaks)


def apply_padding_xy(xy, pad):
    """Map one ``[x, y]`` through a padding-compression report (identity when unused)."""
    if not pad or not pad.get('applied'):
        return [float(xy[0]), float(xy[1])]
    fx = _remap_from_breaks(pad.get('breaks_x') or []) or (lambda v: v)
    fy = _remap_from_breaks(pad.get('breaks_y') or []) or (lambda v: v)
    return [float(fx(xy[0])), float(fy(xy[1]))]


def compress_padding_xy(xy_by_key, max_gap=None, edge_lengths=None):
    """Compress long axis-aligned empty runs in a curated coordinate cloud.

    Curated maps often stretch straight H/V runs only to align columns/rows with the
    rest of the canvas.  Topology, direction, turn order and sibling alignment stay;
    only absolute length of empty stretches is reduced (no rotation/mirroring).

    ``xy_by_key`` is mutated in place.  Returns a report with gaps collapsed,
    length removed, and serialisable break lists for replaying the same remap
    (e.g. on source-path control points).
    """
    if not xy_by_key:
        return dict(applied=False, gaps_x=0, gaps_y=0, removed_x=0.0, removed_y=0.0,
                    max_gap=None, breaks_x=[], breaks_y=[],
                    segments_compressed=0, length_removed=0.0)
    pts = list(xy_by_key.values())
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    if max_gap is None:
        if edge_lengths:
            med = float(np.median(edge_lengths))
        else:
            # Fallback: median nearest-neighbour gap among unique coords per axis.
            gaps = []
            for arr in (sorted(set(round(x, 3) for x in xs)), sorted(set(round(y, 3) for y in ys))):
                for a, b in zip(arr, arr[1:]):
                    if b - a > 1e-6:
                        gaps.append(b - a)
            med = float(np.median(gaps)) if gaps else 100.0
        max_gap = max(med * PADDING_GAP_FACTOR, 1.0)
    fx, rx = build_axis_remap(xs, max_gap)
    fy, ry = build_axis_remap(ys, max_gap)
    if fx is None and fy is None:
        return dict(applied=False, gaps_x=0, gaps_y=0, removed_x=0.0, removed_y=0.0,
                    max_gap=round(float(max_gap), 2), breaks_x=[], breaks_y=[],
                    segments_compressed=0, length_removed=0.0)
    fx = fx or (lambda v: v)
    fy = fy or (lambda v: v)
    for k, xy in list(xy_by_key.items()):
        xy_by_key[k] = [float(fx(xy[0])), float(fy(xy[1]))]
    return dict(applied=True, gaps_x=rx['gaps'], gaps_y=ry['gaps'],
                removed_x=rx['removed'], removed_y=ry['removed'],
                max_gap=round(float(max_gap), 2),
                breaks_x=rx.get('breaks') or [], breaks_y=ry.get('breaks') or [],
                segments_compressed=rx['gaps'] + ry['gaps'],
                length_removed=round(rx['removed'] + ry['removed'], 2))


def _axis_aligned(dx, dy, axis_tol=28.0, angle_deg=8.0):
    """True when a span is effectively vertical or horizontal (padding candidate)."""
    length = float(math.hypot(dx, dy))
    if length < 1e-9:
        return False, False
    # Pixel band OR small angle from the axis (BiGG columns are rarely exact).
    vertical = (abs(dx) <= axis_tol and abs(dy) > abs(dx)) or (
        abs(dy) >= abs(dx) and math.degrees(math.atan2(abs(dx), abs(dy))) <= angle_deg)
    horizontal = (abs(dy) <= axis_tol and abs(dx) > abs(dy)) or (
        abs(dx) >= abs(dy) and math.degrees(math.atan2(abs(dy), abs(dx))) <= angle_deg)
    if vertical and horizontal:
        # Prefer the dominant axis.
        if abs(dy) >= abs(dx):
            horizontal = False
        else:
            vertical = False
    return vertical, horizontal


def compress_empty_axis_edges(xy_by_key, edges, max_gap, axis_tol=28.0, margin=18.0,
                              prefer_move=None):
    """Shorten axis-aligned mid↔met spans whose open corridor holds no other glyphs.

    Global axis remaps cannot collapse a long vertical when other modules occupy the
    same Y band at a different X.  When the thin corridor between the two endpoints
    is empty, length is free (shape and direction kept): pull the more peripheral
    endpoint toward its partner down to ``max_gap``.

    Prefer moving metabolites (``m:`` keys, or ids in ``prefer_move``) over reaction
    midmarkers so shared currency spokes stay anchored on the mid.
    """
    if not xy_by_key or not edges or max_gap is None:
        return dict(applied=False, shortened=0, length_removed=0.0)
    pts = {k: np.asarray(xy_by_key[k], float) for k in xy_by_key}
    if len(pts) < 2:
        return dict(applied=False, shortened=0, length_removed=0.0)
    centroid = np.mean(np.stack(list(pts.values())), axis=0)
    prefer_move = set(prefer_move or ())
    # How many corridor edges touch each key — leave high-degree hubs alone when possible.
    degree = defaultdict(int)
    for ka, kb in edges:
        if ka in pts and kb in pts:
            degree[ka] += 1
            degree[kb] += 1
    # Longest empty corridors first so nested shortenings see freed space.
    scored = []
    for ka, kb in edges:
        if ka not in pts or kb not in pts:
            continue
        a, b = pts[ka], pts[kb]
        dx, dy = float(a[0] - b[0]), float(a[1] - b[1])
        length = float(math.hypot(dx, dy))
        if length <= max_gap + 1e-6:
            continue
        vertical, horizontal = _axis_aligned(dx, dy, axis_tol=axis_tol)
        if not (vertical or horizontal):
            continue
        scored.append((length, ka, kb, vertical))
    scored.sort(reverse=True)
    shortened = 0
    removed = 0.0

    def _corridor_empty(a, b, vertical):
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        # Only the middle of the span is padding; local chemistry near either
        # endpoint (currency spokes, etc.) must not block shortening.
        span = float((hi[1] - lo[1]) if vertical else (hi[0] - lo[0]))
        pad = max(span * 0.12, float(max_gap) * 0.5, 40.0)
        cx = 0.5 * (a[0] + b[0])
        cy = 0.5 * (a[1] + b[1])
        # Strip width: margin plus half the endpoint offset (near-axis drift).
        band = margin + 0.5 * (abs(float(a[0] - b[0])) if vertical else abs(float(a[1] - b[1])))
        for k, p in pts.items():
            if np.allclose(p, a) or np.allclose(p, b):
                continue
            if vertical:
                if abs(p[0] - cx) > band:
                    continue
                if lo[1] + pad < p[1] < hi[1] - pad:
                    return False
            else:
                if abs(p[1] - cy) > band:
                    continue
                if lo[0] + pad < p[0] < hi[0] - pad:
                    return False
        return True

    for length, ka, kb, vertical in scored:
        a, b = pts[ka], pts[kb]
        length = float(np.hypot(*(a - b)))
        if length <= max_gap + 1e-6:
            continue
        if not _corridor_empty(a, b, vertical):
            continue

        def _move_score(key):
            # Lower is better. Never yank shared metabolites (degree > 1) — move the
            # midmarker toward them instead; private mets may move freely.
            is_met = (isinstance(key, str) and key.startswith('m:')) or key in prefer_move
            deg = degree.get(key, 0)
            dist_c = float(np.hypot(*(pts[key] - centroid)))
            if is_met and deg > 1:
                return (2, deg, -dist_c)
            if is_met:
                return (0, deg, -dist_c)
            return (1, deg, -dist_c)

        move_k = min((ka, kb), key=_move_score)
        fixed_k = kb if move_k == ka else ka
        fixed = pts[fixed_k]
        movable = pts[move_k]
        vec = movable - fixed
        dist = float(np.hypot(*vec))
        if dist <= max_gap + 1e-6:
            continue
        u = vec / dist
        new_xy = fixed + u * float(max_gap)
        delta = new_xy - movable
        removed += dist - max_gap
        pts[move_k] = new_xy
        xy_by_key[move_k] = [float(new_xy[0]), float(new_xy[1])]
        # If we moved a midmarker, drag its private (degree-1) spoke metabolites
        # by the same delta so transport heads stay attached.
        if not ((isinstance(move_k, str) and move_k.startswith('m:')) or move_k in prefer_move):
            for ka2, kb2 in edges:
                other = None
                if ka2 == move_k:
                    other = kb2
                elif kb2 == move_k:
                    other = ka2
                if other is None or other not in pts or degree.get(other, 0) != 1:
                    continue
                if not ((isinstance(other, str) and other.startswith('m:')) or other in prefer_move):
                    continue
                pts[other] = pts[other] + delta
                xy_by_key[other] = [float(pts[other][0]), float(pts[other][1])]
        shortened += 1
    return dict(applied=shortened > 0, shortened=shortened,
                length_removed=round(removed, 2), max_gap=round(float(max_gap), 2))


def compress_padding_joint(xy_by_key, edge_lengths=None, edges=None, rounds=3):
    """Alternate global empty-run collapse with empty-corridor edge shortening.

    Each round: axis remap → corridor shorten → axis remap.  Stops when a round
    removes less than 1 px of length.  Mutates ``xy_by_key``.
    """
    report = dict(rounds=0, padding=[], corridors=[], length_removed=0.0, max_gap=None)
    prev = None
    for _ in range(max(1, int(rounds))):
        pad = compress_padding_xy(xy_by_key, edge_lengths=edge_lengths)
        report['padding'].append(pad)
        report['max_gap'] = pad.get('max_gap')
        corridor = dict(applied=False, shortened=0, length_removed=0.0)
        if edges and pad.get('max_gap') is not None:
            corridor = compress_empty_axis_edges(
                xy_by_key, edges, max_gap=float(pad['max_gap']))
            # Re-collapse any global gaps the corridor moves opened.
            if corridor.get('applied'):
                pad2 = compress_padding_xy(
                    xy_by_key, max_gap=float(pad['max_gap']), edge_lengths=edge_lengths)
                report['padding'].append(pad2)
                pad = pad2
        report['corridors'].append(corridor)
        gained = float(pad.get('length_removed') or 0.0) + float(corridor.get('length_removed') or 0.0)
        report['rounds'] += 1
        report['length_removed'] = round(report['length_removed'] + gained, 2)
        if gained < 1.0 or (prev is not None and gained < 0.05 * max(prev, 1.0)):
            break
        prev = gained
    report['applied'] = report['length_removed'] > 0.0
    report['segments_compressed'] = sum(p.get('segments_compressed') or 0 for p in report['padding'])
    report['edges_shortened'] = sum(c.get('shortened') or 0 for c in report['corridors'])
    # Source-path replay uses the final global axis breaks (corridor moves are
    # node-only; mismatched Beziers fall back to restyle/reroute).
    last_pad = next((p for p in reversed(report['padding']) if p.get('breaks_x') is not None
                     or p.get('breaks_y') is not None), {}) if report['padding'] else {}
    report['breaks_x'] = last_pad.get('breaks_x') or []
    report['breaks_y'] = last_pad.get('breaks_y') or []
    report['gaps_x'] = sum(p.get('gaps_x') or 0 for p in report['padding'])
    report['gaps_y'] = sum(p.get('gaps_y') or 0 for p in report['padding'])
    report['removed_x'] = round(sum(p.get('removed_x') or 0 for p in report['padding']), 2)
    report['removed_y'] = round(sum(p.get('removed_y') or 0 for p in report['padding']), 2)
    return report


def bake_same_map_positions(blocks, model_n_reactions=None):
    """Place same-source curated blocks as one rigid group from original map coords.

    When ``source_sha256`` resolves to absolute source-map coordinates and the group
    covers a large share of the model, every glyph in the group is rewritten from those
    coordinates at one common scale (translation only).  Packing then treats the group
    as a single composite tile.

    Without absolute coords, falls back to translating block centroids onto source
    relative offsets × a common internal scale (no compact collapse).
    """
    enrich_absolute_source_xy(blocks)
    by_map = defaultdict(list)
    for i, b in enumerate(blocks or []):
        key = _group_key(b)
        if key and (b.get('source_xy') or b.get('source_sha256')) and b.get('positions'):
            by_map[key].append(i)
    report = dict(groups=0, blocks=0, scales={}, mode={}, skipped=[])
    for map_id, idxs in by_map.items():
        ok, n_rx = _rigid_eligible(idxs, blocks, model_n_reactions=model_n_reactions)
        if not ok:
            report['skipped'].append(dict(key=(map_id[:24] if isinstance(map_id, str) else map_id),
                                          blocks=len(idxs), reactions=n_rx, reason='below_share'))
            continue
        abs_map = absolute_source_coords(blocks[idxs[0]].get('source_sha256')) if blocks[idxs[0]].get('source_sha256') else None
        abs_lookups = {i: (_abs_lookup_for_block(blocks[i], abs_map) if abs_map else {}) for i in idxs}
        n_abs = sum(1 for i in idxs if len(abs_lookups[i]) >= 2)
        label = map_id[:24] if isinstance(map_id, str) else str(map_id)

        if abs_map and n_abs >= 2:
            scale = _common_abs_scale(blocks, idxs, abs_lookups, abs_map=abs_map)
            # Compress alignment-only H/V padding in absolute source coords (shape kept).
            cloud = {}
            for i in idxs:
                for k, xy in abs_lookups[i].items():
                    cloud[k] = list(xy)
            # Global H/V padding only at bake time (preserves BiGG axis/orderliness).
            # Extreme empty corridors are shortened later on the Escher map by
            # translating whole reaction units toward shared pools.
            pad = compress_padding_xy(cloud, edge_lengths=abs_map.get('edge_lengths'))
            for i in idxs:
                abs_lookups[i] = {k: list(cloud[k]) for k in abs_lookups[i] if k in cloud}
            # Rebuild every resolvable glyph from (compressed) absolute coords.
            all_abs = []
            for i in idxs:
                all_abs.extend(abs_lookups[i].values())
            if len(all_abs) < 2:
                report['skipped'].append(dict(key=label, blocks=len(idxs), reactions=n_rx, reason='no_abs_pts'))
                continue
            origin = np.mean(np.asarray(all_abs, float), axis=0)
            # Keep the group's current centroid so packing starts near existing layout.
            cur_all = []
            for i in idxs:
                cur_all.extend(blocks[i]['positions'].values())
            cur_origin = np.mean(np.asarray(cur_all, float), axis=0)
            delta0 = cur_origin - scale * origin
            for i in idxs:
                b = blocks[i]
                lookup = abs_lookups[i]
                new_pos = {}
                for k, xy in lookup.items():
                    new_pos[k] = (scale * np.asarray(xy, float) + delta0).tolist()
                if lookup:
                    abs_c = np.mean(np.asarray(list(lookup.values()), float), axis=0)
                    target_c = scale * abs_c + delta0
                else:
                    target_c = None
                if target_c is not None:
                    # Unresolved keys: preserve offset from old block centroid → new.
                    old_c = np.mean(np.asarray(list(b['positions'].values()), float), axis=0)
                    shift = target_c - old_c
                    for k, xy in b['positions'].items():
                        if k not in new_pos:
                            new_pos[k] = (np.asarray(xy, float) + shift).tolist()
                else:
                    new_pos = dict(b['positions'])
                b['positions'] = new_pos
                b['same_map_rigid'] = map_id
                b['source_xy'] = {k: list(lookup[k]) for k in lookup}
                b['source_xy_absolute'] = True
                b['padding_compression'] = pad
                frame = dict(b.get('source_to_initial_frame') or {})
                frame.update(rotation=np.eye(2).tolist(), scale=scale, reflect=False, rotate=False,
                             source_center=origin.tolist(), target_center=cur_origin.tolist())
                b['source_to_initial_frame'] = frame
            # Snapshot for later restore: relatives must survive composition / arms / relax.
            snap = {}
            for i in idxs:
                for k, xy in blocks[i]['positions'].items():
                    snap[k] = list(xy)
            for i in idxs:
                blocks[i]['same_map_snapshot'] = snap
            report['groups'] += 1
            report['blocks'] += len(idxs)
            report['scales'][label] = round(scale, 4)
            report['mode'][label] = 'absolute_rigid'
            report['padding_compression'] = pad
            continue

        # Fallback: centroid restore from source_xy when centroids actually spread.
        src_c, diameters, scales = [], [], []
        for i in idxs:
            sxy = np.array(list(blocks[i]['source_xy'].values()), float)
            pxy = np.array(list(blocks[i]['positions'].values()), float)
            if len(sxy) == 0:
                continue
            src_c.append(sxy.mean(0))
            diameters.append(float(np.hypot(*(pxy.max(0) - pxy.min(0)))) if len(pxy) else 100.0)
            src_span = float(np.hypot(*np.ptp(sxy, axis=0))) if len(sxy) > 1 else 0.0
            pos_span = float(np.hypot(*np.ptp(pxy, axis=0))) if len(pxy) > 1 else 0.0
            if src_span > 1e-6:
                scales.append(pos_span / src_span)
        if len(src_c) < 2:
            continue
        src_c = np.array(src_c, float)
        if float(np.max(np.ptp(src_c, axis=0))) < 1e-3:
            report['skipped'].append(dict(key=label, blocks=len(idxs), reactions=n_rx, reason='collapsed_source_xy'))
            continue
        common = float(np.median(scales)) if scales else float(np.median(diameters))
        origin = src_c.mean(0)
        cur_centres = []
        for i in idxs:
            pxy = np.array(list(blocks[i]['positions'].values()), float)
            cur_centres.append(pxy.mean(0))
        spring = np.mean(cur_centres, axis=0)
        targets = spring + common * (src_c - origin)
        for j, i in enumerate(idxs):
            b = blocks[i]
            pxy = np.array(list(b['positions'].values()), float)
            delta = targets[j] - pxy.mean(0)
            b['positions'] = {k: (np.asarray(v, float) + delta).tolist() for k, v in b['positions'].items()}
            b['same_map_rigid'] = map_id
        report['groups'] += 1
        report['blocks'] += len(idxs)
        report['scales'][label] = round(common, 4)
        report['mode'][label] = 'centroid_relative'
    return blocks, report


def same_map_packing_groups(blocks):
    """Index lists for rigid same-map composites: key -> [block_index,...] (len>=2 only).

    Only blocks marked ``same_map_rigid`` by ``bake_same_map_positions`` are grouped —
    sharing a map_id / source_xy alone must not merge independently packed tiles into one
    AABB (that inflates the packing canvas).
    """
    by_map = defaultdict(list)
    for i, b in enumerate(blocks or []):
        mid = b.get('same_map_rigid')
        if mid:
            by_map[mid].append(i)
    return {mid: idxs for mid, idxs in by_map.items() if len(idxs) >= 2}


def rigid_reaction_ids(blocks):
    """Reaction ids belonging to any same_map_rigid block."""
    return {rid for b in (blocks or []) if b.get('same_map_rigid') for rid in (b.get('reaction_ids') or [])}


def rigid_block_ids(blocks):
    return {b.get('id') for b in (blocks or []) if b.get('same_map_rigid') and b.get('id')}


# Atlas windows can span distant transports on the source map. Continuity scores each
# structure as one unit at 3× whole-map median spacing, so a single mega-window at 1:1
# BiGG scale looks "split". Emit spatially compact pieces (source midmarker NN × factor)
# so continuity measures whether each drawn piece stayed together.
SPATIAL_STRUCTURE_FACTOR = 1.5


def _midmarker_xy(block):
    """Reaction id → (x, y) from snapshot or block positions."""
    cloud = block.get('same_map_snapshot') or block.get('positions') or {}
    out = {}
    for k, xy in cloud.items():
        if isinstance(k, str) and k.startswith('r:') and xy is not None:
            out[k[2:]] = (float(xy[0]), float(xy[1]))
    return out


def spatial_reaction_pieces(reaction_ids, xy_by_rid, adjacency_factor=SPATIAL_STRUCTURE_FACTOR):
    """Split reactions into geometrically contiguous pieces (median NN × factor)."""
    placed = [r for r in (reaction_ids or []) if r in xy_by_rid]
    if len(placed) < 2:
        return [placed] if placed else []
    xy = np.asarray([xy_by_rid[r] for r in placed], float)
    nn = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(nn, np.inf)
    d = nn.min(axis=1)
    d = d[np.isfinite(d) & (d > 0)]
    spacing = float(np.median(d)) if len(d) else 60.0
    thr = float(adjacency_factor) * max(spacing, 1e-9)
    parent = {r: r for r in placed}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            dx = xy_by_rid[a][0] - xy_by_rid[b][0]
            dy = xy_by_rid[a][1] - xy_by_rid[b][1]
            if (dx * dx + dy * dy) ** 0.5 <= thr:
                union(a, b)
    comps = defaultdict(list)
    for r in placed:
        comps[find(r)].append(r)
    return [rs for rs in comps.values() if len(rs) >= 2]


def expert_structure_entries(blocks, codes=None, adjacency_factor=SPATIAL_STRUCTURE_FACTOR):
    """``metadata.expert_structures`` rows; large windows become spatial pieces."""
    codes = codes or {}
    out = []
    for b in blocks or []:
        rids = list(b.get('reaction_ids') or [])
        if len(rids) < 2:
            continue
        bid = b.get('id') or b.get('template_id')
        code = codes.get(b.get('id'))
        xy = _midmarker_xy(b)
        pieces = spatial_reaction_pieces(rids, xy, adjacency_factor=adjacency_factor) if xy else []
        if len(pieces) <= 1:
            out.append(dict(id=bid, code=code, reactions=rids))
            continue
        for i, rs in enumerate(pieces):
            out.append(dict(id='%s#cc%d' % (bid, i), code=code, reactions=list(rs), parent=bid))
    return out


def clear_rigid_node_overlaps(scene, blocks, min_dist=12.0):
    """Nudge non-rigid glyphs off restored rigid centres (post-restore overlap cleanup).

    ``instances.separate`` runs before the last rigid restore, so absolute snap
    can land a curated metabolite on top of an algorithm glyph that was already
    parked nearby. Move only non-rigid keys.
    """
    if not scene or not scene.get('positions'):
        return scene, dict(moved=0)
    rigid_rx = rigid_reaction_ids(blocks)
    pos = scene['positions']
    parts = scene.get('participants') or {}
    rigid_keys = set()
    for k in pos:
        if k.startswith('r:') and k[2:] in rigid_rx:
            rigid_keys.add(k)
        elif k.startswith('m:') or k.startswith('copy:') or k.startswith('side:'):
            # Mark keys owned exclusively by rigid reactions via participants.
            pass
    for pk, key in (parts.items() if isinstance(parts, dict) else []):
        if not isinstance(key, str) or key not in pos:
            continue
        if isinstance(pk, (tuple, list)) and len(pk) >= 1:
            rid = pk[0]
        elif isinstance(pk, str) and '||' in pk:
            rid = pk.split('||', 1)[0]
        else:
            continue
        if rid in rigid_rx:
            rigid_keys.add(key)
    # Also treat snapshot keys as rigid.
    for b in blocks or []:
        for k in (b.get('same_map_snapshot') or {}):
            if k in pos:
                rigid_keys.add(k)
    keys = sorted(pos)
    if len(keys) < 2:
        return scene, dict(moved=0)
    xy = np.asarray([pos[k] for k in keys], float)
    index = {k: i for i, k in enumerate(keys)}
    rigid_idx = {index[k] for k in rigid_keys if k in index}
    moved = 0
    for _ in range(40):
        hit = False
        for i in list(rigid_idx):
            for j, k in enumerate(keys):
                if j in rigid_idx:
                    continue
                d = xy[j] - xy[i]
                dist = float(np.hypot(d[0], d[1]))
                if dist >= min_dist:
                    continue
                hit = True
                if dist < 1e-6:
                    ang = (i * 17 + j * 31) * 2.399963
                    d = np.array([math.cos(ang), math.sin(ang)])
                    dist = 1.0
                step = d / dist * (min_dist - dist)
                xy[j] = xy[j] + step
                moved += 1
        if not hit:
            break
    for k in keys:
        pos[k] = xy[index[k]].tolist()
    return scene, dict(moved=moved, rigid_keys=len(rigid_keys))


def clear_rigid_node_overlaps_escher(data, blocks, min_dist=12.0):
    """Nudge Escher metabolite nodes so centres stay ≥ ``min_dist`` after rigid restore.

    Prefers moving glyphs that are not exclusive to ``same_map_rigid`` reactions;
    if both ends are rigid (duplicate same-bigg), nudges the non-primary copy.
    Uses a KD-tree so mid/large maps stay cheap (brute pairs cost minutes on iIT341).
    """
    if not data or len(data) < 2:
        return data, dict(moved=0)
    from scipy.spatial import cKDTree
    rigid_rx = rigid_reaction_ids(blocks)
    nodes = data[1].get('nodes') or {}
    reactions = data[1].get('reactions') or {}
    holders = defaultdict(set)
    for rx in reactions.values():
        rid = rx.get('bigg_id')
        for seg in (rx.get('segments') or {}).values():
            for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                if nid in nodes:
                    holders[nid].add(rid)
    rigid_nids = set()
    if rigid_rx:
        for nid, hs in holders.items():
            if hs and hs.issubset(rigid_rx) and nodes[nid].get('node_type') in ('metabolite', 'midmarker'):
                rigid_nids.add(nid)
        for rx in reactions.values():
            if rx.get('bigg_id') not in rigid_rx:
                continue
            for seg in (rx.get('segments') or {}).values():
                for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                    if (nodes.get(nid) or {}).get('node_type') == 'midmarker':
                        rigid_nids.add(nid)
    ids = [nid for nid, n in nodes.items() if n.get('node_type') in ('metabolite', 'midmarker')]
    if len(ids) < 2:
        return data, dict(moved=0)
    moved = 0

    def _can_move(nid):
        if nid not in rigid_nids:
            return True
        n = nodes[nid]
        if n.get('node_type') == 'metabolite' and not n.get('node_is_primary', True):
            return True
        return False

    for _ in range(40):
        xy = np.asarray([[float(nodes[i]['x']), float(nodes[i]['y'])] for i in ids], float)
        pairs = cKDTree(xy).query_pairs(min_dist, output_type='ndarray')
        if len(pairs) == 0:
            break
        hit = False
        for i, j in pairs:
            a, b = ids[int(i)], ids[int(j)]
            d = xy[int(j)] - xy[int(i)]
            dist = float(np.hypot(d[0], d[1]))
            if dist >= min_dist:
                continue
            move_a, move_b = _can_move(a), _can_move(b)
            if not (move_a or move_b):
                if ((nodes[a].get('bigg_id') or '') == (nodes[b].get('bigg_id') or '')
                        and nodes[a].get('node_type') == 'metabolite'):
                    move_b = True
                else:
                    continue
            hit = True
            if dist < 1e-6:
                ang = ((hash(a) + hash(b)) % 360) * math.pi / 180.0
                d = np.array([math.cos(ang), math.sin(ang)])
                dist = 1.0
            step = d / dist * (min_dist - dist)
            # Track midmarker deltas so reaction labels stay with their anchors.
            mid_delta = {}
            if move_a and move_b:
                dx_a, dy_a = -step[0] / 2, -step[1] / 2
                dx_b, dy_b = step[0] / 2, step[1] / 2
                translate_node(nodes[a], dx_a, dy_a, digits=6)
                translate_node(nodes[b], dx_b, dy_b, digits=6)
                if nodes[a].get('node_type') == 'midmarker':
                    mid_delta[a] = (dx_a, dy_a)
                if nodes[b].get('node_type') == 'midmarker':
                    mid_delta[b] = (dx_b, dy_b)
            elif move_a:
                dx_a, dy_a = -step[0], -step[1]
                translate_node(nodes[a], dx_a, dy_a, digits=6)
                if nodes[a].get('node_type') == 'midmarker':
                    mid_delta[a] = (dx_a, dy_a)
            else:
                dx_b, dy_b = step[0], step[1]
                translate_node(nodes[b], dx_b, dy_b, digits=6)
                if nodes[b].get('node_type') == 'midmarker':
                    mid_delta[b] = (dx_b, dy_b)
            if mid_delta:
                for rx in reactions.values():
                    for seg in (rx.get('segments') or {}).values():
                        for end in (seg.get('from_node_id'), seg.get('to_node_id')):
                            if end in mid_delta and nodes.get(end, {}).get('node_type') == 'midmarker':
                                translate_reaction_label(rx, *mid_delta[end], digits=6)
                                break
            moved += 1
        if not hit:
            break
    return data, dict(moved=moved, rigid_nodes=len(rigid_nids))


def restore_rigid_group(scene, blocks):
    """Re-apply same-map absolute relatives; allow only one common translation of the group.

    Composition / landscape reflow / compact-arms / relax may move glyphs.  After those
    stages call this so the shared-source backbone stays source-faithful (1:1 scale).
    """
    rigid_blocks = [b for b in (blocks or []) if b.get('same_map_rigid')]
    if not rigid_blocks or not scene or not scene.get('positions'):
        return scene, dict(restored=0, groups=0)
    pos = scene['positions']
    parts = scene.get('participants') or {}
    # Prefer the bake snapshot; fall back to current block positions.
    snap = None
    for b in rigid_blocks:
        if b.get('same_map_snapshot'):
            snap = b['same_map_snapshot']
            break
    if not snap:
        snap = {}
        for b in rigid_blocks:
            for k, xy in (b.get('positions') or {}).items():
                snap[k] = list(xy)
    rigid_rx = rigid_reaction_ids(blocks)
    rigid_bids = rigid_block_ids(blocks)
    # Common translation from reaction midmarkers present in both snap and scene.
    cur, tgt = [], []
    for rid in rigid_rx:
        k = 'r:' + rid
        if k in pos and k in snap:
            cur.append(pos[k])
            tgt.append(snap[k])
    if len(cur) < 2:
        return scene, dict(restored=0, groups=len({b.get('same_map_rigid') for b in rigid_blocks}))
    delta = np.mean(np.asarray(cur, float), axis=0) - np.mean(np.asarray(tgt, float), axis=0)
    restored = 0
    # Remember pre-restore midmarkers so copy:/side: glyphs can keep their offset to the reaction.
    old_mids = {('r:' + rid): list(pos['r:' + rid]) for rid in rigid_rx if ('r:' + rid) in pos}

    def _set(key, xy):
        nonlocal restored
        if key in pos:
            pos[key] = list(xy)
            restored += 1

    # Restore only snapshot keys (midmarkers + primary mets).  Do NOT stack every copy:
    # glyph onto the primary — that created zero-distance same-metabolite node overlaps.
    for k, xy in snap.items():
        _set(k, (np.asarray(xy, float) + delta).tolist())
    # Translate reaction-local copies with their midmarker so relative stubs stay attached.
    if isinstance(parts, dict):
        for pk, key in parts.items():
            if not isinstance(key, str) or key in snap or key not in pos:
                continue
            if isinstance(pk, (tuple, list)) and len(pk) >= 2:
                rid = pk[0]
            elif isinstance(pk, str) and '||' in pk:
                rid = pk.split('||', 1)[0]
            else:
                continue
            if rid not in rigid_rx:
                continue
            rk = 'r:' + rid
            if rk not in old_mids or rk not in pos:
                continue
            shift = np.asarray(pos[rk], float) - np.asarray(old_mids[rk], float)
            pos[key] = (np.asarray(pos[key], float) + shift).tolist()
            restored += 1
    # Also rewrite block positions so later sync / export stay consistent.
    for b in rigid_blocks:
        new_pos = {}
        for k, xy in (b.get('positions') or {}).items():
            if k in snap:
                new_pos[k] = (np.asarray(snap[k], float) + delta).tolist()
            else:
                new_pos[k] = list(xy)
        b['positions'] = new_pos
        # Keep snapshot in absolute (pre-delta) form so repeated restores stay stable.
        if b.get('same_map_snapshot') is None:
            b['same_map_snapshot'] = dict(snap)
    return scene, dict(restored=restored, groups=len({b.get('same_map_rigid') for b in rigid_blocks}),
                       delta=[round(float(x), 2) for x in delta], n_rx=len(rigid_rx))


def restore_rigid_group_escher(data, blocks):
    """Re-apply same-map absolute midmarker/metabolite positions onto an Escher map.

    Allows one common translation of the whole rigid group (to follow canvas squeeze
    translation) but restores source relatives at the bake scale (1:1 by default).
    """
    rigid_blocks = [b for b in (blocks or []) if b.get('same_map_rigid')]
    if not rigid_blocks or not data or len(data) < 2:
        return data, dict(restored=0)
    snap = None
    for b in rigid_blocks:
        if b.get('same_map_snapshot'):
            snap = b['same_map_snapshot']
            break
    if not snap:
        return data, dict(restored=0)
    nodes = data[1].get('nodes') or {}
    reactions = data[1].get('reactions') or {}
    rigid_rx = rigid_reaction_ids(blocks)
    # Current midmarker positions vs snapshot for common translation.
    cur, tgt = [], []
    mid_node = {}
    for rx in reactions.values():
        rid = rx.get('bigg_id')
        if rid not in rigid_rx:
            continue
        for seg in (rx.get('segments') or {}).values():
            for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                n = nodes.get(nid) or {}
                if n.get('node_type') == 'midmarker':
                    mid_node[rid] = nid
                    sk = 'r:' + rid
                    if sk in snap:
                        cur.append([n['x'], n['y']])
                        tgt.append(snap[sk])
                    break
            if rid in mid_node:
                break
    if len(cur) < 2:
        return data, dict(restored=0)
    delta = np.mean(np.asarray(cur, float), axis=0) - np.mean(np.asarray(tgt, float), axis=0)
    # Pre-restore midmarker xy so duplicate metabolite glyphs can keep their stub offset.
    old_mid_xy = {rid: [float(nodes[nid]['x']), float(nodes[nid]['y'])]
                  for rid, nid in mid_node.items()}
    restored = 0
    for rid, nid in mid_node.items():
        sk = 'r:' + rid
        if sk not in snap:
            continue
        xy = np.asarray(snap[sk], float) + delta
        dx, dy = set_node_xy(nodes[nid], float(xy[0]), float(xy[1]), digits=6)
        # Reaction label rides with the midmarker (midmarkers have no label_x).
        rx = next((r for r in reactions.values() if r.get('bigg_id') == rid), None)
        if rx is not None:
            translate_reaction_label(rx, dx, dy, digits=6)
        restored += 1
    # Metabolites: one primary glyph per bigg_id takes the absolute snap position;
    # additional glyphs for the same metabolite keep their offset to their reaction
    # midmarker (avoids stacking duplicates on top of each other).
    snap_mets = {k[2:]: snap[k] for k in snap if k.startswith('m:')}
    by_met = defaultdict(list)
    for nid, n in nodes.items():
        if n.get('node_type') != 'metabolite':
            continue
        mid = n.get('bigg_id')
        if mid not in snap_mets:
            continue
        holders = set()
        for rx in reactions.values():
            for seg in (rx.get('segments') or {}).values():
                if nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                    holders.add(rx.get('bigg_id'))
        if not holders or not holders.issubset(rigid_rx):
            continue
        by_met[mid].append((nid, holders))
    for mid, entries in by_met.items():
        target = np.asarray(snap_mets[mid], float) + delta
        # Prefer a node flagged primary when present.
        entries_sorted = sorted(entries, key=lambda t: (not nodes[t[0]].get('node_is_primary', True), t[0]))
        primary_nid = entries_sorted[0][0]
        set_node_xy(nodes[primary_nid], float(target[0]), float(target[1]), digits=6)
        restored += 1
        for nid, holders in entries_sorted[1:]:
            rid = next(iter(holders))
            mnid = mid_node.get(rid)
            old = old_mid_xy.get(rid)
            if mnid is None or old is None:
                # Fallback ring so centres stay clear of the primary glyph.
                ang = (hash(nid) % 360) * math.pi / 180.0
                set_node_xy(nodes[nid],
                            float(target[0] + 14.0 * math.cos(ang)),
                            float(target[1] + 14.0 * math.sin(ang)),
                            digits=6)
            else:
                # Translate with the midmarker: keep pre-restore stub vector.
                stub = np.asarray([nodes[nid]['x'], nodes[nid]['y']], float) - np.asarray(old, float)
                # If the stub collapsed (already stacked), push onto a small ring.
                if float(np.hypot(stub[0], stub[1])) < 1.0:
                    ang = (hash(nid) % 360) * math.pi / 180.0
                    stub = np.asarray([14.0 * math.cos(ang), 14.0 * math.sin(ang)], float)
                new_mid = np.asarray([nodes[mnid]['x'], nodes[mnid]['y']], float)
                set_node_xy(nodes[nid],
                            float(new_mid[0] + stub[0]),
                            float(new_mid[1] + stub[1]),
                            digits=6)
            restored += 1
    # Drop Bezier handles on rigid segments so restyle can rebuild from nodes.
    for rx in reactions.values():
        if rx.get('bigg_id') not in rigid_rx:
            continue
        for seg in (rx.get('segments') or {}).values():
            seg['b1'] = None
            seg['b2'] = None
    return data, dict(restored=restored, n_rx=len(rigid_rx),
                      delta=[round(float(x), 2) for x in delta])


def restore_same_map_centres(blocks, centres, sizes):
    """After spring/stress: nudge same-map block centres onto source relative offsets."""
    centres = np.asarray(centres, float)
    by_map = defaultdict(list)
    for i, b in enumerate(blocks or []):
        if _group_key(b) and b.get('source_xy'):
            by_map[_group_key(b)].append(i)
    report = dict(groups=0, blocks=0, scales={})
    for map_id, idxs in by_map.items():
        if len(idxs) < 2:
            continue
        # Rigid groups already placed — do not compact-nudge them apart.
        if all(blocks[i].get('same_map_rigid') for i in idxs):
            continue
        src, valid, scales = [], [], []
        for i in idxs:
            sxy = np.array(list(blocks[i]['source_xy'].values()), float)
            if len(sxy) == 0:
                continue
            src.append(sxy.mean(0))
            valid.append(i)
            if sizes is not None:
                diam = float(np.hypot(*np.asarray(sizes[i], float)[:2]))
            else:
                pxy = np.array(list(blocks[i]['positions'].values()), float)
                diam = float(np.hypot(*(pxy.max(0) - pxy.min(0)))) if len(pxy) else 100.0
            src_span = float(np.hypot(*np.ptp(sxy, axis=0))) if len(sxy) > 1 else 0.0
            scales.append(diam / src_span if src_span > 1e-6 else diam)
        if len(valid) < 2:
            continue
        src = np.array(src, float)
        if float(np.max(np.ptp(src, axis=0))) < 1e-3:
            continue
        common = float(np.median(scales)) if scales else 1.0
        rel = common * (src - src.mean(0))
        spring = centres[valid].mean(0)
        centres[valid] = spring + rel
        report['groups'] += 1
        report['blocks'] += len(valid)
        report['scales'][map_id[:24] if isinstance(map_id, str) else map_id] = round(common, 4)
    return centres, report


def axes_report(blocks, scene=None):
    """Share of curated blocks whose placement keeps source axes (no rot / no reflect).

    Prefers ``source_to_initial_frame`` on each block/proposal.  When ``scene`` is given,
    also checks the residual Kabsch from ``source_xy`` to final scene positions.
    """
    rows = []
    for b in blocks or []:
        frame = b.get('source_to_initial_frame') or {}
        rot = np.asarray(frame.get('rotation') or np.eye(2), float)
        reflect = bool(frame.get('reflect'))
        angle = abs(_angle_deg(rot))
        kept = bool((not reflect) and angle <= AXES_ANGLE_DEG and abs(float(np.linalg.det(rot)) - 1.0) < 0.05)
        # Scene residual check only when the source cloud has enough spread; near-collinear
        # 2–3 point modules make Kabsch reflection/angle unstable and false-positive "mirrored".
        if scene is not None and b.get('source_xy'):
            import source_paths as sp
            src, dst = [], []
            for k, xy in b['source_xy'].items():
                if k in scene['positions']:
                    src.append(xy)
                    dst.append(scene['positions'][k])
            if len(src) >= 4:
                arr = np.asarray(src, float)
                spread = float(np.min(np.ptp(arr, axis=0)))
                if spread > 1e-3 * max(float(np.max(np.ptp(arr, axis=0))), 1.0):
                    fit = sp.similarity(src, dst)
                    if fit is not None:
                        reflect = reflect or bool(fit.get('reflect'))
                        angle = abs(_angle_deg(fit['rot']))
                        kept = bool((not reflect) and angle <= AXES_ANGLE_DEG)
        rows.append(dict(id=b.get('id') or b.get('template_id'), kept=bool(kept), angle_deg=round(float(angle), 2),
                         reflected=bool(reflect), map_id=b.get('map_id')))
    n = len(rows)
    kept_n = sum(1 for r in rows if r['kept'])
    mirrored = sum(1 for r in rows if r['reflected'])
    return dict(blocks=n, axes_kept=kept_n, axes_kept_share=round(kept_n / n, 4) if n else None,
                mirrored=mirrored, threshold_deg=AXES_ANGLE_DEG, rows=rows[:80])
