"""Inter-skeleton refinement after global block placement (not a fifth arrangement).

Whole modules translate only; curated internals keep their relative coordinates (no flip
or rotation). ``copy:<structure>:<metabolite>`` glyphs are protected: each curated
structure keeps its own copy of a shared metabolite, copies are never merged, and they
move only when their structure moves.

Attachment pairs
----------------
Each inter-skeleton connection is an explicit pair
``(block_a, anchor_a, block_b, anchor_b, kind, metabolite_or_reaction_id)``.

* ``shared_metabolite`` — non-currency metabolite present in several modules (hubs with
  more than ``HUB`` modules are ignored, same rule as ``group_weights``).
* ``bridging_reaction`` — a reaction outside every module that touches non-currency
  metabolites in two or more modules.

Objective / double-counting
---------------------------
Live placement is driven by ``group_weights`` and ``block_candidates.module_links``
(``block_weights`` is unused).  Both put weight ``1/(k-1)`` on every module pair that
shares a metabolite present in ``k`` modules, so attraction per metabolite sums to
``k/2`` (linear in ``k``).  Arrangement *counts* (``links_over_250`` /
``links_over_1000``) still add one unweighted link per pair, i.e. ``C(k, 2)`` per
metabolite — on iIT341, ``glu__L_c`` (k=23) alone is ~23% of those links.

This stage normalises per metabolite:

* Build a deterministic MST on the ``k`` structure copies (Kruskal; ties by stable
  block ids) → exactly ``(k - 1)`` undirected edges, not ``C(k, 2)``.
* Weight each MST edge by ``1/(k - 1)`` so metabolite ``M`` contributes
  ``mean(MST edge lengths)`` to ``L`` (one unit of mass), not ``k/2`` and not
  ``C(k, 2)`` distances.  Bridging reactions use the same rule with ``t`` touched
  modules.  Pareto long-link mass uses the same weights.

``L`` and long-link statistics are straight-line anchor distances (not routed
length).  The report also keeps the unweighted MST sum for comparison.

Search
------
Deterministic discrete search only (swap two module centres, or relocate one module to
a feasible free slot).  Fixed attempt budget; never wall-clock.  ``--previous`` pinned
modules do not move.  Accept decisions go through ``accept_move`` (constraint-first
aspect band + Pareto on attachment / packing metrics) so a later
``LayoutQualityEvaluator`` can replace that one function.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

import block_candidates as BC

# Reuse the long-link threshold and aspect bands of the arrangement chooser.
LONG_LINK = 1000.0
ASPECT_HARD = BC.ASPECT_HARD
ASPECT_RANGE = BC.ASPECT_RANGE
BLOCK_GAP = 90.0
# Fixed attempt budget (not wall-clock). Caps swap+relocate trials for reproducibility.
# Sized so mid-size compose stays near the pre-stage runtime while still finding moves.
ATTEMPT_BUDGET = 80
# How many relocate slots to try per movable module (deterministic generators below).
RELOCATE_SLOTS = 6
# Extra slots reserved for modules that still have a cross-structure pair > LONG_LINK.
FAR_RELOCATE_SLOTS = 10
# When arrangements are present, always refine each (report only; winner selection unchanged).
DIAGNOSTIC = False

_EPS = 1e-9


def _dist(a, b) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _item_ref(item) -> np.ndarray:
    """Packed point that lands on the module centre after translation (see plan_global)."""
    lo = np.asarray(item['lo'] if 'lo' in item else (0.0, 0.0), float)
    size = np.asarray(item['size'], float)
    return lo + size / 2.0 - 40.0


def _local_metabolite_offset(item, mid: str, blocks) -> np.ndarray | None:
    """Offset of metabolite ``mid`` from the module centre reference, or None."""
    pts = []
    for bi in item.get('block_indexes') or ():
        if bi >= len(blocks):
            continue
        pos = blocks[bi].get('positions') or {}
        key = 'm:' + mid
        if key in pos:
            pts.append(np.asarray(pos[key], float))
    if not pts and item.get('kind') == 'auto':
        # Automatic modules: metabolite may sit in local layout coords under positions later;
        # before anchors exist, fall back to the module centre (zero local offset).
        return np.zeros(2, float)
    if not pts:
        # Reaction participants listed in _rx but no glyph yet → centre.
        rx = item.get('_rx') or {}
        if any(mid in (r.get('metabolites') or {}) for r in rx.values()):
            return np.zeros(2, float)
        return None
    mean = np.mean(np.stack(pts, axis=0), axis=0)
    return mean - _item_ref(item)


def _module_ids(items) -> list[str]:
    out = []
    for i, it in enumerate(items):
        if it.get('same_map_rigid'):
            out.append('rigid:' + str(it['same_map_rigid']))
        elif it.get('block_indexes'):
            out.append('block:' + ','.join(str(b) for b in it['block_indexes']))
        elif it.get('id') is not None:
            out.append('auto:' + str(it['id']))
        else:
            out.append('item:' + str(i))
    return out


def _owner_modules(items, cur, hub: int) -> dict[str, list[int]]:
    """Non-currency metabolite → sorted module indices that contain it (hubs dropped)."""
    owner = defaultdict(set)
    for gi, it in enumerate(items):
        rx = it.get('_rx') or {}
        for rid, r in rx.items():
            for mid in r.get('metabolites') or ():
                if mid not in cur:
                    owner[mid].add(gi)
    out = {}
    for mid, gs in owner.items():
        gs = sorted(gs)
        if 1 < len(gs) <= hub:
            out[mid] = gs
    return out


def _kruskal_mst(nodes: list[tuple[int, np.ndarray, str]]) -> list[tuple[int, int, float]]:
    """MST edges as (i, j, dist) with i < j. Tie-break by stable ids then indices."""
    n = len(nodes)
    if n < 2:
        return []
    edges = []
    for a in range(n):
        ia, pa, sa = nodes[a]
        for b in range(a + 1, n):
            ib, pb, sb = nodes[b]
            d = _dist(pa, pb)
            lo, hi = (ia, ib) if ia < ib else (ib, ia)
            slo, shi = (sa, sb) if sa <= sb else (sb, sa)
            edges.append((d, slo, shi, lo, hi, a, b))
    edges.sort()
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen = []
    for d, _slo, _shi, _lo, _hi, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        ia, ib = nodes[a][0], nodes[b][0]
        if ia > ib:
            ia, ib = ib, ia
            a, b = b, a
        chosen.append((ia, ib, d))
        if len(chosen) == n - 1:
            break
    return chosen


def build_attachment_pairs(items, centres, model, cur, blocks=None, hub: int = 30):
    """Build attachment pairs for the current module centres.

    Shared metabolites: MST over structure copies (``k`` copies → ``k - 1`` pairs),
    each edge weighted ``1/(k - 1)``.  Bridging outside reactions: same with ``t``.
    """
    centres = np.asarray(centres, float)
    blocks = blocks or []
    ids = _module_ids(items)
    pairs = []
    per_met = {}  # mid -> dict(k, edges, length, mean, all_pairs)

    owner = _owner_modules(items, cur, hub)
    # All-pairs reference (report only): what module_links would count as C(k,2).
    all_pairs_count = 0
    for mid, gs in sorted(owner.items()):
        k = len(gs)
        all_pairs_count += k * (k - 1) // 2
        nodes = []
        for gi in gs:
            off = _local_metabolite_offset(items[gi], mid, blocks)
            if off is None:
                off = np.zeros(2, float)
            xy = centres[gi] + off
            nodes.append((gi, xy, ids[gi]))
            items[gi].setdefault('_anchor_local', {})[mid] = off.tolist()
        mst = _kruskal_mst(nodes)
        w = 1.0 / max(k - 1, 1)
        length = 0.0
        weighted = 0.0
        node_xy = {n[0]: n[1] for n in nodes}
        node_local = {gi: (xy - centres[gi]).tolist() for gi, xy, _sid in nodes}
        for a, b, d in mst:
            pairs.append(dict(
                block_a=a, block_b=b,
                anchor_a=node_xy[a].tolist(), anchor_b=node_xy[b].tolist(),
                local_a=node_local[a], local_b=node_local[b],
                kind='shared_metabolite', metabolite_id=mid, reaction_id=None,
                distance=round(d, 4), weight=w, k=k,
                cross_structure=True,
            ))
            length += d
            weighted += w * d
        per_met[mid] = dict(
            k=k, edges=len(mst), all_pairs=k * (k - 1) // 2,
            length_sum=round(length, 4), length_mean=round(length / max(k - 1, 1), 4),
            contribution_L=round(weighted, 4),
            contribution_edges='k-1', contribution_weight='1/(k-1) per MST edge',
        )

    # Bridging reactions outside every packed module.
    in_mod = {rid for it in items for rid in (it.get('reactions') or ())}
    met_mods = defaultdict(set)
    for gi, it in enumerate(items):
        for rid, r in (it.get('_rx') or {}).items():
            for mid in r.get('metabolites') or ():
                if mid not in cur:
                    met_mods[mid].add(gi)
    for r in model.get('reactions') or ():
        rid = r['id']
        if rid in in_mod:
            continue
        touched = {}
        for mid in r.get('metabolites') or ():
            if mid in cur:
                continue
            for gi in met_mods.get(mid, ()):
                off = _local_metabolite_offset(items[gi], mid, blocks)
                if off is None:
                    continue
                xy = centres[gi] + off
                prev = touched.get(gi)
                if prev is None or (mid, ids[gi]) < (prev[2], ids[gi]):
                    touched[gi] = (xy, off, mid)
        if len(touched) < 2:
            continue
        if len(touched) > hub:
            continue
        t = len(touched)
        w = 1.0 / max(t - 1, 1)
        nodes = [(gi, touched[gi][0], ids[gi]) for gi in sorted(touched)]
        for a, b, d in _kruskal_mst(nodes):
            la, lb = touched[a][1], touched[b][1]
            pairs.append(dict(
                block_a=a, block_b=b,
                anchor_a=(centres[a] + la).tolist(),
                anchor_b=(centres[b] + lb).tolist(),
                local_a=np.asarray(la, float).tolist(),
                local_b=np.asarray(lb, float).tolist(),
                kind='bridging_reaction', metabolite_id=None, reaction_id=rid,
                distance=round(d, 4), weight=w, k=t,
                cross_structure=True,
            ))

    pairs.sort(key=lambda p: (p['kind'], p.get('metabolite_id') or '', p.get('reaction_id') or '',
                              p['block_a'], p['block_b']))
    return pairs, dict(
        per_metabolite=per_met,
        all_pairs_reference=all_pairs_count,
        mst_edges=sum(1 for p in pairs if p['kind'] == 'shared_metabolite'),
        objective=(
            'L = sum over metabolites of (MST edge lengths) × 1/(k-1): each metabolite '
            'contributes its mean MST edge (one unit of mass). MST has k-1 edges, not '
            'C(k,2). Removes group_weights/module_links bias (attraction sum k/2; link '
            'counts C(k,2)). block_weights is unused on main. Straight-line anchors, '
            'not routed length.'
        ),
        copy_policy=(
            'Each curated structure keeps its own copy of a shared metabolite '
            '(copy:<structure>:<metabolite>); copies are never merged and move only '
            'with their structure. The objective does not pull all copies to one point.'
        ),
        bias_removed=True,
    )


def pair_stats(pairs) -> dict[str, Any]:
    """Split normalised objective numbers from unnormalised display quality.

    * ``objective`` — what the search optimises (weight ``1/(k-1)`` per MST edge).
    * ``display`` — real visual lengths/counts (unweighted); normalisation must not
      hide long links or canvas problems.
    """
    empty_obj = dict(total_L=0.0, long_link_mass=0.0,
                     note='per-metabolite normalised (weight 1/(k-1) on each MST edge)')
    empty_disp = dict(
        count=0, total_length=0.0, max=0.0, p90=0.0,
        links_over_250=0, links_over_1000=0, cross_structure_far=0,
        by_kind={'shared_metabolite': 0, 'bridging_reaction': 0},
        note='straight-line inter-structure anchor distance, not routed length',
    )
    if not pairs:
        return dict(objective=empty_obj, display=empty_disp,
                    # Back-compat aliases used by Pareto / older report keys:
                    total_L=0.0, long_links=0.0, total_L_unweighted=0.0,
                    max=0.0, p90=0.0, long_links_count=0, cross_structure_far=0)
    dists = np.array([float(p['distance']) for p in pairs], float)
    weights = np.array([float(p.get('weight', 1.0)) for p in pairs], float)
    long1000 = dists > LONG_LINK
    long250 = dists > 250.0
    cross_far = sum(
        1 for p in pairs
        if p.get('cross_structure') and float(p['distance']) > LONG_LINK
        and p.get('kind') == 'shared_metabolite'
    )
    objective = dict(
        total_L=round(float((weights * dists).sum()), 4),
        long_link_mass=round(float(weights[long1000].sum()), 4),
        note='per-metabolite normalised (weight 1/(k-1) on each MST edge)',
    )
    display = dict(
        count=int(len(dists)),
        total_length=round(float(dists.sum()), 4),
        max=round(float(dists.max()), 4),
        p90=round(float(np.percentile(dists, 90)), 4),
        links_over_250=int(long250.sum()),
        links_over_1000=int(long1000.sum()),
        cross_structure_far=int(cross_far),
        by_kind={
            'shared_metabolite': sum(1 for p in pairs if p['kind'] == 'shared_metabolite'),
            'bridging_reaction': sum(1 for p in pairs if p['kind'] == 'bridging_reaction'),
        },
        note='straight-line inter-structure anchor distance, not routed length',
    )
    return dict(
        objective=objective,
        display=display,
        # Pareto / accept_move aliases (normalised objective + unnormalised extrema):
        total_L=objective['total_L'],
        long_links=objective['long_link_mass'],
        total_L_unweighted=display['total_length'],
        max=display['max'],
        p90=display['p90'],
        long_links_count=display['links_over_1000'],
        cross_structure_far=display['cross_structure_far'],
    )


def _rects(centres, sizes):
    half = sizes / 2.0
    return centres - half, centres + half


def _overlap_count(centres, sizes) -> int:
    lo, hi = _rects(centres, sizes)
    n = len(centres)
    bad = 0
    for i in range(n):
        for j in range(i + 1, n):
            if np.all(lo[i] < hi[j] - _EPS) and np.all(lo[j] < hi[i] - _EPS):
                # positive area overlap
                ov = np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j])
                if np.all(ov > _EPS):
                    bad += 1
    return bad


def _canvas_metrics(centres, sizes):
    lo, hi = _rects(centres, sizes)
    width = float(hi[:, 0].max() - lo[:, 0].min())
    height = float(hi[:, 1].max() - lo[:, 1].min())
    covered = float(np.sum((hi - lo)[:, 0] * (hi - lo)[:, 1]))
    aspect = width / max(height, 1.0)
    area = width * height
    return dict(width=width, height=height, area=area, aspect=aspect,
                compactness=covered / max(area, 1.0))


def _in_aspect_band(aspect: float) -> bool:
    return ASPECT_HARD[0] <= aspect <= ASPECT_HARD[1] or ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]


def evaluate_layout(items, centres, model, cur, blocks=None, hub: int = 30,
                    weights=None, links=None) -> dict[str, Any]:
    """Snapshot metrics used by Pareto accept (and for reports)."""
    centres = np.asarray(centres, float)
    sizes = np.array([it['size'] for it in items], float) + BLOCK_GAP
    pairs, meta = build_attachment_pairs(items, centres, model, cur, blocks=blocks, hub=hub)
    stats = pair_stats(pairs)
    canvas = _canvas_metrics(centres, sizes)
    overlaps = _overlap_count(centres, sizes)
    crossings = 0
    links_over_250 = links_over_1000 = 0
    if links is not None and weights is not None:
        m = BC.metrics(centres, sizes, weights, links)
        crossings = int(m['module_crossings'])
        # Module-centre link counts from arrangement metrics (unnormalised C(k,2) style).
        links_over_250 = int(m.get('links_over_250', 0))
        links_over_1000 = int(m.get('links_over_1000', 0))
    display = dict(stats['display'],
                   structure_overlaps=overlaps,
                   module_crossings=crossings,
                   canvas_area=round(canvas['area'], 2),
                   aspect=round(canvas['aspect'], 4),
                   canvas=[round(canvas['width']), round(canvas['height'])],
                   compactness=round(canvas['compactness'], 4),
                   module_links_over_250=links_over_250,
                   module_links_over_1000=links_over_1000)
    return dict(
        pairs=pairs,
        pair_meta=meta,
        pair_stats=stats,
        objective=stats['objective'],
        display=display,
        # Pareto aliases (normalised objective + unnormalised visual constraints):
        total_L=stats['total_L'],
        total_L_unweighted=stats['total_L_unweighted'],
        max_anchor=stats['max'],
        p90_anchor=stats['p90'],
        long_links=stats['long_links'],
        long_links_count=stats['long_links_count'],
        cross_structure_far=stats['cross_structure_far'],
        module_crossings=crossings,
        overlaps=overlaps,
        canvas_area=round(canvas['area'], 2),
        aspect=round(canvas['aspect'], 4),
        canvas=[round(canvas['width']), round(canvas['height'])],
        compactness=round(canvas['compactness'], 4),
        in_band=_in_aspect_band(canvas['aspect']),
    )


# Search optimises normalised total_L / long_links; other keys are real visual constraints.
_PARETO_KEYS = (
    'total_L', 'max_anchor', 'p90_anchor', 'long_links',
    'module_crossings', 'overlaps', 'canvas_area',
)


def cross_structure_far_delta(before_pairs, after_pairs) -> dict[str, Any]:
    """How many cross-structure shared-metabolite far MST edges shortened.

    Far = straight-line distance > ``LONG_LINK`` (same 1000 px band as
    ``links_over_1000``).  These are the copy-reuse links that only whole-block
    placement can fix (internals never move).
    """
    def key(p):
        return (p.get('metabolite_id'), p['block_a'], p['block_b'])

    before = {
        key(p): float(p['distance'])
        for p in before_pairs
        if p.get('kind') == 'shared_metabolite' and p.get('cross_structure')
    }
    after = {
        key(p): float(p['distance'])
        for p in after_pairs
        if p.get('kind') == 'shared_metabolite' and p.get('cross_structure')
    }
    far_before = {k for k, d in before.items() if d > LONG_LINK}
    far_after = {k for k, d in after.items() if d > LONG_LINK}
    shortened = []
    for k in sorted(far_before, key=lambda x: (-before[x], x[0] or '', x[1], x[2])):
        d0 = before[k]
        d1 = after.get(k)
        if d1 is None:
            continue
        if d1 < d0 - 1e-6:
            shortened.append(dict(
                metabolite_id=k[0], block_a=k[1], block_b=k[2],
                before=round(d0, 2), after=round(d1, 2),
                delta=round(d0 - d1, 2),
            ))
    return dict(
        far_before=len(far_before),
        far_after=len(far_after),
        shortened=len(shortened),
        still_far=len(far_after),
        no_longer_far=len(far_before - far_after),
        newly_far=len(far_after - far_before),
        shortened_rows=shortened[:40],
        threshold=LONG_LINK,
        note=(
            'Skeleton MST attachment pairs (shared_metabolite, cross_structure) '
            'longer than LONG_LINK. Distinct from final_map_cross_structure_far '
            '(drawn edges to multi-copy metabolites; ~12 on main iIT341).'
        ),
    )


def final_map_cross_structure_far(map_json, threshold: float | None = None) -> dict[str, Any]:
    """Far edges on the finished drawing whose metabolite is a reused copy.

    Distinct from skeleton MST ``links_over_1000`` / ``cross_structure_far``:
    this walks routed polylines on the Escher map.  An edge counts when its
    length exceeds ``threshold`` (default ``LONG_LINK``) and the metabolite
    ``bigg_id`` appears on more than one metabolite glyph (a copy reused by
    another structure).  On main iIT341 this yields the reported ~12 edges.
    """
    import layout_quality as LQ

    thr = float(LONG_LINK if threshold is None else threshold)
    body = map_json[1] if isinstance(map_json, list) else map_json
    nodes = body.get('nodes') or {}
    from collections import defaultdict
    by_bigg = defaultdict(list)
    for nid, n in nodes.items():
        if n.get('node_type') == 'metabolite' and n.get('bigg_id'):
            by_bigg[n['bigg_id']].append(nid)
    multi = {mid for mid, nids in by_bigg.items() if len(nids) > 1}
    edges, _ = LQ.build_edges(map_json, None)
    rows = []
    for e in edges:
        pts = e.get('pts')
        if pts is None:
            continue
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        mid = e.get('met')
        if mid not in multi:
            continue
        length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
        if length <= thr:
            continue
        a, b = pts[0], pts[-1]
        rows.append(dict(
            reaction=e.get('rid'), metabolite=mid,
            length_px=round(length, 2),
            straight_px=round(float(np.hypot(*(a - b))), 2),
            n_copies=len(by_bigg[mid]),
            curated=bool(e.get('curated')),
        ))
    rows.sort(key=lambda r: (-r['length_px'], r['reaction'] or '', r['metabolite'] or ''))
    return dict(
        threshold=thr,
        far_count=len(rows),
        multi_copy_metabolites=len(multi),
        edges=rows,
        note=('Final drawn-map edges > threshold whose metabolite bigg_id has '
              'multiple glyphs (copy reused across structures). Not the skeleton MST count.'),
    )


def final_map_far_delta(before_map, after_map, threshold: float | None = None) -> dict[str, Any]:
    """How many final-map multi-copy far edges shortened vs ``before_map``."""
    thr = float(LONG_LINK if threshold is None else threshold)
    before = final_map_cross_structure_far(before_map, threshold=thr)
    after = final_map_cross_structure_far(after_map, threshold=thr)

    def key(r):
        return (r.get('reaction'), r.get('metabolite'))

    b = {key(r): float(r['length_px']) for r in before['edges']}
    a = {key(r): float(r['length_px']) for r in after['edges']}
    # Lengths for every after edge (including those no longer far).
    import layout_quality as LQ
    edges, _ = LQ.build_edges(after_map, None)
    after_len = {}
    for e in edges:
        pts = e.get('pts')
        if pts is None:
            continue
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        after_len[(e.get('rid'), e.get('met'))] = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
    shortened = []
    for k, d0 in sorted(b.items(), key=lambda kv: (-kv[1], kv[0][0] or '', kv[0][1] or '')):
        d1 = after_len.get(k)
        if d1 is None:
            continue
        if d1 < d0 - 1e-6:
            shortened.append(dict(
                reaction=k[0], metabolite=k[1],
                before=round(d0, 2), after=round(d1, 2),
                delta=round(d0 - d1, 2),
                still_far=d1 > thr,
            ))
    return dict(
        far_before=before['far_count'],
        far_after=after['far_count'],
        shortened=len(shortened),
        no_longer_far=len(set(b) - set(a)),
        newly_far=len(set(a) - set(b)),
        shortened_rows=shortened[:40],
        threshold=thr,
        before_edges=before['edges'][:40],
        after_edges=after['edges'][:40],
        note=('Final-map multi-copy far edges vs a baseline map (fresh main). '
              'Distinct from skeleton MST cross_structure_far_edges.'),
    )


def accept_move(before: dict, after: dict) -> tuple[bool, str]:
    """Constraint-first + Pareto accept (swappable for LayoutQualityEvaluator later).

    Hard constraints: no new rectangle overlaps; aspect stays in the existing band
    when the baseline was in-band.  Soft: Pareto improve on ``_PARETO_KEYS`` (no
    weighted sum).
    """
    if after.get('overlaps', 0) > before.get('overlaps', 0):
        return False, 'overlaps increased'
    if before.get('in_band') and not after.get('in_band'):
        return False, 'aspect left allowed band'
    # Canvas / aspect: when baseline in-band, keep after in-band (above). Also reject
    # huge area growth that leaves the soft band even if hard band failed open.
    improved = False
    for key in _PARETO_KEYS:
        bv, av = float(before[key]), float(after[key])
        if av > bv + 1e-6:
            return False, f'{key} worse ({bv} -> {av})'
        if av < bv - 1e-6:
            improved = True
    if not improved:
        return False, 'no Pareto improvement'
    return True, 'Pareto improvement'


def _feasible(centres, sizes, max_overlaps: int = 0) -> bool:
    """True when rectangle overlaps are at most ``max_overlaps`` (usually the start count)."""
    return _overlap_count(centres, sizes) <= max_overlaps


def _swap_candidates(n, pinned):
    for i in range(n):
        if i in pinned:
            continue
        for j in range(i + 1, n):
            if j in pinned:
                continue
            yield ('swap', i, j)


def _relocate_slots(i, centres, sizes, pairs, pinned):
    """Deterministic free-slot proposals for module ``i``.

    Far cross-structure pairs (distance > ``LONG_LINK``) are expanded first with
    more anchor-alignment and compass offsets so long copy-reuse links get slots
    before generic boundary / gap samples truncate the list.
    """
    # (slot, priority): priority 0 = far-pair driven, 1 = other. Within a priority
    # band, nearer proposals are tried first under the fixed relocate budget.
    slots = []
    ci = centres[i]
    si = sizes[i]
    half_i = si / 2.0
    far_pairs = []
    near_pairs = []
    for p in pairs:
        if p['block_a'] == i:
            partner, local, other = p['block_b'], p['local_a'], p['anchor_b']
        elif p['block_b'] == i:
            partner, local, other = p['block_a'], p['local_b'], p['anchor_a']
        else:
            continue
        row = (float(p['distance']), partner, np.asarray(local, float), np.asarray(other, float), p)
        if float(p['distance']) > LONG_LINK:
            far_pairs.append(row)
        else:
            near_pairs.append(row)
    far_pairs.sort(key=lambda t: (-t[0], t[1]))
    near_pairs.sort(key=lambda t: (-t[0], t[1]))

    def _add(pos, priority):
        slots.append((np.asarray(pos, float), priority))

    def _add_partner_slots(partner, local, other, far=False):
        pri = 0 if far else 1
        # Desire: centre + local ≈ other  →  centre ≈ other - local
        desire = other - local
        _add(desire, pri)
        cur = centres[i] + local
        delta = other - cur
        nrm = float(np.linalg.norm(delta))
        if nrm > 1e-6:
            u = delta / nrm
            # Pull toward the partner (more fractions when the pair is still far).
            for scale in (0.25, 0.45, 0.65, 0.85, 1.0) if far else (0.75, 1.0):
                # Fraction of the remaining gap — stays on the line to the partner.
                _add(centres[i] + delta * scale, pri)
            if far:
                # Lateral nudges (perpendicular to the link) around desire / mid-pull.
                perp = np.array([-u[1], u[0]], float)
                lat = half_i.max() + 40.0
                mid = centres[i] + delta * 0.5
                for base in (desire, mid):
                    for sign in (-1.0, 1.0):
                        _add(base + perp * lat * sign, pri)
                # Flush this module against the partner block, matching the partner's anchor axis.
                pj, sj = centres[partner], sizes[partner]
                for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
                    pos = desire.copy()
                    if axis == 0:
                        pos[0] = pj[0] + sign * (sj[0] / 2.0 + half_i[0])
                        pos[1] = other[1] - local[1]
                    else:
                        pos[1] = pj[1] + sign * (sj[1] / 2.0 + half_i[1])
                        pos[0] = other[0] - local[0]
                    _add(pos, pri)

    # 1) Far cross-structure pairs first (widened).
    for _d, partner, local, other, _p in far_pairs:
        _add_partner_slots(partner, local, other, far=True)
    # 2) Remaining attachment partners.
    for _d, partner, local, other, _p in near_pairs:
        _add_partner_slots(partner, local, other, far=False)
    # 3) Flush against other module boundaries.
    for j in range(len(centres)):
        if j == i:
            continue
        cj, sj = centres[j], sizes[j]
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            pos = ci.copy()
            if axis == 0:
                pos[0] = cj[0] + sign * (sj[0] / 2.0 + half_i[0])
                pos[1] = cj[1]
            else:
                pos[1] = cj[1] + sign * (sj[1] / 2.0 + half_i[1])
                pos[0] = cj[0]
            _add(pos, 1)
    # 4) Free-gap samples: corners of the current bounding box expanded by one module.
    lo, hi = _rects(centres, sizes)
    bb0 = lo.min(0) - si
    bb1 = hi.max(0) + si
    for x in (bb0[0], (bb0[0] + bb1[0]) / 2.0, bb1[0]):
        for y in (bb0[1], (bb0[1] + bb1[1]) / 2.0, bb1[1]):
            _add(np.array([x, y], float), 1)
    # Dedup (rounded); far-driven slots keep a higher cap. Nearer first within band.
    limit = RELOCATE_SLOTS + (FAR_RELOCATE_SLOTS if far_pairs else 0)
    seen = set()
    uniq = []
    for s, pri in slots:
        key = (round(float(s[0]), 1), round(float(s[1]), 1))
        if key in seen:
            continue
        if _dist(s, ci) < 1.0:
            continue
        seen.add(key)
        uniq.append((pri, _dist(s, ci), key[0], key[1], s))
    uniq.sort()
    return [t[-1] for t in uniq[:limit]]


def local_search(items, centres, model, cur, blocks=None, hub: int = 30,
                 weights=None, links=None, pinned=(), budget: int = ATTEMPT_BUDGET,
                 log=None) -> tuple[np.ndarray, dict]:
    """Deterministic swap/relocate search with Pareto accept. Returns (centres, report)."""
    centres = np.asarray(centres, float).copy()
    start_centres = centres.copy()
    pinned = set(int(x) for x in pinned)
    sizes = np.array([it['size'] for it in items], float) + BLOCK_GAP
    n = len(items)
    # Placement may already have gap-inflated overlaps; never allow more than the start.
    start_overlaps = _overlap_count(centres, sizes)

    before = evaluate_layout(items, centres, model, cur, blocks=blocks, hub=hub,
                             weights=weights, links=links)
    start = before
    baseline = {k: before[k] for k in list(_PARETO_KEYS) + ['in_band', 'aspect', 'canvas']}

    moves = dict(swap=dict(generated=0, accepted=0, rejected=0, reasons=CounterLike()),
                 relocate=dict(generated=0, accepted=0, rejected=0, reasons=CounterLike()))
    attempts = 0
    # Split the fixed budget so relocate still runs when many swap pairs exist.
    swap_budget = max(1, budget // 2)
    reloc_budget = budget - swap_budget

    # --- swaps (each pair at most once; prefer pairs that currently share a long attachment) ---
    long_touch = defaultdict(int)
    for p in before['pairs']:
        if p['distance'] > LONG_LINK:
            long_touch[p['block_a']] += 1
            long_touch[p['block_b']] += 1
    swap_order = sorted(
        _swap_candidates(n, pinned),
        key=lambda t: (-(long_touch[t[1]] + long_touch[t[2]]), t[1], t[2]),
    )
    for _kind, i, j in swap_order:
        if attempts >= swap_budget:
            break
        attempts += 1
        moves['swap']['generated'] += 1
        trial = centres.copy()
        trial[i], trial[j] = centres[j].copy(), centres[i].copy()
        if not _feasible(trial, sizes, start_overlaps):
            moves['swap']['rejected'] += 1
            moves['swap']['reasons'].add('collision')
            continue
        after = evaluate_layout(items, trial, model, cur, blocks=blocks, hub=hub,
                               weights=weights, links=links)
        ok, reason = accept_move(before, after)
        if ok:
            centres = trial
            before = after
            moves['swap']['accepted'] += 1
            moves['swap']['reasons'].add('accepted:' + reason)
        else:
            moves['swap']['rejected'] += 1
            moves['swap']['reasons'].add(reason)

    # --- relocates ---
    order = [i for i in range(n) if i not in pinned]
    long_touch = defaultdict(int)
    for p in before['pairs']:
        if p['distance'] > LONG_LINK:
            long_touch[p['block_a']] += 1
            long_touch[p['block_b']] += 1
    order.sort(key=lambda i: (-long_touch[i], i))
    reloc_attempts = 0

    for i in order:
        if reloc_attempts >= reloc_budget:
            break
        slots = _relocate_slots(i, centres, sizes, before['pairs'], pinned)
        for slot in slots:
            if reloc_attempts >= reloc_budget:
                break
            reloc_attempts += 1
            attempts += 1
            moves['relocate']['generated'] += 1
            trial = centres.copy()
            trial[i] = slot
            if not _feasible(trial, sizes, start_overlaps):
                moves['relocate']['rejected'] += 1
                moves['relocate']['reasons'].add('collision')
                continue
            after = evaluate_layout(items, trial, model, cur, blocks=blocks, hub=hub,
                                   weights=weights, links=links)
            ok, reason = accept_move(before, after)
            if ok:
                centres = trial
                before = after
                moves['relocate']['accepted'] += 1
                moves['relocate']['reasons'].add('accepted:' + reason)
            else:
                moves['relocate']['rejected'] += 1
                moves['relocate']['reasons'].add(reason)

    # Keep the stage only if the final snapshot Pareto-improves the start
    # (stand-in until LayoutQualityEvaluator owns accept_move).
    final_ok, final_reason = accept_move(start, before)
    if not final_ok:
        centres = start_centres
        before = start

    def _snap(ev):
        return dict(
            objective=ev.get('objective') or (ev.get('pair_stats') or {}).get('objective'),
            display=ev.get('display') or (ev.get('pair_stats') or {}).get('display'),
            pair_stats=ev['pair_stats'], aspect=ev['aspect'], canvas=ev['canvas'],
            total_L=ev['total_L'], total_L_unweighted=ev.get('total_L_unweighted'),
            max_anchor=ev['max_anchor'], p90_anchor=ev['p90_anchor'],
            long_links=ev['long_links'], long_links_count=ev.get('long_links_count'),
            cross_structure_far=ev.get('cross_structure_far'),
            module_crossings=ev['module_crossings'],
            overlaps=ev['overlaps'], canvas_area=ev['canvas_area'],
        )

    before_snap = _snap(start)
    # Compact k-histogram + top metabolites by k (full per-met table is too large for reports).
    k_hist = defaultdict(int)
    for v in start['pair_meta']['per_metabolite'].values():
        k_hist[int(v['k'])] += 1
    top_k = sorted(
        ((m, v) for m, v in start['pair_meta']['per_metabolite'].items()),
        key=lambda kv: (-kv[1]['k'], kv[0]),
    )[:12]
    before_snap.update(
        k_histogram={str(k): k_hist[k] for k in sorted(k_hist)},
        top_metabolites_by_k=[
            dict(metabolite=m, k=v['k'], edges=v['edges'], all_pairs=v['all_pairs'],
                 contribution_L=v['contribution_L'], length_mean=v['length_mean'])
            for m, v in top_k
        ],
        # Do not overwrite the numeric ``objective`` snapshot with the prose string.
        objective_note=start['pair_meta']['objective'],
        copy_policy=start['pair_meta']['copy_policy'],
        bias_removed=start['pair_meta'].get('bias_removed'),
        all_pairs_reference=start['pair_meta'].get('all_pairs_reference'),
        mst_edges=start['pair_meta'].get('mst_edges'),
    )
    cs_delta = cross_structure_far_delta(start['pairs'], before['pairs'])
    return centres, dict(
        before=before_snap,
        after=_snap(before),
        cross_structure_far_edges=cs_delta,
        moves={k: dict(generated=v['generated'], accepted=v['accepted'], rejected=v['rejected'],
                       reasons=v['reasons'].as_dict()) for k, v in moves.items()},
        attempts=attempts, budget=budget, pinned=sorted(pinned),
        accept_interface='accept_move',
        stage_accepted=final_ok, stage_reason=final_reason,
        baseline=baseline,
        _pairs_before=start['pairs'],
        _pairs_after=before['pairs'],
    )


class CounterLike:
    def __init__(self):
        self._c = defaultdict(int)

    def add(self, key):
        self._c[str(key)] += 1

    def as_dict(self):
        return dict(sorted(self._c.items(), key=lambda kv: (-kv[1], kv[0])))


_ARRANGEMENT_NAMES = ('spring', 'stress', 'stress_wide', 'skyline')


def refine(items, centres, model, cur, blocks=None, hub: int = 30,
           weights=None, links=None, pinned=(), arrangements=None,
           arrangement_names=None, diagnostic: bool | None = None,
           log=None) -> tuple[np.ndarray, dict]:
    """Run inter-skeleton refinement on ``centres``. Returns (centres, report).

    When ``arrangements`` holds the four initial canvases, each is refined with the
    same attempt budget and the report records side-by-side display metrics.  That
    diagnostic never changes which arrangement was chosen; the returned centres are
    still the refined winner.  ``diagnostic=False`` skips the extra searches (tests).
    """
    if diagnostic is None:
        # Always refine every initial arrangement when present (report only).
        diagnostic = True if arrangements is not None else DIAGNOSTIC
    centres0 = np.asarray(centres, float).copy()
    refined, report = local_search(
        items, centres0, model, cur, blocks=blocks, hub=hub,
        weights=weights, links=links, pinned=pinned, log=log,
    )
    # Roll the whole stage back when the final state does not Pareto-beat the start.
    if not report['stage_accepted']:
        report = dict(report, rolled_back=True,
                      rollback_reason=report.get('stage_reason') or 'no improvement',
                      after=report['before'])
        refined = centres0
        if log:
            log(f"  inter-skeleton: rolled back ({report['rollback_reason']})")
    else:
        report = dict(report, rolled_back=False)
        if log:
            b, a = report['before'], report['after']
            log(f"  inter-skeleton: L {b['total_L']} -> {a['total_L']}, "
                f"long {b['long_links']} -> {a['long_links']}, "
                f"accepted swap {report['moves']['swap']['accepted']} "
                f"relocate {report['moves']['relocate']['accepted']}")

    report['attachment_pairs_before'] = report['before'].get('display') or report['before'].get('pair_stats')
    report['attachment_pairs_after'] = report['after'].get('display') or report['after'].get('pair_stats')
    report['objective_before'] = report['before'].get('objective')
    report['objective_after'] = report['after'].get('objective')
    report['display_before'] = report['before'].get('display')
    report['display_after'] = report['after'].get('display')
    report['per_metabolite_contribution'] = (
        'Optimisation uses normalised L (MST edges × 1/(k-1)). Reports also keep '
        'unnormalised display quality: total/p90/max length, links_over_250/1000, '
        'overlaps, canvas, crossings. For a metabolite in k structures: MST has '
        '(k-1) edges; module_links would count C(k,2); group_weights sums to k/2.'
    )
    # Drop in-memory pair lists from the JSON report (kept only for cs_delta above).
    report.pop('_pairs_before', None)
    report.pop('_pairs_after', None)

    if diagnostic and arrangements:
        names = list(arrangement_names or _ARRANGEMENT_NAMES)
        diag_rows = []
        best_i, best_L = None, None
        for idx, arr in enumerate(arrangements):
            arr = np.asarray(arr, float)
            c2, r2 = local_search(
                items, arr, model, cur, blocks=blocks, hub=hub,
                weights=weights, links=links, pinned=pinned, log=None,
            )
            final = r2['after'] if r2['stage_accepted'] else r2['before']
            disp = final.get('display') or {}
            name = names[idx] if idx < len(names) else f'arrangement_{idx}'
            row = dict(
                index=idx, name=name,
                stage_accepted=r2['stage_accepted'],
                moves=r2.get('moves'),
                # Objective (normalised).
                total_L=final['total_L'], long_links=final['long_links'],
                # Unnormalised display metrics (side-by-side across starts).
                display=dict(
                    total_length=disp.get('total_length'),
                    max=disp.get('max'), p90=disp.get('p90'),
                    links_over_250=disp.get('links_over_250'),
                    links_over_1000=disp.get('links_over_1000'),
                    cross_structure_far=disp.get('cross_structure_far'),
                    structure_overlaps=disp.get('structure_overlaps'),
                    module_crossings=disp.get('module_crossings'),
                    canvas_area=disp.get('canvas_area'),
                    aspect=disp.get('aspect'), canvas=disp.get('canvas'),
                    compactness=disp.get('compactness'),
                    module_links_over_250=disp.get('module_links_over_250'),
                    module_links_over_1000=disp.get('module_links_over_1000'),
                ),
                aspect=final['aspect'], canvas=final['canvas'],
            )
            diag_rows.append(row)
            if best_L is None or final['total_L'] < best_L - 1e-6:
                best_L, best_i = final['total_L'], idx
            if log:
                d = row['display']
                log(f"  inter-skeleton diagnostic {name}: L {row['total_L']}, "
                    f">1000 {d.get('links_over_1000')}, "
                    f"cs_far {d.get('cross_structure_far')}, "
                    f"accepted={r2['stage_accepted']}")
        winner_L = report['after']['total_L']
        report['diagnostic'] = dict(
            arrangements=diag_rows,
            best_index=best_i,
            best_name=(names[best_i] if best_i is not None and best_i < len(names) else None),
            best_total_L=best_L,
            winning_start_is_best=(best_i is None or abs(best_L - winner_L) <= 1e-4),
            note=('Same ATTEMPT_BUDGET on each initial arrangement (spring / stress / '
                  'stress_wide / skyline). Report only — winner selection unchanged; '
                  'final chooser waits for LayoutQualityEvaluator.'),
        )
    return refined, report


def far_from_reaction_copies(scene_or_blocks, model, cur, zscore: float = 2.5) -> dict:
    """Count metabolite copies whose distance to connected reactions is an outlier.

    Input-quality signal for later island work.  Accepts either a display ``scene``
    with ``positions`` / ``participants`` / ``copies``, or a list of packed ``blocks``.
    """
    reactions = {r['id']: r for r in model.get('reactions') or ()}
    dists = []
    records = []

    if isinstance(scene_or_blocks, dict) and 'positions' in scene_or_blocks:
        pos = scene_or_blocks['positions']
        parts = scene_or_blocks.get('participants') or {}
        for (rid, mid), key in parts.items():
            if mid in cur or key not in pos or ('r:' + rid) not in pos:
                continue
            d = _dist(pos[key], pos['r:' + rid])
            dists.append(d)
            records.append((key, rid, mid, d))
    else:
        for b in scene_or_blocks or ():
            pos = b.get('positions') or {}
            for rid in b.get('reaction_ids') or ():
                rkey = 'r:' + rid
                if rkey not in pos:
                    continue
                r = reactions.get(rid) or {}
                for mid in r.get('metabolites') or ():
                    if mid in cur:
                        continue
                    mkey = 'm:' + mid
                    # Prefer structure-local copy coordinates when present.
                    copy_key = 'copy:' + str(b.get('id')) + ':' + mid
                    if copy_key in pos:
                        mxy = pos[copy_key]
                    elif mkey in pos:
                        mxy = pos[mkey]
                    else:
                        continue
                    d = _dist(mxy, pos[rkey])
                    dists.append(d)
                    records.append((b.get('id'), rid, mid, d))

    if len(dists) < 4:
        return dict(count=0, total=len(dists), threshold=None, zscore=zscore)
    arr = np.asarray(dists, float)
    mu, sigma = float(arr.mean()), float(arr.std())
    if sigma < 1e-9:
        return dict(count=0, total=len(dists), threshold=mu, zscore=zscore)
    thr = mu + zscore * sigma
    outliers = [r for r in records if r[-1] > thr]
    return dict(count=len(outliers), total=len(dists), threshold=round(thr, 2),
                mean=round(mu, 2), std=round(sigma, 2), zscore=zscore)


# Fields that hold absolute canvas coordinates and must share the structure translation.
# ``source_xy`` stays in source-map space and is never translated.
_ABS_POINT_KEYS = ('positions',)
_ABS_NESTED_KEYS = ('paths', 'segments', 'curves', 'routes', 'labels_xy', 'label_positions')


def _shift_xy(obj, delta):
    """In-place: add ``delta`` to any [x,y] / {x,y} / b1/b2 control points found."""
    dx, dy = float(delta[0]), float(delta[1])
    if isinstance(obj, dict):
        if 'x' in obj and 'y' in obj and isinstance(obj['x'], (int, float)):
            obj['x'] = float(obj['x']) + dx
            obj['y'] = float(obj['y']) + dy
        for k in ('b1', 'b2', 'base_point'):
            if k in obj and obj[k] is not None:
                _shift_xy(obj[k], delta)
        for k, v in list(obj.items()):
            if k in ('b1', 'b2', 'base_point', 'x', 'y'):
                continue
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(t, (int, float)) for t in v):
                obj[k] = [float(v[0]) + dx, float(v[1]) + dy]
            elif isinstance(v, (dict, list)):
                _shift_xy(v, delta)
    elif isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(t, (int, float)) for t in obj):
            obj[0] = float(obj[0]) + dx
            obj[1] = float(obj[1]) + dy
        else:
            for item in obj:
                _shift_xy(item, delta)


def translate_structure(block, delta):
    """Translate one curated structure by ``delta`` (full local geometry, one vector).

    Moves metabolite copies, reaction midmarkers, and any stored curve/segment control
    points or anchored label positions that live on the block.  Source-map ``source_xy``
    is left unchanged.  Returns the translation applied.
    """
    delta = np.asarray(delta, float).reshape(2)
    if float(np.hypot(*delta)) < 1e-12:
        return [0.0, 0.0]
    for key in _ABS_POINT_KEYS:
        bag = block.get(key)
        if isinstance(bag, dict):
            block[key] = {k: (np.asarray(v, float) + delta).tolist() for k, v in bag.items()}
    for key in _ABS_NESTED_KEYS:
        if key in block and block[key] is not None:
            _shift_xy(block[key], delta)
    # Anchored labels sometimes sit as label_x/label_y on nested reaction records.
    for key in ('reaction_labels', 'anchored_labels'):
        if key in block and block[key] is not None:
            _shift_xy(block[key], delta)
    return [float(delta[0]), float(delta[1])]


def relative_geometry_signature(block) -> list[tuple]:
    """Sorted pairwise deltas between every absolute point belonging to the structure.

    Identical signatures before/after a move prove the curated internals were
    translated rigidly (no flip, rotation, or relative drift).
    """
    pts = {}
    for k, v in (block.get('positions') or {}).items():
        pts[('pos', k)] = np.asarray(v, float)

    def walk(prefix, obj):
        if isinstance(obj, dict):
            if 'x' in obj and 'y' in obj and isinstance(obj['x'], (int, float)):
                pts[prefix + ('xy',)] = np.array([obj['x'], obj['y']], float)
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    walk(prefix + (str(k),), v)
                elif isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(t, (int, float)) for t in v):
                    pts[prefix + (str(k),)] = np.asarray(v, float)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(prefix + (str(i),), item)

    for key in _ABS_NESTED_KEYS:
        if key in block and block[key] is not None:
            walk((key,), block[key])
    keys = sorted(pts)
    if len(keys) < 2:
        return [(keys[0] if keys else ('empty',), 0.0, 0.0)]
    origin = pts[keys[0]]
    # Pairwise from a stable origin + consecutive chain covers relative layout.
    out = []
    for k in keys[1:]:
        d = pts[k] - origin
        out.append((k, round(float(d[0]), 6), round(float(d[1]), 6)))
    for a, b in zip(keys, keys[1:]):
        d = pts[b] - pts[a]
        out.append(((a, b), round(float(d[0]), 6), round(float(d[1]), 6)))
    return out


def check_curated_internals_unchanged(before_blocks, after_blocks, tol: float = 1e-4) -> dict:
    """Verify every curated structure kept identical relative coordinates."""
    rows = []
    ok = True
    n = min(len(before_blocks), len(after_blocks))
    for i in range(n):
        b0, b1 = before_blocks[i], after_blocks[i]
        s0, s1 = relative_geometry_signature(b0), relative_geometry_signature(b1)
        same = s0 == s1
        if not same and len(s0) == len(s1):
            # Allow tiny float noise.
            same = all(
                a[0] == b[0] and abs(a[1] - b[1]) <= tol and abs(a[2] - b[2]) <= tol
                for a, b in zip(s0, s1)
            )
        rows.append(dict(
            block_id=b1.get('id') or b0.get('id') or i,
            unchanged=bool(same),
            n_points=len((b1.get('positions') or {})),
        ))
        ok = ok and same
    return dict(ok=ok, blocks=len(rows), failed=[r for r in rows if not r['unchanged']],
                note='Relative coords of metabolite copies, midmarkers and stored curve/label '
                     'points must match before/after (pure translation).')


def refine_packed_blocks(model, blocks, cur, hub: int = 30, pinned_reaction_ids=(), log=None):
    """Translate packed curated blocks in place (small-model ELK path).

    Builds one packing item per block, runs the same discrete search as ``refine``,
    and rewrites each block by a pure translation of its full local geometry.
    """
    if len(blocks) < 2:
        return dict(skipped='fewer_than_two_blocks')
    reactions = {r['id']: r for r in model.get('reactions') or ()}
    items = []
    centres = []
    for i, b in enumerate(blocks):
        pts = np.array(list(b['positions'].values()), float)
        lo, hi = pts.min(0), pts.max(0)
        size = hi - lo + 80
        centre = (lo + hi) / 2.0
        items.append(dict(
            kind='curated', reactions=list(b['reaction_ids']), lo=lo, size=size,
            compartment=b.get('container'),
            _rx={rid: reactions[rid] for rid in b['reaction_ids'] if rid in reactions},
            map_id=b.get('map_id'), block_indexes=[i],
        ))
        centres.append(centre)
    centres = np.asarray(centres, float)
    pinned = set()
    pinned_reaction_ids = set(pinned_reaction_ids or ())
    if pinned_reaction_ids:
        for ii, it in enumerate(items):
            rx = set(it['reactions'])
            if rx and rx <= pinned_reaction_ids:
                pinned.add(ii)
    owner = defaultdict(set)
    for gi, it in enumerate(items):
        for rid, r in it['_rx'].items():
            for mid in r.get('metabolites') or ():
                if mid not in cur:
                    owner[mid].add(gi)
    links = defaultdict(int)
    weights = defaultdict(float)
    for mid, gs in owner.items():
        gs = sorted(gs)
        if 1 < len(gs) <= hub:
            w = 1.0 / (len(gs) - 1)
            for a in range(len(gs)):
                for c in range(a + 1, len(gs)):
                    links[gs[a], gs[c]] += 1
                    weights[gs[a], gs[c]] += w
    far_before = far_from_reaction_copies(blocks, model, cur)
    before_sig = [dict(id=b.get('id'), positions={k: list(v) for k, v in (b.get('positions') or {}).items()})
                  for b in blocks]
    # Deep-copy nested absolute geometry for the internals check.
    import copy as _copy
    before_blocks = [_copy.deepcopy(b) for b in blocks]
    # Report-only: same four arrangement starts as plan_global (winner stays ELK pack).
    arrangements = arrangement_names = None
    try:
        import global_layout as _gl
        _, _placement = BC.place(items, weights, links, _gl.layout_blocks, log=None)
        arrangements = (_placement or {}).get('arrangements')
        arrangement_names = [c.get('name') for c in (_placement or {}).get('candidates') or ()]
        if arrangements is not None and len(arrangement_names) != len(arrangements):
            arrangement_names = list(_ARRANGEMENT_NAMES)
    except Exception:
        arrangements = arrangement_names = None
    new_centres, report = refine(
        items, centres, model, cur, blocks=blocks, hub=hub,
        weights=weights, links=links, pinned=pinned,
        arrangements=arrangements, arrangement_names=arrangement_names, log=log,
    )
    if report.get('stage_accepted') and not report.get('rolled_back'):
        for i, (old_c, new_c) in enumerate(zip(centres, new_centres)):
            delta = np.asarray(new_c, float) - np.asarray(old_c, float)
            translate_structure(blocks[i], delta)
    internals = check_curated_internals_unchanged(before_blocks, blocks)
    far_after = far_from_reaction_copies(blocks, model, cur)
    return dict(report, far_from_reaction_before=far_before, far_from_reaction_after=far_after,
                curated_internals=internals)


def routed_attachment_length(map_json, pairs_or_none=None) -> dict:
    """Sum routed polyline lengths for edges that span distinct metabolites (report helper).

    Full pair matching needs the finished scene; this returns total segment length of
    the map as a coarse routed-length companion to straight-line ``L``.  Callers that
    have explicit pair endpoints in the finished map may pass them later.
    """
    try:
        canvas = map_json[1] if isinstance(map_json, list) else map_json
        reactions = canvas.get('reactions') or {}
        total = 0.0
        n_seg = 0
        for r in reactions.values():
            for seg in (r.get('segments') or {}).values():
                b1, b2 = seg.get('b1'), seg.get('b2')
                # Approximate cubic length by control polygon when handles exist.
                nodes = canvas.get('nodes') or {}
                a = nodes.get(seg.get('from_node_id')) or {}
                b = nodes.get(seg.get('to_node_id')) or {}
                if 'x' not in a or 'x' not in b:
                    continue
                pts = [(a['x'], a['y'])]
                if b1:
                    pts.append((b1['x'], b1['y']))
                if b2:
                    pts.append((b2['x'], b2['y']))
                pts.append((b['x'], b['y']))
                for u, v in zip(pts, pts[1:]):
                    total += _dist(u, v)
                    n_seg += 1
        return dict(routed_polyline_length=round(total, 2), segments=n_seg,
                    note='routed segment length on the final map (companion to straight-line L)')
    except Exception as e:
        return dict(error=str(e)[:120])
