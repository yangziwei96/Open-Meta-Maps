"""V2: several arrangements of the modules on the global canvas, one chosen by module-level metrics.

`global_layout.layout_blocks` placed every module (curated blocks and automatic modules, all rigid rectangles) with one spring layout.  For a
large model the result had one weak spot: a bundle of long vertical connections through the middle of the map.  Here the arrangement is
built several ways, each scored on what it will do to the final map, and the best one is the one drawn:

  spring         the V1 arrangement (the previous behaviour; always a candidate, so the choice can never be worse than V1 by these measures)
  stress         stress majorization on the module-connection graph: modules that are N links apart end up ~N module-widths apart
  stress_wide    the same stretched toward 16:9
  skyline        connection-aware skyline with binary-searched target width forced into the 4:3–16:9 band

The module-connection graph: nodes are modules, an edge joins two modules that share a non-currency metabolite; a metabolite shared by more
than `HUB` modules is ignored (ATP/H2O-like hubs must not decide where modules go) and a metabolite shared by k modules adds 1/(k-1) to each
pair.  `links` (unweighted, one per pair per metabolite) is what the metrics count.

Metrics per candidate (all estimated from module rectangles; no routing, no drawing):
  links_over_250 / links_over_1000   module links whose rectangles are further apart than that (an estimate of long cross-module edges)
  module_crossings                   module links whose centre-to-centre segment passes through a third module
  gap_weighted                       sum of link weight x rectangle gap (reported; NOT used alone to choose)
  canvas / compactness / aspect      bounding box, share of it covered by modules, width / height
The choice: prefer candidates inside the hard 4:3–16:9 landscape band; among those, fewest long links, then fewest module crossings
(at equal compactness).  Curated geometry is untouched: only whole modules move.  Deterministic.
"""
import time
from collections import defaultdict

import numpy as np

# Prefer the hard 4:3–16:9 landscape band; keep a soft near-band so stress/spring stay
# eligible when skyline alone would win on aspect but lose on crossings / long links.
ASPECT_RANGE = (1.15, 1.95)
ASPECT_HARD = (4.0 / 3.0, 16.0 / 9.0)
COMPACT_TOLERANCE = 0.80          # a candidate must cover at least this share of the best candidate's compactness (raised)
LONG_TOLERANCE = 0.05             # counts within 5 % are treated as equal and the next criterion decides
COMPACTNESS_WEIGHT = 0.20         # weight for compactness in final scoring
CROSS_WEIGHT = 1.0                # among equal compactness, minimise inter-block link crossings
LINK_WEIGHT = 1.2                 # once aspect holds, weight the long-link count term
GAP_WEIGHT = 1.8                  # once aspect holds, minimise total weighted link length (gap_weighted)


def module_links(groups, cur, hub):
    """groups: [{reactions, _rx}]  ->  (weights {(a, b): w}, links {(a, b): number of shared metabolites})."""
    owner = defaultdict(set)
    for gi, grp in enumerate(groups):
        for rid in grp['reactions']:
            for mid in grp['_rx'][rid]['metabolites']:
                if mid not in cur:
                    owner[mid].add(gi)
    weights, links = defaultdict(float), defaultdict(int)
    for mid, gs in owner.items():
        gs = sorted(gs)
        if 1 < len(gs) <= hub:
            w = 1.0 / (len(gs) - 1)
            for a in range(len(gs)):
                for c in range(a + 1, len(gs)):
                    weights[gs[a], gs[c]] += w
                    links[gs[a], gs[c]] += 1
    return weights, links


def stress_layout(sizes, weights, aspect=1.0, compress=1.0, iterations=300, compact=0.02):
    """Stress majorization (SMACOF) of the module-connection graph with compactness term.
    Target distance between linked modules is the sum of their half diagonals plus a gap;
    unconnected components are placed one after another.  A small inward pull toward the
    centre keeps the layout compact."""
    import networkx as nx
    from scipy.sparse.csgraph import shortest_path
    n = len(sizes)
    sizes = np.asarray(sizes, float)
    radius = np.hypot(sizes[:, 0], sizes[:, 1]) / 2
    W = np.zeros((n, n))
    for (a, b), w in weights.items():
        W[a, b] = W[b, a] = w ** compress if compress != 1.0 else w
    # a strongly shared pair is closer: length 1 / (1 + w)
    L = np.where(W > 0, 1.0 / (1.0 + W), 0.0)
    D = shortest_path(L, directed=False)
    finite = np.isfinite(D)
    big = D[finite].max() if finite.any() else 1.0
    D = np.where(finite, D, big * 1.5 + 1.0)
    unit = float(np.median(radius)) * 2.0 + 70.0  # tighter spacing
    target = D * unit + (radius[:, None] + radius[None, :]) * .7  # closer together
    np.fill_diagonal(target, 0.0)
    # start from a deterministic spiral (no random numbers) stretched to the aspect ratio
    angle = np.arange(n) * 2.399963
    rad = np.sqrt(np.arange(n) + 1.0) * unit * .6  # tighter initial spiral
    xy = np.stack([np.cos(angle) * rad, np.sin(angle) * rad], axis=1)
    weight = 1.0 / np.maximum(target, 1.0) ** 2
    np.fill_diagonal(weight, 0.0)
    wsum = weight.sum(1)
    for it in range(iterations):
        diff = xy[:, None, :] - xy[None, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=2), 1e-6)
        # Gauss-Seidel-free SMACOF update (weighted, majorizing)
        ratio = weight * target / dist
        np.fill_diagonal(ratio, 0.0)
        new = (weight @ xy + ratio.sum(1)[:, None] * xy - ratio @ xy) / wsum[:, None]
        # compactness: pull toward centre (decays over iterations to allow settling)
        centre = new.mean(0)
        pull = compact * (1.0 - it / iterations)
        new = new * (1.0 - pull) + centre * pull
        xy = new
    xy[:, 0] *= aspect ** .5
    xy[:, 1] /= aspect ** .5
    return xy


def _rect_gap(a_lo, a_hi, b_lo, b_hi):
    gap = np.maximum(0.0, np.maximum(b_lo - a_hi, a_lo - b_hi))
    return np.hypot(gap[..., 0], gap[..., 1])


def metrics(centres, sizes, weights, links):
    """The module-level measures of one arrangement (see the module docstring)."""
    half = np.asarray(sizes, float) / 2
    lo, hi = centres - half, centres + half
    pairs = sorted(links)
    a = np.array([p[0] for p in pairs], int)
    b = np.array([p[1] for p in pairs], int)
    cnt = np.array([links[p] for p in pairs], float)
    w = np.array([weights[p] for p in pairs], float)
    gap = _rect_gap(lo[a], hi[a], lo[b], hi[b]) if len(pairs) else np.zeros(0)
    crossings = 0
    for k in range(len(pairs)):
        p, q = centres[a[k]], centres[b[k]]
        d = q - p
        t0, t1 = np.zeros(len(centres)), np.ones(len(centres))
        ok = np.ones(len(centres), bool)
        for ax in (0, 1):
            if abs(d[ax]) < 1e-9:
                ok &= (p[ax] >= lo[:, ax]) & (p[ax] <= hi[:, ax])
            else:
                ta, tb = (lo[:, ax] - p[ax]) / d[ax], (hi[:, ax] - p[ax]) / d[ax]
                t0 = np.maximum(t0, np.minimum(ta, tb))
                t1 = np.minimum(t1, np.maximum(ta, tb))
        ok &= t0 < t1
        ok[a[k]] = ok[b[k]] = False
        crossings += int(ok.any()) * int(cnt[k])
    width, height = (hi.max(0) - lo.min(0))
    covered = float(np.sum((hi - lo)[:, 0] * (hi - lo)[:, 1]))
    return dict(links=int(cnt.sum()), links_over_250=int(cnt[gap > 250].sum()), links_over_1000=int(cnt[gap > 1000].sum()),
                module_crossings=int(crossings), gap_weighted=round(float((w * gap).sum()), 1), gap_median=round(float(np.median(gap)), 1) if len(gap) else 0.0,
                canvas=[round(float(width)), round(float(height))], aspect=round(float(width / max(height, 1.0)), 2),
                compactness=round(covered / max(float(width * height), 1.0), 3))


def choose(rows):
    """rows: [{name, metrics}] -> (index, reason).

    Hard 4:3–16:9 band is preferred whenever any candidate lands in it.  Within the band,
    minimise total weighted link length (``gap_weighted``) so strongly linked blocks sit
    adjacent; then long-link counts, crossings, compactness.
    """
    hard = [i for i, r in enumerate(rows) if ASPECT_HARD[0] <= r['metrics']['aspect'] <= ASPECT_HARD[1]]
    soft = [i for i, r in enumerate(rows) if ASPECT_RANGE[0] <= r['metrics']['aspect'] <= ASPECT_RANGE[1]]
    ok = hard or soft or list(range(len(rows)))
    best_compact = max(rows[i]['metrics']['compactness'] for i in ok)
    ok = [i for i in ok if rows[i]['metrics']['compactness'] >= COMPACT_TOLERANCE * best_compact]
    pool = list(ok)
    steps = []
    if hard:
        steps.append(f'hard_aspect_band: kept {[rows[i]["name"] for i in pool if i in hard] or [rows[i]["name"] for i in pool]}')

    # Within the aspect constraint, choose by total weighted link length first.
    if hard and any(i in hard for i in pool):
        band_pool = [i for i in pool if i in hard] or pool
        low_gap = min(rows[i]['metrics']['gap_weighted'] for i in band_pool)
        gap_keep = [i for i in band_pool
                    if rows[i]['metrics']['gap_weighted'] <= low_gap * (1 + LONG_TOLERANCE) + 1.0]
        if gap_keep:
            steps.append(f'gap_weighted: best {low_gap}, kept {[rows[i]["name"] for i in gap_keep]}')
            pool = gap_keep

    # Eliminate based on links_over_1000 (hard limit)
    for key in ('links_over_1000',):
        low = min(rows[i]['metrics'][key] for i in pool)
        keep = [i for i in pool if rows[i]['metrics'][key] <= low * (1 + LONG_TOLERANCE) + 0.5]
        steps.append(f'{key}: best {low}, kept {[rows[i]["name"] for i in keep]}')
        pool = keep

    # Among remaining: if compactness is within LONG_TOLERANCE of the best, prefer fewer crossings.
    best_comp = max(rows[i]['metrics']['compactness'] for i in pool)
    compact_eq = [i for i in pool if rows[i]['metrics']['compactness'] >= best_comp * (1.0 - LONG_TOLERANCE)]
    if len(compact_eq) > 1:
        low_cross = min(rows[i]['metrics']['module_crossings'] for i in compact_eq)
        cross_keep = [i for i in compact_eq if rows[i]['metrics']['module_crossings'] <= low_cross * (1 + LONG_TOLERANCE) + 0.5]
        if cross_keep:
            steps.append(f'equal_compactness_min_crossings: best {low_cross}, kept {[rows[i]["name"] for i in cross_keep]}')
            pool = cross_keep

    def score(i):
        m = rows[i]['metrics']
        best_250 = min(rows[j]['metrics']['links_over_250'] for j in pool) + 1
        best_cross = min(rows[j]['metrics']['module_crossings'] for j in pool) + 1
        best_comp_s = max(rows[j]['metrics']['compactness'] for j in pool)
        best_gap = min(rows[j]['metrics']['gap_weighted'] for j in pool) + 1.0
        in_band = ASPECT_HARD[0] <= m['aspect'] <= ASPECT_HARD[1]
        link_w = LINK_WEIGHT if in_band else 1.0
        gap_w = GAP_WEIGHT if in_band else 0.5
        s = (gap_w * m['gap_weighted'] / best_gap +
             link_w * m['links_over_250'] / best_250 +
             CROSS_WEIGHT * m['module_crossings'] / best_cross +
             COMPACTNESS_WEIGHT * (1.0 - m['compactness'] / best_comp_s))
        if in_band:
            s *= 0.85
        return s

    if len(pool) > 1:
        scores = {i: score(i) for i in pool}
        winner = min(pool, key=lambda i: scores[i])
        steps.append(f'weighted_score: {[(rows[i]["name"], round(scores[i], 3)) for i in pool]}, winner {rows[winner]["name"]}')
    else:
        winner = pool[0]

    discarded = [rows[i]['name'] for i in range(len(rows)) if i not in ok]
    return winner, dict(rule='hard 4:3-16:9 prefer, min gap_weighted in-band, links_over_1000, equal-compactness min crossings, then weighted score', discarded=discarded, steps=steps)


def _skyline_centres(sizes, weights, links, target_aspect=1.55):
    """Connection-aware skyline packing into the 4:3–16:9 band (tile top-left → centres)."""
    import composition as C
    size_map = {i: (float(sizes[i][0]), float(sizes[i][1])) for i in range(len(sizes))}
    # Prefer chemically linked neighbours sitting adjacent (skyline link term).
    wmap = {}
    for (a, b), w in (weights or {}).items():
        wmap[C._link_key(a, b)] = float(w)
    placed, report = C.pack_tiles_aspect(size_map, gap=90.0, weights=wmap, target_aspect=target_aspect)
    centres = np.zeros((len(sizes), 2), float)
    for i, (x, y) in placed.items():
        centres[i] = (x + size_map[i][0] / 2.0, y + size_map[i][1] / 2.0)
    return centres, report


def place(items, weights, links, layout_blocks, log=None):
    """Build the candidates, score them, return (centres of the winner, report)."""
    sizes = np.array([i['size'] for i in items], float)
    # Skyline (binary-searched width → 4:3–16:9) is a first-class candidate so tall
    # columns cannot win on other metrics when a landscape pack exists.
    specs = [('spring', dict()),
             ('stress', dict(init='stress', aspect=1.5)),
             ('stress_wide', dict(init='stress', aspect=16.0 / 9.0)),
             ('skyline', dict(init='skyline', aspect=1.55))]
    rows, arrangements = [], []
    for name, spec in specs:
        t = time.time()
        sky_report = None
        if spec.get('init') == 'skyline':
            centres, sky_report = _skyline_centres(sizes, weights, links, target_aspect=spec['aspect'])
            # Light overlap separation only — do not re-run gravity that undoes the landscape frame.
            half = (sizes + 90.0) / 2.0
            import global_layout as gl
            centres = gl._separate(np.asarray(centres, float), half, passes=40)
        elif spec.get('init') == 'stress':
            init = stress_layout(sizes, weights, aspect=spec['aspect'], compress=spec.get('compress', 1.0))
            centres = layout_blocks(sizes, weights, aspect=spec['aspect'], init=init)
        else:
            centres = layout_blocks(sizes, weights, aspect=1.5)
        m = metrics(centres, sizes + 90.0, weights, links)
        row = dict(name=name, metrics=m, seconds=round(time.time() - t, 1))
        if sky_report is not None:
            row['skyline'] = {k: sky_report[k] for k in ('aspect', 'in_band', 'target_width', 'width', 'height', 'link_cost', 'crossings', 'widths_tried') if k in sky_report}
        rows.append(row)
        arrangements.append(centres)
        if log:
            log(f'  candidate {name}: >250 {m["links_over_250"]}, >1000 {m["links_over_1000"]}, crossings {m["module_crossings"]}, aspect {m["aspect"]}, compactness {m["compactness"]} ({rows[-1]["seconds"]} s)')
    win, why = choose(rows)
    why['chosen'] = rows[win]['name']
    if log:
        log(f'  chosen arrangement: {rows[win]["name"]}')
    # Keep all four canvases in memory for optional inter-skeleton diagnostics.
    return arrangements[win], dict(candidates=rows, selection=why, arrangements=arrangements)
