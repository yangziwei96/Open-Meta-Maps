"""Pathway-aware hierarchical placement (V7.3 P4): functional regions are placed NEXT TO the region they are most strongly linked to, judged on the finished map.

The failed `regions.py` moved every unit with free forces.  This pass keeps the existing arrangement and only asks, region by region, "is there a better place for this
whole region beside its strongest neighbour?" - the biological link decides who should be adjacent, the geometry only chooses the side and the alignment:

  units       expert blocks and algorithm groups (`regions.build_units`), rigid
  links       shared non-currency metabolites between units (hub metabolites damped, as in `regions`)
  regions     Louvain communities of the unit graph (seeded); the region that holds the biggest unit stays where it is (the anchor of the picture)
  order       the other regions, strongest link to what is already placed first
  candidates  for one region: the four sides (left / right / above / below) of its strongest PLACED neighbour region, each centred on the neighbour's centre line or aligned
              with its top / left edge, at a gap of GAP px; a candidate must be free (rectangle + CLEARANCE against every unit outside the region)
  decision    each candidate is drawn on a COPY of the finished map (the region's glyphs move rigidly; connections that cross the region boundary are re-drawn as Bezier
              curves, expert ones are recorded as regenerated) and the best one is kept only if the balanced gain (`block_relocation._accept`) is positive; otherwise the
              region stays where it is.  Region adjacency (boundary gap between linked regions, in reaction spacings) is reported before / after.
Only whole-region translations; GEM content, protected connections and the inside of every expert reaction are untouched.
"""
import time
from collections import defaultdict

import numpy as np

GAP = 170.0
CLEARANCE = 90.0
MAX_REGIONS = 8
MAX_UNITS_PER_METABOLITE = 6


def _regions(model, scene, blocks, cur):
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    import regions as R
    unit_of, keys_of, reactions_of = R.build_units(model, scene, blocks)
    uids = sorted(u for u in keys_of if keys_of[u])
    index = {u: i for i, u in enumerate(uids)}
    users = defaultdict(set)
    for r in model['reactions']:
        for mid in r['metabolites']:
            if mid not in cur:
                users[mid].add(unit_of[r['id']])
    W = defaultdict(float)
    for mid, us in users.items():
        us = sorted(us)
        if 2 <= len(us) <= MAX_UNITS_PER_METABOLITE:
            for a in range(len(us)):
                for b in range(a + 1, len(us)):
                    W[index[us[a]], index[us[b]]] += 1.0 / (len(us) - 1)
    G = nx.Graph()
    G.add_nodes_from(range(len(uids)))
    for (a, b), w in W.items():
        G.add_edge(a, b, weight=w)
    comms = louvain_communities(G, weight='weight', seed=0) if G.number_of_edges() else [{i} for i in range(len(uids))]
    regs = [sorted(uids[i] for i in c) for c in sorted(comms, key=lambda s: (-len(s), min(s)))]
    return regs, unit_of, keys_of, reactions_of, W, index, uids


def _rect(scene, keys, delta=(0.0, 0.0)):
    pts = np.array([scene['positions'][k] for k in keys if k in scene['positions']], float)
    return pts.min(0) + np.asarray(delta, float) - 60.0, pts.max(0) + np.asarray(delta, float) + 60.0


def _region_unit(model, scene, reactions_of, keys_of, region_units):
    """The movable description of a region: reactions, the glyph keys used ONLY by them, and the connections that cross the region boundary."""
    parts = scene['participants']
    reactions = set().union(*[reactions_of[u] for u in region_units])
    holders = defaultdict(set)
    for (rid, mid), k in parts.items():
        holders[k].add(rid)
    keys = set()
    for u in region_units:
        for k in keys_of[u]:
            if k in scene['positions'] and (not holders.get(k) or holders[k] <= reactions or k.startswith('r:')):
                keys.add(k)
    rx = {r['id']: r for r in model['reactions']}
    cross = []
    for r in reactions:
        for mid in rx[r]['metabolites']:
            k = parts[r, mid]
            if k not in keys and k in scene['positions']:
                cross.append((r, mid))
    return dict(keys=keys, reactions=sorted(reactions), cross=cross, block='region')


def _gap(a, b):
    """Boundary gap between two rectangles (0 when they touch or overlap)."""
    dx = max(a[0][0] - b[1][0], b[0][0] - a[1][0], 0.0)
    dy = max(a[0][1] - b[1][1], b[0][1] - a[1][1], 0.0)
    return float(np.hypot(dx, dy))


MAX_COMPACT_UNIT_REACTIONS = 25


def _step(ctx, unit, nb_keys, exclude_units, keys_of, label, row_extra):
    """One move: put `unit` (a region or a single unit) beside `nb_keys` on the best side, kept only if the balanced gate accepts.  Updates ctx in place; -> row."""
    import block_relocation as br
    import edge_style
    import quality_gate as qg
    data, scene, skip_edges, base, spacing = ctx['data'], ctx['scene'], ctx['skip'], ctx['base'], ctx['spacing']
    mine = _rect(scene, unit['keys'])
    theirs = _rect(scene, nb_keys)
    gap_before = _gap(mine, theirs)
    others = [(u, _rect(scene, keys_of[u])) for u in keys_of if keys_of[u] and u not in exclude_units]
    cx_t, cy_t = (theirs[0] + theirs[1]) / 2
    cx_m, cy_m = (mine[0] + mine[1]) / 2
    wm, hm = mine[1] - mine[0]
    cands = []
    for side in ('right', 'left', 'below', 'above'):
        for align in ('centre', 'edge'):
            if side in ('right', 'left'):
                x = theirs[1][0] + GAP + wm / 2 if side == 'right' else theirs[0][0] - GAP - wm / 2
                y = cy_t if align == 'centre' else theirs[0][1] + hm / 2
            else:
                y = theirs[1][1] + GAP + hm / 2 if side == 'below' else theirs[0][1] - GAP - hm / 2
                x = cx_t if align == 'centre' else theirs[0][0] + wm / 2
            delta = np.array([x - cx_m, y - cy_m])
            lo, hi = _rect(scene, unit['keys'], delta)
            if any((lo < ohi + CLEARANCE).all() and (olo - CLEARANCE < hi).all() for _, (olo, ohi) in others):
                continue
            cands.append((side, align, delta))
    row = dict(label=label, reactions=len(unit['reactions']), gap_before_spacings=round(gap_before / spacing, 1), tries=[], **row_extra)
    best = None
    for side, align, delta in cands:
        d1, detached = br._detach_cross_edges(data, scene, unit)
        d2, redo, _ = qg.translate_module(d1, scene, unit['keys'], delta, set(unit['reactions']), skip_edges)
        sk = [e for e in skip_edges if list(e) not in detached]
        edge_style.apply(d2, sk, only=redo)
        new, s2, i2 = qg.evaluate_map(d2, sk, disconnected=0)
        ok, why = br._accept(ctx['base'], new)
        dj = br.W_CROSS * (new['crossings'] - ctx['base']['crossings']) + br.W_COLL * (new['collisions'] - ctx['base']['collisions']) + br.W_LONG * (new['long_edges'] - ctx['base']['long_edges'])
        row['tries'].append(dict(side=side, align=align, moved_px=round(float(np.hypot(*delta)), 1), decision='accepted' if ok else 'rejected', reason=why,
                                 after={k: new[k] for k in ('collisions', 'crossings', 'long_edges')}))
        ctx['tried'] += 1
        if ok and (best is None or dj < best[0]):
            best = (dj, d2, new, delta, sk, detached, side, align)
    if best is not None:
        dj, d2, new, delta, sk, detached, side, align = best
        ctx['data'], ctx['base'] = d2, new
        skip_edges[:] = sk
        for k in unit['keys']:
            scene['positions'][k] = [scene['positions'][k][0] + float(delta[0]), scene['positions'][k][1] + float(delta[1])]
        row['decision'] = 'accepted'
        row['chosen'] = dict(side=side, align=align, moved_px=round(float(np.hypot(*delta)), 1))
        row['regenerated_expert_edges'] = detached
        ctx['accepted'] += 1
    else:
        row['decision'] = 'rejected' if row['tries'] else 'no free candidate'
        ctx['rejected'] += 1
    row['gap_after_spacings'] = round(_gap(_rect(scene, unit['keys']), theirs) / spacing, 1)
    return row


def _spread(scene, keys_of, region_units):
    """Largest boundary gap between a unit of the region and the rest of the region (how scattered the region is), in px."""
    rects = [_rect(scene, keys_of[u]) for u in region_units if keys_of[u]]
    if len(rects) < 2:
        return 0.0
    return max(min(_gap(a, b) for j, b in enumerate(rects) if j != i) for i, a in enumerate(rects))


def place(data, model, scene, blocks, cur, skip_edges, log=None):
    """Place regions beside linked neighbours. Adjacency tries at most MAX_REGIONS; no wall-clock."""
    import block_relocation as br
    import quality_gate as qg
    t0 = time.time()
    regs, unit_of, keys_of, reactions_of, W, index, uids = _regions(model, scene, blocks, cur)
    report = dict(regions=len(regs), region_sizes=[len(r) for r in regs][:14], compaction=[], rows=[])
    if len(regs) < 2:
        return data, report
    spacing = br._spacing(scene['positions'])
    base, summary, issues = qg.evaluate_map(data, skip_edges, disconnected=0)
    ctx = dict(data=data, scene=scene, skip=skip_edges, base=base, spacing=spacing, tried=0, accepted=0, rejected=0)
    size = {i: sum(len(reactions_of[u]) for u in r) for i, r in enumerate(regs)}
    anchor = max(size, key=lambda i: (size[i], -i))
    region_of_unit = {u: i for i, r in enumerate(regs) for u in r}
    unit_link = defaultdict(float)
    for (a, b), w in W.items():
        unit_link[uids[a], uids[b]] += w
        unit_link[uids[b], uids[a]] += w
    # --- 1. compaction: inside each region the units are brought beside their strongest partner in the same region
    for ri, region in enumerate(regs):
        if len(region) < 2:
            continue
        spread_before = _spread(scene, keys_of, region)
        big = max(region, key=lambda u: (len(reactions_of[u]), u))
        placed_units = {big}
        todo = [u for u in region if u != big]
        info = dict(region=ri, units=len(region), spread_before_spacings=round(spread_before / spacing, 1), moves=[])
        while todo:
            nxt = max(todo, key=lambda u: (sum(unit_link.get((u, v), 0.0) for v in placed_units), len(reactions_of[u]), u))
            w_best = sum(unit_link.get((nxt, v), 0.0) for v in placed_units)
            todo.remove(nxt)
            if w_best <= 0 or len(reactions_of[nxt]) > MAX_COMPACT_UNIT_REACTIONS:
                placed_units.add(nxt)
                continue
            partner = max(placed_units, key=lambda v: unit_link.get((nxt, v), 0.0))
            unit = _region_unit(model, scene, reactions_of, keys_of, [nxt])
            row = _step(ctx, unit, keys_of[partner], {nxt}, keys_of, nxt, dict(partner=partner, link=round(unit_link.get((nxt, partner), 0.0), 2)))
            info['moves'].append({k: row[k] for k in ('label', 'reactions', 'partner', 'link', 'gap_before_spacings', 'gap_after_spacings', 'decision') if k in row})
            placed_units.add(nxt)
        info['spread_after_spacings'] = round(_spread(scene, keys_of, region) / spacing, 1)
        report['compaction'].append(info)
        if log:
            log(f"  region {ri} compaction: spread {info['spread_before_spacings']} -> {info['spread_after_spacings']} spacings ({len(info['moves'])} units tried)")
    # --- 2. region adjacency: each region beside its strongest placed neighbour region
    link = defaultdict(float)
    for (a, b), w in W.items():
        ra, rb = region_of_unit[uids[a]], region_of_unit[uids[b]]
        if ra != rb:
            link[(min(ra, rb), max(ra, rb))] += w
    placed, order = {anchor}, []
    remaining = {i for i in range(len(regs)) if i != anchor}
    while remaining:
        nxt = max(remaining, key=lambda i: (sum(link.get((min(i, j), max(i, j)), 0.0) for j in placed), size[i], -i))
        if sum(link.get((min(nxt, j), max(nxt, j)), 0.0) for j in placed) <= 0:
            break
        order.append(nxt)
        placed.add(nxt)
        remaining.discard(nxt)
    done = {anchor}
    for ri in order[:MAX_REGIONS]:
        nb = max(done, key=lambda j: link.get((min(ri, j), max(ri, j)), 0.0))
        unit = _region_unit(model, scene, reactions_of, keys_of, regs[ri])
        nb_keys = set().union(*[keys_of[u] for u in regs[nb]])
        row = _step(ctx, unit, nb_keys, set(regs[ri]), keys_of, regs[ri][:3], dict(neighbour=regs[nb][:3], link=round(link.get((min(ri, nb), max(ri, nb)), 0.0), 2)))
        report['rows'].append(row)
        done.add(ri)
        if log:
            log(f"  region placement {regs[ri][:2]}...: {row['decision']} (gap {row['gap_before_spacings']} -> {row['gap_after_spacings']} spacings)")
    report.update(tried=ctx['tried'], accepted=ctx['accepted'], rejected=ctx['rejected'], seconds=round(time.time() - t0, 1))
    return ctx['data'], report
