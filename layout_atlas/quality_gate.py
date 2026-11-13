"""Unified quality gate: a candidate change to the final map is accepted only if the final geometry does not get worse, compared tier by tier.

Tiers (a plain lexicographic comparison with small tolerances, no combined score):
  hard    GEM completeness, every algorithm connection still routed (Bézier or orthogonal polyline), no connection of a moved reaction without segments -> never allowed to degrade
  tier 1  disconnected peripheral modules (a real connection that the map does not show), curves through nodes, curves through labels,
          label overlaps, node overlaps                                                                                         -> must not get worse; improvement is decisive
  tier 2  crossings of algorithm connections (algorithm x algorithm + algorithm x curated)                                       -> may not rise by more than TOL_CROSS
  tier 3  connections longer than 250 px, small-angle share etc.                                                                 -> may not rise by more than 5 %

`evaluate_map(data, skip)` runs the final-geometry evaluator; `accept(old, new)` returns (decision, reason); `translate_module(...)` moves a rigid module inside
an Escher map and re-fits its algorithm connections; `peripheral_pass(...)` is the loop that uses them for the peripheral modules.
"""
import copy
import time

import numpy as np

import layout_quality

TOL_CROSS = 2
TOL_LONG = 0.05


def proxy_score(scene, keys, delta, long_connections, clear=85.0):
    """Cheap geometric proxy for a rigid module move: no Bezier restyle, no full evaluate.

    Returns dict(long_gain, collision_est, score).  Used to rank candidates before one full restyle.
    long_connections: [(reaction_id, outside_key, px), ...] in scene units.
    Generic: operates on positions and distances only.
    """
    pos = scene['positions']
    dxy = np.asarray(delta, float)
    # Predicted shortening of the triggering long connections
    long_gain = 0.0
    for rid, outside, px in long_connections:
        rk = 'r:' + rid
        if rk not in pos or outside not in pos:
            continue
        before = float(px)
        after = float(np.hypot(*(np.array(pos[rk], float) + dxy - np.array(pos[outside], float))))
        long_gain += max(0.0, before - after)
    # Rough collision estimate: how many foreign glyphs fall inside the moved AABB (+clear)
    local = np.array([pos[k] for k in keys if k in pos], float)
    if not len(local):
        return dict(long_gain=0.0, collision_est=0, score=0.0)
    moved = local + dxy
    lo, hi = moved.min(0) - clear, moved.max(0) + clear
    keyset = set(keys)
    collision_est = 0
    for k, v in pos.items():
        if k in keyset or k.startswith('side:'):
            continue
        x, y = v
        if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
            collision_est += 1
    # Prefer large shortening with few new overlaps
    score = long_gain - 40.0 * collision_est
    return dict(long_gain=round(long_gain, 1), collision_est=collision_est, score=round(score, 1))


def evaluate_map(data, skip_edges, disconnected=0):
    s, issues = layout_quality.evaluate(data, skip_edges)
    c = s['core']
    x, co, b = c['crossings'], c['collisions'], c['bezier_coverage']
    return dict(
        hard=int(b['share'] < 0.999),
        disconnected=disconnected,
        collisions=co['curves_through_nodes'] + co['curves_through_labels'] + co['label_overlap_pairs'] + co['node_overlap_pairs'],
        crossings=x['algorithm_algorithm'] + x.get('algorithm_curated', x.get('algorithm_expert', 0)),
        long_edges=(c.get('long_lines') or {}).get('over_250', 0),
        detail=dict(through_nodes=co['curves_through_nodes'], through_labels=co['curves_through_labels'], label_overlaps=co['label_overlap_pairs'], node_overlaps=co['node_overlap_pairs'])), s, issues


def accept(old, new):
    if new['hard'] > old['hard']:
        return False, 'hard constraint broken'
    if new['collisions'] > old['collisions']:
        return False, 'more collisions (%d -> %d)' % (old['collisions'], new['collisions'])
    if new['crossings'] > old['crossings'] + TOL_CROSS:
        return False, 'more crossings (%d -> %d)' % (old['crossings'], new['crossings'])
    if new['long_edges'] > old['long_edges'] * (1 + TOL_LONG) + 1:
        return False, 'more long connections (%d -> %d)' % (old['long_edges'], new['long_edges'])
    better = (new['disconnected'] < old['disconnected'], new['collisions'] < old['collisions'], new['crossings'] < old['crossings'], new['long_edges'] < old['long_edges'])
    if any(better):
        return True, 'improves ' + ', '.join(n for n, f in zip(('connection', 'collisions', 'crossings', 'long connections'), better) if f)
    return False, 'no improvement'


def _ids(scene):
    return {k: str(i) for i, k in enumerate(sorted(scene['positions']))}


def translate_module(data, scene, keys, delta, reactions, skip_edges):
    """Move the glyphs `keys` of a rigid module by `delta` (scene units; the map has a constant offset) inside a COPY of the map and re-fit the module's algorithm
    connections.  Expert connections (their controls and multimarkers) move with the glyphs."""
    import edge_style
    d2 = copy.deepcopy(data)
    body = d2[1]
    nodes, rxs = body['nodes'], body['reactions']
    ids = _ids(scene)
    dx, dy = float(delta[0]), float(delta[1])
    moved_nodes = set()
    for k in keys:
        nid = ids.get(k)
        if nid is None or nid not in nodes:
            continue
        n = nodes[nid]
        n['x'] = round(n['x'] + dx, 2)
        n['y'] = round(n['y'] + dy, 2)
        for lk in ('label_x', 'label_y'):
            if lk in n:
                n[lk] = round(n[lk] + (dx if lk == 'label_x' else dy), 2)
        moved_nodes.add(nid)
    redo = set()
    counter = 0
    for rx in rxs.values():
        if rx['bigg_id'] not in reactions:
            continue
        mid = next((s['from_node_id'] for s in rx['segments'].values() if nodes[s['from_node_id']].get('node_type') == 'midmarker'), None)
        rx['label_x'] = round(rx['label_x'] + dx, 2)
        rx['label_y'] = round(rx['label_y'] + dy, 2)
        out = {}
        for key, s in rx['segments'].items():
            out.setdefault(s['from_node_id'], []).append((key, s))
        keep, drop_nodes, straight = {}, set(), []
        for key, s in rx['segments'].items():
            algo = str(key).startswith('es_s') or str(key).startswith('route')
            if not algo:
                for nn in (s['from_node_id'], s['to_node_id']):
                    if nodes[nn].get('node_type') == 'multimarker' and nn not in moved_nodes:
                        nodes[nn]['x'] = round(nodes[nn]['x'] + dx, 2)
                        nodes[nn]['y'] = round(nodes[nn]['y'] + dy, 2)
                        moved_nodes.add(nn)
                for c in ('b1', 'b2'):
                    if s.get(c):
                        s[c] = dict(x=round(s[c]['x'] + dx, 2), y=round(s[c]['y'] + dy, 2))
                keep[key] = s
        # an algorithm connection is the chain reaction point -> (corner points) -> metabolite: replace each by one straight segment, edge_style re-fits it
        for key, s in rx['segments'].items():
            if s['from_node_id'] != mid or key in keep or not (str(key).startswith('es_s') or str(key).startswith('route')):
                continue
            cur_node, chain = s['to_node_id'], [s['from_node_id']]
            while nodes[cur_node].get('node_type') == 'multimarker':
                drop_nodes.add(cur_node)
                nxt = out.get(cur_node)
                if not nxt:
                    break
                cur_node = nxt[0][1]['to_node_id']
            if nodes[cur_node].get('node_type') == 'metabolite':
                counter += 1
                straight.append((f'qg{counter}', dict(from_node_id=mid, to_node_id=cur_node, b1=None, b2=None)))
        for key, s in straight:
            keep[key] = s
        rx['segments'] = keep
        for nn in drop_nodes:
            nodes.pop(nn, None)
        redo.add(rx['bigg_id'])
    return d2, redo, moved_nodes


def peripheral_pass(data, scene, proposals, skip_edges, budget_s=150.0, max_candidates=24, log=None):
    """Try each proposed module move on a copy of the map, keep it only when `accept` says the final geometry is not worse.  Returns (data, report)."""
    import edge_style
    from layout_quality_evaluator import LayoutQualityEvaluator
    t0 = time.time()
    far = len(proposals)
    base, summary, issues = evaluate_map(data, skip_edges, disconnected=far)
    rows, accepted = [], 0
    gate = LayoutQualityEvaluator(
        data, skip_edges=skip_edges, rule='peripheral', require_local_improvement=True,
    )
    for prop in sorted(proposals, key=lambda p: -p.get('connection_px', [0, 0])[0])[:max_candidates]:
        if time.time() - t0 > budget_s:
            rows.append(dict(exchange=prop['exchange'], decision='budget_exhausted'))
            continue
        d2, redo, _ = translate_module(data, scene, prop['keys'], prop['delta'], set(prop['reactions']), skip_edges)
        edge_style.apply(d2, skip_edges, only=redo)
        new, s2, i2 = evaluate_map(d2, skip_edges, disconnected=far - 1)
        if time.time() - t0 > budget_s:
            rows.append(dict(exchange=prop['exchange'], decision='budget_exhausted'))
            continue
        gate.begin_report(base)
        decision = gate.evaluate(d2, metrics=new, disconnected=far - 1)
        ok, why = decision.accepted, decision.reason
        rows.append(dict(exchange=prop['exchange'], transports=prop['transports'], decision='accepted' if ok else 'rejected', reason=why, moved_px=prop.get('moved_px'),
                         before={k: base[k] for k in ('collisions', 'crossings', 'long_edges')}, after={k: new[k] for k in ('collisions', 'crossings', 'long_edges')}))
        if ok:
            data, base, summary, issues = d2, new, s2, i2
            base['disconnected'] = far - 1
            far -= 1
            accepted += 1
            for k in prop['keys']:
                if k in scene['positions']:
                    scene['positions'][k] = [scene['positions'][k][0] + prop['delta'][0], scene['positions'][k][1] + prop['delta'][1]]
    for prop in sorted(proposals, key=lambda p: -p.get('connection_px', [0, 0])[0])[max_candidates:]:
        rows.append(dict(exchange=prop['exchange'], decision='budget_exhausted'))
    return data, dict(proposed=len(proposals), accepted=accepted, rejected=sum(r['decision'] == 'rejected' for r in rows), budget_exhausted=sum(r['decision'] == 'budget_exhausted' for r in rows),
                      seconds=round(time.time() - t0, 1), rows=rows)
