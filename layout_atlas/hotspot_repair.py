"""Post-layout local repair of crossing hotspots.

After Bézier routing and label placement, reuses ``layout_quality``'s shared
crossing detector (``build_edges`` / ``_seg_hits`` / ``CrossingIndex``) to find
the top-k densest 400 px windows and try cheap generic moves.  A move is kept
only when the headline crossing count (algorithm×algorithm + algorithm×curated)
strictly falls and node/label overlaps and curves-through-nodes do not rise.
Curated geometry moves only as a whole-block translation. Previous-map pins
are never moved.

Evaluation is incremental: a spatial segment index is built once; each candidate
only recounts segments of the touched reactions (delta counting). Candidate
generation is pruned to nodes/edges that participate in crossings inside the
window. Each visit tries that window's ranked candidates (regenerating after
an accept); the first accepting window ends the round. Search stops when a
full pass accepts nothing, or when accept / round / mass-scaled try caps bind.
Try caps replace wall-clock: same window-mass coefficients as before, never a
timer. Deterministic and model-agnostic.
"""
from __future__ import annotations

import copy
import math
import time
from collections import defaultdict

import numpy as np

import edge_style
import layout_quality as lq
from core import CURRENCY, chem

DEFAULT_K = 5  # layout_quality.evaluate worst_regions count (tests / explicit k)
WINDOW_PAD = 40.0
# Mass-scaled try budget from window crossings/edges; try ceiling only —
# never as wall-clock. Caller may raise the implied depth via budget_s.
BUDGET_CROSS = 0.05
BUDGET_EDGE = 0.002
BUDGET_FLOOR = 0.8
BUDGET_CAP = 30.0
TRIES_PER_CROSS = 4.0
# Callers still pass legacy wall-clock ``budget_s`` (e.g. 35 from tool_pipeline).
# After mass-scaled tries, that number must NOT divide into auto_budget (~1s) or
# tries explode (~40× → multi-minute hotspot). Cap the scale factor.
BUDGET_SCALE_CAP = 3.0
# Default k=DEFAULT_K matches layout_quality.evaluate worst_regions. Caps:
#   max_accepts / max_rounds from caller (tool_pipeline) or window crossing mass
#   max_tries from window mass × TRIES_PER_CROSS, scaled when budget_s > auto
# Per window visit: ranked candidates with regen after each accept;
# the first accepting window ends the round. Stop after an idle round.
# Never wall-clock.





def _window_budget(windows, k=DEFAULT_K):
    """Try/second budget from window mass (main formula); seconds unused at runtime."""
    wins = list(windows or [])[:k]
    mass_c = sum(int(w.get('crossings') or 0) for w in wins)
    mass_e = sum(int(w.get('edges') or 0) for w in wins)
    seconds = min(BUDGET_CAP, max(BUDGET_FLOOR, BUDGET_CROSS * mass_c + BUDGET_EDGE * mass_e))
    tries = max(8, int(TRIES_PER_CROSS * max(mass_c, 1)))
    return seconds, tries, dict(window_crossings=mass_c, window_edges=mass_e)


def _touches_pins(data, moved_nids, pinned_keys):
    """True when a move relocates a previous-map pin (reaction mid or metabolite)."""
    if not pinned_keys or not moved_nids:
        return False
    body = data[1] if isinstance(data, list) else data
    nodes = body.get('nodes') or {}
    pinned_r = {k[2:] for k in pinned_keys if k.startswith('r:')}
    pinned_m = {k[2:] for k in pinned_keys if k.startswith('m:')}
    if not pinned_r and not pinned_m:
        return False
    if pinned_r:
        for rx in (body.get('reactions') or {}).values():
            rid = rx.get('bigg_id')
            if rid not in pinned_r:
                continue
            for seg in (rx.get('segments') or {}).values():
                for end in ('from_node_id', 'to_node_id'):
                    nid = seg.get(end)
                    if nid in moved_nids and (nodes.get(nid) or {}).get('node_type') == 'midmarker':
                        return True
    for nid in moved_nids:
        n = nodes.get(nid) or {}
        if n.get('node_type') == 'metabolite' and n.get('bigg_id') in pinned_m:
            return True
    return False


def _is_currency(bigg):
    return chem(bigg) in CURRENCY


def _skip_set(data, skip_edges):
    meta = data[0].get('metadata') or {}
    skip = {tuple(e) for e in (skip_edges if skip_edges is not None else meta.get('preserved_edges') or [])}
    skip |= {(p['reaction'], p['metabolite']) for p in (meta.get('source_paths') or [])}
    return skip


def _score(data, skip_edges, structures=None):
    """Full evaluator snapshot (start/end report only — not the per-move path)."""
    summary, issues = lq.evaluate(data, skip_edges, structures=structures)
    cross = summary['core']['crossings']
    coll = summary['core'].get('collisions') or summary['core'].get('overlaps') or {}
    cont = summary['core']['structure_continuity'] or {}
    alg = int(cross.get('counted') or (cross['algorithm_algorithm'] + cross.get('algorithm_curated', cross.get('algorithm_expert', 0))))
    windows = list(summary.get('worst_regions') or [])
    return dict(
        summary=summary,
        issues=issues,
        crossings=alg,
        top_window_crossings=[int(w.get('crossings') or 0) for w in windows[:DEFAULT_K]],
        worst_window=int((summary.get('details') or {}).get('crossing_density', {}).get('worst_window_crossings') or 0),
        node_overlaps=int(coll.get('node_overlap_pairs') or 0),
        label_overlaps=int(coll.get('label_overlap_pairs') or 0),
        continuity_share=float(cont.get('contiguous_pair_share') or 0.0),
        split_structures=int(cont.get('split_structures') or 0),
        length_p90=float((summary['core'].get('length_p90') or {}).get('px') or summary['core'].get('long_lines', {}).get('p90_px') or 0),
        spacing=float((summary['core'].get('length_p90') or {}).get('reaction_spacing_px')
                      or summary['core'].get('long_lines', {}).get('reaction_spacing_px') or 60.0),
        windows=windows,
        currency_involved=int(cross.get('currency_involved') or 0),
        main_metabolite=int(cross.get('main_metabolite') or 0),
    )


def _window_mass(windows, k=None):
    """Crossing/edge mass of the selected hotspot windows (reporting only)."""
    wins = list(windows or [])
    if k is not None:
        wins = wins[:k]
    mass_c = sum(int(w.get('crossings') or 0) for w in wins)
    mass_e = sum(int(w.get('edges') or 0) for w in wins)
    return dict(window_crossings=mass_c, window_edges=mass_e)


def _spacing_from_nodes(nodes):
    from scipy.spatial import cKDTree
    mids = np.array([[n['x'], n['y']] for n in nodes.values() if n.get('node_type') == 'midmarker'], float)
    if len(mids) > 2:
        return float(np.median(cKDTree(mids).query(mids, k=2)[0][:, 1])) or 60.0
    return 60.0


def _expert_reaction_ids(blocks, data):
    ids = {rid for b in (blocks or []) for rid in (b.get('reaction_ids') or [])}
    for s in (data[0].get('metadata') or {}).get('expert_structures') or []:
        ids.update(s.get('reactions') or s.get('reaction_ids') or [])
    return ids


def _scene_ids(scene):
    if not scene or not scene.get('positions'):
        return {}
    return {k: str(i) for i, k in enumerate(sorted(scene['positions']))}


def _reaction_index(body):
    out = {}
    nodes = body['nodes']
    for rx in body['reactions'].values():
        mid = next((s['from_node_id'] for s in rx['segments'].values()
                    if nodes.get(s['from_node_id'], {}).get('node_type') == 'midmarker'), None)
        if mid is not None:
            out[rx['bigg_id']] = (rx, mid)
    return out


def _holders(body):
    h = defaultdict(set)
    for rx in body['reactions'].values():
        for s in rx['segments'].values():
            for nid in (s['from_node_id'], s['to_node_id']):
                if body['nodes'].get(nid, {}).get('node_type') == 'metabolite':
                    h[nid].add(rx['bigg_id'])
    return h


def _in_window(x, y, win, pad=0.0):
    return (win['x'] - pad <= x < win['x'] + win['w'] + pad and
            win['y'] - pad <= y < win['y'] + win['h'] + pad)


def _move_node(n, dx, dy):
    from map_edit import translate_node
    translate_node(n, dx, dy, digits=2)


def _set_xy(n, x, y):
    from map_edit import set_node_xy
    set_node_xy(n, x, y, digits=2)


def _reflect(px, py, cx, cy, axis):
    v = np.array([px - cx, py - cy], float)
    a = np.asarray(axis, float)
    nrm = float(np.hypot(*a))
    if nrm < 1e-9:
        return 2 * cx - px, 2 * cy - py
    a = a / nrm
    parallel = (v @ a) * a
    out = np.array([cx, cy]) + parallel - (v - parallel)
    return float(out[0]), float(out[1])


def _coeff_map(rx):
    out = {}
    for m in rx.get('metabolites') or []:
        out.setdefault(m['bigg_id'], m['coefficient'])
    return out


def _reaction_axis(rx, mid, nodes, coeff):
    subs, prods = [], []
    for s in rx['segments'].values():
        tid = s['to_node_id'] if s['from_node_id'] == mid else None
        if tid is None or nodes.get(tid, {}).get('node_type') != 'metabolite':
            continue
        t = nodes[tid]
        c = coeff.get(t.get('bigg_id'), 0)
        (subs if c < 0 else prods).append(np.array([t['x'], t['y']], float))
    P = np.array([nodes[mid]['x'], nodes[mid]['y']], float)
    if subs and prods:
        u = np.mean(prods, axis=0) - np.mean(subs, axis=0)
    elif prods:
        u = np.mean(prods, axis=0) - P
    elif subs:
        u = P - np.mean(subs, axis=0)
    else:
        u = np.array([1.0, 0.0])
    n = float(np.hypot(*u))
    return u / n if n > 1e-9 else np.array([1.0, 0.0])


def _is_algo_seg(key):
    s = str(key)
    return s.startswith('es_s') or s.startswith('route') or s.startswith('hr') or s.startswith('qg')


def _cheap_restyle_edge(rx, mid, met_nid, nodes):
    """Set simple Bezier controls on the mid→metabolite segment without full edge_style.apply."""
    P = np.array([nodes[mid]['x'], nodes[mid]['y']], float)
    Q = np.array([nodes[met_nid]['x'], nodes[met_nid]['y']], float)
    c1 = P + (Q - P) / 3.0
    c2 = P + 2.0 * (Q - P) / 3.0
    for key, s in list(rx['segments'].items()):
        if s['from_node_id'] == mid and s['to_node_id'] == met_nid:
            s['b1'] = dict(x=round(float(c1[0]), 2), y=round(float(c1[1]), 2))
            s['b2'] = dict(x=round(float(c2[0]), 2), y=round(float(c2[1]), 2))
            return True
        # collapse one-hop multimarker corridor head
        if s['from_node_id'] == mid and _is_algo_seg(key):
            cur = s['to_node_id']
            while nodes.get(cur, {}).get('node_type') == 'multimarker':
                nxt = [ss['to_node_id'] for ss in rx['segments'].values() if ss['from_node_id'] == cur]
                if not nxt:
                    break
                cur = nxt[0]
            if cur == met_nid:
                # replace chain with one segment
                drop = [k for k, ss in rx['segments'].items()
                        if ss['from_node_id'] != mid and (ss['from_node_id'] in {s['to_node_id']} or ss['to_node_id'] == met_nid)]
                # simpler: just rewrite this head if direct
                if nodes.get(s['to_node_id'], {}).get('node_type') == 'metabolite':
                    s['b1'] = dict(x=round(float(c1[0]), 2), y=round(float(c1[1]), 2))
                    s['b2'] = dict(x=round(float(c2[0]), 2), y=round(float(c2[1]), 2))
                    return True
    # create / replace
    rx['segments'][f'hr_{mid}_{met_nid}'] = dict(
        from_node_id=mid, to_node_id=met_nid,
        b1=dict(x=round(float(c1[0]), 2), y=round(float(c1[1]), 2)),
        b2=dict(x=round(float(c2[0]), 2), y=round(float(c2[1]), 2)))
    return True


def _local_node_overlap_ok(data, moved_nids, min_dist=10.0):
    """Reject if a moved node lands within min_dist of a foreign glyph."""
    nodes = data[1]['nodes']
    moved = []
    others = []
    for nid, n in nodes.items():
        if n.get('node_type') not in ('metabolite', 'midmarker'):
            continue
        (moved if nid in moved_nids else others).append((nid, n['x'], n['y']))
    if not moved or not others:
        return True
    O = np.array([[x, y] for _, x, y in others], float)
    for _, x, y in moved:
        d = np.hypot(O[:, 0] - x, O[:, 1] - y)
        if (d < min_dist).any():
            return False
    return True


def _snapshot_nodes(data, nids):
    nodes = data[1]['nodes']
    snap = {}
    for nid in nids:
        n = nodes.get(nid)
        if n is None:
            continue
        snap[nid] = {k: n[k] for k in ('x', 'y', 'label_x', 'label_y') if k in n}
    return snap


def _restore_nodes(data, snap):
    nodes = data[1]['nodes']
    for nid, vals in snap.items():
        if nid in nodes:
            nodes[nid].update(vals)


def _snapshot_reaction_segments(data, rids):
    body = data[1]
    snap = {}
    for rx in body['reactions'].values():
        if rx['bigg_id'] not in rids:
            continue
        snap[rx['bigg_id']] = copy.deepcopy(rx['segments'])
        if 'label_x' in rx:
            snap[rx['bigg_id'] + '::label'] = (rx['label_x'], rx['label_y'])
    return snap


def _restore_reaction_segments(data, snap):
    body = data[1]
    by = {rx['bigg_id']: rx for rx in body['reactions'].values()}
    for key, val in snap.items():
        if key.endswith('::label'):
            rid = key[:-7]
            if rid in by:
                by[rid]['label_x'], by[rid]['label_y'] = val
            continue
        if key in by:
            by[key]['segments'] = copy.deepcopy(val)


def _is_rigid_metabolite(nid, n, holders, skip, inv):
    sk = inv.get(nid, '')
    if sk.startswith('copy:'):
        return True
    bigg = n.get('bigg_id')
    owns = holders.get(nid) or set()
    if owns and bigg and all((rid, bigg) in skip for rid in owns):
        return True
    if owns and bigg and not _is_currency(bigg) and any((rid, bigg) in skip for rid in owns):
        return True
    return False


def _candidate_moves(data, scene, blocks, win, skip, expert, spacing, cross_rids, cross_nids):
    """Cheap moves pruned to reactions/nodes that participate in window crossings."""
    body = data[1]
    nodes = body['nodes']
    rx_index = _reaction_index(body)
    holders = _holders(body)
    ids = _scene_ids(scene)
    inv = {v: k for k, v in ids.items()}
    moves = []
    step = 0.35 * spacing
    focus_rids = set(cross_rids) or set(win.get('reactions') or [])
    focus_nids = set(cross_nids)
    # Ensure currency side glyphs of focus reactions are considered even when
    # the crossing event list did not name their node ids.
    for rid in focus_rids:
        if rid not in rx_index:
            continue
        rx, mid = rx_index[rid]
        focus_nids.add(mid)
        for s in rx['segments'].values():
            for nid in (s['from_node_id'], s['to_node_id']):
                n = nodes.get(nid)
                if n and n.get('node_type') == 'metabolite' and _is_currency(n.get('bigg_id', '')):
                    focus_nids.add(nid)

    # Currency side glyphs of crossing reactions: flip / quarter-turn.
    for nid in sorted(focus_nids):
        n = nodes.get(nid)
        if not n or n.get('node_type') != 'metabolite' or not _is_currency(n.get('bigg_id', '')):
            continue
        if len(holders.get(nid, ())) != 1:
            continue
        rid = next(iter(holders[nid]))
        if rid not in focus_rids or rid not in rx_index or (rid, n['bigg_id']) in skip:
            continue
        mid = rx_index[rid][1]
        P = np.array([nodes[mid]['x'], nodes[mid]['y']], float)
        Q = np.array([n['x'], n['y']], float)
        base = math.atan2(Q[1] - P[1], Q[0] - P[0])
        dist = float(np.hypot(*(Q - P))) or (0.55 * spacing)
        for k, ang_off in enumerate((math.pi, math.pi / 2)):
            ang = base + ang_off
            moves.append(dict(kind='side_rotate', nid=nid, rid=rid,
                              x=float(P[0] + dist * math.cos(ang)),
                              y=float(P[1] + dist * math.sin(ang)),
                              order=(0, rid, nid, k)))

    # Mirror free (non-curated) reactions that participate in crossings.
    for rid in sorted(focus_rids):
        if rid in expert or rid not in rx_index:
            continue
        rx, mid = rx_index[rid]
        axis = _reaction_axis(rx, mid, nodes, _coeff_map(rx))
        moves.append(dict(kind='mirror_reaction', rid=rid, mode='axis', axis=axis.tolist(), order=(1, rid, 'axis')))
        moves.append(dict(kind='mirror_reaction', rid=rid, mode='point', axis=axis.tolist(), order=(1, rid, 'point')))

    # Cheap Bezier re-route (control refresh) for crossing reactions.
    for rid in sorted(focus_rids):
        if rid in rx_index:
            moves.append(dict(kind='reroute', rid=rid, order=(2, rid)))

    # Small free-node nudges for crossing participants that are free.
    # Stable sort; at most the first 12 free participants get 4-direction probes.
    free = []
    for nid in sorted(focus_nids):
        n = nodes.get(nid)
        if not n or n.get('node_type') not in ('metabolite', 'midmarker'):
            continue
        if n.get('node_type') == 'midmarker':
            rid = next((r['bigg_id'] for r in body['reactions'].values()
                        if any(s['from_node_id'] == nid for s in r['segments'].values())), None)
            if rid in expert:
                continue
        else:
            if _is_rigid_metabolite(nid, n, holders, skip, inv):
                continue
            if _is_currency(n.get('bigg_id', '')) and len(holders.get(nid, ())) == 1:
                continue
        free.append(nid)
    for nid in free[:12]:
        n = nodes[nid]
        for i, (dx, dy) in enumerate(((step, 0), (-step, 0), (0, step), (0, -step))):
            moves.append(dict(kind='reposition', nid=nid, x=float(n['x'] + dx), y=float(n['y'] + dy),
                              order=(5, nid, i)))

    if blocks and scene:
        for b in sorted(blocks, key=lambda x: x.get('id') or ''):
            # Absolute same-map rigid groups must not translate — that raises
            # curated×curated crossings while the headline gate ignores them.
            if b.get('same_map_rigid'):
                continue
            br = set(b.get('reaction_ids') or [])
            if not (br & focus_rids):
                continue
            for i, (dx, dy) in enumerate(((step, 0), (-step, 0), (0, step), (0, -step))):
                moves.append(dict(kind='block_translate', block_id=b.get('id'),
                                  delta=(float(dx), float(dy)), order=(6, b.get('id'), i)))

    moves.sort(key=lambda m: m['order'])
    return moves


def _apply_move_inplace(data, scene, blocks, move, skip, expert):
    """Apply move in place; return (undo_fn, changed_rids, moved_nids) or (None, reason, None)."""
    kind = move['kind']
    body = data[1]
    nodes = body['nodes']
    rx_index = _reaction_index(body)
    changed, moved = set(), set()

    if kind == 'block_translate':
        block = next((b for b in (blocks or []) if b.get('id') == move['block_id']), None)
        if block is None or scene is None:
            return None, 'no_block', None
        import quality_gate as qg
        keys = [k for k in (block.get('positions') or {}) if k in (scene.get('positions') or {})]
        rids = set(block.get('reaction_ids') or [])
        for k in list(scene.get('positions') or {}):
            if k.startswith('side:'):
                parts = k.split(':', 2)
                if len(parts) >= 2 and parts[1] in rids:
                    keys.append(k)
        keys = sorted(set(keys))
        if not keys:
            return None, 'empty_block', None
        # Snapshot then translate via quality_gate helper on a deepcopy merge-back.
        before = copy.deepcopy(data)
        d2, redo, _ = qg.translate_module(data, scene, keys, move['delta'], rids, skip)
        data[0], data[1] = d2[0], d2[1]
        try:
            edge_style.apply(data, list(skip), only=set(redo))
        except Exception:
            data[0], data[1] = before[0], before[1]
            return None, 'restyle_failed', None

        def undo():
            data[0], data[1] = before[0], before[1]
        return undo, set(redo), set()

    # Snapshots for undo
    touch_nids = set()
    touch_rids = set()
    if kind == 'side_rotate':
        touch_nids.add(move['nid']); touch_rids.add(move['rid'])
    elif kind == 'mirror_reaction':
        touch_rids.add(move['rid'])
        if move['rid'] in rx_index:
            rx, mid = rx_index[move['rid']]
            touch_nids.add(mid)
            for s in rx['segments'].values():
                for nid in (s['from_node_id'], s['to_node_id']):
                    if nodes.get(nid, {}).get('node_type') in ('metabolite', 'multimarker'):
                        touch_nids.add(nid)
    elif kind == 'reposition':
        touch_nids.add(move['nid'])
        touch_rids.update(_holders(body).get(move['nid'], ()))
        if nodes.get(move['nid'], {}).get('node_type') == 'midmarker':
            for rx in body['reactions'].values():
                if any(s['from_node_id'] == move['nid'] for s in rx['segments'].values()):
                    touch_rids.add(rx['bigg_id'])
    elif kind == 'reroute':
        touch_rids.add(move['rid'])
    else:
        return None, 'unknown', None

    node_snap = _snapshot_nodes(data, touch_nids)
    seg_snap = _snapshot_reaction_segments(data, touch_rids)

    if kind == 'side_rotate':
        n = nodes.get(move['nid'])
        if n is None:
            return None, 'missing', None
        _set_xy(n, move['x'], move['y'])
        if move['rid'] in rx_index:
            rx, mid = rx_index[move['rid']]
            _cheap_restyle_edge(rx, mid, move['nid'], nodes)
        moved.add(move['nid']); changed.add(move['rid'])
    elif kind == 'mirror_reaction':
        rid = move['rid']
        if rid not in rx_index or rid in expert:
            return None, 'not_free', None
        rx, mid = rx_index[rid]
        cx, cy = nodes[mid]['x'], nodes[mid]['y']
        axis = move.get('axis') or [1.0, 0.0]
        seen = {mid}
        for s in rx['segments'].values():
            for nid in (s['from_node_id'], s['to_node_id']):
                if nid in seen:
                    continue
                n = nodes.get(nid)
                if not n or n.get('node_type') not in ('metabolite', 'multimarker'):
                    continue
                met = n.get('bigg_id')
                if n.get('node_type') == 'metabolite' and met and (rid, met) in skip:
                    continue
                seen.add(nid); moved.add(nid)
                if move['mode'] == 'point':
                    _set_xy(n, 2 * cx - n['x'], 2 * cy - n['y'])
                else:
                    x, y = _reflect(n['x'], n['y'], cx, cy, axis)
                    _set_xy(n, x, y)
        if 'label_x' in rx:
            if move['mode'] == 'point':
                rx['label_x'] = round(2 * cx - rx['label_x'], 2)
                rx['label_y'] = round(2 * cy - rx['label_y'], 2)
            else:
                lx, ly = _reflect(rx['label_x'], rx['label_y'], cx, cy, axis)
                rx['label_x'], rx['label_y'] = round(lx, 2), round(ly, 2)
        # Refresh algo controls cheaply
        for s in list(rx['segments'].values()):
            if s['from_node_id'] == mid and nodes.get(s['to_node_id'], {}).get('node_type') == 'metabolite':
                met = nodes[s['to_node_id']].get('bigg_id')
                if (rid, met) not in skip:
                    _cheap_restyle_edge(rx, mid, s['to_node_id'], nodes)
        changed.add(rid)
    elif kind == 'reposition':
        n = nodes.get(move['nid'])
        if n is None:
            return None, 'missing', None
        ox, oy = n['x'], n['y']
        _set_xy(n, move['x'], move['y'])
        dx, dy = move['x'] - ox, move['y'] - oy
        moved.add(move['nid'])
        if n.get('node_type') == 'midmarker':
            for rx in body['reactions'].values():
                if any(s['from_node_id'] == move['nid'] for s in rx['segments'].values()):
                    if 'label_x' in rx:
                        rx['label_x'] = round(float(rx['label_x'] + dx), 2)
                        rx['label_y'] = round(float(rx['label_y'] + dy), 2)
                    changed.add(rx['bigg_id'])
                    for s in rx['segments'].values():
                        if s['from_node_id'] == move['nid'] and nodes.get(s['to_node_id'], {}).get('node_type') == 'metabolite':
                            if (rx['bigg_id'], nodes[s['to_node_id']].get('bigg_id')) not in skip:
                                _cheap_restyle_edge(rx, move['nid'], s['to_node_id'], nodes)
        for rid in _holders(body).get(move['nid'], ()):
            changed.add(rid)
            if rid in rx_index:
                rx, mid = rx_index[rid]
                _cheap_restyle_edge(rx, mid, move['nid'], nodes)
    elif kind == 'reroute':
        rid = move['rid']
        if rid not in rx_index:
            return None, 'missing', None
        rx, mid = rx_index[rid]
        for s in list(rx['segments'].values()):
            if s['from_node_id'] != mid:
                continue
            to = s['to_node_id']
            if nodes.get(to, {}).get('node_type') != 'metabolite':
                continue
            met = nodes[to].get('bigg_id')
            if (rid, met) in skip:
                continue
            # Flip side of a bowed control relative to the chord
            P = np.array([nodes[mid]['x'], nodes[mid]['y']], float)
            Q = np.array([nodes[to]['x'], nodes[to]['y']], float)
            chord = Q - P
            L = float(np.hypot(*chord)) or 1.0
            d = chord / L
            perp = np.array([-d[1], d[0]])
            bow = 0.18 * L
            # Prefer the opposite side of the current midpoint of controls if present
            side = 1.0
            if s.get('b1') and s.get('b2'):
                mid_c = np.array([(s['b1']['x'] + s['b2']['x']) / 2, (s['b1']['y'] + s['b2']['y']) / 2])
                side = -1.0 if float(np.dot(mid_c - P, perp)) >= 0 else 1.0
            c1 = P + d * (L / 3) + perp * bow * side
            c2 = P + d * (2 * L / 3) + perp * bow * side
            s['b1'] = dict(x=round(float(c1[0]), 2), y=round(float(c1[1]), 2))
            s['b2'] = dict(x=round(float(c2[0]), 2), y=round(float(c2[1]), 2))
        changed.add(rid)

    def undo():
        _restore_nodes(data, node_snap)
        _restore_reaction_segments(data, seg_snap)

    return undo, changed, moved


def _metrics_from_index(index, windows, node_ov, lab_ov, continuity, length_p90, spacing):
    cur_n = sum(1 for _i, _j, _x, _y, c in index.events if c)
    main_n = len(index.events) - cur_n
    return dict(
        crossings=index.headline(),
        top_window_crossings=[int(w.get('crossings') or 0) for w in windows[:DEFAULT_K]],
        worst_window=int(windows[0]['crossings']) if windows else 0,
        node_overlaps=node_ov,
        label_overlaps=lab_ov,
        continuity_share=float(continuity.get('contiguous_pair_share') or 0.0),
        split_structures=int(continuity.get('split_structures') or 0),
        length_p90=length_p90,
        spacing=spacing,
        currency_involved=cur_n,
        main_metabolite=main_n,
        windows=windows,
    )


def diagnose_windows(data, skip_edges=None, k=DEFAULT_K):
    """Explain what kinds of crossings dominate the top-k windows (for reporting)."""
    idx = lq.CrossingIndex(data, skip_edges)
    windows = _windows_from_index(idx, k)
    cur_n = sum(1 for _i, _j, _x, _y, c in idx.events if c)
    rows = []
    for win in windows[:k]:
        br = idx.window_breakdown(win)
        rows.append(dict(window=[win['x'], win['y']], crossings=win['crossings'], **br,
                         reactions=win.get('reactions')))
    return dict(headline=idx.headline(), currency_involved=cur_n,
                main_metabolite=len(idx.events) - cur_n, windows=rows)


def repair(data, model=None, scene=None, blocks=None, skip_edges=(), k=DEFAULT_K,
           max_rounds=None, max_accepts=None, budget_evals=None, budget_s=None, log=None,
           structures=None, pinned_keys=None):
    """Greedy local repair of crossing hotspots. Mutates ``data`` when accepting.

    ``k`` is the max number of separated positive-crossing windows (same count as
    ``layout_quality.evaluate`` worst_regions; default ``DEFAULT_K``). Pass
    ``k=None`` to keep every positive window after the 300 px separation filter.
    Each window visit walks ranked candidates and regenerates after an accept;
    the first accepting window ends the round. A full pass with no accept stops
    search. Defaults: ``max_accepts`` = sum of crossings in those windows;
    ``max_rounds`` = ``max_accepts + 1``. Try ceiling from window mass
    (``TRIES_PER_CROSS``), scaled when ``budget_s`` exceeds the mass-derived
    second estimate; ``budget_evals`` overrides that ceiling (tests). Never
    wall-clock. ``pinned_keys`` are previous-map ``r:`` / ``m:`` anchors that
    must not move.
    """
    t0 = time.time()
    skip = _skip_set(data, skip_edges)
    expert = _expert_reaction_ids(blocks, data)
    structures = structures or (data[0].get('metadata') or {}).get('expert_structures')
    pinned_keys = set(pinned_keys or ())

    # Setup (not charged to the search budget): shared index + light metrics (no full evaluate).
    # Use the same skip set as evaluate (preserved + source_paths), not the raw caller list.
    index = lq.CrossingIndex(data, skip)
    node_ov0, lab_ov0 = lq.overlap_counts(data)
    through0 = lq.curves_through_nodes_count(data, skip)
    windows = _windows_from_index(index, k)
    spacing = _spacing_from_nodes(index.nodes)
    lengths = [float(np.sum(np.hypot(*np.diff(e['pts'], axis=0).T))) for e in index.edges if len(e['pts']) > 1]
    length_p90 = float(np.percentile(lengths, 90)) if lengths else 0.0
    from evaluation.continuity import structure_continuity as _structure_continuity
    cont = _structure_continuity(data, structures=structures)
    before = _metrics_from_index(index, windows, node_ov0, lab_ov0, cont, length_p90, spacing)
    mass = _window_mass(windows, k)
    # Caps from the windows we search: each accept removes ≥1 of that crossing mass.
    if max_accepts is None:
        max_accepts = max(1, int(mass['window_crossings'] or before['crossings'] or 1))
    else:
        max_accepts = int(max_accepts)
    if max_rounds is None:
        max_rounds = max_accepts + 1
    else:
        max_rounds = int(max_rounds)
    # Try ceiling from window mass. ``budget_s`` only scales tries when above the
    # mass-derived second estimate — never a wall-clock stop. Cap the scale so
    # legacy 35s caller budgets cannot multiply tries by tens.
    auto_budget, auto_tries, _mass2 = _window_budget(windows, k if k is not None else DEFAULT_K)
    if budget_evals is not None:
        max_tries = int(budget_evals)
    else:
        max_tries = auto_tries
        if budget_s is not None and auto_budget > 0 and float(budget_s) > auto_budget:
            scale = min(float(budget_s) / auto_budget, BUDGET_SCALE_CAP)
            max_tries = max(auto_tries, int(auto_tries * scale))
    setup_s = round(time.time() - t0, 2)
    report = dict(
        enabled=True, k=k, before={kk: before[kk] for kk in (
            'crossings', 'top_window_crossings', 'worst_window', 'node_overlaps', 'label_overlaps',
            'continuity_share', 'split_structures', 'length_p90', 'currency_involved', 'main_metabolite')},
        accepted=[], rejected=0, tries=0, rounds=0, reject_reasons={},
        budget_evals=max_tries, budget_s=budget_s, budget_mass=mass, setup_s=setup_s,
        diagnosis=[], max_rounds=max_rounds, max_accepts=max_accepts,
    )
    report['before']['curves_through_nodes'] = through0
    for win in windows:
        report['diagnosis'].append(index.window_breakdown(win) | dict(
            window=[win['x'], win['y']], listed_crossings=win.get('crossings'),
            reactions=win.get('reactions')))

    accepts = 0
    current_cross = index.headline()
    if log:
        log(f"  hotspot repair: start crossings={before['crossings']} worst_window={report['before']['worst_window']} "
            f"search_budget=idle stop / rounds≤{max_rounds} accepts≤{max_accepts} tries≤{max_tries} "
            f"setup={setup_s}s spacing={spacing:.1f}")

    t_search = time.time()
    for rnd in range(max_rounds):
        if accepts >= max_accepts or (max_tries is not None and report['tries'] >= max_tries):
            break
        report['rounds'] = rnd + 1
        if not windows or all(w.get('crossings', 0) <= 0 for w in windows):
            break
        progressed = False
        for win in windows:
            if accepts >= max_accepts or (max_tries is not None and report['tries'] >= max_tries):
                break
            cross_rids, cross_nids = index.participants_in_window(win)
            if not cross_rids:
                # Fall back to the window's listed reactions so we still try side flips.
                cross_rids = set(win.get('reactions') or [])
            moves = _candidate_moves(data, scene, blocks, win, skip, expert, spacing, cross_rids, cross_nids)
            mi = 0
            while mi < len(moves):
                if accepts >= max_accepts or (max_tries is not None and report['tries'] >= max_tries):
                    break
                move = moves[mi]
                mi += 1
                report['tries'] += 1
                result = _apply_move_inplace(data, scene, blocks, move, skip, expert)
                if result[0] is None:
                    report['rejected'] += 1
                    reason = result[1] if isinstance(result[1], str) else 'apply_failed'
                    report['reject_reasons'][reason] = report['reject_reasons'].get(reason, 0) + 1
                    continue
                undo, changed, moved = result
                if _touches_pins(data, moved, pinned_keys):
                    undo()
                    report['rejected'] += 1
                    report['reject_reasons']['pinned'] = report['reject_reasons'].get('pinned', 0) + 1
                    continue
                if moved and not _local_node_overlap_ok(data, moved):
                    undo()
                    report['rejected'] += 1
                    report['reject_reasons']['node_overlap'] = report['reject_reasons'].get('node_overlap', 0) + 1
                    continue
                new_total, delta, payload = index.delta_after_replace(data, changed)
                # Allow flat crossings when through-nodes may fall; reject only rises.
                if delta > 0:
                    undo()
                    report['rejected'] += 1
                    report['reject_reasons']['no_crossing_drop'] = report['reject_reasons'].get('no_crossing_drop', 0) + 1
                    continue
                # Overlap / through-node check for promising moves.
                n_ov, l_ov = lq.overlap_counts(data)
                if n_ov > node_ov0 or l_ov > lab_ov0:
                    undo()
                    report['rejected'] += 1
                    why = 'node_overlap' if n_ov > node_ov0 else 'label_overlap'
                    report['reject_reasons'][why] = report['reject_reasons'].get(why, 0) + 1
                    continue
                through1 = lq.curves_through_nodes_count(data, skip)
                if through1 > through0:
                    undo()
                    report['rejected'] += 1
                    report['reject_reasons']['curves_through_nodes'] = (
                        report['reject_reasons'].get('curves_through_nodes', 0) + 1
                    )
                    continue
                # Rebuild after every accept so incremental delta drift cannot
                # claim phantom gains (delta_after_replace under-counts some Bezier edits).
                index.rebuild(data)
                real_total = index.headline()
                real_delta = real_total - current_cross
                through_drop = through1 < through0
                crossing_drop = real_delta < 0
                if real_delta > 0 or not (crossing_drop or through_drop):
                    undo()
                    index.rebuild(data)
                    report['rejected'] += 1
                    why = 'no_real_drop' if real_delta > 0 else 'no_through_or_crossing_drop'
                    report['reject_reasons'][why] = report['reject_reasons'].get(why, 0) + 1
                    continue
                current_cross = real_total
                node_ov0, lab_ov0 = n_ov, l_ov
                through0 = through1
                row = dict(
                    kind=move['kind'], order=list(move.get('order') or ()),
                    window=[win['x'], win['y']],
                    before_crossings=current_cross - real_delta, after_crossings=current_cross,
                    delta=real_delta,
                    curves_through_nodes=through0,
                )
                if move.get('rid'):
                    row['reaction'] = move['rid']
                report['accepted'].append(row)
                accepts += 1
                progressed = True
                if log:
                    log(f"  hotspot repair: accepted {move['kind']} Δ={real_delta} "
                        f"crossings→{current_cross} through→{through0}")
                cross_rids, cross_nids = index.participants_in_window(win)
                if not cross_rids:
                    cross_rids = set(win.get('reactions') or [])
                moves = _candidate_moves(data, scene, blocks, win, skip, expert, spacing, cross_rids, cross_nids)
                mi = 0
            if progressed:
                break
        if not progressed:
            break
        windows = _windows_from_index(index, k)

    search_s = round(time.time() - t_search, 2)
    # Light after-metrics from the live index (no full evaluate).
    windows = _windows_from_index(index, k)
    lengths = [float(np.sum(np.hypot(*np.diff(e['pts'], axis=0).T))) for e in index.edges if len(e['pts']) > 1]
    length_p90 = float(np.percentile(lengths, 90)) if lengths else length_p90
    node_ov1, lab_ov1 = lq.overlap_counts(data)
    after = _metrics_from_index(index, windows, node_ov1, lab_ov1, cont, length_p90, spacing)
    report['after'] = {kk: after[kk] for kk in (
        'crossings', 'top_window_crossings', 'worst_window', 'node_overlaps', 'label_overlaps',
        'continuity_share', 'split_structures', 'length_p90', 'currency_involved', 'main_metabolite')}
    report['after']['curves_through_nodes'] = lq.curves_through_nodes_count(data, skip)
    report['accepted_count'] = len(report['accepted'])
    report['search_s'] = search_s
    report['seconds'] = round(time.time() - t0, 2)
    report['improved'] = report['after']['crossings'] < report['before']['crossings']
    report['index_headline'] = current_cross
    if log:
        log(f"  hotspot repair: done accepted={report['accepted_count']} "
            f"crossings {report['before']['crossings']}->{report['after']['crossings']} "
            f"through_nodes {report['before']['curves_through_nodes']}->"
            f"{report['after']['curves_through_nodes']} "
            f"(search {search_s}s / total {report['seconds']}s)")
    return report



def _windows_from_index(index, k=None):
    """Worst crossing windows from the index (same 400/200 boxes as evaluate).

    When ``k`` is None, keep every positive-crossing cell that survives the
    300 px separation filter (problem-sized window list).
    """
    if not index.edges:
        return []
    centres = np.array([[n['x'], n['y']] for n in index.nodes.values()
                        if n.get('node_type') in ('metabolite', 'midmarker')], float)
    if not len(centres):
        return []
    x0, y0 = float(centres[:, 0].min()), float(centres[:, 1].min())
    cells = defaultdict(lambda: dict(crossings=0, reactions=set(), edges=0))
    for i, j, x, y, _c in index.events:
        for key in lq._region_window_keys(x, y, x0, y0):
            c = cells[key]
            c['crossings'] += 1
            c['reactions'].update((index.edges[i]['rid'], index.edges[j]['rid']))
    for e in index.edges:
        mx, my = float(np.mean(e['pts'][:, 0])), float(np.mean(e['pts'][:, 1]))
        for key in lq._region_window_keys(mx, my, x0, y0):
            cells[key]['edges'] += 1
    ranked = sorted(cells.items(), key=lambda kv: (-kv[1]['crossings'], -kv[1]['edges'], kv[0]))
    out, taken = [], []
    for (kx, ky), c in ranked:
        if c['crossings'] <= 0:
            continue
        wx, wy = x0 + kx * lq.REGION_STRIDE, y0 + ky * lq.REGION_STRIDE
        cx, cy = wx + lq.REGION_WINDOW / 2, wy + lq.REGION_WINDOW / 2
        if any(abs(cx - tx) < 300 and abs(cy - ty) < 300 for tx, ty in taken):
            continue
        taken.append((cx, cy))
        out.append(dict(x=round(wx, 1), y=round(wy, 1), w=int(lq.REGION_WINDOW), h=int(lq.REGION_WINDOW),
                        crossings=c['crossings'], edges=c['edges'],
                        reactions=sorted(c['reactions'])[:10]))
        if k is not None and len(out) == k:
            break
    return out
