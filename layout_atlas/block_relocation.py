"""Block-level relocation driven by long cross-block connections (V7.3 P0 / S3b).

`composition` moves visual components, but an expert block that shares a glyph with the main network (EX_fum_e + FUMt2_2 sharing fum_c / mal__L_c with FUM) is
part of the main component and never moves, so its connections to the rest of the network stay thousands of px long (FUMt2_2 -> FUM 457 px = 5 reaction spacings,
TALA -> PGI 1,514 px in e_coli_core).  A Bezier router cannot shorten a connection whose ends are far apart.

What is decided here, from the scene only (no names):
  unit        an expert block (rigid; its glyphs, its reactions and the currency glyphs only its reactions use move together)
  trigger     a connection from a reaction of the block to a non-currency glyph OUTSIDE the block that is longer than TAU x the typical reaction spacing (not a fixed px)
  candidates  the whole block translated so that its connecting reactions sit CLEAR px from the glyphs they connect to, on the side where the block is, or at +-90 degrees,
              each at the nearest free place (nothing else moves; the inside of the block is untouched)
  decision    the same unified gate as the peripheral modules: the block is re-drawn on a COPY of the final map (`quality_gate.translate_module` + `edge_style.apply`),
              and kept only if `quality_gate.accept` says the final geometry is not worse (a long connection that gets short is the improvement).
The distance only proposes a move; the real Bezier curves, crossings, labels and collisions decide it.
"""
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

TAU = 4.0
CLEAR = 85.0
GAP = 120.0
MAX_BLOCK_REACTIONS = 25

# Acceptance of a relocation.  The strict gate (`quality_gate.accept`: no collision may rise, crossings at most +2) vetoed moves that shortened a connection a lot for +1..+3
# collisions.  'balanced' compares the finished maps with one weighted difference on the same final-geometry counts, with hard limits so nothing gets clearly worse:
#   dJ = W_CROSS * d(crossings) + W_COLL * d(collisions) + W_LONG * d(connections > 250 px)     accepted when dJ < -MIN_GAIN
#   hard: Bezier coverage, collisions at most +MAX_COLL_RISE, crossings at most +MAX_CROSS_RISE.
# Weights are a stated judgement of how bad each problem is (a long cross-module connection is the worst readability defect), not an automatic truth.
#
# Rationale for the parameter values (2026-11 adjustment):
#   W_LONG=5.0 (was 4.0): Shortening a long cross-module edge is the primary readability improvement. A connection spanning
#     1000+ px obscures the relationship between modules and is the worst visual defect. Increased weight to favor moves
#     that shorten these connections even at minor cost elsewhere.
#   MIN_GAIN=0.5 (was 1.0): The original threshold was too conservative - a move that shortened a 1500px edge to 300px
#     but added one crossing was often rejected. Now smaller net improvements are accepted.
#   MAX_COLL_RISE=5, MAX_CROSS_RISE=10 (were 3, 6): These are HARD limits, not weights. The original limits were too
#     strict for large models where local crowding is unavoidable. The limits are still restrictive (never accept a
#     change that adds many new problems) but allow beneficial trade-offs.
# These values are data-driven (derived from analyzing iIT341 and iMM904 rejection reasons) but model-independent: they
# express how much worse each problem type is, not what any specific model looks like.
import os as _os
ACCEPT = _os.environ.get('OPM_RELOC_ACCEPT', 'balanced')
W_CROSS, W_COLL, W_LONG = 1.0, 1.5, 5.0
MIN_GAIN = 0.5
MAX_COLL_RISE, MAX_CROSS_RISE = 5, 10


def _accept(old, new):
    import quality_gate as qg
    if ACCEPT != 'balanced':
        return qg.accept(old, new)
    if new['hard'] > old['hard']:
        return False, 'hard constraint broken'
    dc, do, dl = new['crossings'] - old['crossings'], new['collisions'] - old['collisions'], new['long_edges'] - old['long_edges']
    if do > MAX_COLL_RISE:
        return False, 'collisions +%d (limit %d)' % (do, MAX_COLL_RISE)
    if dc > MAX_CROSS_RISE:
        return False, 'crossings +%d (limit %d)' % (dc, MAX_CROSS_RISE)
    dj = W_CROSS * dc + W_COLL * do + W_LONG * dl + 0.0
    if dj < -MIN_GAIN:
        return True, 'weighted gain %.1f (crossings %+d, collisions %+d, long connections %+d)' % (-dj, dc, do, dl)
    return False, 'no weighted gain (%.1f; crossings %+d, collisions %+d, long connections %+d)' % (-dj, dc, do, dl)


def _spacing(pos):
    pts = np.array([v for k, v in pos.items() if k.startswith('r:')], float)
    return float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])) if len(pts) > 2 else 100.0


def find(model, scene, blocks, cur):
    """-> list of units: dict(block, reactions, keys, long=[(reaction, metabolite, px, outside_key)], spacing)."""
    pos, parts = scene['positions'], scene['participants']
    spacing = _spacing(pos)
    holders = defaultdict(set)
    for (rid, mid), key in parts.items():
        holders[key].add(rid)
    rx = {r['id']: r for r in model['reactions']}
    users = defaultdict(set)
    for r in model['reactions']:
        for mm in r['metabolites']:
            users[mm].add(r['id'])
    units = []
    for b in blocks:
        inside = set(b['reaction_ids'])
        if not inside or len(inside) > MAX_BLOCK_REACTIONS:
            continue
        keys = {'r:' + r for r in inside}
        keys.update(k for k in pos if k.startswith('copy:' + b['id'] + ':'))
        for r in inside:
            for mid in rx[r]['metabolites']:
                k = parts[r, mid]
                if holders[k] <= inside:
                    keys.add(k)
        keys = {k for k in keys if k in pos}
        long = []
        for r in inside:
            for mid in rx[r]['metabolites']:
                if mid in cur or 'r:' + r not in pos:
                    continue
                k = parts[r, mid]
                if k not in keys and k in pos:
                    # the glyph is shared with the outside: the connection itself is the long one
                    px = float(np.hypot(*(np.array(pos['r:' + r]) - np.array(pos[k]))))
                    if px > TAU * spacing:
                        long.append((r, mid, round(px, 1), k))
                    continue
                # the block draws its own copy of the metabolite: anchor displacement = distance from this reaction to the nearest glyph of the same metabolite that
                # an OUTSIDE reaction uses (identity of the metabolite, not of the glyph, says where the reaction belongs)
                best = None
                for u in users[mid]:
                    if u in inside or (u, mid) not in parts:
                        continue
                    ku = parts[u, mid]
                    if ku in pos and ku not in keys:
                        d = float(np.hypot(*(np.array(pos['r:' + r]) - np.array(pos[ku]))))
                        if best is None or d < best[0]:
                            best = (d, ku)
                if best is not None and best[0] > TAU * spacing:
                    long.append((r, mid, round(best[0], 1), best[1]))
        if long:
            units.append(dict(block=b['id'], reactions=sorted(inside), keys=keys, long=sorted(long, key=lambda t: -t[2]), spacing=spacing))
    return sorted(units, key=lambda u: -u['long'][0][2])


def candidates(scene, unit, max_candidates=3):
    """Translations (dx, dy) that bring the block next to the glyphs its long connections go to, at the nearest free place."""
    pos = scene['positions']
    keys = unit['keys']
    anchors = np.array([pos[k] for _, _, _, k in unit['long']], float)
    ref = np.array([pos['r:' + r] for r, _, _, _ in unit['long']], float).mean(axis=0)
    centre_a = anchors.mean(axis=0)
    away = ref - centre_a
    n = float(np.hypot(*away))
    away = away / n if n > 1e-6 else np.array([0.0, 1.0])
    local = np.array([pos[k] for k in keys], float)
    size = float(np.hypot(*(local.max(0) - local.min(0))))
    others = np.array([v for k, v in pos.items() if k not in keys and not k.startswith('side:')], float)
    tree = cKDTree(others) if len(others) else None
    out, seen = [], []
    for ang in (0.0, 90.0, -90.0, 180.0):
        a = np.radians(ang)
        d = np.array([away[0] * np.cos(a) - away[1] * np.sin(a), away[0] * np.sin(a) + away[1] * np.cos(a)])
        delta0 = (centre_a + d * (GAP + size / 2.0)) - ref
        for i in range(0, 120):
            r = 0.0 if i == 0 else 38.0 * np.sqrt(i)
            t = i * 2.399963
            delta = delta0 + r * np.array([np.cos(t), np.sin(t)])
            if tree is None or float(tree.query(local + delta)[0].min()) >= CLEAR:
                if not any(np.hypot(*(delta - s)) < 60 for s in seen):
                    seen.append(delta)
                    out.append(delta)
                break
        if len(out) >= max_candidates:
            break
    return out


def _frame_offset(data, scene, keys):
    """Constant offset between scene units and map coordinates (taken from one glyph of the module)."""
    import quality_gate as qg
    ids = qg._ids(scene)
    for k in sorted(keys):
        nid = ids.get(k)
        if nid is not None and nid in data[1]['nodes']:
            n = data[1]['nodes'][nid]
            return np.array([n['x'] - scene['positions'][k][0], n['y'] - scene['positions'][k][1]])
    return np.zeros(2)


def _diagnose(data, scene, unit, delta, old_issues, new_issues, radius=400.0):
    """Where do the NEW obstacle hits and crossings of a candidate appear: within `radius` px of the moved module's new centre, or elsewhere, and which reactions.
    (The issue lists are capped by the evaluator, so the counts are lower bounds.)"""
    pos = scene['positions']
    centre = np.mean([pos[k] for k in unit['keys']], axis=0) + np.asarray(delta, float) + _frame_offset(data, scene, unit['keys'])

    def sig(item):
        if 'kind' in item and 'reactions' in item:
            return ('x', item['kind'], tuple(item['reactions']))
        return ('hit', item.get('kind'), item.get('reaction'), item.get('metabolite'), item.get('label'))
    old = {sig(h) for h in old_issues.get('obstacle_hits', [])} | {sig(c) for c in old_issues.get('crossings', [])}
    near, far, names = 0, 0, set()
    for item in list(new_issues.get('obstacle_hits', [])) + list(new_issues.get('crossings', [])):
        s = sig(item)
        if s in old:
            continue
        d = float(np.hypot(item['x'] - centre[0], item['y'] - centre[1]))
        near, far = (near + 1, far) if d <= radius else (near, far + 1)
        names.update(s[2] if s[0] == 'x' else [s[2]])
    return dict(new_near_module=near, new_elsewhere=far, reactions=sorted(x for x in names if x)[:8])


def relocate(data, model, scene, blocks, cur, skip_edges, max_units=20, log=None):
    """Try the relocation candidates of every unit on copies of the final map; keep what the unified gate accepts.  -> (data, report).

    Stops on a deterministic unit budget, never on wall-clock.  Each unit tries at
    most two ranked candidates; the joint-pair pass keeps its fixed pair/candidate caps.
    """
    import edge_style
    import quality_gate as qg
    t0 = time.time()
    units = find(model, scene, blocks, cur)
    truncated = max(0, len(units) - max_units)
    report = dict(units=len(units), accepted=0, rejected=0, budget_exhausted=truncated,
                  spacing_px=round(units[0]['spacing'], 1) if units else None, rows=[])
    if not units:
        return data, report
    far = len(units)
    rejected_units = []
    base, summary, issues = qg.evaluate_map(data, skip_edges, disconnected=far)
    for unit in units[:max_units]:
        row = dict(block=unit['block'], reactions=unit['reactions'][:6], longest_px=unit['long'][0][2], longest_in_spacings=round(unit['long'][0][2] / unit['spacing'], 1),
                   connections=[(r, m, px) for r, m, px, _ in unit['long'][:4]], tries=[])
        won = False
        # Cheap geometric proxy ranks candidates; only the top few get a full edge_style + evaluate
        longs = [(r, k, px) for r, _m, px, k in unit['long']]
        ranked = sorted(((qg.proxy_score(scene, unit['keys'], d, longs), d) for d in candidates(scene, unit)),
                        key=lambda t: -t[0]['score'])
        for proxy, delta in ranked[:2]:
            if proxy['score'] < 0 and proxy['long_gain'] < unit['spacing']:
                row['tries'].append(dict(moved_px=round(float(np.hypot(*delta)), 1), decision='skipped', reason='proxy no gain', proxy=proxy))
                continue
            d2, redo, _ = qg.translate_module(data, scene, unit['keys'], delta, set(unit['reactions']), skip_edges)
            edge_style.apply(d2, skip_edges, only=redo)
            new, s2, i2 = qg.evaluate_map(d2, skip_edges, disconnected=far - 1)
            ok, why = _accept(base, new)
            row['tries'].append(dict(moved_px=round(float(np.hypot(*delta)), 1), decision='accepted' if ok else 'rejected', reason=why, proxy=proxy,
                                     after={k: new[k] for k in ('collisions', 'crossings', 'long_edges')}, diagnosis=_diagnose(data, scene, unit, delta, issues, i2)))
            if ok:
                data, base, summary, issues = d2, new, s2, i2
                far -= 1
                base['disconnected'] = far
                for k in unit['keys']:
                    scene['positions'][k] = [scene['positions'][k][0] + float(delta[0]), scene['positions'][k][1] + float(delta[1])]
                won = True
                break
        if not won and CLEARING:
            for delta in candidates(scene, unit)[:2]:
                pushes = _clearing_moves(model, scene, blocks, [(unit, delta)])
                if pushes is None:
                    row['tries'].append(dict(moved_px=round(float(np.hypot(*delta)), 1), decision='skipped', reason='clearing infeasible (a large unit is in the way or the push is too long)'))
                    continue
                d1, redo, _ = qg.translate_module(data, scene, unit['keys'], delta, set(unit['reactions']), skip_edges)
                for ks, dp, rs in pushes:
                    d1, r2, _ = qg.translate_module(d1, scene, ks, dp, rs, skip_edges)
                    redo = redo | r2
                edge_style.apply(d1, skip_edges, only=redo)
                new, s2, i2 = qg.evaluate_map(d1, skip_edges, disconnected=far - 1)
                ok, why = _accept(base, new)
                row['tries'].append(dict(moved_px=round(float(np.hypot(*delta)), 1), cleared_units=len(pushes), decision='accepted' if ok else 'rejected', reason=why,
                                         after={k: new[k] for k in ('collisions', 'crossings', 'long_edges')}, diagnosis=_diagnose(data, scene, unit, delta, issues, i2)))
                if ok:
                    data, base, summary, issues = d1, new, s2, i2
                    far -= 1
                    base['disconnected'] = far
                    for k in unit['keys']:
                        scene['positions'][k] = [scene['positions'][k][0] + float(delta[0]), scene['positions'][k][1] + float(delta[1])]
                    for ks, dp, rs in pushes:
                        for k in ks:
                            scene['positions'][k] = [scene['positions'][k][0] + float(dp[0]), scene['positions'][k][1] + float(dp[1])]
                    won = True
                    break
        row['decision'] = 'accepted' if won else 'rejected'
        row['before'] = None
        report['accepted' if won else 'rejected'] += 1
        report['rows'].append(row)
        if not won:
            rejected_units.append(unit)
        if log:
            log(f"  block relocation {unit['block'][:8]} ({unit['reactions'][:3]}): {row['decision']} (longest {row['longest_in_spacings']} spacings)")
    # coupled units: two rejected units that share a non-currency metabolite pull on each other (moving one changes the other's connections), so a move that is
    # not good alone can be good together.  Try joint candidates (a few positions for each, moved together) for the most displaced pairs.
    data, base, issues, far = _joint_pairs(data, model, scene, blocks, cur, skip_edges, rejected_units, base, issues, far, report, log)
    report['seconds'] = round(time.time() - t0, 1)
    return data, report


CLEARING = _os.environ.get('OPM_RELOC_CLEAR', 'off') == 'on'   # experimental, off: on e_coli_core every cleared candidate was worse (collisions +7..+24)
CLEAR_MARGIN = 120.0
MAX_CLEAR_PUSH = 500.0
MAX_NEIGHBOUR_REACTIONS = 25


def _clearing_moves(model, scene, blocks, moving, scene_units=None):
    """Local space-making.  `moving` = [(unit, delta)] (the modules about to be moved).  Every OTHER small unit (an expert block or algorithm group of at most
    MAX_NEIGHBOUR_REACTIONS reactions) whose rectangle meets the target rectangle of a moving module (+ CLEAR_MARGIN) is pushed straight out of it along the axis of
    least penetration (rigid translation, like every other move here).  A larger unit in the way, or a push longer than MAX_CLEAR_PUSH, makes the candidate infeasible
    (returns None): space is made locally, the main network is never shoved around.  -> [(keys, delta, reactions)] for the neighbours."""
    # Clearing is opt-in (OPM_RELOC_CLEAR); unit graph from production regions helper
    import regions as _regions
    unit_of, keys_of, reactions_of = scene_units or _regions.build_units(model, scene, blocks)
    pos = scene['positions']
    moving_reactions = {r for u, _ in moving for r in u['reactions']}
    moving_keys = set().union(*[u['keys'] for u, _ in moving])
    targets = []
    for u, d in moving:
        lo, hi = _box(scene, u['keys'], d)
        targets.append((lo - (CLEAR_MARGIN - 60.0), hi + (CLEAR_MARGIN - 60.0)))
    pushes = []
    for uid_, keys in keys_of.items():
        if not keys or (reactions_of[uid_] & moving_reactions):
            continue
        ks = [k for k in keys if k in pos and k not in moving_keys]
        if not ks:
            continue
        arr = np.array([pos[k] for k in ks], float)
        nlo, nhi = arr.min(0) - 40.0, arr.max(0) + 40.0
        total = np.zeros(2)
        for tlo, thi in targets:
            if not ((nlo < thi).all() and (tlo < nhi).all()):
                continue
            if len(reactions_of[uid_]) > MAX_NEIGHBOUR_REACTIONS:
                return None
            pen = np.minimum(thi - nlo, nhi - tlo)             # penetration along x and y
            axis = 0 if pen[0] < pen[1] else 1
            sign = 1.0 if (nlo[axis] + nhi[axis]) / 2 >= (tlo[axis] + thi[axis]) / 2 else -1.0
            step = np.zeros(2)
            step[axis] = sign * (pen[axis] + 40.0)
            total += step
        if float(np.hypot(*total)) > 0:
            if float(np.hypot(*total)) > MAX_CLEAR_PUSH:
                return None
            pushes.append((set(ks), total, set(reactions_of[uid_])))
    return pushes


JOINT_PAIRS = 4
JOINT_CANDIDATES = 3


def _box(scene, keys, delta):
    arr = np.array([scene['positions'][k] for k in keys], float) + np.asarray(delta, float)
    return arr.min(0) - 60.0, arr.max(0) + 60.0


def _joint_pairs(data, model, scene, blocks, cur, skip_edges, rejected, base, issues, far, report, log):
    import edge_style
    import quality_gate as qg
    rx = {r['id']: r for r in model['reactions']}
    mets = {id(u): {m for r in u['reactions'] for m in rx[r]['metabolites'] if m not in cur} for u in rejected}
    pairs = []
    for i in range(len(rejected)):
        for j in range(i + 1, len(rejected)):
            a, b = rejected[i], rejected[j]
            if mets[id(a)] & mets[id(b)] and not (a['keys'] & b['keys']):
                pairs.append((a['long'][0][2] + b['long'][0][2], a, b))
    pairs.sort(key=lambda x: -x[0])
    report['joint_pairs'] = []
    # Joint search is bounded by JOINT_PAIRS × JOINT_CANDIDATES² gate attempts.
    for _, a, b in pairs[:JOINT_PAIRS]:
        ca = candidates(scene, a)[:JOINT_CANDIDATES]
        cb = candidates(scene, b)[:JOINT_CANDIDATES]
        row = dict(units=[a['reactions'][:4], b['reactions'][:4]], shared=sorted(mets[id(a)] & mets[id(b)])[:6], tries=[])
        won = False
        for da in ca:
            for db in cb:
                (alo, ahi), (blo, bhi) = _box(scene, a['keys'], da), _box(scene, b['keys'], db)
                if (alo < bhi).all() and (blo < ahi).all():
                    continue                       # the two moved modules would sit on each other
                d1, redo1, _ = qg.translate_module(data, scene, a['keys'], da, set(a['reactions']), skip_edges)
                d2, redo2, _ = qg.translate_module(d1, scene, b['keys'], db, set(b['reactions']), skip_edges)
                edge_style.apply(d2, skip_edges, only=redo1 | redo2)
                new, s2, i2 = qg.evaluate_map(d2, skip_edges, disconnected=far - 2)
                ok, why = _accept(base, new)
                row['tries'].append(dict(moved_px=[round(float(np.hypot(*da)), 1), round(float(np.hypot(*db)), 1)], decision='accepted' if ok else 'rejected', reason=why,
                                         after={k: new[k] for k in ('collisions', 'crossings', 'long_edges')}))
                if ok:
                    data, base, issues = d2, new, i2
                    far -= 2
                    base['disconnected'] = far
                    for u, d in ((a, da), (b, db)):
                        for k in u['keys']:
                            scene['positions'][k] = [scene['positions'][k][0] + float(d[0]), scene['positions'][k][1] + float(d[1])]
                    won = True
                    break
            if won:
                break
        if not won and CLEARING:
            for da, db in [(x, y) for x in ca[:2] for y in cb[:2]]:
                (alo, ahi), (blo, bhi) = _box(scene, a['keys'], da), _box(scene, b['keys'], db)
                if (alo < bhi).all() and (blo < ahi).all():
                    continue
                pushes = _clearing_moves(model, scene, blocks, [(a, da), (b, db)])
                if pushes is None:
                    row['tries'].append(dict(moved_px=[round(float(np.hypot(*da)), 1), round(float(np.hypot(*db)), 1)], decision='skipped', reason='clearing infeasible'))
                    continue
                d1, redo1, _ = qg.translate_module(data, scene, a['keys'], da, set(a['reactions']), skip_edges)
                d1, redo2, _ = qg.translate_module(d1, scene, b['keys'], db, set(b['reactions']), skip_edges)
                redo = redo1 | redo2
                for ks, dp, rs in pushes:
                    d1, r3, _ = qg.translate_module(d1, scene, ks, dp, rs, skip_edges)
                    redo = redo | r3
                edge_style.apply(d1, skip_edges, only=redo)
                new, s2, i2 = qg.evaluate_map(d1, skip_edges, disconnected=far - 2)
                ok, why = _accept(base, new)
                row['tries'].append(dict(moved_px=[round(float(np.hypot(*da)), 1), round(float(np.hypot(*db)), 1)], cleared_units=len(pushes), decision='accepted' if ok else 'rejected',
                                         reason=why, after={k: new[k] for k in ('collisions', 'crossings', 'long_edges')}))
                if ok:
                    data, base, issues = d1, new, i2
                    far -= 2
                    base['disconnected'] = far
                    for u, d in ((a, da), (b, db)):
                        for k in u['keys']:
                            scene['positions'][k] = [scene['positions'][k][0] + float(d[0]), scene['positions'][k][1] + float(d[1])]
                    for ks, dp, rs in pushes:
                        for k in ks:
                            scene['positions'][k] = [scene['positions'][k][0] + float(dp[0]), scene['positions'][k][1] + float(dp[1])]
                    won = True
                    break
        row['decision'] = 'accepted' if won else 'rejected'
        report['joint_pairs'].append(row)
        if log:
            log(f"  joint relocation {row['units']}: {row['decision']}")
    return data, base, issues, far


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Constrained expert block relocation triggered by a long exchange + transport submodule
# (OPM_EXPERT_SUBMODULE=on|branch; default branch).
#
# Detection still uses peripheral.find_modules (exchange + simple/branch transport) inside a large
# curated block whose cross-link is abnormally long.  The move itself always translates the WHOLE
# curated block — never pulls EX_mal/MALt2_2 (or EX_fum/FUMt2_2) out of the source structure.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def find_sub(model, scene, blocks, cur, simple=True):
    import peripheral
    pos, parts = scene['positions'], scene['participants']
    spacing = _spacing(pos)
    holders = defaultdict(set)
    for (rid, mid), key in parts.items():
        holders[key].add(rid)
    rx = {r['id']: r for r in model['reactions']}
    block_of = {rid: b for b in blocks for rid in b['reaction_ids']}
    units = []
    for mod in peripheral.find_modules(model, cur, set(), (), simple=simple):
        members = [mod['exchange']] + [t for t, _ in mod['transports']]
        bs = {block_of[m]['id'] for m in members if m in block_of}
        if len(bs) != 1:
            continue                      # members that no structure placed (algorithm-drawn, e.g. an exchange) move with the module; the expert ones must share ONE block
        b = block_of[next(m for m in members if m in block_of)]
        if len(b['reaction_ids']) <= MAX_BLOCK_REACTIONS:
            continue                      # small blocks are handled as a whole by `relocate`
        inside = set(members)
        keys = {'r:' + r for r in inside}
        for r in inside:
            for mid in rx[r]['metabolites']:
                k = parts[r, mid]
                if holders[k] <= inside:
                    keys.add(k)
        keys = {k for k in keys if k in pos}
        long, cross = [], []
        for r in inside:
            for mid in rx[r]['metabolites']:
                k = parts[r, mid]
                if k in keys or k not in pos or 'r:' + r not in pos:
                    continue
                cross.append((r, mid))
                if mid in cur:
                    continue
                px = float(np.hypot(*(np.array(pos['r:' + r]) - np.array(pos[k]))))
                if px > TAU * spacing:
                    long.append((r, mid, round(px, 1), k))
        if long:
            units.append(dict(block=b['id'], reactions=sorted(inside), keys=keys, long=sorted(long, key=lambda t: -t[2]), cross=cross, spacing=spacing, block_size=len(b['reaction_ids'])))
    return sorted(units, key=lambda u: -u['long'][0][2])


def _detach_cross_edges(data, scene, unit):
    """On a COPY of the map: replace each expert connection of the module that crosses the module boundary by one plain segment (reaction point -> metabolite), so that
    edge_style re-fits it as a Bezier after the module has moved.  Returns (data copy, edges as [reaction, metabolite])."""
    import copy
    import quality_gate as qg
    d2 = copy.deepcopy(data)
    body = d2[1]
    nodes, rxs = body['nodes'], body['reactions']
    ids = qg._ids(scene)
    by_id = {r['bigg_id']: r for r in rxs.values()}
    detached, counter = [], 0
    for rid, mid in unit['cross']:
        rxn = by_id.get(rid)
        key = scene['participants'][rid, mid]
        target = ids.get(key)
        if rxn is None or target is None or target not in nodes:
            continue
        mid_node = next((s['from_node_id'] for s in rxn['segments'].values() if nodes[s['from_node_id']].get('node_type') == 'midmarker'), None)
        out = defaultdict(list)
        for sk, s in rxn['segments'].items():
            out[s['from_node_id']].append((sk, s))
        for sk, s in list(out.get(mid_node, [])):
            chain, cur_node, drop = [sk], s['to_node_id'], []
            while nodes[cur_node].get('node_type') == 'multimarker':
                drop.append(cur_node)
                nxt = out.get(cur_node)
                if not nxt:
                    break
                chain.append(nxt[0][0])
                cur_node = nxt[0][1]['to_node_id']
            if cur_node == target:
                for c in chain:
                    rxn['segments'].pop(c, None)
                for n in drop:
                    nodes.pop(n, None)
                counter += 1
                rxn['segments'][f'qx{counter}'] = dict(from_node_id=mid_node, to_node_id=target, b1=None, b2=None)
                detached.append([rid, mid])
                break
    return d2, detached


def _whole_block_unit(model, scene, block, cur, long_hint, spacing):
    """Build a rigid whole-block relocation unit (never a submodule slice).

    Keys cover every reaction of the curated block plus glyphs used only inside it —
    the same membership rule as ``find``, without the small-block size cap.
    """
    pos, parts = scene['positions'], scene['participants']
    holders = defaultdict(set)
    for (rid, mid), key in parts.items():
        holders[key].add(rid)
    rx = {r['id']: r for r in model['reactions']}
    inside = set(block['reaction_ids'])
    keys = {'r:' + r for r in inside if 'r:' + r in pos}
    keys.update(k for k in pos if k.startswith('copy:' + block['id'] + ':'))
    for r in inside:
        if r not in rx:
            continue
        for mid in rx[r]['metabolites']:
            k = parts.get((r, mid))
            if k is not None and holders[k] <= inside and k in pos:
                keys.add(k)
    # Prefer long connections computed on the whole block; fall back to the submodule hint
    # (used only to aim candidates when the block has no external long edge of its own).
    users = defaultdict(set)
    for r in model['reactions']:
        for mm in r['metabolites']:
            users[mm].add(r['id'])
    long = []
    for r in inside:
        if 'r:' + r not in pos or r not in rx:
            continue
        for mid in rx[r]['metabolites']:
            if mid in cur:
                continue
            k = parts.get((r, mid))
            if k is None or k not in pos:
                continue
            if k not in keys:
                px = float(np.hypot(*(np.array(pos['r:' + r]) - np.array(pos[k]))))
                if px > TAU * spacing:
                    long.append((r, mid, round(px, 1), k))
                continue
            best = None
            for u in users[mid]:
                if u in inside or (u, mid) not in parts:
                    continue
                ku = parts[u, mid]
                if ku in pos and ku not in keys:
                    d = float(np.hypot(*(np.array(pos['r:' + r]) - np.array(pos[ku]))))
                    if best is None or d < best[0]:
                        best = (d, ku)
            if best is not None and best[0] > TAU * spacing:
                long.append((r, mid, round(best[0], 1), best[1]))
    if not long and long_hint:
        long = list(long_hint)
    if not long:
        return None
    return dict(block=block['id'], reactions=sorted(inside), keys=keys,
                long=sorted(long, key=lambda t: -t[2]), spacing=spacing,
                block_size=len(inside), cross=[], whole_block=True)


def relocate_sub(data, model, scene, blocks, cur, skip_edges, max_units=12, log=None, simple=True):
    """Relocate whole curated blocks triggered by a long exchange+transport submodule.

    Detection still uses ``find_sub`` (exchange + transport with an abnormally long link inside a
    large expert block), but the move always translates the **entire** block.  Pulling only
    EX_mal/MALt2_2 (or EX_fum/FUMt2_2) out of the curated structure is forbidden.

    Stops on unit / per-unit try caps only (never wall-clock).
    """
    import edge_style
    import quality_gate as qg
    t0 = time.time()
    subs = find_sub(model, scene, blocks, cur, simple=simple)
    block_by_id = {b['id']: b for b in blocks}
    units, seen = [], set()
    for sub in subs:
        bid = sub['block']
        if bid in seen or bid not in block_by_id:
            continue
        seen.add(bid)
        whole = _whole_block_unit(model, scene, block_by_id[bid], cur, sub['long'], sub['spacing'])
        if whole is not None:
            units.append(whole)
    truncated = max(0, len(units) - max_units)
    report = dict(mode='whole-block (triggered by simple transport)' if simple else 'whole-block (triggered by branch transport)',
                  units=len(units), accepted=0, rejected=0, budget_exhausted=truncated,
                  regenerated_expert_edges=[], rows=[], submodule_triggers=len(subs))
    if not units:
        return data, report
    far = len(units)
    base, summary, issues = qg.evaluate_map(data, skip_edges, disconnected=far)
    # Cap expensive whole-block trials: translating a large curated block rarely passes the gate.
    max_tries = 2
    max_move_px = 900.0
    for unit in units[:max_units]:
        row = dict(block=unit['block'], block_size=unit['block_size'], module=unit['reactions'][:8],
                   longest_px=unit['long'][0][2],
                   longest_in_spacings=round(unit['long'][0][2] / unit['spacing'], 1), tries=[])
        won = False
        # Whole-block translate (same path as ``relocate``): no cross-edge detach, no submodule split.
        longs = [(r, k, px) for r, _m, px, k in unit['long']]
        ranked = sorted(((qg.proxy_score(scene, unit['keys'], d, longs), d) for d in candidates(scene, unit)),
                        key=lambda t: -t[0]['score'])
        tried = 0
        for proxy, delta in ranked:
            move = float(np.hypot(*delta))
            if move > max_move_px and unit['block_size'] > 12:
                row['tries'].append(dict(moved_px=round(move, 1), decision='skipped',
                                         reason='whole-block move too far for large curated block'))
                continue
            if tried >= max_tries:
                break
            tried += 1
            d2, redo, _ = qg.translate_module(data, scene, unit['keys'], delta, set(unit['reactions']), skip_edges)
            edge_style.apply(d2, skip_edges, only=redo)
            new, s2, i2 = qg.evaluate_map(d2, skip_edges, disconnected=far - 1)
            ok, why = _accept(base, new)
            row['tries'].append(dict(moved_px=round(move, 1),
                                     decision='accepted' if ok else 'rejected', reason=why,
                                     after={k: new[k] for k in ('collisions', 'crossings', 'long_edges')}))
            if ok:
                data, base, summary, issues = d2, new, s2, i2
                far -= 1
                base['disconnected'] = far
                for k in unit['keys']:
                    scene['positions'][k] = [scene['positions'][k][0] + float(delta[0]),
                                            scene['positions'][k][1] + float(delta[1])]
                won = True
                break
        row['decision'] = 'accepted' if won else 'rejected'
        report['accepted' if won else 'rejected'] += 1
        report['rows'].append(row)
        if log:
            log(f"  curated whole-block relocation {unit['block'][:8]} ({unit['reactions'][:3]}): "
                f"{row['decision']} (longest {row['longest_in_spacings']} spacings, block of {unit['block_size']})")
    report['seconds'] = round(time.time() - t0, 1)
    return data, report
