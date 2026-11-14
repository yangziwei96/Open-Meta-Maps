"""V3: visual instances of metabolites, decided from the global layout.

Until now a metabolite outside the curated structures had ONE visual node (`m:<id>`), however many modules used it.  In the iMM904 global map
66 % of the connections longer than 1,000 px ran to ~30 such metabolites (glutamate, acetyl-CoA, 2-oxoglutarate, pyruvate ...) and 29 to the
biomass reaction.  Curated maps draw such a metabolite once per place where it is used.  This module decides, from the positions that the
global layout produced, how many visual instances each metabolite gets and which reactions each instance serves.  One GEM metabolite stays one
identity (`copies[key]` is always its id); instances are only places where it is drawn.

Operations on the scene made by `drawing.display_scene` (the hashed drawing files are not modified):

  summary reactions   a reaction outside the curated structures with >= SUMMARY_MIN non-currency participants (biomass) gets a private instance
                      of every participant on a ring around it, instead of one long line per participant to nodes all over the map.  Every
                      participant stays drawn and connected; nothing is hidden or merged.  Reason `summary_reaction`.
  reuse / merge       a reaction attaches to the nearest existing instance of its metabolite (a curated copy, a shared node or an earlier
                      instance) when that is within NEAR px.  A shared node that no longer serves any reaction disappears (a merge).
  split               reactions of one metabolite that are further than NEAR from every existing instance are grouped (single linkage,
                      LINK px); a group of >= MIN_CLUSTER reactions (or one reaction farther than FAR) gets a new instance at its centre.
                      Reason `long_edge_split`.  At most MAX_NEW new instances per metabolite.
  bridge share        after proximity reuse/split, a shared `m:` glyph keeps only the attachments that bridge otherwise-separate reaction
                      components (union-find over non-currency edges excluding that metabolite).  Non-bridging attachments become local
                      `inst:` clones beside their reaction.  Reason `bridge_local`.  Packing may still use shared identities as junctions;
                      this pass is the principled share-vs-local policy.
  island attach       an isolated non-curated reaction may attach to a nearby instance of a participant (up to RESCUE px) so packing keeps
                      connectivity — but only when that attachment bridges components (same rule as bridge share).
  structure-local     after packing (compose pipeline), a reaction inside a curated structure prefers that structure's own `copy:` of the
                      metabolite.  When the structure has no suitable copy and the reaction still has a far link to another structure's
                      `copy:`, a new movable `inst:` is placed next to the structure — replacing the island-proximity stretch onto a foreign
                      curated copy.  New `inst:` glyphs are accepted only when participant side-consistency does not worsen vs the current
                      scene (LayoutQualityEvaluator ``rule="instance"``, new-only).  Curated `copy:` / `side:` / pinned glyphs are never
                      created, moved or merged.  Reason `structure_local`.  The far cut is the layout's own edge-length fence
                      (Q3 + 1.5 IQR of non-currency participant lengths, falling back to NEAR).

Curated glyphs and reaction points are never moved here; only the participant each reaction attaches to is chosen (and new `inst:` nodes may
be added).  Currency metabolites keep their per-reaction `side:` glyphs.  Deterministic.  The report gives long-edge counts before / after
(on the scene before relaxation, participant edges of non-currency metabolites) and the reduction per added instance.
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from core import participation_roles

NEAR = 250.0
LINK = 260.0
FAR = 1000.0
MIN_CLUSTER = 2
MAX_NEW = 8
SUMMARY_MIN = 12          # absolute floor for summary-reaction treatment
SUMMARY_DEGREE_PCT = 0.92 # or: reactions whose non-currency degree is at/above this percentile of the model
import os as _os
SUMMARY_LAYOUT = _os.environ.get('OPM_SUMMARY_LAYOUT', 'columns')  # columns fan locally; ring radiates in every direction
SUMMARY_SPACING = 34.0
SPLIT_PLACED = 380.0  # a single reaction that anchor-driven placement put beside a structure gets its own recorded instance of a metabolite that is further than this
RESCUE = 700.0      # an otherwise isolated reaction attaches to the nearest instance used by another reaction up to this far
# High-degree non-currency metabolites (shared across many reactions) get more aggressive local cloning
HUB_METAB_PCT = 0.95
HUB_NEAR = 180.0
HUB_MAX_NEW = 12
LOCAL_OFFSET = np.array([0.0, 55.0])  # same offset as long_edge_split placement


def _side_consistency_ok(model, scene, pc_before):
    """New-only participant side-consistency gate (LayoutQualityEvaluator rule=instance)."""
    from layout_quality_evaluator import (
        accept_participant_consistency,
        measure_scene_participant_consistency,
    )

    pc_after = measure_scene_participant_consistency(model, scene)
    ok, reason = accept_participant_consistency(pc_before, pc_after)
    return ok, reason, pc_after


def _long(model, scene, cur, thresholds=(250.0, 1000.0)):
    pos, parts = scene['positions'], scene['participants']
    counts = [0] * len(thresholds)
    for r in model['reactions']:
        rp = np.array(pos['r:' + r['id']], float)
        for mid in r['metabolites']:
            if mid in cur:
                continue
            d = float(np.hypot(*(np.array(pos[parts[r['id'], mid]], float) - rp)))
            for i, t in enumerate(thresholds):
                counts[i] += d > t
    return counts


def _clusters(points, link):
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if n > 1:
        for i, j in cKDTree(points).query_pairs(link):
            parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _uf_find(parent, x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent, a, b):
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra
        return True
    return False


def _components_excluding_metabolite(model, parts, mid, cur):
    """Union-find over reactions via non-currency participant edges, excluding ``mid``.

    Metabolite identity nodes (``M:<id>``) link reactions that share any other
    non-currency metabolite, matching the spanning ``joins`` idea: a share of
    ``mid`` is useful only when it merges otherwise-separate components.
    """
    parent = {}
    for r in model['reactions']:
        rid = r['id']
        _uf_find(parent, 'R:' + rid)
        for m in r['metabolites']:
            if m in cur or m == mid:
                continue
            key = parts.get((rid, m))
            if key is None or str(key).startswith('side:'):
                continue
            _uf_union(parent, 'R:' + rid, 'M:' + m)
    return parent


def share_joins(model, parts, mid, cur, candidate_rids):
    """Which reactions may share one glyph of ``mid`` (bridge otherwise-separate parts).

    Candidates are considered in the given order (caller should sort by priority).
    The first attachment always opens the shared box; later ones join only when
    ``find(M:mid) != find(R:rid)`` before the union — same greedy spanning rule as
    the automatic-layout ``joins`` set.  Non-members should get local ``inst:``.
    """
    parent = _components_excluding_metabolite(model, parts, mid, cur)
    joins = set()
    for rid in candidate_rids:
        if _uf_find(parent, 'M:' + mid) != _uf_find(parent, 'R:' + rid):
            _uf_union(parent, 'M:' + mid, 'R:' + rid)
            joins.add(rid)
    return joins


def _degree_threshold(model, cur, pct, floor):
    """Data-driven degree cutoff: max(floor, percentile of non-currency participant counts)."""
    degrees = [sum(1 for m in r['metabolites'] if m not in cur) for r in model['reactions']]
    if not degrees:
        return floor
    return max(floor, int(np.percentile(degrees, pct * 100)))


def _hub_metabolites(model, cur, pct):
    """Metabolites used by many reactions (top percentile of usage counts), excluding currency."""
    uses = defaultdict(int)
    for r in model['reactions']:
        for mid in r['metabolites']:
            if mid not in cur:
                uses[mid] += 1
    if not uses:
        return set()
    cut = max(8, int(np.percentile(list(uses.values()), pct * 100)))
    return {m for m, n in uses.items() if n >= cut}


def _edge_far_threshold(model, scene, cur):
    """Far-edge fence from the scene itself: Q3 + 1.5 IQR of non-currency participant lengths."""
    pos, parts = scene['positions'], scene['participants']
    lengths = []
    for r in model['reactions']:
        rk = 'r:' + r['id']
        if rk not in pos:
            continue
        rp = np.array(pos[rk], float)
        for mid in r['metabolites']:
            if mid in cur:
                continue
            key = parts.get((r['id'], mid))
            if key not in pos:
                continue
            lengths.append(float(np.hypot(*(np.array(pos[key], float) - rp))))
    if len(lengths) < 8:
        return NEAR
    q1, q3 = np.percentile(lengths, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


def _rid_block(blocks):
    return {rid: b['id'] for b in (blocks or []) for rid in b.get('reaction_ids') or ()}


def _block_copies(copies, pos):
    """(block_id, metabolite_id) -> copy: key for curated copies present in the scene."""
    out = {}
    for key, mid in copies.items():
        if key.startswith('copy:') and key in pos:
            out[key.split(':')[1], mid] = key
    return out


def _pinned_keys(scene):
    """Hard-pinned glyph keys (previous-layout anchors); never reassigned or moved here."""
    pinned = set()
    for key, hint in (scene.get('anchors') or {}).items():
        if isinstance(hint, dict):
            if hint.get('hard'):
                pinned.add(key)
        else:
            pinned.add(key)
    return pinned


def _detach_protected(scene, rid, mid):
    """Remove one reaction from protected groups of `mid` so a deliberate reattachment does not count as a split.

    A curated reaction that was sharing another structure's `copy:` (or a far `m:`) is not part of that
    glyph's intended protection; peeling it off leaves the remaining members on one instance.
    """
    kept = []
    for rec in scene.get('protected') or []:
        if rec.get('canonical_metabolite_id') != mid or rid not in rec.get('attached_reaction_ids', ()):
            kept.append(rec)
            continue
        members = [r for r in rec['attached_reaction_ids'] if r != rid]
        if len(members) >= 2:
            rec = dict(rec, attached_reaction_ids=members)
            kept.append(rec)
        # Degenerate (<2) groups are dropped.
    scene['protected'] = kept


def _far_foreign_or_shared(cur_key, bid):
    """True when the attachment is another structure's copy, or a shared/movable glyph (m:/inst:)."""
    if cur_key.startswith('copy:'):
        return cur_key.split(':')[1] != bid
    return cur_key.startswith(('inst:', 'm:'))


def _unit_toward(origin, target):
    vec = np.asarray(target, float) - np.asarray(origin, float)
    norm = float(np.hypot(*vec))
    if norm < 1e-6:
        return np.array([0.0, 1.0])
    return vec / norm


def _axis_unit_toward(origin, target):
    """Dominant-axis unit toward the old endpoint — keeps the half-plane and favours orthogonal stubs."""
    unit = _unit_toward(origin, target)
    if abs(unit[0]) >= abs(unit[1]):
        return np.array([1.0 if unit[0] >= 0 else -1.0, 0.0])
    return np.array([0.0, 1.0 if unit[1] >= 0 else -1.0])


def _clear_of(existing, spot, need):
    if existing is None or len(existing) == 0:
        return True
    return float(np.min(np.hypot(*(existing - spot).T))) >= need - 1e-9


def _place_along_corridor(centre, foreign_xy, existing, place_dist, clearance, far_thr, disk_clearance):
    """Place a local stub toward the old endpoint: try the axis-aligned ray first (edge orthogonality), then the true corridor.

    Distances stay at or under the layout far fence so the new stub is not itself a far edge.
    """
    max_dist = max(float(place_dist), float(far_thr))
    distances = []
    d = float(place_dist)
    while d <= max_dist + 1e-9:
        distances.append(d)
        d += clearance
    if not distances or distances[-1] < max_dist - 1e-9:
        distances.append(max_dist)
    # Soft clearance = metabolite label gap; hard floor = Escher primary diameter.
    soft, hard = float(clearance), float(disk_clearance)

    def _search(unit, need):
        angles = [0.0]
        for step in (1, 2, 3, 4):
            ang = step * (np.pi / 4.0)
            angles.extend((ang, -ang))
        for ang in angles:
            c, s = float(np.cos(ang)), float(np.sin(ang))
            rot = np.array([unit[0] * c - unit[1] * s, unit[0] * s + unit[1] * c])
            for dist in distances:
                cand = np.asarray(centre, float) + rot * dist
                if _clear_of(existing, cand, need):
                    return cand
        return None

    # Axis-aligned corridor first (edge orthogonality), then the true corridor (half-plane).
    true = _unit_toward(centre, foreign_xy)
    axis = _axis_unit_toward(centre, foreign_xy)
    for need in (soft, hard):
        for unit in (axis, true):
            hit = _search(unit, need)
            if hit is not None:
                return hit
    # Last resort on the axis ray at place_dist (caller may still reject if hard clearance fails).
    return np.asarray(centre, float) + axis * float(place_dist)


def structure_local(model, scene, blocks, cur, reasons=None):
    """Prefer each curated structure's own metabolite copy; add local inst: only to clear far foreign / far m:/inst: links.

    Does not move or merge curated `copy:`, `side:` or pinned glyphs.  Returns (scene, report) and may modify `scene`.
    """
    pos, labels, copies, parts = scene['positions'], scene['labels'], scene['copies'], scene['participants']
    reasons = reasons if reasons is not None else scene.setdefault('instance_reasons', {})
    rid_block = _rid_block(blocks)
    own_of = _block_copies(copies, pos)
    pinned = _pinned_keys(scene)
    far_thr = _edge_far_threshold(model, scene, cur)
    rx = {r['id']: r for r in model['reactions']}
    reassigned = 0
    # 1. Prefer the structure's own curated copy whenever it exists.
    for rid, bid in sorted(rid_block.items()):
        r = rx.get(rid)
        if r is None or ('r:' + rid) not in pos:
            continue
        for mid in sorted(r['metabolites']):
            if mid in cur:
                continue
            own = own_of.get((bid, mid))
            if not own:
                continue
            cur_key = parts.get((rid, mid))
            if cur_key == own or cur_key in pinned or own in pinned:
                continue
            if cur_key and cur_key.startswith('side:'):
                continue
            _detach_protected(scene, rid, mid)
            parts[rid, mid] = own
            reassigned += 1
    # 2. Far attachments that still lack an own copy: foreign curated copies first, then far m:/inst:.
    def _pending(kind):
        out = defaultdict(list)
        for rid, bid in sorted(rid_block.items()):
            r = rx.get(rid)
            if r is None or ('r:' + rid) not in pos:
                continue
            rp = np.array(pos['r:' + rid], float)
            for mid in sorted(r['metabolites']):
                if mid in cur or own_of.get((bid, mid)):
                    continue
                cur_key = parts.get((rid, mid))
                if cur_key is None or cur_key not in pos or cur_key in pinned:
                    continue
                if cur_key.startswith('side:') or not _far_foreign_or_shared(cur_key, bid):
                    continue
                is_foreign = cur_key.startswith('copy:') and cur_key.split(':')[1] != bid
                is_shared = cur_key.startswith(('inst:', 'm:'))
                if kind == 'foreign' and not is_foreign:
                    continue
                if kind == 'shared' and not is_shared:
                    continue
                dist = float(np.hypot(*(np.array(pos[cur_key], float) - rp)))
                if dist <= far_thr:
                    continue
                out[bid, mid].append((rid, cur_key))
        return out

    # Clearance beside a curated reaction: same defaults separate() uses; disk floor from Escher primary radius.
    import inspect as _inspect
    from layout_gate import _RADIUS as _ESCHER_RADIUS, SPECIES as _ESCHER_SPECIES
    _sep = _inspect.signature(separate).parameters
    reaction_gap = float(_sep['reaction_gap'].default)
    metabolite_gap = float(_sep['metabolite_gap'].default)
    place_dist = max(reaction_gap, metabolite_gap)
    clearance = metabolite_gap
    disk_clearance = 2.0 * float(_ESCHER_RADIUS[_ESCHER_SPECIES][0])
    existing = np.array([pos[k] for k in sorted(pos)], float) if pos else np.zeros((0, 2))

    added = 0
    reused = 0
    skipped = 0
    skipped_side = 0
    frozen_copies = set(scene.get('structure_local_frozen_copies') or ())
    foreign_pending = _pending('foreign')
    shared_pending = _pending('shared')
    from layout_quality_evaluator import measure_scene_participant_consistency
    pc_base = measure_scene_participant_consistency(model, scene)
    # Shared (m:/inst:) groups only reuse an existing local movable instance — new glyphs are reserved for
    # far foreign-copy links (the cross-structure case).  Reuse is applied when it clearly shortens the edge.
    for kind, pending in (('foreign', foreign_pending), ('shared', shared_pending)):
        for (bid, mid), members in sorted(pending.items()):
            rids = [rid for rid, _ in members]
            centre = np.asarray(np.mean([pos['r:' + rid] for rid in rids], axis=0), float)
            local_opts = []
            for t, m2 in copies.items():
                if m2 != mid or t not in pos or t.startswith(('side:', 'copy:')):
                    continue
                # Never reuse another structure's structure-local instance (that recreates cross-structure links).
                if t.startswith('inst:local:'):
                    other = t.split(':')[2] if t.count(':') >= 3 else None
                    if other is not None and other != bid:
                        continue
                d = float(np.hypot(*(np.array(pos[t], float) - centre)))
                if d <= far_thr:
                    local_opts.append((d, t))
            if local_opts:
                best = min(local_opts)[1]
                for rid, old in members:
                    if old.startswith('copy:'):
                        frozen_copies.add(old)
                    _detach_protected(scene, rid, mid)
                    parts[rid, mid] = best
                reused += 1
                continue
            if kind == 'shared':
                skipped += 1
                continue
            foreign_xy = np.mean([pos[fk] for _, fk in members], axis=0)
            spot = _place_along_corridor(
                centre, foreign_xy, existing, place_dist, clearance, far_thr, disk_clearance)
            if not _clear_of(existing, spot, disk_clearance):
                skipped += 1
                continue
            key = f'inst:local:{bid}:{mid}'
            n = 0
            while key in pos:
                n += 1
                key = f'inst:local:{bid}:{mid}:{n}'
            # Tentative new inst: — keep only when side consistency does not worsen (new-only).
            prev_parts = {rid: parts.get((rid, mid)) for rid, _ in members}
            prev_protected = list(scene.get('protected') or [])
            pos[key] = np.asarray(spot, float).tolist()
            labels[key], copies[key] = mid, mid
            for rid, old in members:
                _detach_protected(scene, rid, mid)
                parts[rid, mid] = key
            ok, _why, pc_after = _side_consistency_ok(model, scene, pc_base)
            # Far foreign / shared attachments: prefer a local tuck over keeping a spoke.
            if not ok:
                old_d = []
                for rid, old in members:
                    if old and old in pos and ('r:' + rid) in pos:
                        old_d.append(float(np.hypot(*(
                            np.array(pos['r:' + rid], float) - np.array(pos[old], float)))))
                if old_d and float(np.mean(old_d)) > FAR:
                    ok = True
                    pc_after = pc_base
            if not ok:
                for rid, old in prev_parts.items():
                    if old is None:
                        parts.pop((rid, mid), None)
                    else:
                        parts[rid, mid] = old
                pos.pop(key, None)
                labels.pop(key, None)
                copies.pop(key, None)
                scene['protected'] = prev_protected
                skipped += 1
                skipped_side += 1
                continue
            reasons[key] = 'structure_local'
            existing = np.vstack([existing, spot]) if len(existing) else np.asarray(spot, float).reshape(1, 2)
            for rid, old in members:
                if old.startswith('copy:'):
                    frozen_copies.add(old)
            added += 1
            pc_base = pc_after if pc_after is not None else pc_base
    scene['structure_local_frozen_copies'] = sorted(frozen_copies)
    report = dict(added=added, reused_local=reused, reassigned_to_own_copy=reassigned,
                  skipped_no_clearance=skipped, skipped_side_consistency=skipped_side,
                  far_threshold=round(far_thr, 2),
                  groups=len(foreign_pending) + len(shared_pending), reason='structure_local',
                  frozen_copies=len(frozen_copies))
    return scene, report


def optimize(model, scene, blocks, cur, summary=True, reuse=True, split=True):
    """Returns (new scene, report).  `scene` is not modified."""
    scene = dict(scene, positions={k: list(v) for k, v in scene['positions'].items()}, labels=dict(scene['labels']), copies=dict(scene['copies']),
                 participants=dict(scene['participants']))
    pos, labels, copies, parts = scene['positions'], scene['labels'], scene['copies'], scene['participants']
    reasons = {}
    curated = {rid for b in blocks for rid in b['reaction_ids']}
    before = _long(model, scene, cur)
    rx = {r['id']: r for r in model['reactions']}
    shared_before = {k for k in pos if k.startswith('m:')}
    summary_cut = _degree_threshold(model, cur, SUMMARY_DEGREE_PCT, SUMMARY_MIN)
    hub_mets = _hub_metabolites(model, cur, HUB_METAB_PCT)
    report = dict(summary_reactions=[], added=0, reattached=0, skipped_side_consistency=0,
                  summary_degree_cut=summary_cut, hub_metabolites=len(hub_mets))
    from layout_quality_evaluator import measure_scene_participant_consistency
    pc_base = measure_scene_participant_consistency(model, scene)

    # 1. summary reactions: local stubs/columns instead of map-wide spokes (degree-driven, not name-driven)
    if summary:
        for rid in sorted(rx):
            r = rx[rid]
            core = [m for m in sorted(r['metabolites']) if m not in cur and not parts[rid, m].startswith('copy:')]
            if rid in curated or len(core) < summary_cut:
                continue
            centre = np.array(pos['r:' + rid], float)
            radius = max(160.0, len(core) * 46 / (2 * np.pi))
            subs, prods = [], []
            for m in core:
                for role_c in participation_roles(r['metabolites'][m]):
                    (subs if role_c < 0 else prods).append(m)
            prev_parts = {mid: parts.get((rid, mid)) for mid in core}
            new_keys = []
            if SUMMARY_LAYOUT == 'columns':
                # two columns: substrates in one column on the left of the reaction point, products in one on the right, each ordered by where the
                # metabolite is drawn elsewhere (top to bottom), so the connections form two fans instead of a ring (a ring sends lines in every direction)
                spacing = SUMMARY_SPACING
                half = max(len(subs), len(prods))
                dist = max(200.0, 90.0 + half * spacing * 0.35)

                def yorder(mid):
                    return (pos['m:' + mid][1] if 'm:' + mid in pos else centre[1], mid)
                for members, sign in ((sorted(subs, key=yorder), -1.0), (sorted(prods, key=yorder), 1.0)):
                    n = len(members)
                    for i, mid in enumerate(members):
                        key = f'inst:{rid}:{mid}'
                        pos[key] = [float(centre[0] + sign * dist), float(centre[1] + (i - (n - 1) / 2.0) * spacing)]
                        labels[key], copies[key] = mid, mid
                        parts[rid, mid] = key
                        new_keys.append(key)
            else:
                for members, start in ((subs, 90.0), (prods, -90.0)):
                    for i, mid in enumerate(members):
                        angle = np.radians(start + 180.0 * (i + .5) / max(len(members), 1))
                        key = f'inst:{rid}:{mid}'
                        pos[key] = (centre + radius * np.array([np.cos(angle), np.sin(angle)])).tolist()
                        labels[key], copies[key] = mid, mid
                        parts[rid, mid] = key
                        new_keys.append(key)
            ok, _why, pc_after = _side_consistency_ok(model, scene, pc_base)
            if not ok:
                for mid, old in prev_parts.items():
                    if old is None:
                        parts.pop((rid, mid), None)
                    else:
                        parts[rid, mid] = old
                for key in new_keys:
                    pos.pop(key, None)
                    labels.pop(key, None)
                    copies.pop(key, None)
                report['skipped_side_consistency'] += 1
                continue
            for key in new_keys:
                reasons[key] = 'summary_reaction'
            layout = 'columns' if SUMMARY_LAYOUT == 'columns' else 'ring'
            row = dict(reaction=rid, participants=len(core))
            if layout == 'columns':
                row['layout'] = 'columns'
            report['summary_reactions'].append(row)
            report['added'] += len(core)
            pc_base = pc_after if pc_after is not None else pc_base

    # 2. reuse and split, per metabolite
    uses = defaultdict(list)
    for (rid, mid), key in parts.items():
        if key == 'm:' + mid and mid not in cur:
            uses[mid].append(rid)
    by_mid = defaultdict(list)
    for key, mid in copies.items():
        if not key.startswith('side:') and reasons.get(key) != 'summary_reaction':
            by_mid[mid].append(key)
    per_metabolite = {}
    group_of = {}      # (reaction, metabolite) -> protected record (conditional groups on a shared 'm:' node of a core metabolite)
    for rec in scene.get('protected', []):
        if not rec.get('strict') and rec['visual_instance_id'].startswith('m:') and rec['canonical_metabolite_id'] not in cur:
            for r in rec['attached_reaction_ids']:
                group_of[r, rec['canonical_metabolite_id']] = rec
    moved_groups = Counter()
    for mid in sorted(uses):
        rids = sorted(uses[mid])
        targets = [k for k in by_mid[mid] if k in pos]
        if not targets:
            continue
        # Hub metabolites (top percentile of reaction usage) clone more aggressively so they are not map-wide spokes
        near = HUB_NEAR if mid in hub_mets else NEAR
        max_new = HUB_MAX_NEW if mid in hub_mets else MAX_NEW
        rp = np.array([pos['r:' + rid] for rid in rids], float)
        added = 0
        loose = []
        # protected groups first: identity (same metabolite) -> group compatible -> geometry acceptable -> the whole group reuses an instance or gets one new
        done, handled = set(), set()
        for rid in rids:
            rec = group_of.get((rid, mid))
            if rec is None or id(rec) in done:
                continue
            done.add(id(rec))
            members = [r for r in rec['attached_reaction_ids'] if parts.get((r, mid)) == 'm:' + mid]
            if len(members) < 2:
                continue
            handled.update(members)
            mp = np.array([pos['r:' + r] for r in members], float)
            here = np.array(pos['m:' + mid], float)
            cost_now = float(np.sum(np.hypot(*(mp - here).T)))
            options = []
            for t in targets:
                if t == 'm:' + mid:
                    continue
                if t.startswith('copy:') and rec.get('anchor_region') and t.split(':')[1] != rec['anchor_region']:
                    continue  # a glyph of another curated structure is not this group's anchor
                d = np.hypot(*(mp - np.array(pos[t], float)).T)
                if float(d.max()) <= near and float(d.sum()) <= cost_now:
                    options.append((float(d.sum()), t))
            if options:
                best = min(options)[1]
            elif cost_now > 0 and float(np.min(np.hypot(*(mp - here).T))) > SPLIT_PLACED and added < max_new:
                best = f'inst:{mid}:{added}'
                pos[best] = (mp.mean(axis=0) + LOCAL_OFFSET).tolist()
                labels[best], copies[best] = mid, mid
                prev_parts = {r: parts.get((r, mid)) for r in members}
                for r in members:
                    parts[r, mid] = best
                ok, _why, pc_after = _side_consistency_ok(model, scene, pc_base)
                # Side-consistency must not preserve spokes beyond FAR.
                far_group = float(np.min(np.hypot(*(mp - here).T))) > FAR
                if not ok and far_group:
                    ok = True
                    pc_after = pc_base
                if not ok:
                    for r, old in prev_parts.items():
                        if old is None:
                            parts.pop((r, mid), None)
                        else:
                            parts[r, mid] = old
                    pos.pop(best, None)
                    labels.pop(best, None)
                    copies.pop(best, None)
                    report['skipped_side_consistency'] += 1
                    best = 'm:' + mid
                    moved_groups['kept'] += 1
                else:
                    reasons[best] = 'long_edge_split'
                    targets.append(best)
                    added += 1
                    report['added'] += 1
                    pc_base = pc_after if pc_after is not None else pc_base
                    for r in members:
                        parts[r, mid] = best
                    moved_groups['moved_as_a_whole'] += 1
                    continue
            else:
                best = 'm:' + mid
                moved_groups['kept'] += 1
            if best != 'm:' + mid:
                for r in members:
                    parts[r, mid] = best
                moved_groups['moved_as_a_whole'] += 1
        rids = [r for r in rids if r not in handled]
        rp = np.array([pos['r:' + rid] for rid in rids], float) if rids else np.zeros((0, 2))
        for rid, p in zip(rids, rp):
            tp = np.array([pos[t] for t in targets], float)
            d = np.hypot(*(tp - p).T)
            j = int(np.argmin(d))
            if reuse and d[j] <= near:
                if targets[j] != 'm:' + mid:
                    report['reattached'] += 1
                parts[rid, mid] = targets[j]
            else:
                loose.append((rid, p))
        if loose and split:
            pts = np.array([p for _, p in loose], float)
            for cluster in _clusters(pts, LINK):
                members = [loose[i] for i in cluster]
                centre = np.mean([p for _, p in members], axis=0)
                tp = np.array([pos[t] for t in targets], float)
                mean_d = float(np.mean([np.min(np.hypot(*(tp - p).T)) for _, p in members]))
                placed_here = scene.get('completion_context', {}).get('tiers', {})
                if added < max_new and (len(members) >= MIN_CLUSTER or mean_d > FAR or (mean_d > SPLIT_PLACED and any(rid in placed_here for rid, _ in members))):
                    key = f'inst:{mid}:{added}'
                    pos[key] = (centre + LOCAL_OFFSET).tolist()
                    labels[key], copies[key] = mid, mid
                    prev_parts = {rid: parts.get((rid, mid)) for rid, _ in members}
                    for rid, _ in members:
                        parts[rid, mid] = key
                    ok, _why, pc_after = _side_consistency_ok(model, scene, pc_base)
                    if not ok and mean_d > FAR:
                        ok = True
                        pc_after = pc_base
                    if not ok:
                        for rid, old in prev_parts.items():
                            if old is None:
                                parts.pop((rid, mid), None)
                            else:
                                parts[rid, mid] = old
                        pos.pop(key, None)
                        labels.pop(key, None)
                        copies.pop(key, None)
                        report['skipped_side_consistency'] += 1
                        for rid, p in members:
                            d = np.hypot(*(np.array([pos[t] for t in targets], float) - p).T)
                            parts[rid, mid] = targets[int(np.argmin(d))]
                    else:
                        reasons[key] = 'long_edge_split'
                        targets.append(key)
                        added += 1
                        report['added'] += 1
                        pc_base = pc_after if pc_after is not None else pc_base
                else:
                    for rid, p in members:
                        d = np.hypot(*(np.array([pos[t] for t in targets], float) - p).T)
                        parts[rid, mid] = targets[int(np.argmin(d))]
        if added:
            per_metabolite[mid] = added

    # 2b. Bridge-only share: keep `m:` attachments that merge otherwise-separate components; else local inst:.
    bridge_shared = 0
    bridge_local = 0
    for mid in sorted({m for (_, m), key in parts.items() if key == 'm:' + m and m not in cur}):
        shared_key = 'm:' + mid
        if shared_key not in pos:
            continue
        holders = [rid for (rid, m), key in parts.items() if m == mid and key == shared_key]
        if len(holders) < 2:
            if holders:
                bridge_shared += 1
            continue
        # Prefer reactions closer to the shared glyph as spanning joins (stable, geometry-aware).
        here = np.array(pos[shared_key], float)

        def _join_priority(rid):
            rp = np.array(pos.get('r:' + rid, here), float)
            return (float(np.hypot(*(rp - here))), rid)

        ordered = sorted(holders, key=_join_priority)
        joins = share_joins(model, parts, mid, cur, ordered)
        # Topology may mark a far reaction as a bridge, but a module-scale spoke
        # is still a packing failure (iIT341 after #33: >1000 edges stayed high).
        # Keep shared geometry only when the reaction sits within FAR of the glyph.
        near_joins = set()
        for rid in joins:
            rp = np.array(pos.get('r:' + rid, here), float)
            if float(np.hypot(*(rp - here))) <= FAR:
                near_joins.add(rid)
        joins = near_joins
        bridge_shared += len(joins)
        non = [rid for rid in ordered if rid not in joins]
        if not non:
            continue
        max_new = HUB_MAX_NEW if mid in hub_mets else MAX_NEW
        # Count existing long_edge / bridge locals for this mid so we respect the cap.
        already = sum(1 for k, m2 in copies.items() if m2 == mid and k.startswith('inst:') and reasons.get(k) in ('long_edge_split', 'bridge_local'))
        for rid in non:
            if already >= max_new:
                # Cap exceeded: still prefer a private glyph over a non-bridging share.
                key = f'inst:{mid}:bridge:{rid}'
            else:
                key = f'inst:{mid}:{already}'
                already += 1
            n = 0
            base = key
            while key in pos:
                n += 1
                key = f'{base}:{n}'
            centre = np.array(pos['r:' + rid], float)
            pos[key] = (centre + LOCAL_OFFSET).tolist()
            labels[key], copies[key] = mid, mid
            reasons[key] = 'bridge_local'
            _detach_protected(scene, rid, mid)
            parts[rid, mid] = key
            report['added'] += 1
            bridge_local += 1
            per_metabolite[mid] = per_metabolite.get(mid, 0) + 1
    report['bridge_shares'] = bridge_shared
    report['bridge_local'] = bridge_local

    # 2c. a reaction outside the curated structures whose every glyph is private (shared with no other reaction) is attached to the nearest
    # instance that another reaction uses, whatever the usual NEAR rule says: an island is worse than a longer line — but only when the
    # attachment bridges components (same share policy).  Far stretches onto a foreign curated copy are corrected after packing by
    # structure_local.
    holders = defaultdict(set)
    for (rid, mid), key in parts.items():
        holders[key].add(rid)
    rescued = 0
    for rid in sorted(rx):
        if rid in curated:
            continue
        mine = [(mid, parts[rid, mid]) for mid in sorted(rx[rid]['metabolites']) if mid not in cur]
        if not mine or any(len(holders[k]) > 1 for _, k in mine):
            continue
        rp = np.array(pos['r:' + rid], float)
        done = False
        for mid, k in mine:
            options = [t for t, m2 in copies.items() if m2 == mid and not t.startswith('side:') and reasons.get(t) != 'summary_reaction' and t != k and holders[t] - {rid}]
            if not options:
                continue
            dist = [float(np.hypot(*(np.array(pos[t], float) - rp))) for t in options]
            ranked = sorted(range(len(options)), key=lambda i: dist[i])
            for j in ranked:
                if dist[j] > RESCUE:
                    break
                target = options[j]
                # Sharing onto an existing multi-holder (or shared m:) must bridge components.
                if target.startswith('m:') or len(holders[target] - {rid}) >= 1:
                    peers = sorted(holders[target] - {rid})
                    # Probe: would rid join if peers are already on the shared box?
                    probe = share_joins(model, parts, mid, cur, peers + [rid])
                    if rid not in probe:
                        continue
                holders[k].discard(rid)
                parts[rid, mid] = target
                holders[target].add(rid)
                done = True
                break
            if done:
                break
        rescued += done
    report['isolated_reactions_attached'] = rescued

    # 3. shared nodes that no reaction uses any more were merged into other instances
    used = set(parts.values())
    removed = 0
    for key in sorted(shared_before):
        if key not in used and key in pos:
            for table in (pos, labels, copies):
                table.pop(key, None)
            removed += 1
    scene['instance_reasons'] = reasons
    after = _long(model, scene, cur)
    gained = before[0] - after[0]
    report['protected_groups'] = dict(moved_groups)
    report.update(shared_nodes_merged=removed, reactions_reattached=report.pop('reattached'),
                  edges_over_250=dict(before=int(before[0]), after=int(after[0])), edges_over_1000=dict(before=int(before[1]), after=int(after[1])),
                  long_edge_reduction_per_added_instance=round(gained / max(report['added'], 1), 2),
                  metabolites_split=len(per_metabolite), top_split=sorted(per_metabolite.items(), key=lambda kv: (-kv[1], kv[0]))[:15],
                  thresholds=dict(near=NEAR, link=LINK, far=FAR, min_cluster=MIN_CLUSTER, max_new=MAX_NEW, summary_min=SUMMARY_MIN))
    return scene, report


def overlap_breakdown(scene, radius=10.0):
    """The glyph pairs closer than `radius` (the node-overlap count of the export), by kind: real conflicts between different glyphs; `same_metabolite` are two
    instances of one metabolite lying on each other (a merge candidate); a shared node is ONE glyph and never an overlap."""
    keys = sorted(scene['positions'])
    xy = np.array([scene['positions'][k] for k in keys], float)
    out = Counter()
    for a, b in cKDTree(xy).query_pairs(radius):
        ka, kb = keys[a], keys[b]
        kind = tuple(sorted(k.split(':')[0] for k in (ka, kb)))
        same = ka[0] != 'r' and kb[0] != 'r' and scene['copies'].get(ka) == scene['copies'].get(kb)
        out['|'.join(kind) + ('|same_metabolite' if same else '')] += 1
    return dict(out)


# Gate node-overlap radius is 10 px; keep a hard floor above that even for linked pairs.
NODE_CENTRE_MIN = 12.0


def separate(model, scene, blocks, metabolite_gap=88.0, reaction_gap=70.0, passes=80, side_gap=None,
             node_centre_min=NODE_CENTRE_MIN):
    """Keep the instances this module placed (`inst:`) and the shared nodes (`m:`) from landing on other glyphs: labels are 60-80 px wide, so
    two metabolite glyphs closer than `metabolite_gap`, or a metabolite glyph and a reaction it is not connected to closer than `reaction_gap`,
    have overlapping labels.  Only `inst:` and `m:` glyphs move for the wide gaps; a hard ``node_centre_min`` floor also nudges algorithm
    reactions (``r:`` not inside a curated copy) so gate node-overlap pairs (<10 px) cannot linger — including chemically linked pairs.

    Currency ``side:`` glyphs use the same adaptive local-clearance rule as ``orientation.sides`` unless ``side_gap`` is given explicitly.
    """
    import orientation as orientation_module
    scene = dict(scene, positions={k: list(v) for k, v in scene['positions'].items()})
    pos = scene['positions']
    keys = sorted(pos)
    index = {k: i for i, k in enumerate(keys)}
    xy = np.array([pos[k] for k in keys], float)
    movable = np.array([k.startswith(('inst:', 'm:', 'side:')) for k in keys])
    # Algorithm midmarkers may nudge under the hard floor; curated block reactions stay fixed.
    curated_rx = {('r:' + rid) for b in (blocks or []) for rid in b.get('reaction_ids') or ()}
    rx_movable = np.array([k.startswith('r:') and k not in curated_rx for k in keys])
    is_side = np.array([k.startswith('side:') for k in keys])
    is_rx = np.array([k.startswith('r:') for k in keys])
    if side_gap is None:
        params = scene.get('orientation_spoke_params') or orientation_module.reaction_spoke_params(model, scene)
        side_need = np.array([
            float(params.get(k.split(':', 2)[1], {}).get('gap', orientation_module.side_gap_for(orientation_module.SIDE_MAX)))
            if k.startswith('side:') else 0.0 for k in keys
        ])
    else:
        side_need = np.full(len(keys), float(side_gap))
    linked = defaultdict(set)
    for (rid, mid), key in scene['participants'].items():
        if 'r:' + rid not in index or key not in index:
            continue
        a, b = index['r:' + rid], index[key]
        linked[a].add(b)
        linked[b].add(a)
    start = xy.copy()
    query_r = max(metabolite_gap, reaction_gap, float(node_centre_min),
                  float(side_need.max()) if len(side_need) else 0.0)
    for _ in range(passes):
        pairs = cKDTree(xy).query_pairs(query_r, output_type='ndarray')
        push = np.zeros_like(xy)
        hit = False
        for i, j in pairs:
            can_move = movable[i] or movable[j] or rx_movable[i] or rx_movable[j]
            if not can_move or (is_rx[i] and is_rx[j]):
                continue
            linked_pair = False
            if is_side[i] or is_side[j]:
                need = min(side_need[i] or side_need[j], side_need[j] or side_need[i])
                if is_rx[i] != is_rx[j]:
                    a, b = (i, j) if is_rx[i] else (j, i)
                    if b in linked[a]:
                        linked_pair = True
                        need = node_centre_min
            elif is_rx[i] != is_rx[j]:
                a, b = (i, j) if is_rx[i] else (j, i)
                if b in linked[a]:
                    linked_pair = True
                    need = node_centre_min  # keep centres ≥ gate radius even when chemically linked
                else:
                    need = reaction_gap
            else:
                need = metabolite_gap
            # Always honour the hard centre floor.
            need = max(need, node_centre_min) if linked_pair else need
            d = xy[j] - xy[i]
            dist = float(np.hypot(*d))
            if dist >= need:
                continue
            if dist < 1e-6:
                ang = (i * 17 + j * 31) * 2.399963
                d, dist = np.array([np.cos(ang), np.sin(ang)]), 1.0
            step = d / dist * (need - dist)
            # Prefer moving non-curated glyphs.  For the hard floor, algorithm reactions may move.
            move_i = movable[i] or (linked_pair and rx_movable[i] and not movable[j])
            move_j = movable[j] or (linked_pair and rx_movable[j] and not movable[i])
            if not (move_i or move_j):
                # Fall back: any movable / algorithm endpoint.
                move_i = movable[i] or rx_movable[i]
                move_j = movable[j] or rx_movable[j]
            if not (move_i or move_j):
                continue
            hit = True
            if move_i and move_j:
                push[i] -= step / 2
                push[j] += step / 2
            elif move_i:
                push[i] -= step
            else:
                push[j] += step
        if not hit:
            break
        xy += push * .7
    moved = int((np.hypot(*(xy - start).T) > 1.0).sum())
    for k in keys:
        pos[k] = xy[index[k]].tolist()
    scene['instances_separated'] = moved
    return scene
