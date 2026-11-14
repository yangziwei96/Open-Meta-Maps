"""Build-on-previous layout: pin reaction/metabolite positions from a prior result map JSON.

When enough reactions are already placed (≥ ``PACK_SKIP_SHARE``), module packing is
skipped and only free nodes are completed around the pinned anchors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import participation_roles
from curated_axes import _absolute_from_escher

# Skip block packing when at least this share of model reactions have a previous pin.
PACK_SKIP_SHARE = 0.80
# Keep a local relayout of free (new) nodes only when crossings fall by this much.
CROSSING_IMPROVE_SHARE = 0.25
CROSSING_IMPROVE_MIN = 2
# Offset for parking a leftover reaction that has no drawn participant anchors.
NEW_COMPONENT_GAP = 120.0


def load_previous_anchors(path, model=None):
    """Read reaction midmarkers and primary metabolites from an Escher result map JSON.

    Returns ``{ 'r:<id>': [x, y], 'm:<id>': [x, y] }``.  When ``model`` is given,
    only identifiers present in that model are kept.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    body = payload[1] if isinstance(payload, list) and len(payload) >= 2 else payload
    if not isinstance(body, dict):
        raise ValueError(f'Not an Escher map: {path}')
    abs_map = _absolute_from_escher(body, str(path))
    anchors = {}
    for rid, xy in (abs_map.get('reactions') or {}).items():
        anchors['r:' + rid] = [float(xy[0]), float(xy[1])]
    for mid, xy in (abs_map.get('metabolites') or {}).items():
        anchors['m:' + mid] = [float(xy[0]), float(xy[1])]
    if model is not None:
        rx = {r['id'] for r in model.get('reactions') or []}
        mets = {m['id'] for m in model.get('metabolites') or []}
        anchors = {
            k: v for k, v in anchors.items()
            if (k.startswith('r:') and k[2:] in rx) or (k.startswith('m:') and k[2:] in mets)
        }
    return anchors


def reaction_anchor_share(anchors, model):
    """Share of model reactions that have a previous midmarker pin."""
    rx = model.get('reactions') or []
    if not rx:
        return 0.0
    hits = sum(1 for r in rx if ('r:' + r['id']) in (anchors or {}))
    return hits / len(rx)


def should_skip_packing(anchors, model, threshold=PACK_SKIP_SHARE):
    return reaction_anchor_share(anchors, model) >= float(threshold)


def apply_block_positions(blocks, anchors):
    """Rewrite curated block glyph positions from previous anchors where present."""
    if not anchors:
        return blocks, 0
    n = 0
    for b in blocks or []:
        pos = dict(b.get('positions') or {})
        for k in list(pos):
            if k in anchors:
                pos[k] = list(anchors[k])
                n += 1
        b['positions'] = pos
    return blocks, n


def pin_scene(scene, anchors):
    """Hard-pin previous anchors into a display scene before export.

    Pins ``r:`` / ``m:`` keys and curated ``copy:…:met`` glyphs to the previous
    primary metabolite position.  Currency ``side:`` stubs follow their reaction
    midmarker by the same delta so spokes stay attached.  Must run before
    ``drawing.export`` (which applies one uniform canvas translation); do not
    re-apply absolute previous coordinates onto the exported Escher JSON.
    """
    if not scene or not anchors:
        return scene, 0
    pos = scene.get('positions') or {}
    n = 0
    old_mids = {k: list(pos[k]) for k in pos if k.startswith('r:') and k in anchors}
    for k, xy in anchors.items():
        if k in pos:
            pos[k] = [float(xy[0]), float(xy[1])]
            n += 1
    for k in list(pos):
        if not k.startswith('copy:'):
            continue
        met = k.rsplit(':', 1)[-1]
        mk = 'm:' + met
        if mk in anchors:
            pos[k] = [float(anchors[mk][0]), float(anchors[mk][1])]
            n += 1
    # Keep side stubs attached to pinned midmarkers.
    for k in list(pos):
        if not k.startswith('side:'):
            continue
        # side:<rid>:<met>
        parts = k.split(':', 2)
        if len(parts) < 3:
            continue
        rk = 'r:' + parts[1]
        if rk not in anchors or rk not in old_mids:
            continue
        delta = np.asarray(anchors[rk], float) - np.asarray(old_mids[rk], float)
        pos[k] = (np.asarray(pos[k], float) + delta).tolist()
        n += 1
    scene['positions'] = pos
    return scene, n


def pin_escher(data, anchors):
    """No-op kept for call-site compatibility; scene pinning + export translation is enough.

    Re-applying absolute previous coordinates after ``drawing.export`` mixes the
    previous canvas frame with the new export translation and scrambles edges.
    """
    return data, 0


def filter_previous_map(path, model):
    """Keep the previous Escher drawing for reactions still in ``model``.

    Drops removed reactions and any nodes that no remaining segment references.
    New model reactions are listed in the report for later placement; when the
    model only lost reactions, the filtered map is the previous layout itself.
    """
    from core import uid
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not (isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], dict)):
        raise ValueError(f'Not an Escher map: {path}')
    header = dict(payload[0] or {})
    body = dict(payload[1])
    keep_rx = {r['id'] for r in model.get('reactions') or []}
    reactions = body.get('reactions') or {}
    nodes = body.get('nodes') or {}
    kept, used_nodes = {}, set()
    for rid, rx in reactions.items():
        bigg = rx.get('bigg_id')
        if bigg not in keep_rx:
            continue
        kept[rid] = rx
        for seg in (rx.get('segments') or {}).values():
            for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                if nid is not None:
                    used_nodes.add(str(nid))
    kept_nodes = {nid: dict(nodes[nid]) for nid in used_nodes if nid in nodes}
    # Keep the previous compare-map representative glyph for each remaining metabolite
    # (same rule as evaluation.escher_bridge.escher_positions: last primary wins).  Do not
    # move other clones — that would stretch edges that still use them.
    ref_nid = {}
    for nid, n in nodes.items():
        if n.get('node_type') != 'metabolite' or not n.get('bigg_id'):
            continue
        mid = n['bigg_id']
        if mid not in ref_nid or n.get('node_is_primary'):
            ref_nid[mid] = nid
    present = {n['bigg_id'] for n in kept_nodes.values()
               if n.get('node_type') == 'metabolite' and n.get('bigg_id')}
    for mid, nid in ref_nid.items():
        if mid not in present:
            continue
        if nid not in kept_nodes and nid in nodes:
            kept_nodes[nid] = dict(nodes[nid])
    for nid, n in kept_nodes.items():
        if n.get('node_type') != 'metabolite' or not n.get('bigg_id'):
            continue
        n['node_is_primary'] = (ref_nid.get(n['bigg_id']) == nid)
    drawn = {rx.get('bigg_id') for rx in kept.values() if rx.get('bigg_id')}
    new_reactions = sorted(keep_rx - drawn)
    meta = dict(header.get('metadata') or {})
    meta['model_hash'] = uid(model)
    # Keep source_reactions only for reactions still in the model (graft fills new ones later).
    src = dict(meta.get('source_reactions') or {})
    rx_by_id = {r['id']: r for r in model.get('reactions') or []}
    pruned = {}
    for rid in keep_rx:
        if rid in src:
            pruned[rid] = src[rid]
        elif rid in drawn:
            r = rx_by_id[rid]
            pruned[rid] = dict(
                stoichiometry=dict(r.get('metabolites') or {}),
                lower_bound=r.get('lower_bound', 0),
                upper_bound=r.get('upper_bound', 0))
    meta['source_reactions'] = pruned
    meta['previous_layout'] = dict(
        source=str(path), kept_reactions=len(kept), removed_reactions=len(reactions) - len(kept),
        new_reactions=new_reactions, nodes=len(kept_nodes))
    header['metadata'] = meta
    out = [header, dict(body, nodes=kept_nodes, reactions=kept)]
    report = dict(kept_reactions=len(kept), removed_reactions=len(reactions) - len(kept),
                  new_reactions=new_reactions, nodes=len(kept_nodes))
    return out, report


def missing_model_reactions(path, model):
    """Reaction ids in ``model`` that have no midmarker in the previous map."""
    _, report = filter_previous_map(path, model)
    return list(report.get('new_reactions') or [])


def unanchored_reaction_ids(model, anchors):
    """Model reaction ids with no previous midmarker pin (matched by stable id)."""
    anchors = anchors or {}
    return [r['id'] for r in model.get('reactions') or [] if ('r:' + r['id']) not in anchors]


def keep_local_relayout(crossings_before, crossings_after,
                        share=CROSSING_IMPROVE_SHARE, vmin=CROSSING_IMPROVE_MIN):
    """True only when a local relax cuts crossings enough to justify moving free nodes."""
    if crossings_before is None or crossings_after is None:
        return False
    before, after = int(crossings_before), int(crossings_after)
    drop = before - after
    return drop >= int(vmin) and drop >= float(share) * max(before, 1)


def scene_segment_crossings(model, scene):
    """Straight-segment crossing count for the current display scene (cheap gate signal)."""
    import drawing
    pos = scene.get('positions') or {}
    parts = scene.get('participants') or {}
    segments = []
    for r in model.get('reactions') or []:
        rk = 'r:' + r['id']
        if rk not in pos:
            continue
        for mid in r.get('metabolites') or {}:
            key = parts.get((r['id'], mid))
            if key in pos:
                segments.append((rk, key))
    if len(segments) >= 1200:
        return None
    return int(drawing.crossings(pos, segments) or 0)


def _graft_axis_and_mid(substrates, products, park, index):
    """Place a midmarker so substrates and products leave on opposite sides when possible."""
    def centroid(pts):
        return np.mean(np.asarray(pts, float), axis=0)

    if substrates and products:
        s, p = centroid(substrates), centroid(products)
        axis = p - s
        nrm = float(np.hypot(*axis))
        if nrm < 1e-6:
            axis = np.array([1.0, 0.0])
            nrm = 1.0
        axis = axis / nrm
        mid = (s + p) / 2.0
        # Prefer sitting on the S→P chord; if S/P coincide spatially, offset slightly.
        if float(np.hypot(*(p - s))) < 1e-3:
            mid = s + np.array([-axis[1], axis[0]]) * 40.0
        return mid, axis, 'embedded'
    if substrates or products:
        pts = substrates or products
        c = centroid(pts)
        axis = np.array([1.0, 0.0]) if products else np.array([-1.0, 0.0])
        mid = c + axis * 55.0
        return mid, axis, 'embedded' if len(pts) >= 2 else 'new_component'
    mid = park + np.array([(index % 6) * 90.0, (index // 6) * 90.0])
    return mid, np.array([1.0, 0.0]), 'new_component'


def graft_new_reactions(data, model, new_reaction_ids, min_embed_anchors=2):
    """Add missing model reactions onto a filtered previous Escher map (pin-old / place-new).

    Matched geometry is left untouched.  New reactions sit on the chord between drawn
    substrate and product participants (opposite sides) when possible; otherwise they are
    parked as a new component.  Never skips a leftover — coverage stays complete.
    """
    from core import currency as model_currency, uid
    new_reaction_ids = list(new_reaction_ids or [])
    if not new_reaction_ids:
        return data, dict(placed=0, embedded=0, new_components=0, leftovers=[], midmarkers={})
    header, body = data[0], data[1]
    nodes = dict(body.get('nodes') or {})
    reactions = dict(body.get('reactions') or {})
    cur = set(model_currency(model))
    rx_by_id = {r['id']: r for r in model.get('reactions') or []}
    # Metabolite bigg_id -> node id (prefer primary).
    met_nodes = {}
    for nid, n in nodes.items():
        if n.get('node_type') != 'metabolite' or not n.get('bigg_id'):
            continue
        mid = n['bigg_id']
        if mid not in met_nodes or n.get('node_is_primary'):
            met_nodes[mid] = str(nid)
    xs = [float(n['x']) for n in nodes.values() if 'x' in n]
    ys = [float(n['y']) for n in nodes.values() if 'y' in n]
    if xs and ys:
        lo = np.array([min(xs), min(ys)], float)
        hi = np.array([max(xs), max(ys)], float)
    else:
        lo = np.array([0.0, 0.0]); hi = np.array([0.0, 0.0])
    park = np.array([hi[0] + NEW_COMPONENT_GAP, lo[1]], float)
    next_id = 0
    def fresh(prefix='n'):
        nonlocal next_id
        while True:
            nid = f'{prefix}{next_id}'
            next_id += 1
            if nid not in nodes and nid not in reactions:
                return nid
    embedded = new_components = 0
    grafted_mids = {}
    for i, rid in enumerate(new_reaction_ids):
        r = rx_by_id[rid]
        sign = -1 if r.get('upper_bound', 0) <= 0 and r.get('lower_bound', 0) < 0 else 1
        sub_pts, prod_pts = [], []
        for mid, coeff in r['metabolites'].items():
            if mid in cur:
                continue
            nid = met_nodes.get(mid)
            if not nid or nid not in nodes:
                continue
            xy = np.array([nodes[nid]['x'], nodes[nid]['y']], float)
            for role_c in participation_roles(coeff):
                if role_c * sign < 0:
                    sub_pts.append(xy)
                else:
                    prod_pts.append(xy)
        mid_xy, axis, kind = _graft_axis_and_mid(sub_pts, prod_pts, park, i)
        both_sides = bool(sub_pts) and bool(prod_pts)
        if both_sides and (len(sub_pts) + len(prod_pts)) >= max(1, int(min_embed_anchors)):
            embedded += 1
        else:
            new_components += 1
        mid_nid = fresh('r')
        nodes[mid_nid] = dict(node_type='midmarker', x=float(mid_xy[0]), y=float(mid_xy[1]))
        grafted_mids[rid] = mid_nid
        segs = {}
        sid = 0
        sub_i = prod_i = 0
        for mid, coeff in r['metabolites'].items():
            # Currency always gets a local side stub beside the midmarker (never a long
            # spoke to a distant primary — those dominate add-back crossings).
            reuse = (
                mid not in cur
                and mid in met_nodes
                and met_nodes[mid] in nodes
            )
            met_nid = None
            for role_c in participation_roles(coeff):
                role = 'substrate' if role_c * sign < 0 else 'product'
                if reuse:
                    met_nid = met_nodes[mid]
                elif met_nid is None:
                    met_nid = fresh('m')
                    side = -1.0 if role == 'substrate' else 1.0
                    idx = sub_i if role == 'substrate' else prod_i
                    if role == 'substrate':
                        sub_i += 1
                    else:
                        prod_i += 1
                    perp = np.array([-axis[1], axis[0]])
                    # Currency stubs sit closer; new core mets a bit further out.
                    along = 28.0 if mid in cur else (50.0 + 12.0 * (idx % 3))
                    xy = mid_xy + axis * (side * along) + perp * (16.0 * ((idx // 3) - 0.5))
                    nodes[met_nid] = dict(
                        node_type='metabolite', bigg_id=mid, name=mid,
                        node_is_primary=False if mid in cur else True,
                        x=float(xy[0]), y=float(xy[1]),
                        label_x=float(xy[0] + 10), label_y=float(xy[1] - 10))
                    if mid not in cur:
                        met_nodes[mid] = met_nid
                segs[str(sid)] = dict(from_node_id=mid_nid, to_node_id=met_nid, b1=None, b2=None)
                sid += 1
        reactions[fresh('rx')] = dict(
            bigg_id=rid, name=r.get('name') or rid,
            reversibility=bool(r.get('lower_bound', 0) < 0 < r.get('upper_bound', 0)),
            label_x=float(mid_xy[0] + 10), label_y=float(mid_xy[1] - 10),
            gene_reaction_rule=r.get('gene_reaction_rule', ''), genes=[],
            metabolites=[dict(bigg_id=m, coefficient=c * sign) for m, c in r['metabolites'].items()],
            segments=segs)
    # Expand canvas to include grafted nodes.
    canvas = dict(body.get('canvas') or {})
    xs = [float(n['x']) for n in nodes.values() if 'x' in n]
    ys = [float(n['y']) for n in nodes.values() if 'y' in n]
    if xs and ys:
        pad = 40.0
        canvas = dict(
            x=float(min(xs) - pad), y=float(min(ys) - pad),
            width=float(max(xs) - min(xs) + 2 * pad),
            height=float(max(ys) - min(ys) + 2 * pad))
    meta = dict(header.get('metadata') or {})
    src = dict(meta.get('source_reactions') or {})
    for rid in new_reaction_ids:
        r = rx_by_id[rid]
        src[rid] = dict(
            stoichiometry=r['metabolites'],
            lower_bound=r.get('lower_bound', 0),
            upper_bound=r.get('upper_bound', 0))
    keep_rx = {r['id'] for r in model.get('reactions') or []}
    src = {k: v for k, v in src.items() if k in keep_rx}
    for r in model.get('reactions') or []:
        src.setdefault(r['id'], dict(
            stoichiometry=r['metabolites'],
            lower_bound=r.get('lower_bound', 0),
            upper_bound=r.get('upper_bound', 0)))
    meta['source_reactions'] = src
    meta['model_hash'] = uid(model)
    meta['previous_layout'] = dict(meta.get('previous_layout') or {}, grafted=new_reaction_ids)
    header = dict(header, metadata=meta)
    out = [header, dict(body, nodes=nodes, reactions=reactions, canvas=canvas)]
    return out, dict(
        placed=len(new_reaction_ids), embedded=embedded, new_components=new_components,
        leftovers=list(new_reaction_ids), midmarkers=grafted_mids)


def _escher_segment_list(data):
    """Straight (from, to) node-id pairs for every reaction segment."""
    body = data[1]
    nodes = body.get('nodes') or {}
    segs = []
    for rx in (body.get('reactions') or {}).values():
        for s in (rx.get('segments') or {}).values():
            a, b = s.get('from_node_id'), s.get('to_node_id')
            if a in nodes and b in nodes:
                segs.append((str(a), str(b)))
    return segs


def escher_crossings(data):
    """Straight-segment crossing count on an Escher map (local relax gate)."""
    import drawing
    body = data[1]
    nodes = body.get('nodes') or {}
    pos = {nid: [float(n['x']), float(n['y'])] for nid, n in nodes.items() if 'x' in n and 'y' in n}
    segs = _escher_segment_list(data)
    if len(segs) >= 1200:
        return None
    return int(drawing.crossings(pos, segs) or 0)


def relax_grafted_midmarkers(data, grafted_mids, step=10.0, radii=(1, 2, 3, 4, 6)):
    """Nudge grafted midmarkers (and their private currency stubs) to cut crossings.

    Matched previous geometry is never moved.  Node moves go through ``map_edit``
    helpers so metabolite / midmarker labels stay attached.  Returns (data, report).
    """
    from map_edit import set_node_xy, translate_node, translate_reaction_label
    if not grafted_mids:
        return data, dict(moved=0, crossings_before=None, crossings_after=None)
    header, body = data[0], dict(data[1])
    nodes = {nid: dict(n) for nid, n in (body.get('nodes') or {}).items()}
    reactions = body.get('reactions') or {}
    body['nodes'] = nodes
    data = [header, body]
    before = escher_crossings(data)
    moved = 0
    offsets = []
    for r in radii:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            offsets.append((dx * r * step, dy * r * step))
    # Private nodes of a grafted reaction: metabolite ends that no other reaction uses.
    used = {}
    for rx in reactions.values():
        for s in (rx.get('segments') or {}).values():
            for nid in (s.get('from_node_id'), s.get('to_node_id')):
                used[str(nid)] = used.get(str(nid), 0) + 1

    def private_nodes(rid, mid_nid):
        priv = set()
        for rx in reactions.values():
            if rx.get('bigg_id') != rid:
                continue
            for s in (rx.get('segments') or {}).values():
                for nid in (s.get('from_node_id'), s.get('to_node_id')):
                    nid = str(nid)
                    if nid == mid_nid:
                        continue
                    n = nodes.get(nid) or {}
                    if n.get('node_type') == 'metabolite' and used.get(nid, 0) == 1:
                        priv.add(nid)
        return priv

    def restore_base(base, base_rx_labels):
        for nid, xy in base.items():
            set_node_xy(nodes[nid], xy[0], xy[1])
        for rx, lab in base_rx_labels:
            if 'label_x' in rx:
                rx['label_x'], rx['label_y'] = lab

    for rid, mid_nid in grafted_mids.items():
        mid_nid = str(mid_nid)
        if mid_nid not in nodes:
            continue
        priv = private_nodes(rid, mid_nid)
        base = {nid: [float(nodes[nid]['x']), float(nodes[nid]['y'])] for nid in [mid_nid, *priv]}
        base_rx_labels = [
            (rx, [float(rx['label_x']), float(rx['label_y'])])
            for rx in reactions.values()
            if rx.get('bigg_id') == rid and 'label_x' in rx
        ]
        best = escher_crossings(data)
        best_delta = (0.0, 0.0)
        improved = False
        for dx, dy in offsets:
            restore_base(base, base_rx_labels)
            for nid in base:
                translate_node(nodes[nid], dx, dy)
            for rx, _lab in base_rx_labels:
                translate_reaction_label(rx, dx, dy)
            cur = escher_crossings(data)
            if cur is not None and best is not None and cur < best:
                best = cur
                best_delta = (dx, dy)
                improved = True
        restore_base(base, base_rx_labels)
        if improved:
            dx, dy = best_delta
            for nid in base:
                translate_node(nodes[nid], dx, dy)
            for rx, _lab in base_rx_labels:
                translate_reaction_label(rx, dx, dy)
            moved += 1
    after = escher_crossings(data)
    return data, dict(
        moved=moved, crossings_before=before, crossings_after=after,
        kept=(before is not None and after is not None and after <= before))


def place_leftover_reactions(model, scene, anchors=None, cur=None, min_embed_anchors=2):
    """Pin-old / place-new: park every unanchored reaction so coverage stays complete.

    Reactions with ≥ ``min_embed_anchors`` drawn participant metabolites are embedded
    beside those anchors.  Otherwise the reaction (and any private metabolites) form a
    new component near the canvas — never dropped for lack of anchors.
    """
    from core import currency as model_currency
    if scene is None:
        return scene, dict(placed=0, embedded=0, new_components=0, leftovers=[])
    anchors = anchors or {}
    cur = set(cur if cur is not None else model_currency(model))
    scene = dict(scene, positions={k: list(v) for k, v in (scene.get('positions') or {}).items()})
    pos = scene['positions']
    parts = scene.setdefault('participants', {})
    copies = scene.setdefault('copies', {})
    labels = scene.setdefault('labels', {})
    leftovers = unanchored_reaction_ids(model, anchors)
    if not leftovers:
        # Still ensure every model reaction midmarker exists (defensive).
        leftovers = [r['id'] for r in model.get('reactions') or [] if ('r:' + r['id']) not in pos]
    if not leftovers:
        return scene, dict(placed=0, embedded=0, new_components=0, leftovers=[])

    drawn_met_keys = {}
    for key, mid in copies.items():
        if mid in cur:
            continue
        if key.startswith('side:'):
            continue
        drawn_met_keys.setdefault(mid, []).append(key)

    pts = [v for k, v in pos.items() if not k.startswith('side:')]
    if pts:
        arr = np.asarray(pts, float)
        lo, hi = arr.min(0), arr.max(0)
    else:
        lo, hi = np.array([0.0, 0.0]), np.array([0.0, 0.0])
    park_origin = np.array([hi[0] + NEW_COMPONENT_GAP, lo[1]], float)
    embedded = new_components = 0

    rx_by_id = {r['id']: r for r in model.get('reactions') or []}
    for i, rid in enumerate(leftovers):
        r = rx_by_id[rid]
        rk = 'r:' + rid
        core = [m for m in r['metabolites'] if m not in cur]
        anchor_xy = []
        for mid in core:
            keys = [k for k in drawn_met_keys.get(mid, []) if k in pos]
            if not keys:
                mk = 'm:' + mid
                if mk in pos:
                    keys = [mk]
            if keys:
                key = min(keys, key=lambda k: (0 if k.startswith('m:') else 1, k))
                anchor_xy.append((mid, key, np.asarray(pos[key], float)))
        if len(anchor_xy) >= max(1, int(min_embed_anchors)):
            target = np.mean([p for _, _, p in anchor_xy], axis=0)
            direction = np.array([0.0, 1.0])
            if len(anchor_xy) >= 2:
                delta = anchor_xy[0][2] - anchor_xy[1][2]
                n = float(np.hypot(*delta))
                if n > 1e-6:
                    direction = np.array([-delta[1], delta[0]], float) / n
            goal = target + direction * 55.0
            embedded += 1
        elif anchor_xy:
            goal = anchor_xy[0][2] + np.array([0.0, 70.0])
            new_components += 1
        else:
            goal = park_origin + np.array([(i % 6) * 90.0, (i // 6) * 90.0])
            new_components += 1
        old = np.asarray(pos.get(rk, goal), float)
        pos[rk] = [float(goal[0]), float(goal[1])]
        labels[rk] = rid
        for j, mid in enumerate(r['metabolites']):
            key = parts.get((rid, mid))
            if key is None:
                if mid in cur:
                    key = 'side:' + rid + ':' + mid
                elif mid in drawn_met_keys and any(k in pos for k in drawn_met_keys[mid]):
                    key = next(k for k in drawn_met_keys[mid] if k in pos)
                else:
                    key = 'm:' + mid
                parts[rid, mid] = key
                copies[key] = mid
                labels.setdefault(key, mid)
            if key.startswith('side:'):
                pos[key] = [float(goal[0] + 20.0 * (j % 3 - 1)), float(goal[1] + 40.0)]
                continue
            if key in pos and mid in drawn_met_keys and key in drawn_met_keys.get(mid, []):
                # Already-drawn participant — attach only; do not move the pin.
                continue
            delta = np.asarray(pos.get(key, old + np.array([0.0, 80.0])), float) - old
            if float(np.hypot(*delta)) < 1e-6:
                delta = np.array([0.0, 80.0 + 15.0 * j])
            pos[key] = (goal + delta).tolist()
            drawn_met_keys.setdefault(mid, []).append(key)

    scene['positions'] = pos
    return scene, dict(
        placed=len(leftovers), embedded=embedded, new_components=new_components,
        leftovers=list(leftovers))
