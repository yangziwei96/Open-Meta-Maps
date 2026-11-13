"""Final composition: keep large curated modules intact; shelf the rest.

Large curated modules (area ≥ data-derived percentile, spatially clustered) mark visual
components that stay put — never re-tiled, split or deformed.  Related outside components
are pulled to anchors; independents are shelved in uniform rows with an aspect-aware side.
Empty-band squeeze (pipeline OPM_SQUEEZE) collapses finished-map corridors.  After the
final orthogonal pass, ``nudge_frontier_escher`` may pull free content-bbox tip glyphs
inward when in-repo gate metrics (re-checked after re-routing) do not worsen;
curated-structure and ``--previous`` pins stay fixed.

Shelf / squeeze ideas follow MetaCarto (CC BY 4.0; doi:10.64898/2026.09.19.752882);
independent reimplementation.  A/B: OPM_COMPOSE=off.
"""
from collections import defaultdict
import math

import numpy as np
from scipy.spatial import cKDTree

from core import chem, comp
import orientation as _orientation

# Tip step = one primary diameter; face band = marker radius (nodes whose disk
# touches the content bbox face are tips).
_TIP_PRIMARY_R = float(_orientation.layout_glyph_radius('primary'))
_TIP_SECONDARY_R = float(_orientation.layout_glyph_radius('secondary'))
_TIP_MARKER_R = float(_orientation.layout_glyph_radius('marker'))

MAIN_MIN = 20
FAR_FACTOR = 0.5
FAR_MIN = 700.0        # legacy fixed tolerance (OPM_COMPOSE_LONG=off)
TAU = 3.5              # a related component is a relocation candidate when its distance to its anchors exceeds TAU x the typical reaction spacing + FAR_FACTOR x its own size
import os
LONG = os.environ.get('OPM_COMPOSE_LONG', 'on') == 'on'
CLEAR = 110.0
SHELF_GAP = 120.0
ROW_WIDTH_MIN = 2000.0  # floor only; actual row width follows main-network size and target aspect
TARGET_ASPECT = 1.55   # target width/height for shelf + packers (centre of 4:3–16:9)
ASPECT_MIN = 4.0 / 3.0
ASPECT_MAX = 16.0 / 9.0
SQUEEZE_K = 1.75
# Collapse empty corridors even when that leaves the aspect band, if an empty run
# is at least this fraction of the content bbox on that axis (fake landscape from voids).
SQUEEZE_FORCE_EMPTY_FRAC = 0.08
# Flush shelf standoff (was 400 px and created a huge empty corridor beside main).
SHELF_STANDOFF = SHELF_GAP
LINK_COST = 1.8   # once aspect is in-band, prefer chemically linked tiles sitting adjacent
TILE_GAP = 120.0
LARGE_AREA_PCT = 75.0
CROSS_COST = 0.85  # among equal-compactness slots / widths, minimise estimated link crossings
NODE_ASPECT_LOOPS = 14  # binary-search width iterations for pack_tiles_aspect
STRETCH_MAX = 1.35  # cap anisotropic pack stretch so aspect faking cannot explode the canvas
# Compact long transport / exchange arms (scene + escher): leave at most TARGET px.
ARM_MAX = 220.0
ARM_TARGET = 150.0
# Pocket pack: search empty cells inside/flush with the main AABB before external shelving.
POCKET_CELL_FRAC = 0.55
POCKET_PAD_CELLS = 1


def components(model, scene, cur):
    """Connected visual components: reaction ids -> dict(reactions, keys)."""
    parts = scene['participants']
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for r in model['reactions']:
        rid = r['id']
        find('r:' + rid)
        for mid in r['metabolites']:
            key = parts[rid, mid]
            if key.startswith('side:'):
                continue
            parent[find(key)] = find('r:' + rid)
    groups = defaultdict(lambda: dict(reactions=[], keys=set()))
    for r in model['reactions']:
        rid = r['id']
        g = groups[find('r:' + rid)]
        g['reactions'].append(rid)
        g['keys'].add('r:' + rid)
        for mid in r['metabolites']:
            g['keys'].add(parts[rid, mid])
    return list(groups.values())


def _free_spot(local, target, tree, clear=CLEAR, tries=260):
    if tree is None:
        return target
    for i in range(tries):
        r = 0.0 if i == 0 else 45.0 * np.sqrt(i)
        a = i * 2.399963
        d = target + r * np.array([np.cos(a), np.sin(a)])
        if float(tree.query(local + d)[0].min()) >= clear:
            return d
    return None


def _aligned_spot(local, target, anchors, size, tree, centre):
    """Free place for a component next to its anchors.  Candidates: free spots on the 8 compass sides of the anchor cluster (distance = half the component size + a gap)
    plus the plain nearest free spot; score = distance to the anchor cluster + a bonus for sharing a centre line (x or y) with the nearest anchor, so the module lines up
    with what it belongs to instead of sitting at a random angle."""
    nearest = anchors[int(np.argmin(np.hypot(*(anchors - centre).T)))]
    r = size / 2.0 + 160.0
    cands = []
    plain = _free_spot(local, target, tree)
    if plain is not None:
        cands.append(plain)
    for k in range(8):
        a = k * np.pi / 4
        d = _free_spot(local, target + r * np.array([np.cos(a), np.sin(a)]), tree)
        if d is not None:
            cands.append(d)
    if not cands:
        return None

    def score(c):
        dx, dy = abs(c[0] - nearest[0]), abs(c[1] - nearest[1])
        return float(np.hypot(*(c - target))) + 0.6 * min(dx, dy)
    return min(cands, key=score)


def compose(model, scene, blocks, cur):
    """-> (new scene, report).  Moves small components; `scene` is not modified."""
    scene = dict(scene, positions={k: list(v) for k, v in scene['positions'].items()})
    pos, parts = scene['positions'], scene['participants']
    comps = components(model, scene, cur)
    rx = {r['id']: r for r in model['reactions']}
    metab_users = defaultdict(set)         # core metabolite -> reaction ids
    for r in model['reactions']:
        for mid in r['metabolites']:
            if mid not in cur:
                metab_users[mid].add(r['id'])
    comp_of = {rid: i for i, c in enumerate(comps) for rid in c['reactions']}
    tiles = build_tiles(model, scene, blocks, cur)
    large_ids = classify_large_tiles(tiles)
    # Large curated tiles are reported for diagnostics.  The immovable "main" set stays the
    # largest visual components (MAIN_MIN) so related/independent classification matches the
    # previous main behaviour — expanding main with every large tile caused extra related
    # pulls and crossing/overlap regressions on iIT341.
    report = dict(components=len(comps), tiles=len(tiles),
                  curated_tiles=sum(t['kind'] == 'curated' for t in tiles),
                  auto_tiles=sum(t['kind'] == 'auto' for t in tiles),
                  large_tiles=len(large_ids), small_tiles=len(tiles) - len(large_ids),
                  large_area_pct=LARGE_AREA_PCT,
                  moved_related=0, arranged_independent=0, kept_near=0, failed=0, gathered=0, moves=[])
    main = [i for i, c in enumerate(comps) if len(c['reactions']) >= MAIN_MIN]
    # Freeze EVERY visual component that touches a same-map rigid backbone reaction.
    # Landscape reflow used to treat 2-reaction transport fragments as movable and
    # skyline-pack them away from the rest of the shared source map.
    rigid_rx = {rid for b in (blocks or []) if b.get('same_map_rigid') for rid in (b.get('reaction_ids') or [])}
    if rigid_rx:
        for i, c in enumerate(comps):
            if i in main:
                continue
            if any(rid in rigid_rx for rid in c['reactions']):
                main.append(i)
        report['same_map_rigid_reactions'] = len(rigid_rx)
    report['main'] = len(main)
    if not main:
        scene['composition'] = report
        return scene, report
    movable_idx = {i for i in range(len(comps)) if i not in set(main)}
    rpts = np.array([pos[k] for k in pos if k.startswith('r:')], float)
    spacing = float(np.median(cKDTree(rpts).query(rpts, k=2)[0][:, 1])) if len(rpts) > 2 else 100.0
    report['spacing_px'] = round(spacing, 1)

    def tree_without(skip_keys):
        pts = [pos[k] for k in pos if k not in skip_keys]
        return cKDTree(np.array(pts, float)) if pts else None
    cur_keys_of = {i: {k for k in comps[i]['keys'] if k in pos} for i in range(len(comps))}
    # classify the movable components: a component is "related" only when it shares a core metabolite with a SETTLED component (the main network, or a
    # component that has already been brought next to it); components whose only partners are other stray components are settled in later rounds, and those
    # that never reach the main network are independent
    settled = set(main)
    related, independent = [], []
    pending = sorted(movable_idx, key=lambda i: (-len(comps[i]['reactions']), comps[i]['reactions'][0]))
    anchors_of = {}
    for i in pending:
        anchors_of[i] = {(mid, u) for rid in comps[i]['reactions'] for mid in rx[rid]['metabolites'] if mid not in cur for u in metab_users[mid] if comp_of[u] != i}
    order_rounds = []
    changed = True
    while changed and pending:
        changed = False
        for i in list(pending):
            if any(comp_of[u] in settled for _, u in anchors_of[i]):
                order_rounds.append(i)
                settled.add(i)
                pending.remove(i)
                changed = True
    rank = {i: -1 for i in main}
    rank.update({i: n for n, i in enumerate(order_rounds)})
    related = [(i, {(m, u) for m, u in anchors_of[i] if comp_of[u] in rank and rank[comp_of[u]] < rank[i]}) for i in order_rounds]
    independent = [(i, anchors_of[i]) for i in pending]
    # one occupancy tree for everything that is currently placed; updated as components move
    placed_pts = {k: list(v) for k, v in pos.items() if not k.startswith('side:')}

    def build_tree(exclude):
        pts = np.array([v for k, v in placed_pts.items() if k not in exclude], float)
        return cKDTree(pts) if len(pts) else None

    def move(i, delta):
        for k in cur_keys_of[i]:
            for table in (pos, placed_pts):
                if k in table:
                    table[k] = [table[k][0] + float(delta[0]), table[k][1] + float(delta[1])]
    # ---- related components: next to their anchors
    for i, anchors in related:
        keys = cur_keys_of[i]
        local = np.array([pos[k] for k in keys], float)
        size = float(np.hypot(*(local.max(0) - local.min(0))))
        centre = local.mean(0)
        anchor_pts = np.array([pos['r:' + u] for _, u in anchors if 'r:' + u in pos], float)
        if not len(anchor_pts):
            continue
        # the anchors that are nearest to the component decide where it goes
        d = np.hypot(*(anchor_pts - centre).T)
        near = anchor_pts[np.argsort(d)[:max(1, min(6, len(anchor_pts)))]]
        target_c = near.mean(0)
        dist_now = float(np.hypot(*(centre - target_c)))
        limit = (FAR_FACTOR * size + TAU * spacing) if LONG else (1.5 * size + FAR_MIN)
        if dist_now <= limit:
            report['kept_near'] += 1
            continue
        tree = build_tree(keys)
        if LONG:
            spot = _aligned_spot(local - centre, target_c, near, size, tree, centre)
        else:
            spot = _free_spot(local - centre, target_c, tree)
        if spot is None or (LONG and float(np.hypot(*(spot - target_c))) > 0.7 * dist_now):
            report['failed'] += 1          # no free place clearly closer to the anchors
            continue
        # Prefer east/west attachments when the current content bbox is already tall, so
        # related pulls do not grow a portrait column that the landscape packer then loses.
        cur_asp, _ = _scene_aspect(placed_pts)
        if cur_asp < ASPECT_MIN and spot is not None:
            ew = []
            for sign in (-1.0, 1.0):
                cand = _free_spot(local - centre, target_c + np.array([sign * (size / 2.0 + 180.0), 0.0]), tree)
                if cand is not None:
                    ew.append(cand)
            if ew:
                spot = min(ew, key=lambda c: float(np.hypot(*(c - target_c))))
        delta = spot - centre
        move(i, delta)
        report['moved_related'] += 1
        if len(report['moves']) < 400:
            report['moves'].append(dict(kind='related', reactions=comps[i]['reactions'][:4], moved_px=round(float(np.hypot(*delta)), 1), anchors=sorted({u for _, u in anchors})[:4]))
    # ---- independent components: fill empty pockets beside/inside main, then flush shelf
    if independent:
        main_pts = np.array([placed_pts[k] for i in main for k in cur_keys_of[i] if k in placed_pts], float)
        lo, hi = main_pts.min(0), main_pts.max(0)
        W = max(float(hi[0] - lo[0]), ROW_WIDTH_MIN)
        order = sorted(independent, key=lambda t: (_compartment(comps[t[0]], rx), -len(comps[t[0]]['reactions']), comps[t[0]]['reactions'][0]))
        boxes = []
        labels = scene.get('labels', {})
        try:
            import drawing
            text_w = lambda s: float(drawing.FONT.getlength(s))
        except Exception:
            text_w = lambda s: 6.5 * len(s)
        pads = {}
        for i, _ in order:
            ks = [k for k in cur_keys_of[i] if k in pos]
            arr = np.array([pos[k] for k in ks], float)
            # the visible rectangle: glyph points plus the widest label (a label sits beside its glyph) and a margin for the curves
            lw = min(max([text_w(str(labels.get(k, ''))) for k in ks] + [0.0]), 220.0)
            pads[i] = lw
            boxes.append((i, arr.min(0), arr.max(0) - arr.min(0) + np.array([60.0 + lw, 60.0])))
        # Pocket-pack into empty space flush with the main AABB before any external shelf.
        prefer_c = main_pts.mean(0)
        pocket_boxes = sorted(
            [(i, float(size[0]), float(size[1])) for i, _bmin, size in boxes],
            key=lambda t: (-t[1] * t[2], t[0]))
        pocket_placed, pocket_rep = pack_into_main_pockets(
            pocket_boxes, main_pts, spacing, prefer_c=prefer_c)
        report['pocket_pack'] = pocket_rep
        remaining = []
        for i, bmin, size in boxes:
            if i in pocket_placed:
                origin = pocket_placed[i]
                # bmin is the component's current min corner; move so min lands on origin.
                delta = np.array([origin[0], origin[1]], float) - bmin
                move(i, delta)
                report['arranged_independent'] += 1
                # Keep occupancy current for later shelf checks.
                for k in cur_keys_of[i]:
                    if k in pos:
                        placed_pts[k] = list(pos[k])
            else:
                remaining.append((i, bmin, size))
        boxes = remaining
        if not boxes:
            report['shelf'] = dict(side='pocket', rows=0, width=0, height=0,
                                   components=0, pocket=pocket_rep.get('placed', 0))
            report['shelf_check'] = dict(rect_overlaps=0, overlaps_with_main_network=0)
        # Binary-search shelf row width so the combined main+shelf frame lands in 4:3–16:9.
        total_area = sum(b[2][0] * b[2][1] for b in boxes) if boxes else 0.0
        widest_box = (max(b[2][0] for b in boxes) + SHELF_GAP) if boxes else SHELF_GAP
        main_w, main_h = float(hi[0] - lo[0]), float(hi[1] - lo[1])

        def shelf_layout(row_w):
            x = y = 0.0
            row_h = 0.0
            layout = []
            prev_group = None
            for i, bmin, size in boxes:
                g = _compartment(comps[i], rx)
                if x > 0 and (x + size[0] > row_w or (prev_group is not None and g != prev_group and x > row_w * 0.5)):
                    x, y, row_h = 0.0, y + row_h + SHELF_GAP, 0.0
                layout.append((i, bmin, np.array([x, y])))
                x += size[0] + SHELF_GAP
                row_h = max(row_h, size[1])
                prev_group = g
            block_w, block_h = float(row_w), float(y + row_h)
            return layout, block_w, block_h, row_h

        def combined_options(block_w, block_h):
            g = SHELF_STANDOFF
            return {
                'below': (lo[0], hi[1] + g, max(main_w, block_w) / max(main_h + block_h + g, 1.0)),
                'above': (lo[0], lo[1] - g - block_h, max(main_w, block_w) / max(main_h + block_h + g, 1.0)),
                'right': (hi[0] + g, lo[1], (main_w + block_w + g) / max(main_h, block_h, 1.0)),
                'left': (lo[0] - g - block_w, lo[1], (main_w + block_w + g) / max(main_h, block_h, 1.0)),
            }

        if boxes:
            def shelf_score(row_w):
                layout, block_w, block_h, row_h = shelf_layout(row_w)
                options = combined_options(block_w, block_h)
                # Prefer compact combined bbox among in-band sides, then aspect near TARGET.
                def side_key(s):
                    asp = options[s][2]
                    in_band = ASPECT_MIN - 1e-6 <= asp <= ASPECT_MAX + 1e-6
                    ox, oy, _ = options[s]
                    comb_w = max(hi[0], ox + block_w) - min(lo[0], ox)
                    comb_h = max(hi[1], oy + block_h) - min(lo[1], oy)
                    return (0 if in_band else 1,
                            comb_w * comb_h,
                            abs(math.log(max(asp, 1e-9) / TARGET_ASPECT)))
                side = min(options, key=side_key)
                aspect = options[side][2]
                in_band = ASPECT_MIN - 1e-6 <= aspect <= ASPECT_MAX + 1e-6
                return (0 if in_band else 1,
                        abs(math.log(max(aspect, 1e-9) / TARGET_ASPECT)),
                        block_w * block_h,
                        side, layout, block_w, block_h, row_h, aspect)

            lo_w = max(widest_box, ROW_WIDTH_MIN * 0.5)
            hi_w = max(lo_w * 1.05, math.sqrt(max(total_area * ASPECT_MAX, 1.0)) * 1.6,
                       math.sqrt(max(total_area * TARGET_ASPECT, 1.0)) * 1.35, float(W))
            best_shelf = None
            a, b = lo_w, hi_w
            probes = [lo_w, hi_w, math.sqrt(max(total_area * TARGET_ASPECT, 1.0)),
                      math.sqrt(max(total_area * ASPECT_MIN, 1.0)),
                      math.sqrt(max(total_area * ASPECT_MAX, 1.0))]
            for _ in range(NODE_ASPECT_LOOPS):
                mid = 0.5 * (a + b)
                probes.append(mid)
                sc = shelf_score(mid)
                aspect = sc[-1]
                if aspect < TARGET_ASPECT:
                    a = mid
                else:
                    b = mid
                if best_shelf is None or sc[:3] < best_shelf[:3]:
                    best_shelf = sc
            for tw in probes:
                sc = shelf_score(max(tw, lo_w))
                if best_shelf is None or sc[:3] < best_shelf[:3]:
                    best_shelf = sc
            _in, _log, _area, side, layout, block_w, block_h, row_h, _asp = best_shelf
            options = combined_options(block_w, block_h)
            ox, oy = options[side][0], options[side][1]
            for i, bmin, off in layout:
                ks = [k for k in cur_keys_of[i] if k in pos]
                arr = np.array([pos[k] for k in ks], float)
                delta = np.array([ox, oy]) + off - arr.min(0) + np.array([30.0 + pads.get(i, 0.0) / 2.0, 30.0])
                move(i, delta)
                report['arranged_independent'] += 1
            rects = []
            for i, _, _ in layout:
                arr = np.array([pos[k] for k in cur_keys_of[i] if k in pos], float)
                rects.append((arr.min(0)[0] - 30, arr.min(0)[1] - 30,
                              arr.max(0)[0] + pads.get(i, 0.0) + 30, arr.max(0)[1] + 30))
            overlaps = sum(1 for a in range(len(rects)) for b in range(a + 1, len(rects))
                           if rects[a][0] < rects[b][2] and rects[b][0] < rects[a][2]
                           and rects[a][1] < rects[b][3] and rects[b][1] < rects[a][3])
            main_rect = (lo[0], lo[1], hi[0], hi[1])
            with_main = sum(1 for r in rects if r[0] < main_rect[2] and main_rect[0] < r[2]
                            and r[1] < main_rect[3] and main_rect[1] < r[3])
            report['shelf_check'] = dict(rect_overlaps=overlaps, overlaps_with_main_network=with_main)
            report['shelf'] = dict(
                side=side, rows=int(round(block_h / max(row_h + SHELF_GAP, 1))),
                width=round(block_w), height=round(block_h), components=len(layout),
                pocket=pocket_rep.get('placed', 0))
    report['gathered'] = report['moved_related']
    # After related/independent moves: pull true exterior stubs into free regions
    # of the main AABB. Keep this mild — aggressive teleports raise crossings and
    # split curated structures (the initial pocket_pack already places most tiles).
    iter_reps = []
    prev_ids = None
    for _it in range(2):
        main_pts = np.array(
            [placed_pts[k] for i in main for k in cur_keys_of[i] if k in placed_pts], float)
        if len(main_pts) < 2:
            break
        mlo, mhi = main_pts.min(0), main_pts.max(0)
        outliers = []
        for i in movable_idx:
            ks = [k for k in cur_keys_of[i] if k in placed_pts]
            if len(ks) < 1:
                continue
            arr = np.array([placed_pts[k] for k in ks], float)
            clo, chi = arr.min(0), arr.max(0)
            overlap_w = max(0.0, min(chi[0], mhi[0]) - max(clo[0], mlo[0]))
            overlap_h = max(0.0, min(chi[1], mhi[1]) - max(clo[1], mlo[1]))
            area = max(float((chi[0] - clo[0]) * (chi[1] - clo[1])), 1.0)
            n_rx_i = len(comps[i]['reactions'])
            # Only small stubs that barely touch main (or sit in the outer 8% margin).
            centre = 0.5 * (clo + chi)
            core_lo = mlo + 0.08 * (mhi - mlo)
            core_hi = mhi - 0.08 * (mhi - mlo)
            outside_core = (centre[0] < core_lo[0] or centre[0] > core_hi[0]
                            or centre[1] < core_lo[1] or centre[1] > core_hi[1])
            barely = overlap_w * overlap_h < 0.20 * area
            if not (barely or (n_rx_i <= 3 and outside_core and overlap_w * overlap_h < 0.55 * area)):
                continue
            bw = float(max(chi[0] - clo[0], 40.0) + 60.0)
            bh = float(max(chi[1] - clo[1], 40.0) + 60.0)
            outliers.append((i, bw, bh, clo.copy()))
        if not outliers:
            break
        boxes = sorted([(i, bw, bh) for i, bw, bh, _ in outliers],
                       key=lambda t: (-t[1] * t[2], t[0]))
        placed, prep = pack_into_main_pockets(
            boxes, main_pts, spacing, prefer_c=main_pts.mean(0))
        ids = frozenset(placed)
        if not placed or ids == prev_ids:
            iter_reps.append(dict(placed=len(placed), missed=len(boxes) - len(placed),
                                  **{k: v for k, v in prep.items()
                                     if k not in ('placed', 'missed')}))
            break
        prev_ids = ids
        moved_px = 0.0
        accepted = 0
        prefer = main_pts.mean(0)
        for i, bw, bh, clo in outliers:
            if i not in placed:
                continue
            origin = placed[i]
            old_c = clo + 0.5 * np.array([bw, bh], float)
            new_c = np.array([origin[0] + 0.5 * bw, origin[1] + 0.5 * bh], float)
            if float(np.hypot(*(new_c - prefer))) > float(np.hypot(*(old_c - prefer))) + 40.0:
                continue
            delta = np.array([origin[0], origin[1]], float) - clo
            dlen = float(np.hypot(*delta))
            # Cap single teleports — multi-thousand-px hops create spoke crossings.
            if dlen < 1.0 or dlen > 1200.0:
                continue
            moved_px += dlen
            move(i, delta)
            accepted += 1
            report['arranged_independent'] += 1
        iter_reps.append(dict(placed=accepted, missed=len(boxes) - accepted,
                              moved_px=round(moved_px, 1),
                              **{k: v for k, v in prep.items()
                                 if k not in ('placed', 'missed')}))
        if moved_px < 1.0 or accepted == 0:
            break
    if iter_reps:
        report['interior_repack'] = iter_reps
    # Final landscape reflow: translation-only skyline of rigid visual components so the
    # content bbox lands in 4:3–16:9 even when related-pulls left a tall column.
    scene, reflow = reflow_landscape(model, scene, blocks, cur)
    report['landscape_reflow'] = reflow
    scene['composition'] = report
    return scene, report


def _scene_aspect(pos):
    pts = np.array([v for k, v in pos.items() if not k.startswith('side:')], float)
    if len(pts) < 2:
        return 1.0, pts
    lo, hi = pts.min(0), pts.max(0)
    return float((hi[0] - lo[0]) / max(hi[1] - lo[1], 1e-9)), pts


def _occupancy_grid(points, cell, pad_cells=POCKET_PAD_CELLS):
    """Binary occupancy grid covering ``points`` with a one-cell halo."""
    pts = np.asarray(points, float)
    if len(pts) < 1:
        return None
    cell = max(float(cell), 1.0)
    lo = pts.min(0) - pad_cells * cell
    hi = pts.max(0) + pad_cells * cell
    nx = max(int(math.ceil((hi[0] - lo[0]) / cell)) + 1, 2)
    ny = max(int(math.ceil((hi[1] - lo[1]) / cell)) + 1, 2)
    # Cap grid size for mid GEMs.
    while nx * ny > 250_000 and cell < 1e6:
        cell *= 1.5
        nx = max(int(math.ceil((hi[0] - lo[0]) / cell)) + 1, 2)
        ny = max(int(math.ceil((hi[1] - lo[1]) / cell)) + 1, 2)
    grid = np.zeros((ny, nx), dtype=np.uint8)
    for x, y in pts:
        ix = int((x - lo[0]) / cell)
        iy = int((y - lo[1]) / cell)
        if 0 <= ix < nx and 0 <= iy < ny:
            grid[iy, ix] = 1
            # Dilate one cell so glyphs keep clearance.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < nx and 0 <= jy < ny:
                        grid[jy, jx] = 1
    return dict(grid=grid, lo=lo, cell=cell, nx=nx, ny=ny)


def _mark_rect_occupied(occ, x0, y0, w, h):
    if occ is None:
        return
    grid, lo, cell = occ['grid'], occ['lo'], occ['cell']
    ix0 = max(0, int((x0 - lo[0]) / cell))
    iy0 = max(0, int((y0 - lo[1]) / cell))
    ix1 = min(occ['nx'], int(math.ceil((x0 + w - lo[0]) / cell)) + 1)
    iy1 = min(occ['ny'], int(math.ceil((y0 + h - lo[1]) / cell)) + 1)
    grid[iy0:iy1, ix0:ix1] = 1


def _find_pocket_origin(occ, size_w, size_h, prefer_c, content_lo=None, content_hi=None,
                        prefer_void=False):
    """Top-left origin for a ``size_w×size_h`` box inside empty cells nearest ``prefer_c``.

    When ``content_lo/hi`` are set, prefer slots that pull the combined aspect toward
    the 4:3–16:9 band (widen a tall map, avoid growing height on an already-wide map).
    ``prefer_void`` ranks large empty neighbourhoods before tuck/centre (safer when
    packing many tiles on large maps).
    """
    if occ is None:
        return None
    grid, lo, cell = occ['grid'], occ['lo'], occ['cell']
    nw = max(1, int(math.ceil(size_w / cell)))
    nh = max(1, int(math.ceil(size_h / cell)))
    if nw >= occ['nx'] or nh >= occ['ny']:
        return None
    empty = (grid == 0).astype(np.int32)
    integ = np.pad(empty, ((1, 0), (1, 0)), mode='constant')
    integ = integ.cumsum(0).cumsum(1)
    best = None
    step = 1 if occ['nx'] * occ['ny'] < 40_000 else 2
    want_wide = False
    want_tall = False
    if content_lo is not None and content_hi is not None:
        cw = float(content_hi[0] - content_lo[0])
        ch = float(content_hi[1] - content_lo[1])
        asp = cw / max(ch, 1e-9)
        want_wide = asp < ASPECT_MIN - 1e-3
        want_tall = asp > ASPECT_MAX + 1e-3
    for iy in range(0, occ['ny'] - nh + 1, step):
        for ix in range(0, occ['nx'] - nw + 1, step):
            total = (integ[iy + nh, ix + nw] - integ[iy, ix + nw]
                     - integ[iy + nh, ix] + integ[iy, ix])
            if total < nw * nh:
                continue
            x0 = lo[0] + ix * cell
            y0 = lo[1] + iy * cell
            cx = x0 + size_w / 2.0
            cy = y0 + size_h / 2.0
            dist = (cx - prefer_c[0]) ** 2 + (cy - prefer_c[1]) ** 2
            aspect_pen = 0.0
            expand_pen = 0.0
            void_pen = 0.0
            if content_lo is not None and content_hi is not None:
                nlo0 = min(content_lo[0], x0)
                nhi0 = max(content_hi[0], x0 + size_w)
                nlo1 = min(content_lo[1], y0)
                nhi1 = max(content_hi[1], y0 + size_h)
                nasp = (nhi0 - nlo0) / max(nhi1 - nlo1, 1e-9)
                # Prefer pockets inside the current content AABB (fill voids, don't
                # inflate blank/empty by parking tiles in the outer search pad).
                expand_pen = (
                    max(0.0, content_lo[0] - x0) + max(0.0, (x0 + size_w) - content_hi[0])
                    + max(0.0, content_lo[1] - y0) + max(0.0, (y0 + size_h) - content_hi[1]))
                if want_wide:
                    # Prefer growing width (east/west of content), penalise growing height.
                    aspect_pen = max(0.0, ASPECT_MIN - nasp) * 1e6 + max(0.0, (y0 + size_h) - content_hi[1]) * 10
                    # Needed width growth is cheap; other expansion stays costly.
                    expand_pen = (
                        max(0.0, content_lo[1] - y0) + max(0.0, (y0 + size_h) - content_hi[1])
                        + 0.05 * (max(0.0, content_lo[0] - x0) + max(0.0, (x0 + size_w) - content_hi[0])))
                elif want_tall:
                    aspect_pen = max(0.0, nasp - ASPECT_MAX) * 1e6 + max(0.0, (x0 + size_w) - content_hi[0]) * 10
                    expand_pen = (
                        max(0.0, content_lo[0] - x0) + max(0.0, (x0 + size_w) - content_hi[0])
                        + 0.05 * (max(0.0, content_lo[1] - y0) + max(0.0, (y0 + size_h) - content_hi[1])))
                else:
                    aspect_pen = abs(math.log(max(nasp, 1e-9) / TARGET_ASPECT)) * 1e3
            ring = 2
            iy0 = max(0, iy - ring); iy1 = min(occ['ny'], iy + nh + ring)
            ix0 = max(0, ix - ring); ix1 = min(occ['nx'], ix + nw + ring)
            neigh = integ[iy1, ix1] - integ[iy0, ix1] - integ[iy1, ix0] + integ[iy0, ix0]
            void_pen = -float(neigh)
            if prefer_void:
                score = (aspect_pen, expand_pen, void_pen, dist, iy, ix)
            else:
                # Prefer slots tucked against occupied cells, then near the main
                # centre so TR floaters pull inward (small-map / few-tile packs).
                ring1 = 1
                iy0b = max(0, iy - ring1); iy1b = min(occ['ny'], iy + nh + ring1)
                ix0b = max(0, ix - ring1); ix1b = min(occ['nx'], ix + nw + ring1)
                area_cells = (iy1b - iy0b) * (ix1b - ix0b)
                empty_border = (integ[iy1b, ix1b] - integ[iy0b, ix1b]
                                - integ[iy1b, ix0b] + integ[iy0b, ix0b])
                border_empty = max(0, int(empty_border) - int(total))
                border_cells = max(1, area_cells - nw * nh)
                occupied_contact = border_cells - border_empty
                tuck_pen = -float(occupied_contact)
                score = (aspect_pen, expand_pen, tuck_pen, dist, iy, ix)
            if best is None or score < best[0]:
                best = (score, x0, y0)
    if best is None:
        return None
    return best[1], best[2]


def pack_into_main_pockets(boxes, main_pts, spacing, prefer_c=None, prefer_void=False):
    """Place boxes into empty pockets inside/flush with the main AABB.

    ``boxes``: list of ``(id, size_w, size_h)`` largest-first preferred.
    Returns ``{id: (x, y)}`` top-left placements for those that fit; others omitted.
    """
    if not boxes or len(main_pts) < 2:
        return {}, dict(placed=0, missed=len(boxes))
    if prefer_void is False and len(boxes) > 20:
        prefer_void = True
    cell = max(float(spacing) * POCKET_CELL_FRAC, 20.0)
    # Search primarily inside the main AABB; only a small flush strip outside so
    # small components fill the largest free regions before the canvas grows.
    pts = np.asarray(main_pts, float)
    lo, hi = pts.min(0), pts.max(0)
    pad = min(float(spacing) * 1.25 + SHELF_GAP, 0.08 * max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 1.0))
    # Seed occupancy with main points only; pocket search covers the padded AABB.
    seed = []
    for x, y in pts:
        seed.append((x, y))
    # Ring of sentinel points on the outer pad edge so pockets stay near main.
    for t in np.linspace(lo[0] - pad, hi[0] + pad, 12):
        seed.append((t, lo[1] - pad))
        seed.append((t, hi[1] + pad))
    for t in np.linspace(lo[1], hi[1], 8):
        seed.append((lo[0] - pad, t))
        seed.append((hi[0] + pad, t))
    occ = _occupancy_grid(seed, cell)
    # Clear interior of the padded frame is searchable; re-mark main glyphs.
    if occ is not None:
        occ['grid'][:] = 0
        for x, y in pts:
            ix = int((x - occ['lo'][0]) / occ['cell'])
            iy = int((y - occ['lo'][1]) / occ['cell'])
            if 0 <= ix < occ['nx'] and 0 <= iy < occ['ny']:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        jx, jy = ix + dx, iy + dy
                        if 0 <= jx < occ['nx'] and 0 <= jy < occ['ny']:
                            occ['grid'][jy, jx] = 1
    prefer_c = prefer_c if prefer_c is not None else pts.mean(0)
    content_lo, content_hi = lo.copy(), hi.copy()
    placed = {}
    for bid, bw, bh in boxes:
        origin = _find_pocket_origin(occ, bw, bh, prefer_c,
                                    content_lo=content_lo, content_hi=content_hi,
                                    prefer_void=prefer_void)
        if origin is None:
            continue
        placed[bid] = origin
        _mark_rect_occupied(occ, origin[0], origin[1], bw, bh)
        content_lo[0] = min(content_lo[0], origin[0])
        content_lo[1] = min(content_lo[1], origin[1])
        content_hi[0] = max(content_hi[0], origin[0] + bw)
        content_hi[1] = max(content_hi[1], origin[1] + bh)
    return placed, dict(placed=len(placed), missed=len(boxes) - len(placed),
                        cell=round(cell, 1), search_pad=round(pad, 1),
                        aspect_after=round(float((content_hi[0] - content_lo[0])
                                                 / max(content_hi[1] - content_lo[1], 1e-9)), 3))


def reflow_landscape(model, scene, blocks, cur):
    """Bring the scene content bbox into 4:3–16:9 without deforming curated glyphs.

    Safe tools only (never tear the main packed network):
      1. Rigid 90° page rotate when ``1/aspect`` lands in the hard band.
      2. Otherwise freeze MAIN visual components and skyline-pack only the
         non-main components beside them with binary-searched width so the
         combined frame is landscape (translation only).
    """
    scene = dict(scene, positions={k: list(v) for k, v in scene['positions'].items()})
    pos = scene['positions']
    aspect0, _ = _scene_aspect(pos)
    report = dict(applied=False, aspect_before=round(aspect0, 3), page_rotated_90=False)
    in_band0 = ASPECT_MIN - 1e-3 <= aspect0 <= ASPECT_MAX + 1e-3
    # When already in-band we still fall through to pocket-pack far non-main
    # components into empty space beside main (compactness), unless there is
    # nothing movable.

    # Portrait → page rotate only when the rotated aspect is in-band (avoid 0.27 → 3.7).
    if (not in_band0) and aspect0 < ASPECT_MIN - 1e-3:
        rot_aspect = 1.0 / max(aspect0, 1e-9)
        if ASPECT_MIN - 1e-3 <= rot_aspect <= ASPECT_MAX + 1e-3:
            for k, (x, y) in list(pos.items()):
                pos[k] = [float(y), float(-x)]
            xs = [v[0] for v in pos.values()]
            ys = [v[1] for v in pos.values()]
            ox, oy = min(xs), min(ys)
            for k in pos:
                pos[k] = [pos[k][0] - ox + 50.0, pos[k][1] - oy + 50.0]
            aspect1, _ = _scene_aspect(pos)
            report.update(applied=True, aspect_after=round(aspect1, 3),
                          reason='page_rotate_90', page_rotated_90=True)
            return scene, report

    comps = components(model, scene, cur)
    main = [i for i, c in enumerate(comps) if len(c['reactions']) >= MAIN_MIN]
    rigid_rx = {rid for b in (blocks or []) if b.get('same_map_rigid') for rid in (b.get('reaction_ids') or [])}
    if rigid_rx:
        for i, c in enumerate(comps):
            if i not in main and any(rid in rigid_rx for rid in c['reactions']):
                main.append(i)
    movable = [i for i in range(len(comps)) if i not in set(main)]
    if not main or not movable:
        report.update(aspect_after=round(aspect0, 3),
                      reason='already_in_band' if in_band0 else 'no_movable_or_main')
        return scene, report
    main_pts = np.array([pos[k] for i in main for k in comps[i]['keys']
                         if k in pos and not k.startswith('side:')], float)
    if len(main_pts) < 2:
        report.update(aspect_after=round(aspect0, 3), reason='main_empty')
        return scene, report
    mlo, mhi = main_pts.min(0), main_pts.max(0)
    main_w, main_h = float(mhi[0] - mlo[0]), float(mhi[1] - mlo[1])

    sizes, centres, keys_of = {}, {}, {}
    for i in movable:
        ks = [k for k in comps[i]['keys'] if k in pos and not k.startswith('side:')]
        if len(ks) < 1:
            continue
        arr = np.array([pos[k] for k in ks], float)
        lo, hi = arr.min(0), arr.max(0)
        sizes[i] = (float(max(hi[0] - lo[0], 40.0) + 60.0), float(max(hi[1] - lo[1], 40.0) + 60.0))
        centres[i] = arr.mean(0)
        keys_of[i] = ks
    if len(sizes) < 1:
        report.update(aspect_after=round(aspect0, 3), reason='no_sized_movable')
        return scene, report
    weights = island_link_weights(model, comps, cur)
    w_use = {k: v for k, v in weights.items() if k[0] in sizes and k[1] in sizes}
    parts = scene.get('participants') or {}
    rx = {r['id']: r for r in model['reactions']}
    origin_centres = {i: np.asarray(centres[i], float).copy() for i in centres}
    origin_pos = {k: list(v) for k, v in pos.items()}

    def _apply_placed(placed_map, origin_xy):
        for k, v in origin_pos.items():
            pos[k] = list(v)
        for i, (x, y) in placed_map.items():
            w, h = sizes[i]
            target = np.array([origin_xy[0] + x + w / 2.0, origin_xy[1] + y + h / 2.0], float)
            delta = target - origin_centres[i]
            move_keys = set(keys_of[i])
            for rid in comps[i]['reactions']:
                move_keys.add('r:' + rid)
                for mid in rx.get(rid, {}).get('metabolites', {}):
                    k = parts.get((rid, mid))
                    if k:
                        move_keys.add(k)
            for k in move_keys:
                if k in pos:
                    pos[k] = [origin_pos[k][0] + float(delta[0]), origin_pos[k][1] + float(delta[1])]
        return _scene_aspect(pos)[0]

    # Prefer pocket-packing movables into empty space flush with main before an external shelf.
    # When already in-band, only pull far outliers in (not every non-main tile) — packing
    # hundreds of tiles into pockets collapses large maps toward a square and raises empty/cross.
    prefer_c = main_pts.mean(0)
    span = max(float(np.hypot(*(mhi - mlo))), 1.0)
    if in_band0:
        pack_ids = [i for i in sizes
                    if float(np.hypot(*(centres[i] - prefer_c))) > 0.32 * span]
    else:
        pack_ids = list(sizes)
    pocket_boxes = sorted(
        [(i, sizes[i][0], sizes[i][1]) for i in pack_ids],
        key=lambda t: (-t[1] * t[2], t[0]))
    pocket_tl, pocket_rep = pack_into_main_pockets(
        pocket_boxes, main_pts, max(spacing_from_main(main_pts), 40.0),
        prefer_c=prefer_c, prefer_void=len(pocket_boxes) > 20)
    if in_band0 and not pocket_tl:
        report.update(aspect_after=round(aspect0, 3), reason='already_in_band',
                      pocket=pocket_rep)
        return scene, report
    if pocket_tl:
        pocket_keys = {i: list(keys_of[i]) for i in pocket_tl}
        pocket_placed = {i: (x, y) for i, (x, y) in pocket_tl.items()}
        # Absolute top-lefts with origin (0,0): _apply_placed targets top-left + size/2.
        aspect_p = _apply_placed(pocket_placed, (0.0, 0.0))
        leftover = [i for i in sizes if i not in pocket_tl]
        report['pocket'] = dict(pocket_rep, leftover=len(leftover))
        aspect_p_in = ASPECT_MIN - 1e-3 <= aspect_p <= ASPECT_MAX + 1e-3

        def _pocket_done(reason, left):
            report.update(applied=bool(pocket_tl), aspect_after=round(aspect_p, 3),
                          reason=reason,
                          pack=dict(side='pocket', movable=len(pocket_tl) - len(left),
                                    leftover=len(left), aspect=round(aspect_p, 3),
                                    in_band=ASPECT_MIN - 1e-3 <= aspect_p <= ASPECT_MAX + 1e-3))
            return scene, report

        # If every movable fitted in a pocket but the frame is still tall/narrow,
        # peel the largest pocketed tiles onto an east shelf so width can catch up
        # to 4:3 without reopening empty corridors inside main.
        if not leftover and not aspect_p_in and aspect_p < ASPECT_MIN - 1e-3:
            peel_order = sorted(pocket_tl, key=lambda i: (-sizes[i][1] * sizes[i][0], i))
            peel_n = max(1, (len(peel_order) + 2) // 3)
            leftover = peel_order[:peel_n]
            keep = {i for i in pocket_tl if i not in set(leftover)}
            pocket_placed = {i: pocket_tl[i] for i in keep}
            aspect_p = _apply_placed(pocket_placed, (0.0, 0.0))
            report['pocket'] = dict(pocket_rep, leftover=len(leftover), peeled=len(leftover))
            pocket_keys = {i: list(keys_of[i]) for i in keep}
        elif aspect_p_in and not leftover:
            return _pocket_done('main_pocket_pack', [])
        elif in_band0 and not leftover:
            # Started in-band; pocket-pack only (no external re-shelf that reopens gaps).
            return _pocket_done('main_pocket_pack', [])
        elif not leftover:
            return _pocket_done('main_pocket_pack', [])

        if leftover:
            sizes = {i: sizes[i] for i in leftover}
            centres = {i: centres[i] for i in leftover}
            keys_of = {i: keys_of[i] for i in leftover}
            w_use = {k: v for k, v in w_use.items() if k[0] in sizes and k[1] in sizes}
            main_pts = np.array(
                [pos[k] for i in main for k in comps[i]['keys']
                 if k in pos and not k.startswith('side:')]
                + [pos[k] for i, ks in pocket_keys.items() for k in ks if k in pos],
                float)
            if len(main_pts) >= 2:
                mlo, mhi = main_pts.min(0), main_pts.max(0)
                main_w, main_h = float(mhi[0] - mlo[0]), float(mhi[1] - mlo[1])
            origin_centres = {i: np.asarray(centres[i], float).copy() for i in centres}
            origin_pos = {k: list(v) for k, v in pos.items()}
        else:
            return _pocket_done('main_pocket_pack', [])

    # Binary-search shelf width; try right/left/below/above origins; pick best combined aspect.
    widest = max(w for w, h in sizes.values()) + SHELF_GAP
    area = sum((w + SHELF_GAP) * (h + SHELF_GAP) for w, h in sizes.values())
    order = sorted(sizes, key=lambda i: (-sizes[i][0] * sizes[i][1], i))
    best = None
    width_trials = []
    g = SHELF_STANDOFF
    for ar in (TARGET_ASPECT, ASPECT_MIN, ASPECT_MAX, 1.45):
        base = max(math.sqrt(max(area * ar, 1.0)), widest)
        for scale in (0.7, 0.85, 1.0, 1.2):
            width_trials.append(max(base * scale, widest))
    a, b = widest, max(width_trials)
    for _ in range(10):
        mid = 0.5 * (a + b)
        width_trials.append(mid)
        placed_try = skyline_pack(order, sizes, SHELF_GAP, mid, weights=w_use)
        bw, bh = skyline_extent(placed_try, sizes, SHELF_GAP)
        comb = (main_w + bw + g) / max(main_h, bh, 1.0)
        if comb < TARGET_ASPECT:
            a = mid
        else:
            b = mid
    seen = set()
    for tw in sorted(width_trials):
        key = round(tw, 1)
        if key in seen:
            continue
        seen.add(key)
        placed_try = skyline_pack(order, sizes, SHELF_GAP, tw, weights=w_use)
        bw, bh = skyline_extent(placed_try, sizes, SHELF_GAP)
        options = {
            'right': (mhi[0] + g, mlo[1], (main_w + bw + g) / max(main_h, bh, 1.0)),
            'left': (mlo[0] - g - bw, mlo[1], (main_w + bw + g) / max(main_h, bh, 1.0)),
            'below': (mlo[0], mhi[1] + g, max(main_w, bw) / max(main_h + bh + g, 1.0)),
            'above': (mlo[0], mlo[1] - g - bh, max(main_w, bw) / max(main_h + bh + g, 1.0)),
        }
        for side, (ox, oy, comb) in options.items():
            asp = _apply_placed(placed_try, (ox, oy))
            in_band_c = ASPECT_MIN - 1e-3 <= asp <= ASPECT_MAX + 1e-3
            # Prefer in-band, then smaller content area (compact), then aspect near target.
            pts_now = np.array([v for k, v in pos.items() if not k.startswith('side:')], float)
            area_now = float((pts_now[:, 0].max() - pts_now[:, 0].min())
                             * (pts_now[:, 1].max() - pts_now[:, 1].min())) if len(pts_now) else 1e18
            score = (0 if in_band_c else 1,
                     area_now,
                     abs(math.log(max(asp, 1e-9) / TARGET_ASPECT)),
                     _pack_link_cost(placed_try, sizes, SHELF_GAP, w_use))
            cand = dict(placed=placed_try, origin=(ox, oy), side=side, aspect=asp,
                        score=score, target_width=tw, in_band=in_band_c)
            if best is None or cand['score'] < best['score']:
                best = cand
    if best is None:
        report.update(aspect_after=round(aspect0, 3), reason='pack_failed')
        return scene, report
    aspect1 = _apply_placed(best['placed'], best['origin'])
    report.update(applied=True, aspect_after=round(aspect1, 3),
                  pack=dict(side=best['side'], target_width=round(best['target_width']),
                            aspect=round(aspect1, 3), in_band=best['in_band'],
                            movable=len(best['placed']),
                            pocket=pocket_rep.get('placed', 0) if pocket_tl else 0),
                  reason='main_frozen_shelf_reflow')
    return scene, report


def spacing_from_main(main_pts):
    pts = np.asarray(main_pts, float)
    if len(pts) < 3:
        return 100.0
    return float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])) or 100.0


def fit_canvas_to_content(data, pad=40.0):
    """Shrink ``data[1]['canvas']`` to the content bbox plus ``pad`` on each side.

    Translation-free: only the canvas frame changes, so map units stay absolute.
    """
    xs, ys = _coord_lists(data[1])
    if not xs:
        return dict(applied=False)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    before = dict(data[1].get('canvas') or {})
    data[1]['canvas'] = dict(
        x=x0 - pad, y=y0 - pad,
        width=(x1 - x0) + 2 * pad,
        height=(y1 - y0) + 2 * pad)
    return dict(applied=True, before=before, after=dict(data[1]['canvas']),
                content=[round(x1 - x0, 1), round(y1 - y0, 1)])


def _escher_content_nodes(body):
    return [(nid, n) for nid, n in (body.get('nodes') or {}).items()
            if 'x' in n and n.get('node_type') in ('metabolite', 'midmarker', 'multimarker')]


def _escher_content_bbox(body):
    nodes = _escher_content_nodes(body)
    if not nodes:
        return 0.0, 0.0, 0.0, 0.0
    xs = [n['x'] for _, n in nodes]
    ys = [n['y'] for _, n in nodes]
    return min(xs), min(ys), max(xs), max(ys)


def _content_aspect_ratio(body):
    """min(side)/max(side) of the content bbox (higher = closer to square)."""
    x0, y0, x1, y1 = _escher_content_bbox(body)
    w, h = abs(x1 - x0), abs(y1 - y0)
    if w <= 1e-12 or h <= 1e-12:
        return 1.0
    return float(min(w, h) / max(w, h))


def glyph_circle_overlapping_pairs(data):
    """Count metabolite/midmarker disk pairs that intersect (touching allowed).

    Radii match ``orientation`` primary / stub / marker sizes. Neighbour query only.
    """
    body = data[1] if isinstance(data, list) else data
    pts, rs = [], []
    for n in (body.get('nodes') or {}).values():
        t = n.get('node_type')
        if t == 'metabolite':
            r = _TIP_PRIMARY_R if n.get('node_is_primary') else _TIP_SECONDARY_R
        elif t == 'midmarker':
            r = _TIP_MARKER_R
        else:
            continue
        pts.append((float(n['x']), float(n['y'])))
        rs.append(float(r))
    if len(pts) < 2:
        return 0
    pts_a = np.asarray(pts, float)
    rs_a = np.asarray(rs, float)
    tree = cKDTree(pts_a)
    pairs = 0
    max_r = float(max(rs_a))
    for i, (p, ri) in enumerate(zip(pts_a, rs_a)):
        for j in tree.query_ball_point(p, float(ri + max_r) + 1e-9):
            if j <= i:
                continue
            if float(np.hypot(*(pts_a[i] - pts_a[j]))) < float(ri + rs_a[j]):
                pairs += 1
    return int(pairs)


def _curated_structure_node_ids(data):
    """Escher node ids that belong to curated-structure reactions (must stay put)."""
    curated_rx = set()
    for s in (data[0].get('metadata') or {}).get('expert_structures') or []:
        curated_rx.update(s.get('reactions') or s.get('reaction_ids') or [])
    if not curated_rx:
        return set()
    body = data[1]
    nodes = body.get('nodes') or {}
    out = set()
    for rxn in (body.get('reactions') or {}).values():
        if rxn.get('bigg_id') not in curated_rx:
            continue
        for seg in (rxn.get('segments') or {}).values():
            for end in ('from_node_id', 'to_node_id'):
                nid = seg.get(end)
                if nid is not None and nid in nodes:
                    out.add(nid)
    return out


def _previous_pinned_node_ids(data, previous_anchors):
    """Escher node ids locked by ``--previous`` ``r:`` / ``m:`` pins."""
    if not previous_anchors:
        return set()
    body = data[1]
    nodes = body.get('nodes') or {}
    out = set()
    mid_of_rx = {}
    mets_of = defaultdict(list)
    for nid, n in nodes.items():
        nt = n.get('node_type')
        if nt == 'metabolite' and n.get('bigg_id'):
            mets_of[n['bigg_id']].append(nid)
    for rxn in (body.get('reactions') or {}).values():
        rid = rxn.get('bigg_id')
        if not rid:
            continue
        for seg in (rxn.get('segments') or {}).values():
            for end in ('from_node_id', 'to_node_id'):
                nid = seg.get(end)
                n = nodes.get(nid) or {}
                if n.get('node_type') == 'midmarker':
                    mid_of_rx[rid] = nid
                    break
    for key in previous_anchors:
        sk = str(key)
        if sk.startswith('r:'):
            nid = mid_of_rx.get(sk[2:])
            if nid is not None:
                out.add(nid)
        elif sk.startswith('m:'):
            out.update(mets_of.get(sk[2:]) or [])
    return out


def _expand_move_with_multimarkers(body, move):
    """Keep midmarkers and their multimarkers moving together."""
    nodes = body.get('nodes') or {}
    out = set(move)
    changed = True
    while changed:
        changed = False
        for rxn in (body.get('reactions') or {}).values():
            for seg in (rxn.get('segments') or {}).values():
                a, b = seg.get('from_node_id'), seg.get('to_node_id')
                na, nb = nodes.get(a) or {}, nodes.get(b) or {}
                types = {na.get('node_type'), nb.get('node_type')}
                if types != {'midmarker', 'multimarker'}:
                    continue
                if a in out and b not in out:
                    out.add(b)
                    changed = True
                elif b in out and a not in out:
                    out.add(a)
                    changed = True
    return out


# Tip-nudge no-worse tolerance (same scale as official layout-benchmark gates).
_GATE_EPS = 1e-4


def relocation_gate_not_worse(before, after, eps=_GATE_EPS):
    """True when tip-nudge gate metrics do not worsen beyond ``eps``.

    Crossings / angular / ELD / overlaps ↓; PC / ortho / aspect_ratio ↑.
    Missing (None) fields are skipped.  Shared with ``layout_quality_evaluator``.
    """
    from layout_quality_evaluator import TIP_DIRS, metrics_not_worse
    return metrics_not_worse(before, after, TIP_DIRS, eps)


def _escher_gate_metrics(data, model, pathway=None):
    """Tip-nudge gates from compose-safe counters (no evaluation / gdMetriX).

    PC / ELD / angular from ``orthogonal_route.gate_metrics``; crossings from
    ``layout_quality.algorithm_crossings``; orthogonality from
    ``orthogonal_route.piece_axis_alignment`` on the same drawn pieces.
    """
    import layout_quality as lq
    import orthogonal_route as ortho
    gm = ortho.gate_metrics(data, pathway=pathway)
    return dict(
        crossings=int(lq.algorithm_crossings(data)[0]),
        participant_consistency=gm.get('participant_consistency'),
        angular_resolution=gm.get('angular_resolution'),
        edge_length_deviation=gm.get('edge_length_deviation'),
        edge_orthogonality=ortho.piece_axis_alignment(data, pathway=pathway),
        aspect_ratio=_content_aspect_ratio(data[1]),
        overlapping_pairs=glyph_circle_overlapping_pairs(data),
    )


def _apply_tip_move(tbody, move, dx, dy):
    """Translate ``move`` nodes; shift only the near-end handle of attached segments."""
    nodes = tbody.get('nodes') or {}
    moved_mids = {
        nid for nid in move
        if (nodes.get(nid) or {}).get('node_type') == 'midmarker'}
    for rxn in (tbody.get('reactions') or {}).values():
        segs = rxn.get('segments') or {}
        mid_of_rx = None
        for seg in segs.values():
            for end in ('from_node_id', 'to_node_id'):
                nid = seg.get(end)
                if (nodes.get(nid) or {}).get('node_type') == 'midmarker':
                    mid_of_rx = nid
        for seg in segs.values():
            a, b = seg.get('from_node_id'), seg.get('to_node_id')
            # Escher: b1 is the control near from_node; b2 near to_node.
            if a in move:
                pt = seg.get('b1')
                if pt:
                    pt['x'] = float(pt['x']) + dx
                    pt['y'] = float(pt['y']) + dy
            if b in move:
                pt = seg.get('b2')
                if pt:
                    pt['x'] = float(pt['x']) + dx
                    pt['y'] = float(pt['y']) + dy
        if mid_of_rx in moved_mids and 'label_x' in rxn:
            rxn['label_x'] = float(rxn['label_x']) + dx
            rxn['label_y'] = float(rxn['label_y']) + dy
    for nid in move:
        n = nodes[nid]
        n['x'] = float(n['x']) + dx
        n['y'] = float(n['y']) + dy
        if 'label_x' in n:
            n['label_x'] = float(n['label_x']) + dx
            n['label_y'] = float(n['label_y']) + dy


def _reactions_touching(tbody, move):
    """Reaction bigg_ids with a segment end in ``move``."""
    out = set()
    for rxn in (tbody.get('reactions') or {}).values():
        rid = rxn.get('bigg_id')
        if not rid:
            continue
        for seg in (rxn.get('segments') or {}).values():
            if seg.get('from_node_id') in move or seg.get('to_node_id') in move:
                out.add(rid)
                break
    return out


def nudge_frontier_escher(data, model, previous_anchors=None, skip_edges=(),
                          gem_path=None):
    """Pull free content-bbox tip glyphs inward when post-routing gates allow.

    Runs after the final orthogonal pass. Each candidate is scored against the
    original pre-nudge baseline (limits do not accumulate). Cheap area / overlap /
    aspect / crossing checks reject before re-routing; surviving trials re-route
    only reactions that touch moved nodes. Curated-structure and ``--previous``
    pins stay fixed. Tip step is one primary-glyph diameter; the face band is
    one marker radius (glyph disk touching the bbox face).
    """
    import copy as _copy
    import edge_style
    import layout_quality as lq
    import orthogonal_route as ortho
    if not isinstance(data[0], dict):
        data[0] = {}
    data[0].setdefault('metadata', {})
    body = data[1]
    x0, y0, x1, y1 = _escher_content_bbox(body)
    if x1 <= x0 or y1 <= y0:
        return dict(accepted=0, rejected=0, applied=False)
    step = 2.0 * _TIP_PRIMARY_R
    frozen = _curated_structure_node_ids(data) | _previous_pinned_node_ids(
        data, previous_anchors)
    asp0 = (x1 - x0) / max(y1 - y0, 1e-9)
    pathway = ortho.load_pathway(gem_path) if gem_path else None
    baseline = _escher_gate_metrics(data, model, pathway=pathway)
    accepted = rejected = 0
    face_delta = {
        'right': (-float(step), 0.0),
        'left': (float(step), 0.0),
        'top': (0.0, -float(step)),
        'bottom': (0.0, float(step)),
    }
    face_order = ('top', 'bottom', 'right', 'left') if asp0 <= TARGET_ASPECT \
        else ('right', 'left', 'top', 'bottom')
    # Nodes whose marker-sized disk touches a content-bbox face.
    edge_tol = _TIP_MARKER_R
    skip = list(skip_edges or ())
    while True:
        progressed = False
        x0, y0, x1, y1 = _escher_content_bbox(body)
        area0 = (x1 - x0) * (y1 - y0)
        for face in face_order:
            if face == 'right':
                tips = [nid for nid, n in _escher_content_nodes(body)
                        if n['x'] >= x1 - edge_tol]
            elif face == 'left':
                tips = [nid for nid, n in _escher_content_nodes(body)
                        if n['x'] <= x0 + edge_tol]
            elif face == 'top':
                tips = [nid for nid, n in _escher_content_nodes(body)
                        if n['y'] >= y1 - edge_tol]
            else:
                tips = [nid for nid, n in _escher_content_nodes(body)
                        if n['y'] <= y0 + edge_tol]
            tips = [nid for nid in tips if nid not in frozen]
            if not tips:
                continue
            trial = _copy.deepcopy(data)
            tbody = trial[1]
            move = _expand_move_with_multimarkers(tbody, set(tips))
            move = {nid for nid in move if nid not in frozen}
            if not move:
                continue
            dx, dy = face_delta[face]
            _apply_tip_move(tbody, move, dx, dy)
            tx0, ty0, tx1, ty1 = _escher_content_bbox(tbody)
            area1 = (tx1 - tx0) * (ty1 - ty0)
            if area1 >= area0:
                rejected += 1
                continue
            # Cheap rejects before re-routing (same directions as the full gate).
            if glyph_circle_overlapping_pairs(trial) > int(baseline['overlapping_pairs']):
                rejected += 1
                continue
            if float(_content_aspect_ratio(tbody)) + _GATE_EPS < float(baseline['aspect_ratio']):
                rejected += 1
                continue
            if int(lq.algorithm_crossings(trial)[0]) > int(baseline['crossings']):
                rejected += 1
                continue
            only = _reactions_touching(tbody, move)
            edge_style.apply(
                trial, skip, mode='orthogonal', only=only or None,
                gem_path=gem_path, pathway=pathway)
            gate1 = _escher_gate_metrics(trial, model, pathway=pathway)
            if not relocation_gate_not_worse(baseline, gate1):
                rejected += 1
                continue
            # Keep the trial's re-route metadata; do not route again.
            data[0], data[1] = trial[0], trial[1]
            body = data[1]
            accepted += 1
            progressed = True
            break
        if not progressed:
            break
    return dict(accepted=accepted, rejected=rejected, applied=accepted > 0,
                step=float(step), mode='tip_nudge')


def compact_empty_corridors_escher(data, max_gap=None, spacing=None):
    """Shorten extreme empty axis-aligned mid↔met edges on a finished Escher map.

    For a long empty corridor to a *shared* metabolite, translate the whole
    reaction unit (midmarker + private spoke metabolites) toward that shared
    pool — never yank the shared glyph, and never leave private heads behind.
    Private (degree-1) metabolites may move alone.
    """
    import curated_axes as ca
    body = data[1]
    nodes = body.get('nodes') or {}
    reactions = body.get('reactions') or {}
    # Metabolite node -> reaction bigg_ids that touch it.
    holders = defaultdict(set)
    rx_parts = {}  # rid -> (mid_id, [met_ids])
    for rx in reactions.values():
        rid = rx.get('bigg_id') or ''
        mid_id = None
        mets = []
        for seg in (rx.get('segments') or {}).values():
            for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                n = nodes.get(str(nid)) or nodes.get(nid) or {}
                sid = str(nid)
                if n.get('node_type') == 'midmarker':
                    mid_id = sid
                elif n.get('node_type') == 'metabolite' and 'x' in n:
                    mets.append(sid)
                    holders[sid].add(rid)
        if mid_id is not None:
            rx_parts[rid] = (mid_id, mets)
    edge_lens = []
    candidates = []  # (length, rid, mid_id, met_id, vertical)
    for rid, (mid_id, mets) in rx_parts.items():
        mid = nodes.get(mid_id) or {}
        if 'x' not in mid:
            continue
        for met_id in mets:
            met = nodes.get(met_id) or {}
            if 'x' not in met:
                continue
            dx = float(met['x'] - mid['x']); dy = float(met['y'] - mid['y'])
            length = float(math.hypot(dx, dy))
            if length < 1e-6:
                continue
            edge_lens.append(length)
            vertical, horizontal = ca._axis_aligned(dx, dy)
            if not (vertical or horizontal):
                continue
            candidates.append((length, rid, mid_id, met_id, vertical))
    if not candidates:
        return dict(applied=False, shortened=0)
    if max_gap is None:
        med = float(np.median(edge_lens)) if edge_lens else 150.0
        max_gap = max(med * 2.2, float(spacing or med) * 1.5, 180.0)
    # Extreme empty columns only — milder thresholds were tilting axis share and
    # splitting curated structures via follow-up arm pulls.
    min_len = float(max_gap) * 3.0
    candidates = [c for c in candidates if c[0] >= min_len]
    if not candidates:
        return dict(applied=False, shortened=0, max_gap=round(float(max_gap), 2),
                    min_len=round(min_len, 1))
    # Corridor emptiness uses all mid+met points.
    cloud = {str(nid): [float(n['x']), float(n['y'])]
             for nid, n in nodes.items()
             if n.get('node_type') in ('metabolite', 'midmarker') and 'x' in n}
    shortened = 0
    removed = 0.0
    moved = set()
    max_shorten = 2
    candidates.sort(reverse=True)
    for length, rid, mid_id, met_id, vertical in candidates:
        if shortened >= max_shorten:
            break
        mid = nodes.get(mid_id); met = nodes.get(met_id)
        if mid is None or met is None or 'x' not in mid or 'x' not in met:
            continue
        a = np.array([mid['x'], mid['y']], float)
        b = np.array([met['x'], met['y']], float)
        length = float(np.hypot(*(a - b)))
        if length < min_len:
            continue
        # Middle-span emptiness (ignore endpoint-local chemistry).
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        span = float((hi[1] - lo[1]) if vertical else (hi[0] - lo[0]))
        pad = max(span * 0.12, float(max_gap) * 0.5, 40.0)
        cx, cy = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        band = 18.0 + 0.5 * (abs(a[0] - b[0]) if vertical else abs(a[1] - b[1]))
        blocked = False
        for k, xy in cloud.items():
            if k in (mid_id, met_id):
                continue
            p = np.asarray(xy, float)
            if vertical:
                if abs(p[0] - cx) > band:
                    continue
                if lo[1] + pad < p[1] < hi[1] - pad:
                    blocked = True; break
            else:
                if abs(p[1] - cy) > band:
                    continue
                if lo[0] + pad < p[0] < hi[0] - pad:
                    blocked = True; break
        if blocked:
            continue
        shared = len(holders.get(met_id, ())) > 1
        if shared and not vertical:
            # Horizontal shared corridors are often inter-block metabolic links;
            # collapsing them splits structures and costs continuity.
            continue
        if shared:
            # Translate the whole reaction unit toward the shared pool: mid + every
            # other metabolite on this reaction (including shared _e/_p partners).
            # Companion EX/uptake reactions that only hold those moved partners are
            # dragged by the same delta so their spokes stay short.
            # Leave a little more than max_gap so the mid does not dive into the
            # dense chemistry around the shared pool (extra spoke crossings).
            target_len = max(float(max_gap) * 1.35, 320.0)
            target_len = min(target_len, length - 1.0)
            target = b + (a - b) * (target_len / length)
            delta = target - a
            move_ids = {mid_id}
            partner_shared = []
            for sid in rx_parts[rid][1]:
                if sid == met_id:
                    continue
                move_ids.add(sid)
                if len(holders.get(sid, ())) > 1:
                    partner_shared.append(sid)
            for sid in partner_shared:
                for other_rid in holders.get(sid, ()):
                    if other_rid == rid or other_rid not in rx_parts:
                        continue
                    omid, omets = rx_parts[other_rid]
                    # Simple EX/uptake companions only (≤3 metabolite heads).
                    if len(omets) > 3:
                        continue
                    move_ids.add(omid)
                    for o in omets:
                        if o != met_id:
                            move_ids.add(o)
            for sid in move_ids:
                n = nodes.get(sid)
                if n is None or 'x' not in n:
                    continue
                ox, oy = float(n['x']), float(n['y'])
                n['x'] = ox + float(delta[0]); n['y'] = oy + float(delta[1])
                if 'label_x' in n:
                    n['label_x'] = float(n.get('label_x', ox) + delta[0])
                    n['label_y'] = float(n.get('label_y', oy) + delta[1])
                cloud[sid] = [n['x'], n['y']]
                moved.add(sid)
            # Keep reaction labels on moved midmarkers.
            for r in reactions.values():
                if r.get('bigg_id') in (rid,) or (
                        r.get('bigg_id') and rx_parts.get(r.get('bigg_id'), (None,))[0] in move_ids):
                    mid_moved = rx_parts.get(r.get('bigg_id'), (None,))[0]
                    if mid_moved in move_ids and 'label_x' in r:
                        r['label_x'] = float(nodes[mid_moved]['x'])
                        r['label_y'] = float(nodes[mid_moved]['y'])
            removed += length - target_len
            shortened += 1
        else:
            # Private met: slide it toward the mid.
            target = a + (b - a) * (float(max_gap) / length)
            ox, oy = float(met['x']), float(met['y'])
            met['x'], met['y'] = float(target[0]), float(target[1])
            if 'label_x' in met:
                met['label_x'] = float(met.get('label_x', ox) + (met['x'] - ox))
                met['label_y'] = float(met.get('label_y', oy) + (met['y'] - oy))
            cloud[met_id] = [met['x'], met['y']]
            moved.add(met_id)
            removed += length - float(max_gap)
            shortened += 1
    for rx in reactions.values():
        for seg in (rx.get('segments') or {}).values():
            if str(seg.get('from_node_id')) in moved or str(seg.get('to_node_id')) in moved:
                seg['b1'] = None
                seg['b2'] = None
    return dict(applied=shortened > 0, shortened=shortened,
                length_removed=round(removed, 2), max_gap=round(float(max_gap), 2),
                min_len=round(min_len, 1), moved_nodes=len(moved))


def _link_key(a, b):
    return (a, b) if (str(a), a) <= (str(b), b) else (b, a)


def _seg_hits_rect(x0, y0, x1, y1, rx, ry, rw, rh):
    """True when the open segment (x0,y0)-(x1,y1) crosses the axis-aligned rectangle interior."""
    t_enter, t_leave = 0.0, 1.0
    for a0, a1, lo, hi in ((x0, x1, rx, rx + rw), (y0, y1, ry, ry + rh)):
        da = a1 - a0
        if abs(da) < 1e-12:
            if a0 < lo or a0 > hi:
                return False
            continue
        ta, tb = (lo - a0) / da, (hi - a0) / da
        if ta > tb:
            ta, tb = tb, ta
        t_enter = max(t_enter, ta)
        t_leave = min(t_leave, tb)
        if t_enter > t_leave:
            return False
    return t_leave > t_enter and t_leave > 1e-6 and t_enter < 1.0 - 1e-6


def island_link_weights(model, comps, cur):
    """Undirected weights between visual islands = shared non-currency metabolites."""
    rx = {r['id']: r for r in model['reactions']}
    users = defaultdict(set)
    for i, c in enumerate(comps):
        for rid in c['reactions']:
            for mid in rx[rid]['metabolites']:
                if mid not in cur:
                    users[mid].add(i)
    weights = defaultdict(float)
    for mid, islands in users.items():
        islands = sorted(islands)
        for a in range(len(islands)):
            for b in range(a + 1, len(islands)):
                weights[_link_key(islands[a], islands[b])] += 1.0
    return dict(weights)


def skyline_pack(order, sizes, gap, target_width, weights=None, link_cost=LINK_COST):
    """Pack axis-aligned tiles.  ``sizes`` maps id -> (w, h); returns id -> (x, y) top-left.

    Candidate x = 0 and every skyline segment start/end.  Compactness first (lowest y, then
    smallest x); when ``weights`` are given, among near-best compact slots prefer the one that
    shortens chemical links to already-placed partners (tile sits next to its strongest neighbour).
    """
    weights = weights or {}
    skyline = []  # (x0, x1, height)
    placed = {}

    def height_at(x0, x1):
        top = 0.0
        for a, b, h in skyline:
            if a < x1 and x0 < b:
                top = max(top, h)
        return top

    def centre(name, xy):
        w, h = sizes[name]
        return (xy[0] + w / 2.0, xy[1] + h / 2.0)

    # Prim-style order when links exist: seed with first of `order` (largest), then strongest link.
    remaining = [m for m in order if m in sizes]
    if not remaining:
        return placed
    seq = [remaining.pop(0)]
    remaining = set(remaining)
    while remaining:
        def pick(m):
            w = sum(weights.get(_link_key(m, p), 0.0) for p in seq)
            area = sizes[m][0] * sizes[m][1]
            return (w, area)
        nxt = max(remaining, key=pick)
        remaining.remove(nxt)
        seq.append(nxt)

    for name in seq:
        w0, h0 = sizes[name]
        w, h = float(w0) + gap, float(h0) + gap
        cand = {0.0} | {a for a, _, _ in skyline} | {b for _, b, _ in skyline}
        # Prefer slots flush with already-placed chemical partners (right / left-aligned).
        for p, pxy in placed.items():
            if weights.get(_link_key(name, p), 0.0) <= 0:
                continue
            pw, _ph = sizes[p]
            cand.add(pxy[0])
            cand.add(pxy[0] + pw + gap)
            cand.add(max(0.0, pxy[0] + pw + gap - w))
        candidates = sorted(cand)
        scored = []
        for x in candidates:
            if x + w > target_width + 1e-9 and x > 0.0:
                continue
            y = height_at(x, x + w)
            scored.append((x, y))
        if not scored:
            x, y = 0.0, height_at(0.0, w)
            scored = [(x, y)]
        # Compactness floor: keep slots whose y is within one tile-height of the best y.
        best_y = min(y for _, y in scored)
        band = best_y + max(h0, gap)
        compact = [(x, y) for x, y in scored if y <= band + 1e-9]
        if not compact:
            compact = scored

        def slot_score(xy):
            x, y = xy
            # Primary: compactness.  Then chemical link length; among equal length, fewer
            # centre-to-centre crossings through already-placed tiles.
            link = 0.0
            crosses = 0
            cx, cy = centre(name, (x, y))
            for p, pxy in placed.items():
                wgt = weights.get(_link_key(name, p), 0.0)
                if wgt <= 0:
                    continue
                px, py = centre(p, pxy)
                link += wgt * math.hypot(cx - px, cy - py)
                # Segment (cx,cy)-(px,py) through a third placed tile?
                for q, qxy in placed.items():
                    if q == p:
                        continue
                    qw, qh = sizes[q]
                    qx0, qy0 = qxy
                    if _seg_hits_rect(cx, cy, px, py, qx0, qy0, qw + gap, qh + gap):
                        crosses += 1
                        break
            return (y, link_cost * link, CROSS_COST * crosses, x)

        best_x, best_y = min(compact, key=slot_score)
        placed[name] = (best_x, best_y)
        skyline.append((best_x, best_x + w, best_y + h))
    return placed


def skyline_extent(placed, sizes, gap):
    if not placed:
        return 0.0, 0.0
    right = max(placed[n][0] + sizes[n][0] + gap for n in placed)
    bottom = max(placed[n][1] + sizes[n][1] + gap for n in placed)
    return float(right), float(bottom)


def target_pack_width(sizes, gap, aspect=TARGET_ASPECT):
    """max(sqrt(sum((w+g)(h+g)) * aspect), widest + g)."""
    if not sizes:
        return 1.0
    area = sum((float(w) + gap) * (float(h) + gap) for w, h in sizes.values())
    widest = max(float(w) for w, h in sizes.values())
    return max(math.sqrt(max(area * aspect, 1.0)), widest + gap)


def pack_two_level(groups, sizes, gap, aspect=TARGET_ASPECT, weights=None):
    """Two-level skyline: pack members inside each group, then pack the group blocks.

    ``groups``: ordered list of (group_id, [member_ids...]).
    ``sizes``: member_id -> (w, h).
    ``weights``: optional {(a,b): w} between member ids (chemical links).
    """
    weights = weights or {}
    block_local = {}
    block_size = {}
    member_of = {}
    for gid, members in groups:
        if not members:
            continue
        local_sizes = {m: sizes[m] for m in members if m in sizes}
        if not local_sizes:
            continue
        for m in local_sizes:
            member_of[m] = gid
        local, _rep = pack_tiles_aspect(local_sizes, gap, weights, target_aspect=aspect)
        bw, bh = skyline_extent(local, local_sizes, gap)
        block_local[gid] = local
        block_size[gid] = (bw, bh)

    if not block_size:
        return {}, dict(groups=0, tiles=0, width=0, height=0, target_aspect=aspect)

    # Aggregate inter-group weights from member links.
    gweights = defaultdict(float)
    for (a, b), w in weights.items():
        ga, gb = member_of.get(a), member_of.get(b)
        if ga is None or gb is None or ga == gb:
            continue
        gweights[_link_key(ga, gb)] += w

    block_pos, _grep = pack_tiles_aspect(block_size, gap, dict(gweights), target_aspect=aspect)

    abs_pos = {}
    for gid, local in block_local.items():
        ox, oy = block_pos[gid]
        for m, (x, y) in local.items():
            abs_pos[m] = (ox + x, oy + y)
    bw, bh = skyline_extent(block_pos, block_size, gap)
    return abs_pos, dict(groups=len(block_size), tiles=len(abs_pos),
                         width=round(bw), height=round(bh), target_aspect=aspect,
                         aspect=round(bw / max(bh, 1.0), 3), link_pairs=len(weights))


# --------------------------------------------------------------------------- empty-band squeeze
def _coord_lists(body):
    xs, ys = [], []
    for node in body['nodes'].values():
        xs.append(node['x']); ys.append(node['y'])
        if 'label_x' in node:
            xs.append(node['label_x']); ys.append(node['label_y'])
    for reaction in body['reactions'].values():
        if 'label_x' in reaction:
            xs.append(reaction['label_x']); ys.append(reaction['label_y'])
        for segment in (reaction.get('segments') or {}).values():
            for key in ('b1', 'b2'):
                pt = segment.get(key)
                if pt:
                    xs.append(pt['x']); ys.append(pt['y'])
    for label in (body.get('text_labels') or {}).values():
        xs.append(label['x']); ys.append(label['y'])
    return xs, ys


def build_axis_remap(values, max_gap, max_shift=None):
    """Monotone piecewise-linear map that shrinks empty runs longer than ``max_gap``.

    Returns a callable, or None when nothing collapses.  Order along the axis is preserved,
    so the remap cannot introduce crossings.  ``max_shift`` caps total collapse (largest
    gaps first) so a partial squeeze can stay inside the aspect band.
    """
    if not values:
        return None
    points = sorted(set(round(float(v), 3) for v in values))
    if len(points) < 2:
        return None
    # Collect compressible gaps; optionally spend a collapse budget on the largest first.
    gaps = []
    for prev, cur in zip(points, points[1:]):
        gap = cur - prev
        if gap > max_gap:
            gaps.append((cur, gap - max_gap))
    if not gaps:
        return None
    if max_shift is not None and max_shift > 0:
        # Spend budget on the widest corridors first (stable by position).
        order = sorted(range(len(gaps)), key=lambda i: (-gaps[i][1], gaps[i][0]))
        take = {i: 0.0 for i in range(len(gaps))}
        budget = float(max_shift)
        for i in order:
            if budget <= 1e-9:
                break
            use = min(gaps[i][1], budget)
            take[i] = use
            budget -= use
        gaps = [(gaps[i][0], take[i]) for i in range(len(gaps)) if take[i] > 1e-9]
        if not gaps:
            return None
    shift, breaks = 0.0, []
    for position, amount in sorted(gaps, key=lambda g: g[0]):
        shift += amount
        breaks.append((position, shift))

    def remap(value):
        total = 0.0
        for position, accumulated in breaks:
            if value >= position:
                total = accumulated
            else:
                break
        return value - total

    return remap


def _apply_axis_remap(body, fx, fy):
    for node in body['nodes'].values():
        node['x'] = fx(node['x']); node['y'] = fy(node['y'])
        if 'label_x' in node:
            node['label_x'] = fx(node['label_x']); node['label_y'] = fy(node['label_y'])
    for reaction in body['reactions'].values():
        if 'label_x' in reaction:
            reaction['label_x'] = fx(reaction['label_x']); reaction['label_y'] = fy(reaction['label_y'])
        for segment in (reaction.get('segments') or {}).values():
            for key in ('b1', 'b2'):
                pt = segment.get(key)
                if pt:
                    pt['x'] = fx(pt['x']); pt['y'] = fy(pt['y'])
    for label in (body.get('text_labels') or {}).values():
        label['x'] = fx(label['x']); label['y'] = fy(label['y'])


def squeeze_escher(data, max_gap):
    """Collapse empty bands on a finished Escher map.  -> dict(dx, dy, max_gap).

    Prefers collapses that keep the content bbox inside the 4:3–16:9 landscape band:
    if a full X+Y squeeze would leave the map portrait/too-wide, keep the squeeze on the
    long axis only.
    """
    import copy as _copy
    body = data[1]
    xs, ys = _coord_lists(body)
    if not xs:
        return dict(dx=0.0, dy=0.0, max_gap=max_gap, applied=False)
    before_w = max(xs) - min(xs)
    before_h = max(ys) - min(ys)
    before_aspect = before_w / max(before_h, 1e-9)
    force_empty = False
    fx = build_axis_remap(xs, max_gap)
    fy = build_axis_remap(ys, max_gap)
    if fx is None and fy is None:
        return dict(dx=0.0, dy=0.0, max_gap=max_gap, applied=False)

    def _trial_remap(src, fx_use, fy_use):
        trial = _copy.deepcopy(src)
        _apply_axis_remap(trial[1],
                          fx_use if fx_use else (lambda v: v),
                          fy_use if fy_use else (lambda v: v))
        xs2, ys2 = _coord_lists(trial[1])
        w = max(xs2) - min(xs2); h = max(ys2) - min(ys2)
        return trial, w, h, w / max(h, 1e-9)

    def _trial(use_x, use_y):
        return _trial_remap(data,
                            fx if use_x and fx else None,
                            fy if use_y and fy else None)

    candidates = []
    for use_x, use_y, tag in ((True, True, 'xy'), (False, True, 'y'), (True, False, 'x')):
        if use_x and fx is None:
            continue
        if use_y and fy is None:
            continue
        if not use_x and not use_y:
            continue
        trial, w, h, asp = _trial(use_x, use_y)
        in_band = ASPECT_MIN - 1e-3 <= asp <= ASPECT_MAX + 1e-3
        # Prefer in-band; then closer to TARGET; then more total collapse.
        collapse = (before_w - w) + (before_h - h)
        score = (0 if in_band else 1,
                 abs(math.log(max(asp, 1e-9) / TARGET_ASPECT)),
                 -collapse)
        candidates.append(dict(trial=trial, w=w, h=h, aspect=asp, tag=tag, score=score,
                               use_x=use_x, use_y=use_y))
    if not candidates:
        return dict(dx=0.0, dy=0.0, max_gap=max_gap, applied=False)
    # Prefer keeping an in-band aspect when the map started in-band — unless empty
    # corridors dominate the bbox (packing used voids to fake landscape).  Then
    # force the densest collapse even if aspect leaves the band.
    force_empty = False
    if ASPECT_MIN - 1e-3 <= before_aspect <= ASPECT_MAX + 1e-3:
        in_band_cands = [c for c in candidates if ASPECT_MIN - 1e-3 <= c['aspect'] <= ASPECT_MAX + 1e-3]
        if in_band_cands:
            candidates = in_band_cands
        else:
            max_empty_x = _max_empty_run(xs, max_gap)
            max_empty_y = _max_empty_run(ys, max_gap)
            frac = max(max_empty_x / max(before_w, 1e-9), max_empty_y / max(before_h, 1e-9))
            if frac >= SQUEEZE_FORCE_EMPTY_FRAC:
                force_empty = True
                # Prefer densest collapse that still stays in-band; only leave the
                # band when no in-band candidate exists (then stop at ASPECT_MIN).
                band_ok = [c for c in candidates
                           if ASPECT_MIN - 1e-3 <= c['aspect'] <= ASPECT_MAX + 1e-3]
                if band_ok:
                    for c in band_ok:
                        collapse = (before_w - c['w']) + (before_h - c['h'])
                        c['score'] = (0, -collapse, abs(math.log(max(c['aspect'], 1e-9) / TARGET_ASPECT)))
                    candidates = band_ok
                else:
                    # Partial collapse toward the nearer band edge.
                    partial = None
                    if before_aspect > ASPECT_MIN and fx is not None:
                        room = before_w - ASPECT_MIN * before_h
                        if room > 1.0:
                            fx_p = build_axis_remap(xs, max_gap, max_shift=room)
                            if fx_p is not None:
                                trial, w, h, asp = _trial_remap(data, fx_p, None)
                                partial = dict(trial=trial, w=w, h=h, aspect=asp, tag='force_partial_x',
                                               score=(0, 0, 0), use_x=True, use_y=False)
                    if partial is None and before_aspect < ASPECT_MAX and fy is not None:
                        room = before_h - before_w / ASPECT_MAX
                        if room > 1.0:
                            fy_p = build_axis_remap(ys, max_gap, max_shift=room)
                            if fy_p is not None:
                                trial, w, h, asp = _trial_remap(data, None, fy_p)
                                partial = dict(trial=trial, w=w, h=h, aspect=asp, tag='force_partial_y',
                                               score=(0, 0, 0), use_x=False, use_y=True)
                    if partial is not None:
                        candidates = [partial]
                    else:
                        for c in candidates:
                            collapse = (before_w - c['w']) + (before_h - c['h'])
                            c['score'] = (0, -collapse,
                                          abs(math.log(max(c['aspect'], 1e-9) / TARGET_ASPECT)))
            else:
                # Partial squeeze on the long axis: collapse only as far as the
                # nearer band edge so landscape stays in 4:3–16:9.
                partial = None
                if before_aspect > ASPECT_MIN and fx is not None:
                    # Width room down to ASPECT_MIN (keep height).
                    room = before_w - ASPECT_MIN * before_h
                    if room > 1.0:
                        fx_p = build_axis_remap(xs, max_gap, max_shift=room)
                        if fx_p is not None:
                            trial, w, h, asp = _trial_remap(data, fx_p, None)
                            partial = dict(trial=trial, w=w, h=h, aspect=asp, tag='partial_x',
                                           score=(0, 0, 0), use_x=True, use_y=False)
                if before_aspect < ASPECT_MAX and fy is not None and partial is None:
                    room = before_h - before_w / ASPECT_MAX
                    if room > 1.0:
                        fy_p = build_axis_remap(ys, max_gap, max_shift=room)
                        if fy_p is not None:
                            trial, w, h, asp = _trial_remap(data, None, fy_p)
                            partial = dict(trial=trial, w=w, h=h, aspect=asp, tag='partial_y',
                                           score=(0, 0, 0), use_x=False, use_y=True)
                if partial is None:
                    return dict(dx=0.0, dy=0.0, max_gap=round(max_gap, 1), applied=False,
                                mode='skipped_keep_aspect', aspect_before=round(before_aspect, 3),
                                aspect_after=round(before_aspect, 3))
                candidates = [partial]
    best = min(candidates, key=lambda c: c['score'])
    data[0], data[1] = best['trial'][0], best['trial'][1]
    body = data[1]
    xs2, ys2 = _coord_lists(body)
    canvas = body.get('canvas') or {}
    pad_x = max((canvas.get('width', before_w) - before_w) / 2.0, 40.0)
    pad_y = max((canvas.get('height', before_h) - before_h) / 2.0, 40.0)
    body['canvas'] = dict(
        x=min(xs2) - pad_x, y=min(ys2) - pad_y,
        width=(max(xs2) - min(xs2)) + 2 * pad_x,
        height=(max(ys2) - min(ys2)) + 2 * pad_y)
    return dict(dx=round(before_w - (max(xs2) - min(xs2)), 1),
                dy=round(before_h - (max(ys2) - min(ys2)), 1),
                max_gap=round(max_gap, 1), applied=True,
                mode=('force_empty_' + best['tag']) if force_empty else best['tag'],
                aspect_before=round(before_aspect, 3),
                aspect_after=round(best['aspect'], 3))


def _max_empty_run(values, max_gap):
    """Largest gap along a 1-D coordinate list that exceeds ``max_gap`` (else 0)."""
    if not values:
        return 0.0
    points = sorted(set(round(float(v), 3) for v in values))
    best = 0.0
    for prev, cur in zip(points, points[1:]):
        gap = cur - prev
        if gap > max_gap:
            best = max(best, gap)
    return best


def _is_transport_mid(mid):
    """Heuristic: extracellular / periplasm id or classic transport suffix."""
    if not mid:
        return False
    if mid.endswith('_e') or mid.endswith('_p'):
        return True
    return False


def compact_long_arms_scene(model, scene, cur, max_arm=ARM_MAX, target=ARM_TARGET):
    """Pull long mid→metabolite arms in toward the reaction (scene positions).

    Transport / exchange chains and private metabolite glyphs that were stretched by
    a common scale or a distant shared copy are shortened to ``target`` px.  Shared
    multi-reaction metabolites are not moved; the reaction midmarker is pulled instead
    when the arm is a transport/exchange spoke.  -> (scene, report).
    """
    scene = dict(scene, positions={k: list(v) for k, v in scene['positions'].items()})
    pos, parts = scene['positions'], scene.get('participants') or {}
    rx = {r['id']: r for r in model['reactions']}
    holders = defaultdict(set)
    for (rid, mid), key in parts.items():
        holders[key].add(rid)
    shortened = 0
    pulled_mids = 0
    details = []
    for rid, r in rx.items():
        rk = 'r:' + rid
        if rk not in pos:
            continue
        rxy = np.asarray(pos[rk], float)
        core = [m for m in r['metabolites'] if m not in cur]
        if not core:
            continue
        # Transport if reaction touches ≥2 compartments among non-currency mets.
        comps = {comp(m) for m in core if comp(m)}
        is_transport = len(comps) >= 2 or any(_is_transport_mid(m) for m in core)
        for mid in core:
            key = parts.get((rid, mid))
            if not key or key not in pos or key.startswith('side:'):
                continue
            mxy = np.asarray(pos[key], float)
            vec = mxy - rxy
            dist = float(np.hypot(*vec))
            if dist <= max_arm + 1e-6:
                continue
            # Prefer compacting private glyphs; always compact transport/exchange arms.
            private = len(holders.get(key, ())) <= 1
            if not (private or is_transport or _is_transport_mid(mid)):
                continue
            u = vec / dist
            new_len = float(target)
            if private or _is_transport_mid(mid):
                pos[key] = (rxy + u * new_len).tolist()
                shortened += 1
                details.append(dict(reaction=rid, metabolite=mid, before=round(dist, 1),
                                    after=round(new_len, 1), moved='metabolite'))
            else:
                # Shared cytosolic pool: pull the transporter midmarker toward it.
                pos[rk] = (mxy - u * new_len).tolist()
                rxy = np.asarray(pos[rk], float)
                pulled_mids += 1
                details.append(dict(reaction=rid, metabolite=mid, before=round(dist, 1),
                                    after=round(new_len, 1), moved='midmarker'))
                break  # midmarker moved; re-measure remaining arms next pass if needed
    report = dict(shortened=shortened, pulled_midmarkers=pulled_mids,
                  max_arm=max_arm, target=target, details=details[:40])
    return scene, report


def compact_long_arms_escher(data, cur=None, max_arm=ARM_MAX, target=ARM_TARGET):
    """Same compact on a finished Escher map (nodes move; Bezier controls drop).

    Private metabolite glyphs move toward their midmarker.  Shared extracellular /
    periplasm pools move once toward the nearest holding midmarker (so ACt2r + EX
    cannot yank the pool back and forth).  ``b1``/``b2`` are set to null (schema
    requires the keys) so later restyle / source-path stamp can rebuild.
    """
    cur = cur or set()
    body = data[1]
    nodes, reactions = body['nodes'], body['reactions']
    # metabolite node id -> reaction bigg_ids that touch it
    holders = defaultdict(set)
    mid_of = {}
    for rx in reactions.values():
        rid = rx['bigg_id']
        mid_id = next((s['from_node_id'] for s in rx['segments'].values()
                       if nodes.get(s['from_node_id'], {}).get('node_type') == 'midmarker'), None)
        if mid_id is not None:
            mid_of[rid] = mid_id
        for s in (rx.get('segments') or {}).values():
            for nid in (s['from_node_id'], s['to_node_id']):
                n = nodes.get(nid) or {}
                if n.get('node_type') == 'metabolite':
                    holders[nid].add(rid)
    shortened = 0
    details = []
    moved_shared = set()

    def _clear_seg(seg):
        seg['b1'] = None
        seg['b2'] = None

    # Pass 1: shared transport/exchange pools — one move toward the nearest mid.
    for tid, rids in list(holders.items()):
        if len(rids) < 2 or tid in moved_shared:
            continue
        t = nodes.get(tid) or {}
        mid = t.get('bigg_id')
        if not mid or not (_is_transport_mid(mid) or mid in cur):
            # Shared cytosolic pools are left alone (packing/instances own those).
            if not _is_transport_mid(mid or ''):
                continue
        mxy = np.array([t['x'], t['y']], float)
        cands = []
        for rid in rids:
            mid_id = mid_of.get(rid)
            if mid_id is None:
                continue
            rxy = np.array([nodes[mid_id]['x'], nodes[mid_id]['y']], float)
            dist = float(np.hypot(*(mxy - rxy)))
            if dist > max_arm:
                cands.append((dist, rid, mid_id, rxy))
        if not cands:
            continue
        # Place the pool next to the nearest mid, then pull every other holder
        # midmarker in so EX ↔ transporter chains stay short on both sides.
        dist, rid, mid_id, rxy = min(cands, key=lambda c: c[0])
        u = (mxy - rxy) / dist
        new_xy = rxy + u * float(target)
        t['x'], t['y'] = float(new_xy[0]), float(new_xy[1])
        if 'label_x' in t:
            t['label_x'] = t['x'] + (t.get('label_x', t['x']) - mxy[0])
            t['label_y'] = t['y'] + (t.get('label_y', t['y']) - mxy[1])
        moved_shared.add(tid)
        mxy = new_xy
        for rid2, mid2 in ((r, mid_of[r]) for r in rids if r in mid_of):
            rxy2 = np.array([nodes[mid2]['x'], nodes[mid2]['y']], float)
            vec2 = mxy - rxy2
            d2 = float(np.hypot(*vec2))
            if d2 <= max_arm + 1e-6:
                continue
            u2 = vec2 / d2
            new_mid = mxy - u2 * float(target)
            nodes[mid2]['x'], nodes[mid2]['y'] = float(new_mid[0]), float(new_mid[1])
            rx2 = next((r for r in reactions.values() if r['bigg_id'] == rid2), None)
            if rx2 is not None and 'label_x' in rx2:
                rx2['label_x'], rx2['label_y'] = nodes[mid2]['x'], nodes[mid2]['y']
            shortened += 1
            details.append(dict(reaction=rid2, metabolite=mid, before=round(d2, 1),
                                after=round(float(target), 1), moved='shared_midmarker'))
        # Clear controls on every segment into this node.
        for rx in reactions.values():
            for seg in rx['segments'].values():
                if seg['to_node_id'] == tid or seg['from_node_id'] == tid:
                    _clear_seg(seg)
        shortened += 1
        details.append(dict(reaction=rid, metabolite=mid, before=round(dist, 1),
                            after=round(float(target), 1), moved='shared_metabolite'))

    # Pass 2: private long arms (including private transport).
    for rx in reactions.values():
        rid = rx['bigg_id']
        mid_id = mid_of.get(rid)
        if mid_id is None:
            continue
        mid_n = nodes[mid_id]
        rxy = np.array([mid_n['x'], mid_n['y']], float)
        coeff = {m['bigg_id']: m['coefficient'] for m in (rx.get('metabolites') or []) if isinstance(m, dict)}
        mets = [m for m in coeff if m not in cur]
        comps = {comp(m) for m in mets if comp(m)}
        is_transport = len(comps) >= 2 or any(_is_transport_mid(m) for m in mets)
        for key, seg in list(rx['segments'].items()):
            if seg['from_node_id'] != mid_id:
                continue
            tid = seg['to_node_id']
            if tid in moved_shared:
                continue
            t = nodes.get(tid) or {}
            if t.get('node_type') != 'metabolite':
                continue
            mid = t.get('bigg_id')
            if not mid:
                continue
            if mid in cur and not _is_transport_mid(mid):
                continue
            mxy = np.array([t['x'], t['y']], float)
            vec = mxy - rxy
            dist = float(np.hypot(*vec))
            if dist <= max_arm + 1e-6:
                continue
            private = len(holders.get(tid, ())) <= 1
            if not (private or is_transport or _is_transport_mid(mid)):
                continue
            if not private:
                continue  # shared non-transport handled above or skipped
            u = vec / dist
            new_xy = rxy + u * float(target)
            t['x'], t['y'] = float(new_xy[0]), float(new_xy[1])
            if 'label_x' in t:
                t['label_x'] = t['x'] + (t.get('label_x', t['x']) - mxy[0])
                t['label_y'] = t['y'] + (t.get('label_y', t['y']) - mxy[1])
            _clear_seg(seg)
            shortened += 1
            details.append(dict(reaction=rid, metabolite=mid, before=round(dist, 1),
                                after=round(float(target), 1), moved='metabolite'))
    return dict(shortened=shortened, max_arm=max_arm, target=target, details=details[:40])


def reaction_spacing_escher(data):
    """Median nearest-neighbour distance of reaction midmarkers (fallback 100)."""
    pts = []
    for n in data[1]['nodes'].values():
        if n.get('node_type') == 'midmarker':
            pts.append((n['x'], n['y']))
    if len(pts) < 3:
        # approximate from metabolite nodes
        pts = [(n['x'], n['y']) for n in data[1]['nodes'].values() if 'x' in n][:2000]
    if len(pts) < 3:
        return 100.0
    arr = np.array(pts, float)
    return float(np.median(cKDTree(arr).query(arr, k=2)[0][:, 1]))


def _free_spot(local, target, tree, clear=CLEAR, tries=260):
    if tree is None:
        return target
    for i in range(tries):
        r = 0.0 if i == 0 else 45.0 * np.sqrt(i)
        a = i * 2.399963
        d = target + r * np.array([np.cos(a), np.sin(a)])
        if float(tree.query(local + d)[0].min()) >= clear:
            return d
    return None


def _aligned_spot(local, target, anchors, size, tree, centre):
    """Free place next to anchors; prefer alignment with the nearest anchor centre-line."""
    nearest = anchors[int(np.argmin(np.hypot(*(anchors - centre).T)))]
    r = size / 2.0 + 160.0
    cands = []
    plain = _free_spot(local, target, tree)
    if plain is not None:
        cands.append(plain)
    for k in range(8):
        a = k * np.pi / 4
        d = _free_spot(local, target + r * np.array([np.cos(a), np.sin(a)]), tree)
        if d is not None:
            cands.append(d)
    if not cands:
        return None

    def score(c):
        dx, dy = abs(c[0] - nearest[0]), abs(c[1] - nearest[1])
        return float(np.hypot(*(c - target))) + 0.6 * min(dx, dy)

    return min(cands, key=score)


# --------------------------------------------------------------------------- tiles + aspect-constrained pack
def _tile_local_geometry(rids, keys, pos, labels, text_w, margin):
    """Pack size/centre from reaction midmarkers, plus only nearby stubs.

    Shared metabolites and far side/inst glyphs must not inflate the reserved slot —
    a single reaction with a map-spanning stub would otherwise reserve a 20k×50k tile.
    """
    local = ['r:' + rid for rid in rids if 'r:' + rid in pos]
    if not local:
        return None
    arr = np.array([pos[k] for k in local], float)
    lo, hi = arr.min(0), arr.max(0)
    # Pad scales with the reaction cluster so dense multi-reaction tiles keep their local mets.
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
    pad = max(margin * 2.0, 0.35 * span, 120.0)
    for k in keys:
        if k in local or k not in pos:
            continue
        pt = np.array(pos[k], float)
        if lo[0] - pad <= pt[0] <= hi[0] + pad and lo[1] - pad <= pt[1] <= hi[1] + pad:
            local.append(k)
    arr = np.array([pos[k] for k in local], float)
    lo, hi = arr.min(0), arr.max(0)
    lw = min(max([text_w(str(labels.get(k, ''))) for k in local] + [0.0]), 220.0)
    size = (float(max(hi[0] - lo[0] + margin + lw, 40.0)),
            float(max(hi[1] - lo[1] + margin, 40.0)))
    return dict(lo=lo, centre=arr.mean(0), size=size, lw=lw, local_keys=set(local))

def _text_width():
    try:
        import drawing
        return lambda s: float(drawing.FONT.getlength(s))
    except Exception:
        return lambda s: 6.5 * len(s)


def build_tiles(model, scene, blocks, cur):
    """Curated blocks as rigid tiles + leftover visual components among uncovered reactions."""
    pos, parts = scene['positions'], scene['participants']
    rx = {r['id']: r for r in model['reactions']}
    labels = scene.get('labels', {})
    text_w = _text_width()
    margin = 60.0
    tiles = []
    covered = set()

    for bi, b in enumerate(blocks or []):
        rids = [rid for rid in (b.get('reaction_ids') or []) if rid in rx and 'r:' + rid in pos]
        if not rids:
            continue
        covered.update(rids)
        keys = set()
        for rid in rids:
            keys.add('r:' + rid)
            for mid in rx[rid]['metabolites']:
                key = parts.get((rid, mid))
                if key and key in pos:
                    keys.add(key)
        keys = {k for k in keys if k in pos}
        geom = _tile_local_geometry(rids, keys, pos, labels, text_w, margin)
        if geom is None:
            continue
        tiles.append(dict(
            id='block:%s' % (b.get('id') or bi),
            kind='curated',
            reactions=sorted(rids),
            keys=keys,
            lo=geom['lo'], centre=geom['centre'], size=geom['size'], lw=geom['lw'],
            local_keys=geom['local_keys']))

    # Leftover reactions: components joined only through non-currency glyphs (currency hubs
    # would otherwise glue distant autos into one map-spanning tile).
    leftover = [r['id'] for r in model['reactions'] if r['id'] not in covered and 'r:' + r['id'] in pos]
    if leftover:
        parent = {}

        def find(a):
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for rid in leftover:
            find('r:' + rid)
            for mid in rx[rid]['metabolites']:
                if mid in cur:
                    continue
                key = parts.get((rid, mid))
                if key is None or key.startswith('side:'):
                    continue
                find(key)
                parent[find(key)] = find('r:' + rid)
        groups = defaultdict(list)
        for rid in leftover:
            groups[find('r:' + rid)].append(rid)
        for gi, rids in enumerate(sorted(groups.values(), key=lambda g: (-len(g), g[0]))):
            keys = set()
            for rid in rids:
                keys.add('r:' + rid)
                for mid in rx[rid]['metabolites']:
                    key = parts.get((rid, mid))
                    if key and key in pos:
                        keys.add(key)
            keys = {k for k in keys if k in pos}
            geom = _tile_local_geometry(rids, keys, pos, labels, text_w, margin)
            if geom is None:
                continue
            tiles.append(dict(
                id='auto:%d' % gi, kind='auto', reactions=sorted(rids),
                keys=keys, lo=geom['lo'], centre=geom['centre'], size=geom['size'], lw=geom['lw'],
                local_keys=geom['local_keys']))
    return tiles


def _pack_link_cost(placed, sizes, gap, weights):
    """Sum of weight × centre distance for chemically linked placed tiles."""
    if not weights or not placed:
        return 0.0
    centres = {}
    for n, (x, y) in placed.items():
        w, h = sizes[n]
        centres[n] = (x + w / 2.0, y + h / 2.0)
    total = 0.0
    for (a, b), wgt in weights.items():
        if a not in centres or b not in centres or wgt <= 0:
            continue
        ax, ay = centres[a]
        bx, by = centres[b]
        total += float(wgt) * math.hypot(ax - bx, ay - by)
    return total


def _pack_cross_cost(placed, sizes, gap, weights):
    """Count weighted links whose centre segment passes through a third tile."""
    if not weights or len(placed) < 3:
        return 0
    centres = {}
    rects = {}
    for n, (x, y) in placed.items():
        w, h = sizes[n]
        centres[n] = (x + w / 2.0, y + h / 2.0)
        rects[n] = (x, y, w + gap, h + gap)
    hits = 0
    for (a, b), wgt in weights.items():
        if a not in centres or b not in centres or wgt <= 0:
            continue
        ax, ay = centres[a]
        bx, by = centres[b]
        for q, (rx, ry, rw, rh) in rects.items():
            if q == a or q == b:
                continue
            if _seg_hits_rect(ax, ay, bx, by, rx, ry, rw, rh):
                hits += 1
                break
    return hits


def pack_tiles_aspect(sizes, gap, weights, target_aspect=TARGET_ASPECT):
    """Binary-search skyline target widths; pick best cost under the 4:3–16:9 constraint.

    Cost among in-band packs: link length (primary) then estimated crossings then area.
    Out-of-band packs are only kept when no in-band width exists.
    """
    if not sizes:
        return {}, dict(width=0, height=0, aspect=target_aspect, target_width=0)
    widest = max(float(w) for w, h in sizes.values()) + gap
    area = sum((float(w) + gap) * (float(h) + gap) for w, h in sizes.values())
    order = sorted(sizes, key=lambda i: (-sizes[i][0] * sizes[i][1], str(i)))
    # Search bracket: from a single column (widest) up to a wide landscape shelf.
    lo = widest
    hi = max(lo * 1.05, math.sqrt(max(area * ASPECT_MAX, 1.0)) * 1.6, math.sqrt(max(area * target_aspect, 1.0)) * 1.35)
    trials = [lo, hi, target_pack_width(sizes, gap, target_aspect),
              target_pack_width(sizes, gap, ASPECT_MIN),
              target_pack_width(sizes, gap, ASPECT_MAX)]
    # Binary-search refinement: evaluate midpoints, keep probing the side that
    # moves packed aspect toward TARGET_ASPECT / into the hard band.
    a, b = lo, hi
    for _ in range(NODE_ASPECT_LOOPS):
        mid = 0.5 * (a + b)
        trials.append(mid)
        placed = skyline_pack(order, sizes, gap, mid, weights=weights)
        bw, bh = skyline_extent(placed, sizes, gap)
        aspect = bw / max(bh, 1e-9)
        if aspect < target_aspect:
            a = mid  # too tall → widen
        else:
            b = mid  # too wide → narrow
    # A few extra probes around the closed-form target for non-monotonic packs.
    seed = target_pack_width(sizes, gap, target_aspect)
    for scale in (0.85, 0.95, 1.05, 1.15, 1.3):
        trials.append(max(seed * scale, widest))

    seen, uniq = set(), []
    for tw in sorted(trials):
        key = round(float(tw), 1)
        if key in seen or tw < widest - 1e-9:
            continue
        seen.add(key)
        uniq.append(float(tw))

    def _stretch_to_band(placed, aspect):
        """Anisotropic stretch of top-lefts so the packed frame enters [ASPECT_MIN, ASPECT_MAX].

        Stretching only expands an axis (never compresses), so tiles cannot newly overlap.
        """
        if not placed:
            return placed, aspect, False
        if ASPECT_MIN - 1e-6 <= aspect <= ASPECT_MAX + 1e-6:
            return placed, aspect, False
        bw, bh = skyline_extent(placed, sizes, gap)
        # Aim slightly inside the band: extent uses (size+gap) so a exact-edge
        # target can land a hair outside after top-left scaling.
        if aspect < ASPECT_MIN:
            target = min(ASPECT_MIN * 1.04, TARGET_ASPECT)
            sx, sy = (target * bh) / max(bw, 1e-9), 1.0
        else:
            target = max(ASPECT_MAX * 0.97, TARGET_ASPECT)
            sx, sy = 1.0, bw / max(target * bh, 1e-9)
        # Only allow expansion, and cap so voids cannot invent a landscape frame.
        sx, sy = max(sx, 1.0), max(sy, 1.0)
        sx, sy = min(sx, STRETCH_MAX), min(sy, STRETCH_MAX)
        if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
            return placed, aspect, False
        out = {n: (x * sx, y * sy) for n, (x, y) in placed.items()}
        nbw, nbh = skyline_extent(out, sizes, gap)
        return out, nbw / max(nbh, 1e-9), True

    best = None
    # Cap the search so a pathological weight field cannot pick an 80k-wide shelf.
    seed_cap = target_pack_width(sizes, gap, target_aspect)
    hi_cap = max(widest * 4.0, seed_cap * 2.5)
    uniq = [tw for tw in uniq if tw <= hi_cap + 1e-9] or uniq[:1]
    for tw in uniq:
        placed = skyline_pack(order, sizes, gap, tw, weights=weights)
        bw, bh = skyline_extent(placed, sizes, gap)
        aspect = bw / max(bh, 1e-9)
        placed, aspect, stretched = _stretch_to_band(placed, aspect)
        bw, bh = skyline_extent(placed, sizes, gap)
        in_band = ASPECT_MIN - 1e-6 <= aspect <= ASPECT_MAX + 1e-6
        link = _pack_link_cost(placed, sizes, gap, weights)
        crosses = _pack_cross_cost(placed, sizes, gap, weights)
        # Prefer in-band; then shorter chemical links; then smaller area (dense pack);
        # then fewer crossings; then closer to target.
        band_pen = 0.0 if in_band else abs(math.log(max(aspect, 1e-9) / max(min(max(aspect, ASPECT_MIN), ASPECT_MAX), 1e-9)))
        score = (
            0 if in_band else 1,
            band_pen,
            link,
            bw * bh,
            CROSS_COST * crosses,
            abs(math.log(max(aspect, 1e-9) / target_aspect)),
        )
        cand = dict(placed=placed, width=bw, height=bh, aspect=aspect,
                    target_width=tw, in_band=in_band, score=score,
                    link_cost=link, crossings=crosses, stretched=stretched)
        if best is None or cand['score'] < best['score']:
            best = cand
    report = dict(width=round(best['width']), height=round(best['height']),
                  aspect=round(best['aspect'], 3), target_aspect=target_aspect,
                  target_width=round(best['target_width']), in_band=best['in_band'],
                  tiles=len(best['placed']), gap=gap,
                  link_cost=round(best['link_cost'], 1), crossings=best['crossings'],
                  widths_tried=len(uniq), stretched=bool(best.get('stretched')))
    return best['placed'], report


def classify_large_tiles(tiles, area_pct=LARGE_AREA_PCT):
    """Curated tiles at/above ``area_pct`` that form one spatial cluster around the largest.

    Area cut is data-derived (percentile of curated tile areas).  Candidates far from the
    growing core stay movable — only the compact main-metabolism cluster is frozen.
    """
    curated = [t for t in tiles if t['kind'] == 'curated']
    if not curated:
        return set()
    areas = [float(t['size'][0] * t['size'][1]) for t in curated]
    cut = float(np.percentile(areas, area_pct))
    cands = [t for t in curated if float(t['size'][0] * t['size'][1]) >= cut - 1e-9]
    if not cands:
        cands = [max(curated, key=lambda t: t['size'][0] * t['size'][1])]
    seed = max(cands, key=lambda t: t['size'][0] * t['size'][1])
    large = {seed['id']}
    by_id = {t['id']: t for t in cands}
    if any('centre' not in t for t in cands):
        # Unit tests / callers without centres: keep every area-qualified curated tile.
        return {t['id'] for t in cands}

    def diag(t):
        return float(math.hypot(t['size'][0], t['size'][1]))

    changed = True
    while changed:
        changed = False
        cores = [by_id[i] for i in large]
        for t in cands:
            if t['id'] in large:
                continue
            # Join if centre-to-centre is within a generous multiple of the two diagonals.
            for c in cores:
                lim = 0.75 * (diag(t) + diag(c)) + 400.0
                if float(np.hypot(*(np.array(t['centre'], float) - np.array(c['centre'], float)))) <= lim:
                    large.add(t['id'])
                    changed = True
                    break
    return large


def _compartment(c, rx):
    cnt = defaultdict(int)
    for rid in c['reactions']:
        for mid in rx[rid]['metabolites']:
            cnt[comp(mid)] += 1
    return max(cnt, key=cnt.get) if cnt else ''
