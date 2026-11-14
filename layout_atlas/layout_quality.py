"""Final-geometry measures used by the quality gate and the Layout quality panel.

One computation path: ``evaluate()`` feeds the quality gate, the tool panel, and
``python3 project.py evaluate`` (via ``evaluation.score``). Project 22
layout-benchmark / layout-transfer-check field names are used where they exist;
gdMetriX extras stay CLI-only (optional GPL import, never vendored).

User-facing wording uses **curated**, not "expert". Internal aliases
``algorithm_expert`` / ``expert_expert`` remain for the quality gate.

Also contains metric_details and assert_chemistry (used by compose).
"""
import numpy as np

import source_paths as sp

try:
    from core import CURRENCY, chem
except ImportError:  # pragma: no cover
    CURRENCY, chem = set(), lambda mid: mid


def _is_currency_met(mid: str | None) -> bool:
    return bool(mid) and chem(mid) in CURRENCY


def expert_geometry(blocks, scene):
    """Distortion of curated structures between their planned positions (`block['positions']`) and the final scene."""
    rows = []
    for b in blocks:
        src, dst = [], []
        for rid in b['reaction_ids']:
            k = 'r:' + rid
            if k in b['positions'] and k in scene['positions']:
                src.append(b['positions'][k])
                dst.append(scene['positions'][k])
        if len(src) < 3:
            continue
        fit = sp.similarity(src, dst)
        if fit is None:
            continue
        shift = float(np.mean(np.hypot(*(np.array(dst) - np.array(src)).T)))
        rows.append((fit['rms'], fit['max'], shift, len(src)))
    if not rows:
        return dict(structures=0)
    rms = np.array([r[0] for r in rows])
    weights = np.array([r[3] for r in rows], float)
    return dict(structures=len(rows), distortion_rms_px_mean=round(float(np.average(rms, weights=weights)), 2), distortion_rms_px_max=round(float(rms.max()), 2),
                distortion_max_px=round(float(max(r[1] for r in rows)), 2), mean_displacement_px=round(float(np.average([r[2] for r in rows], weights=weights)), 1),
                note='residual after translation, rotation and uniform scaling of each structure')


# ---------------------------------------------------------------------------------------------- final-geometry evaluator (V7, minimal version)
# Everything below is computed from the FINAL map.json (the same segments, Bezier controls and label offsets that the SVG draws), never from the layout
# stage's straight lines or rectangles.  Crossings are counted on the sampled final curves, by source: algorithm × algorithm, algorithm × curated,
# curated × curated; and by metabolite role: main-metabolite vs currency-involved (ecmapper / Project 22).
from collections import defaultdict as _dd

CELL = 200.0
REGION_WINDOW = 400.0
REGION_STRIDE = 200.0


def _region_window_keys(x, y, x0, y0):
    """Indices (kx, ky) of 400 px windows (stride 200) that contain (x, y).

    Window (kx, ky) covers ``[x0 + 200·kx, x0 + 200·kx + 400) × [y0 + 200·ky, …)``,
    matching the rectangle drawn in the tool UI.  An event at x belongs to
    ``k ∈ {⌊(x−x0)/200⌋, ⌊(x−x0)/200⌋ − 1}`` (and likewise for y).
    """
    fx = int((x - x0) // REGION_STRIDE)
    fy = int((y - y0) // REGION_STRIDE)
    keys = []
    for kx in (fx - 1, fx):
        for ky in (fy - 1, fy):
            wx0 = x0 + kx * REGION_STRIDE
            wy0 = y0 + ky * REGION_STRIDE
            if wx0 <= x < wx0 + REGION_WINDOW and wy0 <= y < wy0 + REGION_WINDOW:
                keys.append((kx, ky))
    return keys


def _grid_add(grid, idx, lo, hi):
    for gx in range(int(lo[0] // CELL), int(hi[0] // CELL) + 1):
        for gy in range(int(lo[1] // CELL), int(hi[1] // CELL) + 1):
            grid[gx, gy].append(idx)


def _grid_query(grid, lo, hi):
    out = set()
    for gx in range(int(lo[0] // CELL), int(hi[0] // CELL) + 1):
        for gy in range(int(lo[1] // CELL), int(hi[1] // CELL) + 1):
            out.update(grid.get((gx, gy), ()))
    return out


def _final_obstacles(data):
    """Glyph and label boxes with kind ('node' / 'label') and the node id each belongs to."""
    import drawing
    font = drawing.FONT
    body = data[1]
    hidden = set((data[0].get('metadata') or {}).get('hidden_metabolite_labels') or [])
    rects, owner, kind, label_of = [], [], [], []
    for nid, n in body['nodes'].items():
        t = n.get('node_type')
        if t == 'metabolite':
            rects.append((n['x'] - er_NODE, n['y'] - er_NODE, n['x'] + er_NODE, n['y'] + er_NODE))
            owner.append(nid); kind.append('node'); label_of.append(n.get('bigg_id'))
            if 'label_x' in n and not n.get('label_hidden') and nid not in hidden:
                w = font.getlength(n.get('bigg_id', ''))
                rects.append((n['label_x'] - 2, n['label_y'] - 14, n['label_x'] + w + 2, n['label_y'] + 4))
                owner.append(nid); kind.append('label'); label_of.append(n.get('bigg_id'))
        elif t == 'midmarker':
            rects.append((n['x'] - er_NODE + 1, n['y'] - er_NODE + 1, n['x'] + er_NODE - 1, n['y'] + er_NODE - 1))
            owner.append(nid); kind.append('node'); label_of.append(None)
    for rx in body['reactions'].values():
        mid = next((s['from_node_id'] for s in rx['segments'].values() if body['nodes'][s['from_node_id']].get('node_type') == 'midmarker'), None)
        if mid is None or 'label_x' not in rx or rx.get('label_hidden'):
            continue
        w = font.getlength(rx['bigg_id'])
        rects.append((rx['label_x'] - 2, rx['label_y'] - 14, rx['label_x'] + w + 2, rx['label_y'] + 4))
        owner.append(mid); kind.append('label'); label_of.append(rx['bigg_id'])
    return np.array(rects, float).reshape(-1, 4), owner, kind, label_of


er_NODE = 7.0


def _seg_hits(a, b, S):
    """Indices (into S [k,4]) of the segments properly crossed by a-b, with intersection points."""
    if not len(S):
        return [], np.zeros((0, 2))
    p, r = np.array(a, float), np.array(b, float) - np.array(a, float)
    q, s = S[:, :2], S[:, 2:] - S[:, :2]
    den = r[0] * s[:, 1] - r[1] * s[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = ((q[:, 0] - p[0]) * s[:, 1] - (q[:, 1] - p[1]) * s[:, 0]) / den
        u = ((q[:, 0] - p[0]) * r[1] - (q[:, 1] - p[1]) * r[0]) / den
    ok = (np.abs(den) > 1e-9) & (t > 1e-3) & (t < 1 - 1e-3) & (u > 1e-3) & (u < 1 - 1e-3)
    idx = np.flatnonzero(ok)
    return idx, p + np.outer(t[idx], r)


def evaluate(data, skip_edges=None, orientation_kinds=None, max_issues=4000, structures=None):
    """-> (summary, issues).  `summary` holds the six core indicators, the details and the worst local regions; `issues` lists where the problems are.

    ``structures``: optional reused curated structures ``[{id, reactions|reaction_ids}]`` (or id→ids).
    When omitted, ``metadata.expert_structures`` is used, else a fallback from curated-derived connectivity.
    """
    import edge_routing as er
    import orientation as orientation_module
    from evaluation.continuity import structure_continuity as _structure_continuity
    from evaluation.map_rules import collect_map_rule_details
    meta = data[0].get('metadata', {})
    skip = {tuple(e) for e in (skip_edges if skip_edges is not None else meta.get('preserved_edges') or [])}
    skip |= {(p['reaction'], p['metabolite']) for p in (meta.get('source_paths') or [])}
    routed = {(p['reaction'], p['metabolite']): p for p in (meta.get('routed_edges') or [])}
    body = data[1]
    nodes, reactions = body['nodes'], body['reactions']
    rects, owner, kind, label_of = _final_obstacles(data)
    owner_arr = np.array(owner, dtype=object)
    grid_r = _dd(list)
    for i, r in enumerate(rects):
        _grid_add(grid_r, i, r[:2], r[2:])
    edges, nodes = build_edges(data, skip_edges)
    algo = [e for e in edges if not e['curated']]
    # ---- crossings on the sampled final curves (shared detector; also used by hotspot repair)
    cross, all_cross_events, cross_points, per_edge_cross = detect_crossings(edges, nodes, max_issues=max_issues)
    # ---- obstacles on the final curves (algorithm connections only)
    node_hits, label_hits, hit_points = 0, 0, []
    for e in algo:
        mask_own = np.isin(owner_arr, e['ends'])
        found_node = found_label = False
        n = len(e['pts']) - 1
        for si, (a, b) in enumerate(zip(e['pts'][:-1], e['pts'][1:])):
            cand = np.array(sorted(_grid_query(grid_r, np.minimum(a, b) - 2, np.maximum(a, b) + 2)), dtype=int)
            cand = cand[~mask_own[cand]] if len(cand) else cand
            if not len(cand):
                continue
            hit = (er._blocked(a, b, rects[cand], 0.0) if (si == 0 or si == n - 1) else er._hits(a, b, rects[cand], 0.0))
            for c in cand[np.flatnonzero(hit)]:
                if kind[c] == 'node' and not found_node:
                    found_node = True
                    hit_points.append(dict(kind='node', reaction=e['rid'], metabolite=e['met'], x=round(float(a[0]), 1), y=round(float(a[1]), 1)))
                if kind[c] == 'label' and not found_label:
                    found_label = True
                    hit_points.append(dict(kind='label', reaction=e['rid'], metabolite=e['met'], label=label_of[c], x=round(float(a[0]), 1), y=round(float(a[1]), 1)))
        node_hits += int(found_node); label_hits += int(found_label)
    # ---- glyph overlaps and label collisions
    from scipy.spatial import cKDTree
    centres = np.array([[n['x'], n['y']] for n in nodes.values() if n.get('node_type') in ('metabolite', 'midmarker')], float)
    node_overlap = list(cKDTree(centres).query_pairs(10)) if len(centres) > 1 else []
    lab_idx = [i for i, k in enumerate(kind) if k == 'label']
    label_overlap = 0
    for ii, i in enumerate(lab_idx):
        for j in _grid_query(grid_r, rects[i][:2], rects[i][2:]):
            if j > i and kind[j] == 'label' and rects[j][0] < rects[i][2] and rects[i][0] < rects[j][2] and rects[j][1] < rects[i][3] and rects[i][1] < rects[j][3]:
                label_overlap += 1
    # ---- length, detour, Bezier representation, curve anomalies
    lengths, detours = [], []
    bez_edges = poly_edges = ortho_edges = straight_edges = hooks = sharp = selfx = 0
    anomalies = []
    for e in algo:
        d = np.diff(e['pts'], axis=0)
        L = float(np.sum(np.hypot(d[:, 0], d[:, 1])))
        chord = float(np.hypot(*(e['pts'][-1] - e['pts'][0])))
        lengths.append(L)
        if chord >= 40:
            detours.append(L / chord)
        if e['curved'] and all(e['curved']):
            bez_edges += 1
            worst, total = _turn_stats(e['pts'])
            if total > 200:
                hooks += 1; anomalies.append(dict(kind='hook', reaction=e['rid'], metabolite=e['met']))
            elif worst > 70:
                sharp += 1; anomalies.append(dict(kind='sharp_turn', reaction=e['rid'], metabolite=e['met']))
            if _self_cross(e['pts']):
                selfx += 1; anomalies.append(dict(kind='self_intersection', reaction=e['rid'], metabolite=e['met']))
        else:
            if len(e['curved']) == 1:
                straight_edges += 1
            else:
                poly_edges += 1
                if _orthogonal(e['pts']):
                    ortho_edges += 1
    all_len = np.array([float(np.sum(np.hypot(*np.diff(e['pts'], axis=0).T))) for e in edges]) if edges else np.zeros(1)
    mids = np.array([[n['x'], n['y']] for n in nodes.values() if n.get('node_type') == 'midmarker'], float)
    spacing = float(np.median(cKDTree(mids).query(mids, k=2)[0][:, 1])) if len(mids) > 2 else 60.0
    # ---- worst local regions (400 px windows, stride 200; counted area == drawn box)
    # Crossings here match the Connection-crossings headline: algorithm×algorithm and algorithm×curated only
    # (same-reaction and curated×curated are tracked elsewhere, not in these windows).
    allx = centres[:, 0]; ally = centres[:, 1]
    windows = []
    if len(centres):
        x0, y0 = float(allx.min()), float(ally.min())
        region_cross_kinds = ('algorithm_algorithm', 'algorithm_curated', 'algorithm_expert')
        events = [(c['x'], c['y'], 'crossing', c['reactions']) for c in all_cross_events if c['kind'] in region_cross_kinds] + \
                 [(h['x'], h['y'], 'obstacle', [h['reaction']]) for h in hit_points] + \
                 [(float(centres[i][0]), float(centres[i][1]), 'node_overlap', []) for i, _ in node_overlap]
        mid_pts = [(float(np.mean(e['pts'][:, 0])), float(np.mean(e['pts'][:, 1]))) for e in edges]
        cells = _dd(lambda: dict(crossings=0, obstacles=0, node_overlaps=0, edges=0, reactions=set()))
        for x, y, what, rs in events:
            for key in _region_window_keys(x, y, x0, y0):
                c = cells[key]
                c['crossings' if what == 'crossing' else 'obstacles' if what == 'obstacle' else 'node_overlaps'] += 1
                c['reactions'].update(rs)
        for x, y in mid_pts:
            for key in _region_window_keys(x, y, x0, y0):
                cells[key]['edges'] += 1
        ranked = sorted(cells.items(), key=lambda kv: (-(kv[1]['crossings'] + kv[1]['obstacles'] + kv[1]['node_overlaps']), -kv[1]['edges'], kv[0]))
        taken = []
        for (kx, ky), c in ranked:
            wx, wy = x0 + kx * REGION_STRIDE, y0 + ky * REGION_STRIDE
            cx, cy = wx + REGION_WINDOW / 2.0, wy + REGION_WINDOW / 2.0
            if c['crossings'] + c['obstacles'] + c['node_overlaps'] == 0 or any(abs(cx - tx) < 300 and abs(cy - ty) < 300 for tx, ty in taken):
                continue
            taken.append((cx, cy))
            windows.append(dict(x=round(wx, 1), y=round(wy, 1), w=int(REGION_WINDOW), h=int(REGION_WINDOW),
                                crossings=c['crossings'], obstacle_hits=c['obstacles'], node_overlaps=c['node_overlaps'],
                                edges=c['edges'], reactions=sorted(c['reactions'])[:10]))
            if len(windows) == 5:
                break
    ori = orientation_module.summary(data, orientation_kinds)
    # ---- packing diagnostics on reaction midmarkers (isolation / densest cell)
    mids_xy = mids if len(mids) else np.zeros((0, 2))
    isolated_n = 0
    densest_cell = None
    outlier_rows = []
    if len(mids_xy) > 8:
        cell = max(spacing * 4.0, 50.0)
        gx = ((mids_xy[:, 0] - mids_xy[:, 0].min()) // cell).astype(int)
        gy = ((mids_xy[:, 1] - mids_xy[:, 1].min()) // cell).astype(int)
        d5 = cKDTree(mids_xy).query(mids_xy, k=min(6, len(mids_xy)))[0][:, -1]
        med = float(np.median(d5))
        outliers = np.flatnonzero(d5 > 8.0 * max(med, 1.0))
        counts = _dd(int)
        for a_, b_ in zip(gx.tolist(), gy.tolist()):
            counts[a_, b_] += 1
        isolated_n = int(len(outliers))
        densest_cell = int(max(counts.values()))
        outlier_rows = [dict(x=round(float(mids_xy[i][0]), 1), y=round(float(mids_xy[i][1]), 1), neighbour5_px=round(float(d5[i]))) for i in outliers[:200]]
    n_alg = max(len(algo), 1)
    counted_cross = cross['algorithm_algorithm'] + cross['algorithm_curated']
    continuity = _structure_continuity(data, structures=structures)
    # Keep the card payload small; full per-structure rows go under details.
    continuity_core = {k: continuity.get(k) for k in (
        'contiguous_pair_share', 'pairs', 'contiguous_pairs', 'structures', 'split_structures',
        'split_structure_share', 'reaction_spacing_px', 'adjacency_factor', 'note')}
    # Participant placement: ≥120° (transfer-check first guess); core = non-currency only.
    place_all = ori.get('all_participants') or {}
    place_core = ori.get('core_participants') or {}
    p90_px = float(np.percentile(all_len, 90)) if len(all_len) else 0.0
    med_len = float(np.median(all_len)) if len(all_len) else 0.0
    max_rel = round(float(np.max(all_len) / max(med_len, 1e-9)), 3) if len(all_len) else None
    rules = collect_map_rule_details(data, edges, all_len, spacing)
    # One content bbox for map size / aspect / blank·utilisation (see map_rules.content_bbox).
    bbox_w, bbox_h = rules.get('width'), rules.get('height')
    blank_share = rules.get('blank_space_share')
    space_util = rules.get('space_utilisation')
    comp = {}
    if bbox_w is not None and bbox_h is not None:
        comp = dict(
            canvas=[round(float(bbox_w)), round(float(bbox_h))],
            aspect=rules.get('aspect_ratio'),
            aspect_ratio=rules.get('aspect_ratio'),
            aspect_ratio_target=rules.get('aspect_ratio_target'),
            aspect_ratio_vs_target=rules.get('aspect_ratio_vs_target'),
            blank_space_share=blank_share,
            space_utilisation=space_util,
            largest_empty_rect_share=rules.get('largest_empty_rect_share'),
            cell_px=rules.get('cell_px'),
            isolated_reactions=isolated_n,
            max_reactions_in_one_cell=densest_cell,
        )
    cofactor = dict(
        currency_hub_violations=rules.get('currency_hub_violations'),
        currency_long_arc_share=rules.get('currency_long_arc_share'),
        currency_glyphs=rules.get('currency_glyphs'),
        hub_threshold_spacings=rules.get('hub_threshold_spacings'),
        long_threshold_spacings=rules.get('long_threshold_spacings'),
    )
    overlaps = dict(
        overlapping_pairs=len(node_overlap) + label_overlap,
        node_overlap_pairs=len(node_overlap),
        label_overlap_pairs=label_overlap,
        curves_through_nodes=node_hits,
        curves_through_labels=label_hits,
    )
    bezier = dict(share=round(bez_edges / n_alg, 3), algorithm_edges=len(algo), bezier=bez_edges, straight_lines=straight_edges,
                  polylines_with_bends=poly_edges, orthogonal_polylines=ortho_edges)
    warnings = []
    if bezier['share'] < 0.999:
        warnings.append(dict(code='bezier_coverage', message='algorithm connections as Bézier must be 100%', share=bezier['share']))
    cross_out = dict(
        counted=counted_cross,
        total=counted_cross + cross['curated_curated'] + cross['same_reaction'] + cross['same_reaction_curated'],
        same_reaction=cross['same_reaction'],
        same_reaction_curated=cross['same_reaction_curated'],
        same_reaction_expert=cross['same_reaction_curated'],  # alias
        algorithm_algorithm=cross['algorithm_algorithm'],
        algorithm_curated=cross['algorithm_curated'],
        curated_curated=cross['curated_curated'],
        algorithm_expert=cross['algorithm_curated'],  # alias for quality_gate
        expert_expert=cross['curated_curated'],  # alias
        main_metabolite=cross['main_metabolite'],
        currency_involved=cross['currency_involved'],
        per_100_edges=round(100.0 * counted_cross / n_alg, 2),
    )
    line_shape = dict(
        detour_median=round(float(np.median(detours)), 3) if detours else None,
        detour_p90=round(float(np.percentile(detours, 90)), 3) if detours else None,
        detour_over_1_6=int(sum(d > 1.6 for d in detours)),
        hooks=hooks, sharp_turns=sharp, self_intersections=selfx,
        bends_per_arc=rules.get('bends_per_arc'), bent_arc_share=rules.get('bent_arc_share'))
    cycles = dict(cycle_circularity=rules.get('cycle_circularity'), cycle_direction=rules.get('cycle_direction'),
                  cycle_direction_consistency=rules.get('cycle_direction_consistency'),
                  cycle_order_consistency=rules.get('cycle_order_consistency'),
                  cycles=rules.get('cycles'), cycle_len=rules.get('cycle_len'))
    orderliness = dict(
        main_segment_axis_share=rules.get('main_segment_axis_share'),
        main_segment_axis_share_length=rules.get('main_segment_axis_share_length'),
        reaction_axis_share=rules.get('reaction_axis_share'),
        chain_turn_median_deg=rules.get('chain_turn_median_deg'),
        metabolite_grid_share=rules.get('metabolite_grid_share'),
        segments=rules.get('orderliness_segments'),
        reactions=rules.get('orderliness_reactions'),
        chain_turns=rules.get('orderliness_chain_turns'),
        metabolites=rules.get('orderliness_metabolites'),
        grid_tol_px=rules.get('orderliness_grid_tol_px'),
    )
    unrelated = dict(
        node_overlaps=rules.get('unrelated_node_overlaps'),
        edge_through_node=rules.get('unrelated_edge_through_node'),
        collinear_edge_overlaps=rules.get('unrelated_collinear_edge_overlaps'),
        total=rules.get('unrelated_overlaps_total'),
    )
    crossing_density = dict(worst_window_crossings=windows[0]['crossings'] if windows else 0)
    # Packing diagnosis from the same crossing/edge pass (no second detect_crossings).
    ac_report = algorithm_curated_crossing_report(cross=cross, events=all_cross_events, top=40)
    long_links_after = longest_links_report(data=data, edges=edges, nodes=nodes, top=10)
    details = dict(
        algorithm_curated_crossings=ac_report,
        longest_links=dict(after=long_links_after),
        line_shape=line_shape,
        cycles=cycles,
        orderliness=orderliness,
        main_segment_axis_share=orderliness.get('main_segment_axis_share'),
        main_segment_axis_share_length=orderliness.get('main_segment_axis_share_length'),
        reaction_axis_share=orderliness.get('reaction_axis_share'),
        chain_turn_median_deg=orderliness.get('chain_turn_median_deg'),
        metabolite_grid_share=orderliness.get('metabolite_grid_share'),
        flow_main_dir=rules.get('flow_main_dir'),
        flow_main_dir_share=rules.get('flow_main_dir_share'),
        exchange_periphery_share=rules.get('exchange_periphery_share'),
        blank_space=dict(
            blank_space_share=blank_share,
            space_utilisation=space_util,
            largest_empty_rect_share=rules.get('largest_empty_rect_share'),
            cell_px=rules.get('cell_px'),
            spacing_px=rules.get('spacing_px'),
        ),
        blank_space_share=blank_share,
        space_utilisation=space_util,
        largest_empty_rect_share=rules.get('largest_empty_rect_share'),
        unrelated_overlaps=unrelated,
        aspect_ratio=rules.get('aspect_ratio'),
        aspect_ratio_target=rules.get('aspect_ratio_target'),
        aspect_ratio_vs_target=rules.get('aspect_ratio_vs_target'),
        species_in_other_compartment=rules.get('species_in_other_compartment'),
        thresholds_note=rules.get('thresholds_note'),
        crossing_density=crossing_density,
        orientation=ori, composition=comp, structure_continuity=continuity,
        edges_footnote=dict(algorithm=len(algo), curated=len(edges) - len(algo), total=len(edges)),
        warnings=warnings,
        bezier_coverage=bezier,
    )
    details['panel'] = _detailed_metrics_panel(details, line_shape=line_shape, cycles=cycles,
                                               unrelated=unrelated, crossing_density=crossing_density,
                                               composition=comp)
    summary = dict(
        edges=dict(total=len(edges), algorithm=len(algo), curated=len(edges) - len(algo), expert=len(edges) - len(algo)),
        core=dict(
            crossings=cross_out,
            overlaps=overlaps,
            collisions=overlaps,  # alias for quality_gate
            long_lines=dict(
                p90_reaction_spacings=round(p90_px / max(spacing, 1.0), 2),
                edge_len_max_rel=max_rel,
                p90_px=round(p90_px, 1),
                reaction_spacing_px=round(spacing, 1),
                over_250=int((all_len > 250).sum()) if len(all_len) else 0,
            ),
            length_p90=dict(px=round(p90_px, 1), in_reaction_spacings=round(p90_px / max(spacing, 1.0), 2), reaction_spacing_px=round(spacing, 1)),
            participant_placement=dict(
                inout_opposite_share=place_all.get('opposite_sides_share'),
                inout_opposite_share_core=place_core.get('opposite_sides_share'),
                reactions_all=place_all.get('reactions'),
                reactions_core=place_core.get('reactions'),
                threshold_deg=ori.get('threshold_deg', 120.0),
            ),
            core_orientation=dict(opposite_sides_share=place_core.get('opposite_sides_share'), reactions=place_core.get('reactions')),
            cofactor_clones=cofactor,
            structure_continuity=continuity_core,
            bezier_coverage=bezier),  # not a card; quality-gate hard check + warning
        details=details,
        worst_regions=windows,
        note='computed from the final result map JSON; crossings are counted on sampled final curves, source-split '
             '(algorithm×algorithm + algorithm×curated; curated×curated not in the headline) and role-split '
             '(main-metabolite vs currency-involved); worst_regions use uncapped counted crossings in true '
             '400 px / stride-200 boxes; structure continuity scores reused curated structures; '
             '120° opposite-side threshold is a Project 22 first guess; '
             'Detailed metrics keep only layout goals (one content bbox for size/aspect/blank)')
    issues = dict(isolated_reactions=outlier_rows, crossings=cross_points, obstacle_hits=hit_points, curve_anomalies=anomalies, worst_regions=windows,
                  node_overlaps=[dict(x=round(float(centres[i][0]), 1), y=round(float(centres[i][1]), 1)) for i, _ in node_overlap][:max_issues],
                  split_structures=[d for d in continuity.get('details') or [] if d.get('components', 1) > 1][:max_issues],
                  warnings=warnings)
    return summary, issues



STAGE_NOT_YET = 'not yet optimised'


def _row(label, value, *, code=None, tip=None, better=None, target=None, stage=None, miss=False):
    """One Detailed-metrics table row (English label is the i18n key in the tool page)."""
    return dict(
        label=label,
        value=value,
        code=code,
        tip=tip,
        better=better,
        target=target,
        stage=stage,
        miss=bool(miss),
    )


def _pct(v):
    if v is None:
        return '—'
    return f'{round(100 * float(v))}%'


def _na(v):
    return '—' if v is None or v == '' else v


def _detailed_metrics_panel(details, *, line_shape, cycles, unrelated, crossing_density, composition):
    """Grouped rows for the Layout quality → Detailed metrics panel.

    Show a metric on the main table only when it is a layout target (including
    backlog / not-yet-optimised targets). Non-target diagnostics go under Debug.
    Pipeline diagnostics (composition_pass / peripheral) are attached later by
    the tool UI when those keys exist on ``details``.
    """
    ls, cy, uo, cd, comp = line_shape, cycles, unrelated, crossing_density, composition or {}
    worst = (cd or {}).get('worst_window_crossings')
    uo_total = (uo or {}).get('total')
    detour_over = ls.get('detour_over_1_6')
    hooks = ls.get('hooks')
    sharp = ls.get('sharp_turns')
    selfx = ls.get('self_intersections')
    bends = ls.get('bends_per_arc')
    exch = details.get('exchange_periphery_share')
    circ = cy.get('cycle_circularity')
    flow = details.get('flow_main_dir_share')
    seg_axis = details.get('main_segment_axis_share')
    rx_axis = details.get('reaction_axis_share')
    chain_turn = details.get('chain_turn_median_deg')
    grid = details.get('metabolite_grid_share')
    species = details.get('species_in_other_compartment')
    blank = details.get('blank_space_share')
    util = details.get('space_utilisation')
    if util is None and blank is not None:
        util = round(1.0 - float(blank), 4)
    aspect = details.get('aspect_ratio')
    aspect_t = details.get('aspect_ratio_target')
    aspect_d = details.get('aspect_ratio_vs_target')
    largest = details.get('largest_empty_rect_share')
    isolated = comp.get('isolated_reactions')
    densest = comp.get('max_reactions_in_one_cell')
    canvas = comp.get('canvas')

    # Targets currently measured; stage is STAGE_NOT_YET until a pipeline owns them.
    cycle_is_target = True
    flow_is_target = False  # panel target is false; kept under Debug diagnostics
    from evaluation.map_rules import GRID_TOL_FLOOR_PX, GRID_TOL_NN_FRAC, ORDERLINESS_GOALS
    og = ORDERLINESS_GOALS

    structure_rows = [
        _row(
            'Exchange reactions on the periphery',
            _pct(exch),
            code='exchange_periphery_share',
            tip='Share of exchange/demand reactions that sit in the outer band of their component. Higher is better; target most exchanges outside the core.',
            better='higher', target='≥ 50%',
            stage='composition & peripherals',
            miss=(exch is not None and exch < 0.5),
        ),
        _row(
            'Axis-aligned main edges',
            _pct(seg_axis),
            code='main_segment_axis_share',
            tip=(
                f'Share of main (non-cofactor) reaction–metabolite edges whose path is ≥50% '
                f'within ±10° of H/V (count-weighted: each edge once, independent of length). '
                f'Higher is better; goal {round(100 * og["main_segment_axis_share"])}% on curated '
                f'BiGG e_coli_core. Length-weighted share is reported separately.'
            ),
            better='higher',
            target=f'≥ {round(100 * og["main_segment_axis_share"])}% (curated e_coli_core)',
            stage=STAGE_NOT_YET,
            miss=(seg_axis is not None and seg_axis < og['main_segment_axis_share']),
        ),
        _row(
            'Axis-aligned reaction axes',
            _pct(rx_axis),
            code='reaction_axis_share',
            tip=f'Share of reactions whose substrate→product axis (non-cofactor) is within ±10° of horizontal or vertical. Higher is better; goal {round(100 * og["reaction_axis_share"])}% on curated BiGG e_coli_core.',
            better='higher',
            target=f'≥ {round(100 * og["reaction_axis_share"])}% (curated e_coli_core)',
            stage=STAGE_NOT_YET,
            miss=(rx_axis is not None and rx_axis < og['reaction_axis_share']),
        ),
        _row(
            'Grid-aligned metabolites',
            _pct(grid),
            code='metabolite_grid_share',
            tip=f'Share of main metabolite glyphs that share an x or y with another nearby main glyph (tolerance {GRID_TOL_NN_FRAC:g} × median nearest-neighbour main-metabolite distance, floor {GRID_TOL_FLOOR_PX:g} px). Higher is better; goal {round(100 * og["metabolite_grid_share"])}% on curated BiGG e_coli_core.',
            better='higher',
            target=f'≥ {round(100 * og["metabolite_grid_share"])}% (curated e_coli_core)',
            stage=STAGE_NOT_YET,
            miss=(grid is not None and grid < og['metabolite_grid_share']),
        ),
        _row(
            'Median chain turn angle',
            '—' if chain_turn is None else f'{chain_turn}°',
            code='chain_turn_median_deg',
            tip=f'Median turn at a main metabolite shared by exactly two reactions (0° = the two midmarkers sit on opposite sides of the metabolite). Lower is better; goal {og["chain_turn_median_deg"]:g}° on curated BiGG e_coli_core.',
            better='lower',
            target=f'{og["chain_turn_median_deg"]:g}° (curated e_coli_core)',
            stage=STAGE_NOT_YET,
            miss=(chain_turn is not None and chain_turn > og['chain_turn_median_deg']),
        ),
    ]
    if cycle_is_target:
        structure_rows.append(_row(
            'Cycle circularity / direction / order',
            f"{_na(circ)} / {_na(cy.get('cycle_direction_consistency'))} ({_na(cy.get('cycle_direction'))}) / {_na(cy.get('cycle_order_consistency'))}",
            code='cycle_circularity',
            tip='How round and consistently oriented drawn cycles are (isoperimetric quotient and neighbour order). Higher is better; goal higher circularity and neighbour-order consistency.',
            better='higher', target='higher circularity & consistency',
            stage=STAGE_NOT_YET,
            miss=(circ is not None and circ < 0.35),
        ))
    structure_rows.append(_row(
        'Species in another compartment',
        _na(species) if species is None else f'{round(float(species), 3)}',
        code='species_in_other_compartment',
        tip='Share of metabolite glyphs that fall inside another compartment hull. Lower is better; target 0.',
        better='lower', target='0',
        stage=STAGE_NOT_YET,
        miss=(species is not None and float(species) > 0.05),
    ))

    groups = [
        dict(id='connections', label='Connections', collapsed=False, rows=[
            _row(
                'Local crossing hotspot',
                _na(worst),
                code='worst_window_crossings',
                tip='Most algorithm×algorithm / algorithm×curated crossings inside one 400 px window (stride 200). Lower is better; target as few as possible (headline crossings are on the cards).',
                better='lower', target='as few as possible',
                stage='Bézier routing · quality gate',
                miss=(worst is not None and worst > 8),
            ),
            _row(
                'Unrelated overlaps',
                f"{_na((uo or {}).get('node_overlaps'))} / {_na((uo or {}).get('edge_through_node'))} / {_na((uo or {}).get('collinear_edge_overlaps'))}",
                code='unrelated_overlaps',
                tip='Overlaps between reactions that share no non-currency metabolite (node · edge-through-node · collinear edge). Lower is better; target 0.',
                better='lower', target='0',
                stage='spacing / relaxation · Bézier routing',
                miss=(uo_total is not None and float(uo_total) > 0),
            ),
        ]),
        dict(id='line_shape', label='Line shape', collapsed=False, rows=[
            _row(
                'Detour (median · p90 · above 1.6)',
                f"{_na(ls.get('detour_median'))} / {_na(ls.get('detour_p90'))} / {_na(detour_over)}",
                code='detour',
                tip='Path length ÷ straight chord for algorithm connections. Lower is better; Bézier routing penalises detour above 1.6.',
                better='lower', target='≤ 1.6',
                stage='Bézier routing',
                miss=(detour_over is not None and detour_over > 0) or (ls.get('detour_p90') is not None and ls.get('detour_p90') > 1.6),
            ),
            _row(
                'Hooks / sharp turns / self-intersections',
                f"{_na(hooks)} / {_na(sharp)} / {_na(selfx)}",
                code='curve_anomalies',
                tip='Curve-shape defects on algorithm Béziers (large winding hooks, sharp corners, self-crossings). Lower is better; target 0.',
                better='lower', target='0',
                stage='Bézier routing',
                miss=((hooks or 0) + (sharp or 0) + (selfx or 0)) > 0,
            ),
            _row(
                'Bends per connection',
                f"{_na(bends)}" + (f" · bent {_pct(ls.get('bent_arc_share'))}" if ls.get('bent_arc_share') is not None else ''),
                code='bends_per_arc',
                tip='Mean direction changes per connection. Lower is better on curated Bézier maps; target near 0.',
                better='lower', target='near 0',
                stage='Bézier routing',
                miss=(bends is not None and bends > 0.5),
            ),
        ]),
        dict(id='structure_flow', label='Structure & flow', collapsed=False, rows=structure_rows),
        dict(id='space_shape', label='Space & shape', collapsed=False, rows=[
            _row(
                'Map size',
                (f"{canvas[0]} × {canvas[1]} px" if canvas else '—'),
                code='content_bbox',
                tip='Width × height of the content-node bounding box (metabolites + reaction midmarkers). Same box as aspect and blank space.',
                better=None, target='readable, not sprawling',
                stage='composition & packing',
                miss=False,
            ),
            _row(
                'Aspect ratio (W/H) vs target',
                f"{_na(aspect)} · target {_na(aspect_t)} · |Δ| {_na(aspect_d)}",
                code='aspect_ratio',
                tip='Content width ÷ height versus target 1.0. Closer to the target is better.',
                better='closer to target', target='1.0',
                stage='composition & packing',
                miss=(aspect_d is not None and float(aspect_d) > 0.5),
            ),
            _row(
                'Blank space / space utilisation',
                f"{_pct(blank)} blank · {_pct(util)} used",
                code='blank_space_share',
                tip='Blank cells versus occupied cells on one raster of the content box (utilisation = 1 − blank). Lower blank / higher utilisation is better.',
                better='lower blank', target='balanced packing',
                stage='composition & packing',
                miss=(blank is not None and float(blank) > 0.7),
            ),
            _row(
                'Largest empty rectangle',
                _pct(largest),
                code='largest_empty_rect_share',
                tip='Largest empty axis-aligned block as a share of the blank-space grid. Lower is better; large voids mean weak packing.',
                better='lower', target='small voids',
                stage='composition & packing',
                miss=(largest is not None and float(largest) > 0.35),
            ),
            _row(
                'Isolated reactions',
                _na(isolated),
                code='isolated_reactions',
                tip='Reactions whose 5th-nearest neighbour is farther than 8 × the median 5th-neighbour distance. Lower is better; target 0.',
                better='lower', target='0',
                stage='composition & packing',
                miss=(isolated is not None and isolated > 0),
            ),
            _row(
                'Most reactions in one cell',
                _na(densest),
                code='max_reactions_in_one_cell',
                tip='Peak reaction count in one spacing-scaled packing cell. Lower local peaks are better; flags overcrowding.',
                better='lower', target='no extreme peaks',
                stage='composition & packing',
                miss=(densest is not None and densest > 20),
            ),
        ]),
    ]
    repro = details.get('reproducibility')
    if isinstance(repro, dict) and (repro.get('procrustes_disparity') is not None or repro.get('mean_displacement') is not None
                                    or repro.get('procrustes_mean') is not None or repro.get('mean_displacement_mean') is not None):
        groups.append(dict(id='reproducibility', label='Reproducibility', collapsed=False, rows=[
            _row(
                'Reproducibility (Procrustes / mean displacement)',
                f"{_na(repro.get('procrustes_disparity', repro.get('procrustes_mean')))} / {_na(repro.get('mean_displacement', repro.get('mean_displacement_mean')))}",
                code='reproducibility',
                tip='Displacement versus a compare-map (evaluate --compare-map). Lower is better; target stable across runs.',
                better='lower', target='stable across runs',
                stage=STAGE_NOT_YET,
                miss=False,
            ),
        ]))
    debug_rows = []
    if not flow_is_target and (flow is not None or details.get('flow_main_dir')):
        debug_rows.append(_row(
            'Main flow-direction share',
            _pct(flow) + (f" ({details.get('flow_main_dir')})" if details.get('flow_main_dir') else ''),
            code='flow_main_dir_share',
            tip='Share of reactant→product axes aligned with the dominant reading direction.',
            better=None, target='n/a (debug)',
            stage='debug only',
            miss=False,
        ))
    if not cycle_is_target and circ is not None:
        debug_rows.append(_row(
            'Cycle circularity / direction / order',
            f"{_na(circ)} / {_na(cy.get('cycle_direction_consistency'))} ({_na(cy.get('cycle_direction'))}) / {_na(cy.get('cycle_order_consistency'))}",
            code='cycle_circularity',
            tip='Isoperimetric quotient and neighbour-order consistency for drawn cycles.',
            better=None, target='n/a (debug)',
            stage='debug only',
            miss=False,
        ))
    if debug_rows:
        groups.append(dict(id='debug', label='Debug diagnostics', collapsed=True, debug=True, rows=debug_rows))
    return dict(groups=groups, stage_not_yet=STAGE_NOT_YET)


def _turn_stats(pts):
    v = np.diff(pts, axis=0)
    n = np.hypot(v[:, 0], v[:, 1])
    ok = n > 1e-9
    v = v[ok] / n[ok][:, None]
    if len(v) < 2:
        return 0.0, 0.0
    ang = np.degrees(np.arccos(np.clip((v[:-1] * v[1:]).sum(1), -1, 1)))
    cross = v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0]
    return float(ang.max()), float(abs(np.where(cross >= 0, ang, -ang).sum()))


def _self_cross(pts):
    for i in range(len(pts) - 1):
        for j in range(i + 2, len(pts) - 1):
            a, b, c, d = pts[i], pts[i + 1], pts[j], pts[j + 1]
            r, s = b - a, d - c
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-9:
                continue
            t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
            u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
            if 1e-3 < t < 1 - 1e-3 and 1e-3 < u < 1 - 1e-3:
                return True
    return False


def _orthogonal(pts):
    """Straight pieces that turn by ~90 degrees at every corner (a right-angle polyline)."""
    v = np.diff(pts, axis=0)
    n = np.hypot(v[:, 0], v[:, 1])
    v = v[n > 1e-6] / n[n > 1e-6][:, None]
    corners = 0
    for a, b in zip(v[:-1], v[1:]):
        ang = float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
        if ang > 20:
            if abs(ang - 90) > 15:
                return False
            corners += 1
    return corners >= 1


def _skip_edge_set(data, skip_edges=None):
    meta = data[0].get('metadata', {})
    skip = {tuple(e) for e in (skip_edges if skip_edges is not None else meta.get('preserved_edges') or [])}
    skip |= {(p['reaction'], p['metabolite']) for p in (meta.get('source_paths') or [])}
    return skip


def _edge_records_for_reaction(rx, nodes, skip):
    """One edge record per mid→metabolite terminal path (same geometry as ``evaluate``)."""
    import edge_routing as er
    import line_render
    out = []
    for path in line_render._terminals(rx, nodes):
        if nodes[path[-1]].get('node_type') != 'metabolite':
            continue
        pts = np.array(er._path_points(rx, path, nodes), float)
        segs = [s for s in rx['segments'].values() if s['from_node_id'] in path and s['to_node_id'] in path]
        curved = [bool(s.get('b1') and s.get('b2')) for s in segs]
        met = nodes[path[-1]].get('bigg_id')
        curated = (rx['bigg_id'], met) in skip
        currency = _is_currency_met(met) or (nodes[path[-1]].get('node_is_primary') is False)
        out.append(dict(rid=rx['bigg_id'], met=met, path=path, pts=pts, curated=curated, expert=curated,
                        currency=currency, curved=curved, ends=(path[0], path[-1])))
    return out


def build_edges(data, skip_edges=None):
    """Edge records used by ``evaluate`` for curve crossings (shared with hotspot repair)."""
    skip = _skip_edge_set(data, skip_edges)
    body = data[1]
    nodes, reactions = body['nodes'], body['reactions']
    edges = []
    for rx in reactions.values():
        edges.extend(_edge_records_for_reaction(rx, nodes, skip))
    return edges, nodes


def build_edges_for_rids(data, rids, skip_edges=None):
    """Rebuild edge records only for the given reaction bigg_ids (incremental repair)."""
    skip = _skip_edge_set(data, skip_edges)
    body = data[1]
    nodes = body['nodes']
    want = set(rids)
    edges = []
    for rx in body['reactions'].values():
        if rx['bigg_id'] in want:
            edges.extend(_edge_records_for_reaction(rx, nodes, skip))
    return edges, nodes


def detect_crossings(edges, nodes, max_issues=4000):
    """Same segment-crossing detector as ``evaluate`` (algorithm/curated split, shared-node rule)."""
    seg_a, seg_b, seg_edge = [], [], []
    for i, e in enumerate(edges):
        for a, b in zip(e['pts'][:-1], e['pts'][1:]):
            seg_a.append(a); seg_b.append(b); seg_edge.append(i)
    S = np.array([[a[0], a[1], b[0], b[1]] for a, b in zip(seg_a, seg_b)], float).reshape(-1, 4)
    seg_edge = np.array(seg_edge)
    grid_s = _dd(list)
    for k, (a, b) in enumerate(zip(seg_a, seg_b)):
        _grid_add(grid_s, k, np.minimum(a, b), np.maximum(a, b))
    cross = dict(algorithm_algorithm=0, algorithm_curated=0, curated_curated=0, same_reaction=0, same_reaction_curated=0,
                 main_metabolite=0, currency_involved=0)
    cross_points, all_cross_events, per_edge_cross = [], [], _dd(int)
    for k, (a, b) in enumerate(zip(seg_a, seg_b)):
        i = seg_edge[k]
        cand = np.array(sorted(c for c in _grid_query(grid_s, np.minimum(a, b), np.maximum(a, b)) if seg_edge[c] > i), dtype=int)
        if not len(cand):
            continue
        idx, pts = _seg_hits(a, b, S[cand])
        for c, pt in zip(cand[idx], pts):
            j = seg_edge[c]
            if edges[i]['rid'] == edges[j]['rid']:
                centre = nodes[edges[i]['ends'][0]]
                if float(np.hypot(pt[0] - centre['x'], pt[1] - centre['y'])) > 14.0:
                    both_curated = edges[i]['curated'] and edges[j]['curated']
                    cross['same_reaction_curated' if both_curated else 'same_reaction'] += 1
                    per_edge_cross[i] += 1; per_edge_cross[j] += 1
                    ev = dict(kind='same_reaction_curated' if both_curated else 'same_reaction',
                              reactions=[edges[i]['rid']], metabolites=[edges[i]['met'], edges[j]['met']],
                              x=round(float(pt[0]), 1), y=round(float(pt[1]), 1))
                    all_cross_events.append(ev)
                    if len(cross_points) < max_issues:
                        cross_points.append(ev)
                continue
            if set(edges[i]['ends']) & set(edges[j]['ends']):
                continue
            ei, ej = edges[i]['curated'], edges[j]['curated']
            kindc = 'curated_curated' if ei and ej else ('algorithm_algorithm' if not ei and not ej else 'algorithm_curated')
            cross[kindc] += 1
            if kindc != 'curated_curated':
                if edges[i]['currency'] or edges[j]['currency']:
                    cross['currency_involved'] += 1
                else:
                    cross['main_metabolite'] += 1
            per_edge_cross[i] += 1; per_edge_cross[j] += 1
            legacy = {'algorithm_curated': 'algorithm_expert', 'curated_curated': 'expert_expert'}.get(kindc, kindc)
            ev = dict(kind=kindc, kind_legacy=legacy, reactions=[edges[i]['rid'], edges[j]['rid']],
                      metabolites=[edges[i]['met'], edges[j]['met']],
                      x=round(float(pt[0]), 1), y=round(float(pt[1]), 1),
                      currency_involved=bool(edges[i]['currency'] or edges[j]['currency']))
            if kindc == 'algorithm_curated':
                if edges[i]['curated'] and not edges[j]['curated']:
                    ev['curated_rid'], ev['algorithm_rid'] = edges[i]['rid'], edges[j]['rid']
                    ev['curated_met'], ev['algorithm_met'] = edges[i]['met'], edges[j]['met']
                else:
                    ev['curated_rid'], ev['algorithm_rid'] = edges[j]['rid'], edges[i]['rid']
                    ev['curated_met'], ev['algorithm_met'] = edges[j]['met'], edges[i]['met']
            all_cross_events.append(ev)
            if len(cross_points) < max_issues:
                cross_points.append(ev)
    return cross, all_cross_events, cross_points, per_edge_cross


def algorithm_crossings(data, skip_edges=None):
    """Headline crossing count (algo×algo + algo×curated)."""
    edges, nodes = build_edges(data, skip_edges)
    cross, events, _, _ = detect_crossings(edges, nodes, max_issues=0)
    return int(cross['algorithm_algorithm'] + cross['algorithm_curated']), cross, events


def algorithm_curated_crossing_report(data=None, skip_edges=None, top=40, cross=None, events=None):
    """Which algorithm reactions cross curated edges, and why.

    Reasons:
      spoke/currency   — at least one edge is a currency spoke
      main_metabolite  — both edges are non-currency (core) links
    Each sample also records the algorithm/curated reaction + metabolite endpoints.

    Pass ``cross`` / ``events`` from a prior ``detect_crossings`` (e.g. inside
    ``evaluate``) to avoid a second full crossing pass on large maps.
    """
    from collections import Counter
    if cross is None or events is None:
        edges, nodes = build_edges(data, skip_edges)
        cross, events, _, _ = detect_crossings(edges, nodes, max_issues=0)
    ac = [e for e in events if e.get('kind') == 'algorithm_curated']
    by_algo = Counter()
    why = Counter()
    samples = []
    for e in ac:
        arid = e.get('algorithm_rid') or (e.get('reactions') or [None])[0]
        by_algo[arid] += 1
        reason = 'spoke/currency' if e.get('currency_involved') else 'main_metabolite'
        why[reason] += 1
        if len(samples) < top:
            samples.append(dict(
                algorithm_rid=arid, curated_rid=e.get('curated_rid'),
                algorithm_met=e.get('algorithm_met'), curated_met=e.get('curated_met'),
                reason=reason,
                why=('Algorithm reaction %s’s link to %s crosses curated reaction %s’s link to %s'
                     % (arid, e.get('algorithm_met'), e.get('curated_rid'), e.get('curated_met'))
                     + (' (currency spoke)' if e.get('currency_involved') else ' (main metabolite)')),
                x=e.get('x'), y=e.get('y')))
    top_algos = [{'algorithm_rid': rid, 'crossings': n,
                  'why': next((s['why'] for s in samples if s['algorithm_rid'] == rid), None)}
                 for rid, n in by_algo.most_common(top)]
    return dict(count=len(ac), headline=int(cross['algorithm_algorithm'] + cross['algorithm_curated']),
                by_reason=dict(why), top_algorithm_reactions=top_algos, samples=samples)


def _block_aabbs_from_map(data):
    """structure id -> (lo, hi) from curated midmarkers (+ optional module boxes)."""
    body = data[1]
    nodes = body['nodes']
    structs = (data[0].get('metadata') or {}).get('expert_structures') or []
    rid_block = {}
    for s in structs:
        sid = s.get('id') or s.get('code') or s.get('template_id')
        for rid in s.get('reactions') or []:
            rid_block[rid] = sid
    # Midmarker xy per reaction.
    mid_xy = {}
    for rx in body['reactions'].values():
        rid = rx['bigg_id']
        for seg in rx['segments'].values():
            for nid in (seg['from_node_id'], seg['to_node_id']):
                n = nodes.get(nid) or {}
                if n.get('node_type') == 'midmarker':
                    mid_xy[rid] = (float(n['x']), float(n['y']))
                    break
            if rid in mid_xy:
                break
    from collections import defaultdict
    pts = defaultdict(list)
    for rid, sid in rid_block.items():
        if rid in mid_xy:
            pts[sid].append(mid_xy[rid])
    # Expand each block AABB with metabolites of its curated edges that sit near the mids.
    for e_pts_sid in list(pts):
        pass
    aabbs = {}
    pad = 80.0
    for sid, xy in pts.items():
        arr = np.asarray(xy, float)
        if len(arr) < 1:
            continue
        lo, hi = arr.min(0) - pad, arr.max(0) + pad
        aabbs[sid] = (lo, hi)
    return aabbs, rid_block


def _classify_link(e, nodes, aabbs, rid_block):
    """intra_curated_block | inter_curated_block | algorithm | curated_unscoped."""
    if not e.get('curated'):
        return 'algorithm'
    rid = e['rid']
    bid = rid_block.get(rid)
    mid_id, met_id = e['ends']
    met = nodes.get(met_id) or {}
    mxy = np.array([float(met.get('x', 0)), float(met.get('y', 0))], float)

    def owners(xy):
        hit = []
        for sid, (lo, hi) in aabbs.items():
            if lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]:
                hit.append(sid)
        return hit

    met_owners = owners(mxy)
    if bid is None:
        return 'curated_unscoped'
    if met_owners == [bid] or (bid in met_owners and len(met_owners) == 1):
        return 'intra_curated_block'
    if met_owners and bid not in met_owners:
        return 'inter_curated_block'
    if bid in met_owners and len(met_owners) > 1:
        return 'inter_curated_block'  # shared glyph / multi-block met
    # Met outside every block AABB: still part of this reaction's curated drawing if near its mid.
    mid = nodes.get(mid_id) or {}
    midx = np.array([float(mid.get('x', 0)), float(mid.get('y', 0))], float)
    if bid in aabbs:
        lo, hi = aabbs[bid]
        # Grow own box; if met falls inside grown box → intra (rigid body + label margin).
        grow = 220.0
        if lo[0] - grow <= mxy[0] <= hi[0] + grow and lo[1] - grow <= mxy[1] <= hi[1] + grow:
            return 'intra_curated_block'
    if float(np.hypot(*(mxy - midx))) < 250.0:
        return 'intra_curated_block'
    return 'inter_curated_block'


def longest_links_report(data=None, skip_edges=None, top=10, edges=None, nodes=None):
    """Top-N longest reaction–metabolite edges with endpoints (for long-line diagnosis).

    Pass ``edges`` / ``nodes`` from a prior ``build_edges`` to reuse evaluate’s work.
    Classifies links as intra_curated_block (unchanged by translation-only pack) vs
    inter_curated_block vs algorithm, and reports p90 excluding pure intra-block edges.
    """
    if edges is None or nodes is None:
        edges, nodes = build_edges(data, skip_edges)
    aabbs, rid_block = ({}, {})
    if data is not None:
        aabbs, rid_block = _block_aabbs_from_map(data)
    rows = []
    for e in edges:
        pts = e.get('pts')
        if pts is None:
            continue
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
        a, b = e['ends']
        na, nb = nodes.get(a, {}), nodes.get(b, {})
        klass = _classify_link(e, nodes, aabbs, rid_block) if aabbs else (
            'algorithm' if not e.get('curated') else 'curated_unscoped')
        rows.append(dict(
            reaction=e['rid'], metabolite=e['met'], length_px=round(length, 1),
            curated=bool(e.get('curated')), currency=bool(e.get('currency')),
            link_class=klass,
            from_xy=[round(float(na.get('x', 0)), 1), round(float(na.get('y', 0)), 1)],
            to_xy=[round(float(nb.get('x', 0)), 1), round(float(nb.get('y', 0)), 1)],
            from_type=na.get('node_type'), to_type=nb.get('node_type')))
    rows.sort(key=lambda r: -r['length_px'])
    lengths = [r['length_px'] for r in rows]
    ex = [r['length_px'] for r in rows if r['link_class'] != 'intra_curated_block']
    from collections import Counter
    return dict(
        top=rows[:top], n_edges=len(rows),
        p90=round(float(np.percentile(lengths, 90)), 1) if lengths else 0.0,
        p90_ex_intra_curated=round(float(np.percentile(ex, 90)), 1) if ex else 0.0,
        by_class=dict(Counter(r['link_class'] for r in rows)),
        above_p90_by_class=dict(Counter(
            r['link_class'] for r in rows
            if lengths and r['length_px'] >= float(np.percentile(lengths, 90)))),
        note='intra_curated_block edges are rigid under translation-only packing; '
             'p90_ex_intra_curated is the packing-relevant long-line gate')


def overlap_counts(data):
    """Node and label overlap pair counts (same geometry as ``evaluate``)."""
    from scipy.spatial import cKDTree
    body = data[1]
    nodes = body['nodes']
    centres = np.array([[n['x'], n['y']] for n in nodes.values() if n.get('node_type') in ('metabolite', 'midmarker')], float)
    node_overlap = list(cKDTree(centres).query_pairs(10)) if len(centres) > 1 else []
    rects, owner, kind, label_of = _final_obstacles(data)
    grid_r = _dd(list)
    for i, r in enumerate(rects):
        _grid_add(grid_r, i, r[:2], r[2:])
    lab_idx = [i for i, k in enumerate(kind) if k == 'label']
    label_overlap = 0
    for i in lab_idx:
        for j in _grid_query(grid_r, rects[i][:2], rects[i][2:]):
            if j > i and kind[j] == 'label' and rects[j][0] < rects[i][2] and rects[i][0] < rects[j][2] and rects[j][1] < rects[i][3] and rects[i][1] < rects[j][3]:
                label_overlap += 1
    return len(node_overlap), label_overlap


def _pair_is_headline(ei, ej):
    """True if the pair contributes to the Connection-crossings / window headline."""
    if ei['rid'] == ej['rid']:
        return False
    if set(ei['ends']) & set(ej['ends']):
        return False
    return not (ei['curated'] and ej['curated'])


def _count_pair_crossings(pts_i, ends_i, rid_i, pts_j, ends_j, rid_j, nodes):
    """How many proper crossings between two sampled polylines (same geometry as detect_crossings)."""
    if rid_i == rid_j:
        return 0
    if set(ends_i) & set(ends_j):
        return 0
    n = 0
    for a, b in zip(pts_i[:-1], pts_i[1:]):
        S = np.array([[c[0], c[1], d[0], d[1]] for c, d in zip(pts_j[:-1], pts_j[1:])], float).reshape(-1, 4)
        if not len(S):
            continue
        idx, _pts = _seg_hits(a, b, S)
        n += len(idx)
    return n


class CrossingIndex:
    """Reusable spatial index over sampled edge segments for incremental hotspot repair.

    Built once from ``build_edges`` / the same ``_seg_hits`` geometry as ``evaluate``.
    After a local move, recount only edges whose geometry changed (and their neighbours
    in the segment grid) and apply a delta to the headline crossing total.
    """

    def __init__(self, data, skip_edges=None):
        self.skip_edges = skip_edges
        self.rebuild(data)

    def rebuild(self, data):
        self.data = data
        self.edges, self.nodes = build_edges(data, self.skip_edges)
        self.by_rid = _dd(list)
        for i, e in enumerate(self.edges):
            self.by_rid[e['rid']].append(i)
        self.seg_a, self.seg_b, self.seg_edge = [], [], []
        for i, e in enumerate(self.edges):
            for a, b in zip(e['pts'][:-1], e['pts'][1:]):
                self.seg_a.append(a); self.seg_b.append(b); self.seg_edge.append(i)
        self.S = np.array([[a[0], a[1], b[0], b[1]] for a, b in zip(self.seg_a, self.seg_b)], float).reshape(-1, 4)
        self.seg_edge = np.array(self.seg_edge, dtype=int) if self.seg_edge else np.zeros(0, dtype=int)
        self.grid = _dd(list)
        for k, (a, b) in enumerate(zip(self.seg_a, self.seg_b)):
            _grid_add(self.grid, k, np.minimum(a, b), np.maximum(a, b))
        # Per-edge headline crossing counts (each pair counted once on the lower index).
        self.edge_cross = np.zeros(len(self.edges), dtype=int)
        self.total = 0
        self.events = []  # (i, j, x, y, currency)
        self._recompute_all_pairs()

    def _recompute_all_pairs(self):
        self.edge_cross[:] = 0
        self.total = 0
        self.events = []
        if not len(self.S):
            return
        for k, (a, b) in enumerate(zip(self.seg_a, self.seg_b)):
            i = int(self.seg_edge[k])
            cand = [c for c in _grid_query(self.grid, np.minimum(a, b), np.maximum(a, b)) if self.seg_edge[c] > i]
            if not cand:
                continue
            cand = np.array(sorted(cand), dtype=int)
            idx, pts = _seg_hits(a, b, self.S[cand])
            for c, pt in zip(cand[idx], pts):
                j = int(self.seg_edge[c])
                if not _pair_is_headline(self.edges[i], self.edges[j]):
                    continue
                self.edge_cross[i] += 1
                self.edge_cross[j] += 1
                self.total += 1
                self.events.append((i, j, float(pt[0]), float(pt[1]),
                                    bool(self.edges[i]['currency'] or self.edges[j]['currency'])))

    def headline(self):
        return int(self.total)

    def edge_indices_for_reactions(self, rids):
        out = []
        for rid in rids:
            out.extend(self.by_rid.get(rid, ()))
        return sorted(set(out))

    def participants_in_window(self, win):
        """Reaction ids and node ids that participate in a headline crossing inside ``win``."""
        rids, nids = set(), set()
        x0, y0, w, h = win['x'], win['y'], win['w'], win['h']
        for i, j, x, y, _cur in self.events:
            if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue
            for e in (self.edges[i], self.edges[j]):
                rids.add(e['rid'])
                nids.update(e['ends'])
        return rids, nids

    def window_breakdown(self, win):
        """Classify headline crossings inside the window (currency vs main; long vs short)."""
        x0, y0, w, h = win['x'], win['y'], win['w'], win['h']
        cur = main = long_edge = 0
        spacing = 60.0
        mids = np.array([[n['x'], n['y']] for n in self.nodes.values() if n.get('node_type') == 'midmarker'], float)
        if len(mids) > 2:
            from scipy.spatial import cKDTree
            spacing = float(np.median(cKDTree(mids).query(mids, k=2)[0][:, 1])) or 60.0
        long_thresh = 3.0 * spacing
        for i, j, x, y, currency in self.events:
            if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue
            if currency:
                cur += 1
            else:
                main += 1
            for e in (self.edges[i], self.edges[j]):
                d = e['pts']
                L = float(np.sum(np.hypot(*np.diff(d, axis=0).T))) if len(d) > 1 else 0.0
                if L >= long_thresh:
                    long_edge += 1
                    break
        return dict(total=cur + main, currency_involved=cur, main_metabolite=main,
                    involves_long_edge=long_edge, spacing_px=round(spacing, 1))

    def _crossings_of_edges(self, edge_ids, against_all=True):
        """Headline crossings involving any edge in ``edge_ids`` (each pair once)."""
        if not edge_ids or not len(self.S):
            return 0, []
        focus = set(edge_ids)
        # segments belonging to focus edges
        focus_segs = [k for k, ei in enumerate(self.seg_edge) if int(ei) in focus]
        count = 0
        pairs = []
        seen = set()
        for k in focus_segs:
            a, b = self.seg_a[k], self.seg_b[k]
            i = int(self.seg_edge[k])
            cand = list(_grid_query(self.grid, np.minimum(a, b), np.maximum(a, b)))
            if not cand:
                continue
            cand = np.array(sorted(set(cand)), dtype=int)
            idx, pts = _seg_hits(a, b, self.S[cand])
            for c, pt in zip(cand[idx], pts):
                j = int(self.seg_edge[c])
                if j == i:
                    continue
                if not against_all and j not in focus:
                    continue
                # count each unordered pair once
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                if not _pair_is_headline(self.edges[i], self.edges[j]):
                    continue
                # Avoid double-count when both edges are in focus: only emit when
                # the other segment is not also being scanned as a focus seg with higher k,
                # or simply always use seen.
                seen.add(key)
                count += 1
                pairs.append(key)
        return count, pairs

    def delta_after_replace(self, data, changed_rids):
        """Recount crossings after local geometry edits to ``changed_rids``.

        Only rebuilds edge polylines for those reactions and queries them against
        the existing segment grid (skipping stale segments of the moved edges).
        Returns ``(new_total, delta, payload)``; call ``commit(payload)`` to keep.
        """
        changed_rids = set(changed_rids)
        if not changed_rids:
            return int(self.total), 0, None
        old_ids = []
        for rid in changed_rids:
            old_ids.extend(self.by_rid.get(rid, ()))
        old_ids = sorted(set(old_ids))
        old_set = set(old_ids)
        # Cheap: reuse stored events instead of re-querying old geometry.
        old_involving = sum(1 for i, j, _x, _y, _c in self.events if i in old_set or j in old_set)

        new_partial, new_nodes = build_edges_for_rids(data, changed_rids, self.skip_edges)
        # Crossings among the new partial edges themselves.
        new_involving = 0
        new_events = []  # (local_a, local_b, x, y, currency) among partial, plus (None, global_j, ...)
        partial_segs = []  # (local_edge_idx, a, b)
        for li, e in enumerate(new_partial):
            for a, b in zip(e['pts'][:-1], e['pts'][1:]):
                partial_segs.append((li, a, b))
        # partial × partial (each segment–segment hit counts once, matching detect_crossings)
        for si, (li, a, b) in enumerate(partial_segs):
            others = [t for t in range(si + 1, len(partial_segs)) if partial_segs[t][0] != li]
            if not others:
                continue
            S = np.array([[partial_segs[t][1][0], partial_segs[t][1][1],
                           partial_segs[t][2][0], partial_segs[t][2][1]] for t in others], float).reshape(-1, 4)
            idx, pts = _seg_hits(a, b, S)
            for t_i, pt in zip(idx, pts):
                lj = partial_segs[others[t_i]][0]
                if li > lj:
                    continue  # unordered pair once
                if not _pair_is_headline(new_partial[li], new_partial[lj]):
                    continue
                new_involving += 1
                new_events.append(('partial', li, lj, float(pt[0]), float(pt[1]),
                                   bool(new_partial[li]['currency'] or new_partial[lj]['currency'])))
        # partial × unchanged (via existing grid; skip stale changed-edge segments)
        if len(self.S) and partial_segs:
            for li, a, b in partial_segs:
                cand = list(_grid_query(self.grid, np.minimum(a, b), np.maximum(a, b)))
                if not cand:
                    continue
                keep = np.array(sorted({c for c in cand if int(self.seg_edge[c]) not in old_set}), dtype=int)
                if not len(keep):
                    continue
                idx, pts = _seg_hits(a, b, self.S[keep])
                for c, pt in zip(keep[idx], pts):
                    j = int(self.seg_edge[c])
                    if not _pair_is_headline(new_partial[li], self.edges[j]):
                        continue
                    new_involving += 1
                    new_events.append(('vs', li, j, float(pt[0]), float(pt[1]),
                                       bool(new_partial[li]['currency'] or self.edges[j]['currency'])))
        new_total = self.total - old_involving + new_involving
        payload = dict(
            changed_rids=changed_rids,
            old_ids=old_ids,
            new_partial=new_partial,
            new_nodes=new_nodes,
            new_involving=new_involving,
            new_events=new_events,
            new_total=int(new_total),
        )
        return int(new_total), int(new_involving - old_involving), payload

    def commit(self, payload):
        """Apply a successful ``delta_after_replace`` payload (splice edges + refresh segments)."""
        if not payload:
            return
        old_set = set(payload['old_ids'])
        # Drop events that touched the old changed edges.
        kept_events = [(i, j, x, y, c) for i, j, x, y, c in self.events if i not in old_set and j not in old_set]
        # Build new edge list: keep unchanged, append new partial.
        new_edges = [e for i, e in enumerate(self.edges) if i not in old_set]
        base = len(new_edges)
        new_edges.extend(payload['new_partial'])
        # Remap kept event indices: old index -> new index for unchanged edges.
        remap = {}
        ni = 0
        for oi in range(len(self.edges)):
            if oi in old_set:
                continue
            remap[oi] = ni
            ni += 1
        remapped = []
        for i, j, x, y, c in kept_events:
            remapped.append((remap[i], remap[j], x, y, c))
        # Append new events with remapped / new indices.
        for kind, a, b, x, y, cur in payload['new_events']:
            if kind == 'partial':
                remapped.append((base + a, base + b, x, y, cur))
            else:  # vs partial×global
                remapped.append((base + a, remap[b], x, y, cur))
        self.edges = new_edges
        self.nodes = payload['new_nodes']
        self.by_rid = _dd(list)
        for i, e in enumerate(self.edges):
            self.by_rid[e['rid']].append(i)
        self.seg_a, self.seg_b, self.seg_edge = [], [], []
        for i, e in enumerate(self.edges):
            for a, b in zip(e['pts'][:-1], e['pts'][1:]):
                self.seg_a.append(a); self.seg_b.append(b); self.seg_edge.append(i)
        self.S = np.array([[a[0], a[1], b[0], b[1]] for a, b in zip(self.seg_a, self.seg_b)], float).reshape(-1, 4)
        self.seg_edge = np.array(self.seg_edge, dtype=int) if self.seg_edge else np.zeros(0, dtype=int)
        self.grid = _dd(list)
        for k, (a, b) in enumerate(zip(self.seg_a, self.seg_b)):
            _grid_add(self.grid, k, np.minimum(a, b), np.maximum(a, b))
        self.events = remapped
        self.total = len(self.events)
        self.edge_cross = np.zeros(len(self.edges), dtype=int)
        for i, j, _x, _y, _c in self.events:
            self.edge_cross[i] += 1
            self.edge_cross[j] += 1


def metric_details(path):
    """Actual exported glyphs/labels, excluding routing control points (moved from geometry_experiment)."""
    import drawing
    from core import read
    data = read(path)
    content = data[1]
    hidden = set((data[0].get('metadata') or {}).get('hidden_metabolite_labels') or [])
    nodes = {k: v for k, v in content['nodes'].items() if v['node_type'] != 'multimarker'}
    xy = {k: [v['x'], v['y']] for k, v in nodes.items()}
    reaction_of = {}  # node id -> first reaction (map order) with a segment on it
    for r in content['reactions'].values():
        for seg in r['segments'].values():
            reaction_of.setdefault(seg['from_node_id'], r)
            reaction_of.setdefault(seg['to_node_id'], r)
    boxes = {}
    for key, v in nodes.items():
        if v['node_type'] == 'metabolite':
            if key in hidden or v.get('label_hidden'):
                continue
            label = v['bigg_id']
            x = v['label_x']
            y = v['label_y']
        else:
            rx = reaction_of[key]
            if rx.get('label_hidden'):
                continue
            label = rx['bigg_id']
            x = rx['label_x']
            y = rx['label_y']
        b = drawing.FONT.getbbox(label)
        boxes[key] = [x, y - (b[3] - b[1]), x + drawing.FONT.getlength(label), y]
    canvas = content['canvas']
    values = np.array(list(xy.values()))
    box_keys = list(boxes)
    B = np.array([boxes[k] for k in box_keys], float).reshape(-1, 4)
    node_label = 0
    for vx, vy in values:  # label box overlapping a node centre (±4 px)
        node_label += int(np.count_nonzero((B[:, 0] < vx + 4) & (B[:, 2] > vx - 4) & (B[:, 1] < vy + 4) & (B[:, 3] > vy - 4)))
    box_index = {k: i for i, k in enumerate(box_keys)}
    intersections = 0
    for r in content['reactions'].values():
        for seg in r['segments'].values():  # segment through a label box (slab test against every box)
            a, b = seg['from_node_id'], seg['to_node_id']
            av = np.array([content['nodes'][a]['x'], content['nodes'][a]['y']], float)
            delta = np.array([content['nodes'][b]['x'], content['nodes'][b]['y']], float) - av
            lower = np.zeros(len(B))
            upper = np.ones(len(B))
            for axis in (0, 1):
                if abs(delta[axis]) < 1e-9:
                    upper[(av[axis] <= B[:, axis]) | (av[axis] >= B[:, axis + 2])] = -1
                    continue
                lo = (B[:, axis] - av[axis]) / delta[axis]
                hi = (B[:, axis + 2] - av[axis]) / delta[axis]
                lower = np.maximum(lower, np.minimum(lo, hi))
                upper = np.minimum(upper, np.maximum(lo, hi))
            hit = lower < upper - 1e-8
            for k in {a, b} & box_index.keys():
                hit[box_index[k]] = False
            intersections += int(np.count_nonzero(hit))
    return dict(visible_nodes=len(nodes), label_node_intersections=int(node_label),
                segment_label_intersections=intersections,
                canvas_width=canvas['width'], canvas_height=canvas['height'],
                canvas_area_per_reaction=canvas['width'] * canvas['height'] / len(content['reactions']),
                node_coordinate_extent=np.ptp(values, axis=0).tolist())


def assert_chemistry(model, path):
    """Validate that the exported map matches the model chemistry (moved from geometry_experiment)."""
    from core import read, uid
    data = read(path)
    lookup = {r['bigg_id']: r for r in data[1]['reactions'].values()}
    assert set(lookup) == {r['id'] for r in model['reactions']}
    for r in model['reactions']:
        sign = -1 if r['upper_bound'] <= 0 and r['lower_bound'] < 0 else 1
        assert {m['bigg_id']: m['coefficient'] for m in lookup[r['id']]['metabolites']} == {k: sign * v for k, v in r['metabolites'].items()}
        assert data[0]['metadata']['source_reactions'][r['id']]['stoichiometry'] == r['metabolites']
    assert data[0]['metadata']['model_hash'] == uid(model)
