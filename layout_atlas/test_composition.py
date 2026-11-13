"""Unit tests for skyline packing and empty-band squeeze."""
import numpy as np

import composition as C


def test_skyline_pack_fills_beside_tall_tile():
    # Tall tile on the left; a short tile should tuck beside it (not only below).
    sizes = {0: (100.0, 300.0), 1: (80.0, 60.0), 2: (80.0, 60.0)}
    order = [0, 1, 2]
    placed = C.skyline_pack(order, sizes, gap=10.0, target_width=220.0)
    assert placed[0] == (0.0, 0.0)
    # At least one short tile shares the top band (y == 0).
    assert min(placed[1][1], placed[2][1]) == 0.0


def test_skyline_target_width_formula():
    sizes = {0: (100.0, 50.0), 1: (80.0, 40.0)}
    gap = 10.0
    tw = C.target_pack_width(sizes, gap, aspect=1.5)
    area = (110 * 60) + (90 * 50)
    assert abs(tw - max((area * 1.5) ** 0.5, 100 + 10)) < 1e-6


def test_pack_two_level_places_all_tiles():
    sizes = {i: (50.0 + 10 * i, 40.0) for i in range(5)}
    groups = [('a', [0, 1, 2]), ('b', [3, 4])]
    placed, report = C.pack_two_level(groups, sizes, gap=8.0, aspect=1.5)
    assert set(placed) == set(sizes)
    assert report['tiles'] == 5
    assert report['groups'] == 2


def test_squeeze_collapses_wide_gap_without_reordering():
    data = [
        {},
        {
            'nodes': {
                'a': {'node_type': 'metabolite', 'x': 0.0, 'y': 0.0},
                'b': {'node_type': 'metabolite', 'x': 100.0, 'y': 0.0},
                'c': {'node_type': 'metabolite', 'x': 5000.0, 'y': 0.0},
                'd': {'node_type': 'metabolite', 'x': 5100.0, 'y': 0.0},
            },
            'reactions': {
                'r1': {
                    'label_x': 50.0, 'label_y': 0.0,
                    'segments': {
                        's1': {
                            'from_node_id': 'a', 'to_node_id': 'b',
                            'b1': {'x': 20.0, 'y': 10.0}, 'b2': {'x': 80.0, 'y': 10.0},
                        }
                    },
                },
                'r2': {
                    'label_x': 5050.0, 'label_y': 0.0,
                    'segments': {
                        's1': {
                            'from_node_id': 'c', 'to_node_id': 'd',
                            'b1': {'x': 5020.0, 'y': 10.0}, 'b2': {'x': 5080.0, 'y': 10.0},
                        }
                    },
                },
            },
            'text_labels': {},
            'canvas': {'x': 0, 'y': 0, 'width': 5200, 'height': 100},
        },
    ]
    before = [data[1]['nodes'][k]['x'] for k in ('a', 'b', 'c', 'd')]
    out = C.squeeze_escher(data, max_gap=200.0)
    assert out['applied']
    assert out['dx'] > 1000
    after = [data[1]['nodes'][k]['x'] for k in ('a', 'b', 'c', 'd')]
    # Order preserved (monotone).
    assert after == sorted(after)
    assert after[0] <= after[1] <= after[2] <= after[3]
    # Far gap gone; local pairs still separated.
    assert after[2] - after[1] <= 200.0 + 1e-6
    assert after[1] - after[0] == before[1] - before[0]


def test_build_axis_remap_none_when_dense():
    assert C.build_axis_remap([0.0, 10.0, 20.0, 30.0], max_gap=50.0) is None


def test_skyline_prefers_linked_neighbour():
    # A links strongly to B; after placing A (large), B should sit beside it, not far below C.
    sizes = {0: (200.0, 200.0), 1: (80.0, 80.0), 2: (80.0, 80.0)}
    weights = {C._link_key(0, 1): 10.0, C._link_key(0, 2): 0.0}
    placed = C.skyline_pack([0, 1, 2], sizes, gap=10.0, target_width=400.0, weights=weights)
    c0 = (placed[0][0] + 100, placed[0][1] + 100)
    c1 = (placed[1][0] + 40, placed[1][1] + 40)
    c2 = (placed[2][0] + 40, placed[2][1] + 40)
    d01 = ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5
    d02 = ((c0[0] - c2[0]) ** 2 + (c0[1] - c2[1]) ** 2) ** 0.5
    assert d01 <= d02 + 1e-6


def test_pack_tiles_aspect_lands_in_band():
    # Many square tiles → packed rectangle should sit inside the 4:3–16:9 band.
    sizes = {i: (120.0, 100.0) for i in range(12)}
    placed, report = C.pack_tiles_aspect(sizes, gap=20.0, weights={}, target_aspect=1.5)
    assert set(placed) == set(sizes)
    assert C.ASPECT_MIN - 0.05 <= report['aspect'] <= C.ASPECT_MAX + 0.05


def test_tile_local_geometry_ignores_far_shared_met():
    # Far shared mets and far side/inst stubs must not inflate the packed tile size.
    pos = {
        'r:A': [0.0, 0.0],
        'r:B': [100.0, 0.0],
        'm:local': [50.0, 40.0],
        'm:far': [5000.0, 5000.0],
        'side:A:x': [8000.0, 100.0],
        'inst:A:y': [100.0, 9000.0],
    }
    keys = set(pos)
    labels = {k: k for k in keys}
    geom = C._tile_local_geometry(['A', 'B'], keys, pos, labels, lambda s: 10.0, 60.0)
    assert geom is not None
    assert geom['size'][0] < 400.0
    assert geom['size'][1] < 400.0
    assert 'm:far' not in geom['local_keys']
    assert 'side:A:x' not in geom['local_keys']
    assert 'inst:A:y' not in geom['local_keys']


def test_classify_large_tiles_uses_area_percentile():
    tiles = [
        dict(id='a', kind='curated', size=(100.0, 100.0), reactions=['r1']),
        dict(id='b', kind='curated', size=(200.0, 200.0), reactions=['r2', 'r3']),
        dict(id='c', kind='curated', size=(50.0, 50.0), reactions=['r4']),
        dict(id='d', kind='auto', size=(500.0, 500.0), reactions=['r5']),
    ]
    large = C.classify_large_tiles(tiles, area_pct=75.0)
    assert 'b' in large
    assert 'd' not in large  # auto never in core
    assert 'c' not in large


def test_squeeze_force_collapses_empty_when_aspect_held_by_void():
    # Two tall clusters with a wide empty mid corridor: aspect already in-band
    # only because of the void. Squeeze must still collapse it (force_empty_*).
    data = [
        {},
        {
            'nodes': {
                'a': {'node_type': 'metabolite', 'x': 0.0, 'y': 0.0},
                'b': {'node_type': 'metabolite', 'x': 600.0, 'y': 2600.0},
                'c': {'node_type': 'metabolite', 'x': 3400.0, 'y': 0.0},
                'd': {'node_type': 'metabolite', 'x': 4000.0, 'y': 2600.0},
            },
            'reactions': {},
            'text_labels': {},
            'canvas': {'x': 0, 'y': 0, 'width': 4100, 'height': 2700},
        },
    ]
    before_w = 4000.0
    before_h = 2600.0
    assert C.ASPECT_MIN <= before_w / before_h <= C.ASPECT_MAX
    out = C.squeeze_escher(data, max_gap=200.0)
    assert out['applied']
    # Prefer collapsing the void while staying in 4:3–16:9 (force_partial_*) when a
    # full force_empty_* pass would leave the band; either mode must shrink width.
    assert out['mode'].startswith('force_')
    assert out['dx'] > 400
    xs = [n['x'] for n in data[1]['nodes'].values()]
    ys = [n['y'] for n in data[1]['nodes'].values()]
    new_w = max(xs) - min(xs)
    new_h = max(ys) - min(ys)
    assert new_w < before_w - 400
    assert C.ASPECT_MIN - 0.05 <= new_w / max(new_h, 1e-9) <= C.ASPECT_MAX + 0.05


def test_compact_long_arms_escher_shortens_transport():
    data = [
        {},
        {
            'nodes': {
                'm': {'node_type': 'midmarker', 'x': 0.0, 'y': 0.0},
                'e': {'node_type': 'metabolite', 'x': 0.0, 'y': 700.0, 'bigg_id': 'acald_e',
                      'node_is_primary': True},
                'c': {'node_type': 'metabolite', 'x': 0.0, 'y': -700.0, 'bigg_id': 'acald_c',
                      'node_is_primary': True},
            },
            'reactions': {
                '1': {
                    'bigg_id': 'ACALDt', 'name': 'ACALDt', 'label_x': 0, 'label_y': 0,
                    'metabolites': [
                        {'bigg_id': 'acald_e', 'coefficient': -1},
                        {'bigg_id': 'acald_c', 'coefficient': 1},
                    ],
                    'segments': {
                        '0': {'from_node_id': 'm', 'to_node_id': 'e',
                              'b1': {'x': 0, 'y': 200}, 'b2': {'x': 0, 'y': 500}},
                        '1': {'from_node_id': 'm', 'to_node_id': 'c'},
                    },
                },
            },
            'text_labels': {},
            'canvas': {'x': -50, 'y': -800, 'width': 100, 'height': 1600},
        },
    ]
    out = C.compact_long_arms_escher(data, cur=set(), max_arm=220.0, target=150.0)
    assert out['shortened'] == 2
    assert abs(data[1]['nodes']['e']['y'] - 150.0) < 1e-6
    assert abs(data[1]['nodes']['c']['y'] + 150.0) < 1e-6
    assert data[1]['reactions']['1']['segments']['0']['b1'] is None
    assert data[1]['reactions']['1']['segments']['0']['b2'] is None


def test_build_axis_remap_respects_max_shift():
    # Gaps: 90 between 10→100 and 90 between 110→200; max_gap 20 → 70+70 compressible.
    vals = [0.0, 10.0, 100.0, 110.0, 200.0]
    f = C.build_axis_remap(vals, max_gap=20.0, max_shift=30.0)
    assert f is not None
    assert abs((vals[-1] - f(vals[-1])) - 30.0) < 1e-6
    # Order preserved
    mapped = [f(v) for v in vals]
    assert mapped == sorted(mapped)


def test_squeeze_partial_stays_in_aspect_band():
    """Wide landscape held by one X void: partial squeeze lands on ASPECT_MIN, not skip."""
    # Content clusters at x≈0 and x≈3000 with height ~1000 → aspect ~3 before;
    # after capping to ASPECT_MIN the width should be ~4/3 * height.
    nodes = {
        'a': {'x': 0.0, 'y': 0.0, 'node_type': 'metabolite'},
        'b': {'x': 50.0, 'y': 1000.0, 'node_type': 'metabolite'},
        'c': {'x': 3000.0, 'y': 0.0, 'node_type': 'metabolite'},
        'd': {'x': 3050.0, 'y': 1000.0, 'node_type': 'metabolite'},
    }
    data = [{'metadata': {}}, {'nodes': nodes, 'reactions': {}, 'canvas': {'x': 0, 'y': 0, 'width': 3100, 'height': 1100}}]
    out = C.squeeze_escher(data, max_gap=200.0)
    assert out.get('applied'), out
    xs = [n['x'] for n in data[1]['nodes'].values()]
    ys = [n['y'] for n in data[1]['nodes'].values()]
    aspect = (max(xs) - min(xs)) / max(max(ys) - min(ys), 1e-9)
    assert C.ASPECT_MIN - 0.02 <= aspect <= C.ASPECT_MAX + 0.02, aspect


def test_pack_into_main_pockets_places_beside_main():
    # Main cluster at origin; one free box should tuck into the empty pad, not far away.
    main = [(0.0, 0.0), (40.0, 0.0), (0.0, 40.0), (40.0, 40.0),
            (80.0, 0.0), (80.0, 40.0), (0.0, 80.0), (40.0, 80.0), (80.0, 80.0)]
    boxes = [(1, 50.0, 40.0)]
    placed, rep = C.pack_into_main_pockets(boxes, main, spacing=40.0, prefer_c=(40.0, 40.0))
    assert 1 in placed, rep
    x, y = placed[1]
    # Within a modest halo of the main AABB.
    assert -80 <= x <= 160 and -80 <= y <= 160


def test_reflow_landscape_in_band_keeps_near_tiles():
    """In-band frame: pocket far outliers only; do not shelf near leftovers to the side.

    Regression for e_coli TR floaters / iIT341 spokes: leftover used to mean
    ``all sizes − pocketed``, which yanked near tiles onto an external shelf.
    """
    # Landscape main ~1080×720 (aspect 1.5). Chain reactions through a shared
    # backbone metabolite so they form one MAIN_MIN+ component.
    reactions = []
    parts = {}
    pos = {}
    n_main = max(C.MAIN_MIN, 30)
    cols = 6
    for i in range(n_main):
        rid = f'M{i}'
        mid = f'm{i}'
        mets = {'backbone': -1.0, mid: 1.0} if i else {mid: -1.0, 'backbone': 1.0}
        reactions.append({'id': rid, 'metabolites': mets})
        for m in mets:
            parts[rid, m] = f'm:{m}'
        x = (i % cols) * 180.0
        y = (i // cols) * 180.0
        pos[f'r:{rid}'] = [x, y]
        pos[f'm:{mid}'] = [x + 40.0, y + 20.0]
    pos['m:backbone'] = [500.0, 360.0]
    # Near free component (inside main AABB, well under FAR_OUTLIER_FRAC · span).
    reactions.append({'id': 'NEAR', 'metabolites': {'n': -1.0}})
    parts['NEAR', 'n'] = 'm:n'
    pos['r:NEAR'] = [500.0, 400.0]
    pos['m:n'] = [540.0, 420.0]
    near_before = list(pos['r:NEAR'])
    # Far free component to the right; combined aspect stays in 4:3–16:9 so the
    # in-band path runs, while distance still exceeds FAR_OUTLIER_FRAC · span.
    reactions.append({'id': 'FAR', 'metabolites': {'f': -1.0}})
    parts['FAR', 'f'] = 'm:f'
    pos['r:FAR'] = [1200.0, 360.0]
    pos['m:f'] = [1240.0, 380.0]
    model = {'reactions': reactions}
    scene = {'positions': pos, 'participants': parts, 'labels': {}}
    cur = set()
    out, rep = C.reflow_landscape(model, scene, blocks=[], cur=cur)
    assert C.ASPECT_MIN - 1e-3 <= rep.get('aspect_before', 0) <= C.ASPECT_MAX + 1e-3, rep
    assert rep.get('reason') == 'main_pocket_pack', rep
    assert rep.get('pack', {}).get('side') == 'pocket', rep
    assert rep.get('pocket', {}).get('near_kept', 0) >= 1, rep
    # Near tile must not be skyline-shelved far from its original pose.
    near_after = out['positions']['r:NEAR']
    assert abs(near_after[0] - near_before[0]) < 1e-6, (near_before, near_after, rep)
    assert abs(near_after[1] - near_before[1]) < 1e-6, (near_before, near_after, rep)


def test_fit_canvas_to_content_shrinks_frame():
    data = [
        {},
        {
            'nodes': {
                'a': {'node_type': 'metabolite', 'x': 100.0, 'y': 200.0},
                'b': {'node_type': 'metabolite', 'x': 400.0, 'y': 500.0},
            },
            'reactions': {},
            'text_labels': {},
            'canvas': {'x': 0, 'y': 0, 'width': 5000, 'height': 4000},
        },
    ]
    out = C.fit_canvas_to_content(data, pad=40.0)
    assert out['applied']
    can = data[1]['canvas']
    assert abs(can['x'] - 60.0) < 1e-6
    assert abs(can['y'] - 160.0) < 1e-6
    assert abs(can['width'] - 380.0) < 1e-6
    assert abs(can['height'] - 380.0) < 1e-6


def test_glyph_circle_overlapping_pairs_counts_disks():
    data = [{}, {'nodes': {
        '1': {'node_type': 'metabolite', 'x': 0.0, 'y': 0.0, 'node_is_primary': True},
        '2': {'node_type': 'metabolite', 'x': 30.0, 'y': 0.0, 'node_is_primary': True},
        '3': {'node_type': 'midmarker', 'x': 200.0, 'y': 0.0},
    }, 'reactions': {}}]
    assert C.glyph_circle_overlapping_pairs(data) == 1


def test_tip_gate_directions():
    from layout_quality_evaluator import METRIC_EPS, TIP_DIRS, metrics_not_worse
    before = dict(crossings=2, overlapping_pairs=1, participant_consistency=0.8,
                  angular_resolution=0.2, edge_length_deviation=0.5,
                  edge_orthogonality=0.6, aspect_ratio=0.7, curves_through_nodes=3)
    assert "curves_through_nodes" in TIP_DIRS
    assert metrics_not_worse(before, before, TIP_DIRS, METRIC_EPS)
    assert not metrics_not_worse(before, dict(before, crossings=3), TIP_DIRS, METRIC_EPS)
    assert not metrics_not_worse(before, dict(before, aspect_ratio=0.5), TIP_DIRS, METRIC_EPS)
    assert not metrics_not_worse(before, dict(before, overlapping_pairs=2), TIP_DIRS, METRIC_EPS)
    assert not metrics_not_worse(before, dict(before, curves_through_nodes=4), TIP_DIRS, METRIC_EPS)
    assert metrics_not_worse(
        before, dict(before, crossings=1, aspect_ratio=0.8, edge_orthogonality=0.7,
                     curves_through_nodes=2),
        TIP_DIRS, METRIC_EPS)


def test_nudge_skips_curated_and_previous_pins():
    data = [{
        'metadata': {'expert_structures': [{'id': 's1', 'reactions': ['R']}]},
    }, {
        'canvas': {'x': 0, 'y': 0, 'width': 400, 'height': 200},
        'nodes': {
            'm1': {'node_type': 'metabolite', 'x': 0.0, 'y': 50.0, 'bigg_id': 'a',
                   'node_is_primary': True},
            'mid': {'node_type': 'midmarker', 'x': 100.0, 'y': 50.0},
            'm2': {'node_type': 'metabolite', 'x': 300.0, 'y': 50.0, 'bigg_id': 'b',
                   'node_is_primary': True},
        },
        'reactions': {
            'r1': {'bigg_id': 'R', 'segments': {
                's1': {'from_node_id': 'm1', 'to_node_id': 'mid'},
                's2': {'from_node_id': 'mid', 'to_node_id': 'm2'},
            }},
        },
    }]
    model = {'reactions': [{'id': 'R', 'metabolites': {'a': -1.0, 'b': 1.0}}]}
    before = {nid: (n['x'], n['y']) for nid, n in data[1]['nodes'].items()}
    C.nudge_frontier_escher(data, model, previous_anchors={'m:b': [300.0, 50.0]})
    after = {nid: (n['x'], n['y']) for nid, n in data[1]['nodes'].items()}
    assert after == before


def test_nudge_frontier_escher_shrinks_when_safe():
    # Long right stub: shortening it shrinks the bbox and improves ELD.
    data = [{}, {
        'canvas': {'x': 0, 'y': 0, 'width': 500, 'height': 200},
        'nodes': {
            'm1': {'node_type': 'metabolite', 'x': 0.0, 'y': 50.0, 'bigg_id': 'a',
                   'node_is_primary': True},
            'mid': {'node_type': 'midmarker', 'x': 100.0, 'y': 50.0},
            'multi': {'node_type': 'multimarker', 'x': 140.0, 'y': 50.0},
            'm2': {'node_type': 'metabolite', 'x': 400.0, 'y': 50.0, 'bigg_id': 'b',
                   'node_is_primary': True},
            'm3': {'node_type': 'metabolite', 'x': 100.0, 'y': 150.0, 'bigg_id': 'c',
                   'node_is_primary': True},
        },
        'reactions': {
            'r1': {'bigg_id': 'R', 'label_x': 100.0, 'label_y': 40.0, 'segments': {
                's1': {'from_node_id': 'm1', 'to_node_id': 'mid'},
                's2': {'from_node_id': 'mid', 'to_node_id': 'multi',
                       'b1': {'x': 110.0, 'y': 50.0}, 'b2': {'x': 130.0, 'y': 50.0}},
                's3': {'from_node_id': 'multi', 'to_node_id': 'm2',
                       'b1': {'x': 220.0, 'y': 50.0}, 'b2': {'x': 320.0, 'y': 50.0}},
            }},
            'r2': {'bigg_id': 'R2', 'label_x': 100.0, 'label_y': 120.0, 'segments': {
                's1': {'from_node_id': 'm3', 'to_node_id': 'mid'},
            }},
        },
    }]
    model = {'reactions': [
        {'id': 'R', 'metabolites': {'a': -1.0, 'b': 1.0}},
        {'id': 'R2', 'metabolites': {'c': -1.0, 'a': 1.0}},
    ]}
    before = C._escher_content_bbox(data[1])
    label_r2 = (data[1]['reactions']['r2']['label_x'],
                data[1]['reactions']['r2']['label_y'])
    m2_x = data[1]['nodes']['m2']['x']
    rep = C.nudge_frontier_escher(data, model)
    after = C._escher_content_bbox(data[1])
    assert rep['accepted'] >= 1
    assert (after[2] - after[0]) * (after[3] - after[1]) < (
        before[2] - before[0]) * (before[3] - before[1])
    assert data[1]['nodes']['m2']['x'] < m2_x
    # Unrelated reaction label stays put (its mid did not move).
    assert (data[1]['reactions']['r2']['label_x'],
            data[1]['reactions']['r2']['label_y']) == label_r2


def test_tip_move_shifts_only_near_end_handle():
    data = [{}, {
        'nodes': {
            'mid': {'node_type': 'midmarker', 'x': 0.0, 'y': 0.0},
            'm': {'node_type': 'metabolite', 'x': 100.0, 'y': 0.0, 'bigg_id': 'x',
                  'node_is_primary': True},
        },
        'reactions': {'r1': {'bigg_id': 'R', 'segments': {
            's1': {'from_node_id': 'mid', 'to_node_id': 'm',
                   'b1': {'x': 30.0, 'y': 0.0}, 'b2': {'x': 70.0, 'y': 0.0}},
        }}},
    }]
    C._apply_tip_move(data[1], {'m'}, -40.0, 0.0)
    seg = data[1]['reactions']['r1']['segments']['s1']
    assert seg['b1'] == {'x': 30.0, 'y': 0.0}
    assert seg['b2'] == {'x': 30.0, 'y': 0.0}
    assert data[1]['nodes']['m']['x'] == 60.0


def test_tip_gate_uses_baseline_not_last_accept():
    # A second accept fine vs the last state but past the baseline eps fails.
    from layout_quality_evaluator import METRIC_EPS, TIP_DIRS, metrics_not_worse
    baseline = dict(crossings=2, overlapping_pairs=1, participant_consistency=0.8,
                    angular_resolution=0.2, edge_length_deviation=0.5,
                    edge_orthogonality=0.9, aspect_ratio=0.7)
    after_first = dict(baseline, edge_orthogonality=0.9 - 5e-5)
    after_second = dict(baseline, edge_orthogonality=0.9 - 1.2e-4)
    assert metrics_not_worse(baseline, after_first, TIP_DIRS, METRIC_EPS)
    assert metrics_not_worse(after_first, after_second, TIP_DIRS, METRIC_EPS)
    assert not metrics_not_worse(baseline, after_second, TIP_DIRS, METRIC_EPS)
