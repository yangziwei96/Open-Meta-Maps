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


def test_compose_freezes_previous_pinned_island():
    # Fully --previous-pinned islands stay put while packing moves other leftovers.
    model = {
        'reactions': [
            {'id': 'R1', 'metabolites': {'A_c': -1, 'B_c': 1}},
            {'id': 'R2', 'metabolites': {'B_c': -1, 'C_c': 1}},
            {'id': 'R3', 'metabolites': {'C_c': -1, 'D_c': 1}},
            {'id': 'R4', 'metabolites': {'D_c': -1, 'E_c': 1}},
            {'id': 'R5', 'metabolites': {'E_c': -1, 'F_c': 1}},
            {'id': 'R6', 'metabolites': {'F_c': -1, 'G_c': 1}},
            {'id': 'R7', 'metabolites': {'G_c': -1, 'H_c': 1}},
            {'id': 'R8', 'metabolites': {'H_c': -1, 'I_c': 1}},
            {'id': 'T1', 'metabolites': {'X_c': -1, 'Y_c': 1}},
        ],
        'metabolites': [],
    }
    pins = {'r:T1': [5000.0, 5000.0], 'm:X_c': [5000.0, 5040.0], 'm:Y_c': [5000.0, 4960.0]}
    scene = dict(
        positions={
            'r:R1': [0.0, 0.0], 'r:R2': [100.0, 0.0], 'r:R3': [200.0, 0.0], 'r:R4': [300.0, 0.0],
            'r:R5': [400.0, 0.0], 'r:R6': [500.0, 0.0], 'r:R7': [600.0, 0.0], 'r:R8': [700.0, 0.0],
            'm:A_c': [0.0, 40.0], 'm:B_c': [50.0, 40.0], 'm:C_c': [150.0, 40.0], 'm:D_c': [250.0, 40.0],
            'm:E_c': [350.0, 40.0], 'm:F_c': [450.0, 40.0], 'm:G_c': [550.0, 40.0], 'm:H_c': [650.0, 40.0],
            'm:I_c': [700.0, 40.0],
            'r:T1': [5000.0, 5000.0], 'm:X_c': [5000.0, 5040.0], 'm:Y_c': [5000.0, 4960.0],
        },
        labels={},
        copies={
            'm:A_c': 'A_c', 'm:B_c': 'B_c', 'm:C_c': 'C_c', 'm:D_c': 'D_c', 'm:E_c': 'E_c',
            'm:F_c': 'F_c', 'm:G_c': 'G_c', 'm:H_c': 'H_c', 'm:I_c': 'I_c',
            'm:X_c': 'X_c', 'm:Y_c': 'Y_c',
        },
        participants={
            ('R1', 'A_c'): 'm:A_c', ('R1', 'B_c'): 'm:B_c',
            ('R2', 'B_c'): 'm:B_c', ('R2', 'C_c'): 'm:C_c',
            ('R3', 'C_c'): 'm:C_c', ('R3', 'D_c'): 'm:D_c',
            ('R4', 'D_c'): 'm:D_c', ('R4', 'E_c'): 'm:E_c',
            ('R5', 'E_c'): 'm:E_c', ('R5', 'F_c'): 'm:F_c',
            ('R6', 'F_c'): 'm:F_c', ('R6', 'G_c'): 'm:G_c',
            ('R7', 'G_c'): 'm:G_c', ('R7', 'H_c'): 'm:H_c',
            ('R8', 'H_c'): 'm:H_c', ('R8', 'I_c'): 'm:I_c',
            ('T1', 'X_c'): 'm:X_c', ('T1', 'Y_c'): 'm:Y_c',
        },
    )
    out, rep = C.compose(model, scene, blocks=[], cur=set(), previous_pins=pins)
    assert rep.get('previous_pinned_reactions') == 1
    assert out['positions']['r:T1'] == [5000.0, 5000.0]
    assert out['positions']['m:X_c'] == [5000.0, 5040.0]
    # Compose must not stamp scene['anchors'] (previous-relayout revert fallback).
    assert 'anchors' not in out

