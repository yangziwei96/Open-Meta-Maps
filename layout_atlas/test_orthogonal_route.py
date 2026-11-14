"""Orthogonal polyline routing for official edge_orthogonality."""
import numpy as np

import edge_style
import orthogonal_route as ortho


def test_ortho_candidates_are_axis_parallel_and_bounded():
    paths = ortho.ortho_candidates([0.0, 0.0], [100.0, 80.0], budget=12)
    assert paths
    assert len(paths) <= 12
    for path in paths:
        assert 2 <= len(path) <= 4  # ends + ≤2 bends
        for a, b in zip(path[:-1], path[1:]):
            assert abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6


def test_choose_ortho_prefers_clean_L():
    path, score = ortho.choose_ortho([0, 0], [40, 30], np.zeros((0, 4)), np.zeros((0, 4)), max_cross=0)
    assert path is not None
    assert score[0] == 0  # no hits
    assert score[1] == 0  # no crossings
    assert len(path) - 2 <= 2
    assert abs(path[0][0] - 0) < 1e-9 and abs(path[-1][0] - 40) < 1e-9


def test_choose_ortho_respects_max_cross():
    # A horizontal barrier segment that an L-route via (40,0) would cross.
    S = np.array([[20.0, -10.0, 20.0, 10.0]], float)
    path, score = ortho.choose_ortho([0, 0], [40, 30], np.zeros((0, 4)), S, max_cross=0)
    # May still find the other L via (0,30) which does not cross the barrier.
    if path is not None:
        assert score[1] == 0
    path2, _ = ortho.choose_ortho([0, 0], [40, 30], np.zeros((0, 4)), S, max_cross=-1)
    assert path2 is None


def test_choose_ortho_respects_max_length():
    # Near-axis: 0-bend snap length ≈ 40; a full L would be longer.
    path, _ = ortho.choose_ortho(
        [0, 0], [40, 2], np.zeros((0, 4)), np.zeros((0, 4)),
        max_cross=0, max_length=42.0)
    assert path is not None
    assert ortho._path_length(path) <= 42.0 + 1e-6
    none, _ = ortho.choose_ortho(
        [0, 0], [40, 30], np.zeros((0, 4)), np.zeros((0, 4)),
        max_cross=0, max_length=50.0)  # chord≈50 but L-route=70
    assert none is None


def test_length_deviation_helper():
    assert ortho._length_deviation([10.0, 10.0, 10.0]) == 0.0
    d = ortho._length_deviation([10.0, 20.0])
    assert abs(d - 1.0 / 3.0) < 1e-9


def test_chord_near_axis_cone():
    assert ortho._chord_near_axis([0, 0], [100, 0]) is True
    assert ortho._chord_near_axis([0, 0], [100, 50]) is True   # tan^{-1}(0.5)<30°? 0.5<0.577
    assert ortho._chord_near_axis([0, 0], [100, 80]) is False  # ratio 0.8 > tan30


def test_gate_ok_respects_angular_directions():
    pre = dict(crossings=10, edge_length_deviation=0.5,
               angular_resolution=0.54, participant_consistency=0.83)
    # AR is lower-better; a rise is a regression.
    bad_ar = dict(pre, angular_resolution=0.541)
    assert ortho._gate_ok(pre, bad_ar) is False
    # PC is higher-better.
    bad_pc = dict(pre, participant_consistency=0.82)
    assert ortho._gate_ok(pre, bad_pc) is False
    ok = dict(pre, angular_resolution=0.539,
              participant_consistency=0.84, crossings=9)
    assert ortho._gate_ok(pre, ok) is True


def test_gate_metrics_no_optional_imports():
    """Accept gates must not import layout_benchmark or gdMetriX."""
    import sys
    blocked = {'layout_benchmark', 'gdMetriX', 'gdmetrix'}
    before = {k for k in sys.modules if any(k == b or k.startswith(b + '.') for b in blocked)}
    data = [
        {'metadata': {'expert_structures': [], 'preserved_edges': [], 'source_paths': []}},
        {
            'nodes': {
                'm': {'node_type': 'midmarker', 'x': 0.0, 'y': 0.0},
                'a': {'node_type': 'metabolite', 'x': 40.0, 'y': 0.0, 'bigg_id': 'a_c',
                      'node_is_primary': True, 'label_x': 45.0, 'label_y': 0.0},
                'b': {'node_type': 'metabolite', 'x': -40.0, 'y': 0.0, 'bigg_id': 'b_c',
                      'node_is_primary': True, 'label_x': -45.0, 'label_y': 0.0},
            },
            'reactions': {
                '0': {
                    'bigg_id': 'RNEW', 'label_x': 2.0, 'label_y': -8.0,
                    'metabolites': [
                        {'bigg_id': 'a_c', 'coefficient': 1},
                        {'bigg_id': 'b_c', 'coefficient': -1},
                    ],
                    'segments': {
                        '1': {'from_node_id': 'm', 'to_node_id': 'a', 'b1': None, 'b2': None},
                        '2': {'from_node_id': 'm', 'to_node_id': 'b', 'b1': None, 'b2': None},
                    },
                }
            },
            'canvas': {'width': 200, 'height': 200},
            'text_labels': {},
        },
    ]
    m = ortho._gate_metrics(data)
    after = {k for k in sys.modules if any(k == b or k.startswith(b + '.') for b in blocked)}
    assert after == before
    assert set(m) >= {'crossings', 'angular_resolution', 'participant_consistency',
                      'edge_length_deviation'}
    assert m['crossings'] == 0.0
    assert m['participant_consistency'] == 1.0


def test_should_ortho_skips_short_cofactor_stubs():
    assert ortho.should_ortho('R1', 'g6p_c', 200.0, set()) is True
    assert ortho.should_ortho('R1', 'atp_c', 40.0, set()) is False
    assert ortho.should_ortho('R1', 'g6p_c', 200.0, {'R1'}) is True  # long on curated rx
    assert ortho.should_ortho('R1', 'g6p_c', 50.0, {'R1'}) is False


def test_crossing_cap_is_pre_orthogonal_count():
    """No absolute model-specific crossing floor — cap equals the map's own count."""
    data = [
        {'metadata': {'expert_structures': [], 'preserved_edges': [], 'source_paths': []}},
        {
            'nodes': {
                'm': {'node_type': 'midmarker', 'x': 0.0, 'y': 0.0},
                'a': {'node_type': 'metabolite', 'x': 40.0, 'y': 0.0, 'bigg_id': 'a_c',
                      'node_is_primary': True, 'label_x': 45.0, 'label_y': 0.0},
            },
            'reactions': {
                '0': {
                    'bigg_id': 'RNEW', 'label_x': 2.0, 'label_y': -8.0,
                    'metabolites': [{'bigg_id': 'a_c', 'coefficient': 1}],
                    'segments': {
                        '1': {'from_node_id': 'm', 'to_node_id': 'a', 'b1': None, 'b2': None},
                    },
                }
            },
            'canvas': {'width': 200, 'height': 200},
            'text_labels': {},
        },
    ]
    before = ortho._piece_crossing_count(data)
    report = edge_style.apply(data, mode='orthogonal')
    assert report['crossing_cap'] == before
    assert ortho._piece_crossing_count(data) <= before


def test_edge_style_writes_multimarkers_for_de_novo():
    # Short near-axis diagonal among long axis-aligned neighbours: L-detour moves
    # its length toward the pre-pass mean, so the length gate accepts a bend.
    data = [
        {'metadata': {'expert_structures': [], 'preserved_edges': [], 'source_paths': []}},
        {
            'nodes': {
                'm': {'node_type': 'midmarker', 'x': 0.0, 'y': 0.0},
                'a': {'node_type': 'metabolite', 'x': 100.0, 'y': 40.0, 'bigg_id': 'g6p_c',
                      'node_is_primary': True, 'label_x': 110.0, 'label_y': 40.0},
                'c': {'node_type': 'metabolite', 'x': -160.0, 'y': 0.0, 'bigg_id': 'g3p_c',
                      'node_is_primary': True, 'label_x': -170.0, 'label_y': 0.0},
                'd': {'node_type': 'metabolite', 'x': 0.0, 'y': 160.0, 'bigg_id': 'f6p_c',
                      'node_is_primary': True, 'label_x': 0.0, 'label_y': 170.0},
                'b': {'node_type': 'metabolite', 'x': 15.0, 'y': 20.0, 'bigg_id': 'atp_c',
                      'node_is_primary': False, 'label_x': 20.0, 'label_y': 20.0},
            },
            'reactions': {
                '0': {
                    'bigg_id': 'HEX1', 'label_x': 5.0, 'label_y': -10.0,
                    'metabolites': [
                        {'bigg_id': 'g6p_c', 'coefficient': 1},
                        {'bigg_id': 'g3p_c', 'coefficient': -1},
                        {'bigg_id': 'f6p_c', 'coefficient': -1},
                        {'bigg_id': 'atp_c', 'coefficient': -1},
                    ],
                    'segments': {
                        '1': {'from_node_id': 'm', 'to_node_id': 'a', 'b1': None, 'b2': None},
                        '2': {'from_node_id': 'm', 'to_node_id': 'c', 'b1': None, 'b2': None},
                        '3': {'from_node_id': 'm', 'to_node_id': 'd', 'b1': None, 'b2': None},
                        '4': {'from_node_id': 'm', 'to_node_id': 'b', 'b1': None, 'b2': None},
                    },
                }
            },
            'canvas': {'width': 400, 'height': 400},
            'text_labels': {},
        },
    ]
    report = edge_style.apply(data, mode='orthogonal')
    assert report['orthogonal'] >= 1
    assert report['stubs'] >= 1
    routed = {(r['reaction'], r['metabolite'])
              for r in (data[0].get('metadata') or {}).get('routed_edges') or []
              if r.get('kind') == 'orthogonal'}
    assert routed
    rx = data[1]['reactions']['0']
    nodes = data[1]['nodes']
    import line_render
    for path in line_render._terminals(rx, nodes):
        met = nodes[path[-1]]
        if (rx['bigg_id'], met.get('bigg_id')) not in routed:
            continue
        pts = [(nodes[n]['x'], nodes[n]['y']) for n in path]
        for a, b in zip(pts[:-1], pts[1:]):
            assert abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6
    assert any(n.get('node_type') == 'multimarker' for n in nodes.values())


def test_orthogonal_apply_keeps_metabolite_label_offsets():
    from map_edit import set_node_xy

    data = [
        {'metadata': {'expert_structures': [], 'preserved_edges': [], 'source_paths': []}},
        {
            'nodes': {
                'm': {'node_type': 'midmarker', 'x': 0.0, 'y': 0.0},
                'a': {'node_type': 'metabolite', 'x': 100.0, 'y': 80.0, 'bigg_id': 'g6p_c',
                      'node_is_primary': True, 'label_x': 110.0, 'label_y': 90.0},
                'b': {'node_type': 'metabolite', 'x': 20.0, 'y': 25.0, 'bigg_id': 'atp_c',
                      'node_is_primary': False, 'label_x': 25.0, 'label_y': 30.0},
            },
            'reactions': {
                '0': {
                    'bigg_id': 'HEX1', 'label_x': 5.0, 'label_y': -10.0,
                    'metabolites': [
                        {'bigg_id': 'g6p_c', 'coefficient': 1},
                        {'bigg_id': 'atp_c', 'coefficient': -1},
                    ],
                    'segments': {
                        '1': {'from_node_id': 'm', 'to_node_id': 'a', 'b1': None, 'b2': None},
                        '2': {'from_node_id': 'm', 'to_node_id': 'b', 'b1': None, 'b2': None},
                    },
                }
            },
            'canvas': {'width': 400, 'height': 400},
            'text_labels': {},
        },
    ]
    before = (data[1]['nodes']['a']['label_x'] - data[1]['nodes']['a']['x'],
              data[1]['nodes']['a']['label_y'] - data[1]['nodes']['a']['y'])
    edge_style.apply(data, mode='orthogonal')
    # map_edit.set_node_xy keeps the label glued; orthogonal routing must not drift it.
    set_node_xy(data[1]['nodes']['a'], data[1]['nodes']['a']['x'], data[1]['nodes']['a']['y'])
    after = (data[1]['nodes']['a']['label_x'] - data[1]['nodes']['a']['x'],
             data[1]['nodes']['a']['label_y'] - data[1]['nodes']['a']['y'])
    assert after == before
