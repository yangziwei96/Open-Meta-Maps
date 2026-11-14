"""Labels must stay attached when nodes move (rigid restore / overlap clear)."""
import math

import numpy as np

import curated_axes as CA
import label_placement as LP
from map_edit import set_node_xy, translate_node


def _met_label_dists(data):
    out = []
    for n in data[1]['nodes'].values():
        if n.get('node_type') != 'metabolite' or 'label_x' not in n or n.get('label_hidden'):
            continue
        out.append(math.hypot(n['label_x'] - n['x'], n['label_y'] - n['y']))
    return out


def _rx_label_dists(data):
    nodes = data[1]['nodes']
    out = []
    for rx in data[1]['reactions'].values():
        if 'label_x' not in rx or rx.get('label_hidden'):
            continue
        mid = next((s['from_node_id'] for s in rx['segments'].values()
                    if nodes[s['from_node_id']].get('node_type') == 'midmarker'), None)
        if mid is None:
            continue
        mn = nodes[mid]
        out.append(math.hypot(rx['label_x'] - mn['x'], rx['label_y'] - mn['y']))
    return out


def test_translate_node_keeps_label_offset():
    n = dict(x=10.0, y=20.0, label_x=18.0, label_y=24.0)
    translate_node(n, 100.0, -50.0)
    assert abs(n['x'] - 110.0) < 1e-9 and abs(n['y'] + 30.0) < 1e-9
    assert abs(n['label_x'] - 118.0) < 1e-9 and abs(n['label_y'] + 26.0) < 1e-9
    set_node_xy(n, 0.0, 0.0)
    assert abs(n['label_x'] - 8.0) < 1e-9 and abs(n['label_y'] - 4.0) < 1e-9


def test_restore_rigid_group_escher_moves_labels_with_nodes():
    """Absolute snap must translate metabolite and reaction labels by the same delta."""
    data = [{}, {
        'nodes': {
            'm1': {'node_type': 'metabolite', 'bigg_id': 'fum_c', 'x': 100.0, 'y': 2000.0,
                   'label_x': 110.0, 'label_y': 2004.0, 'node_is_primary': True},
            'r1': {'node_type': 'midmarker', 'x': 50.0, 'y': 1900.0},
            'm2': {'node_type': 'metabolite', 'bigg_id': 'succ_c', 'x': 200.0, 'y': 2100.0,
                   'label_x': 210.0, 'label_y': 2104.0, 'node_is_primary': True},
            'r2': {'node_type': 'midmarker', 'x': 150.0, 'y': 2100.0},
        },
        'reactions': {
            '0': {'bigg_id': 'FRD7', 'label_x': 55.0, 'label_y': 1905.0,
                  'segments': {'s0': {'from_node_id': 'r1', 'to_node_id': 'm1', 'b1': None, 'b2': None}}},
            '1': {'bigg_id': 'SUCDi', 'label_x': 155.0, 'label_y': 2105.0,
                  'segments': {'s0': {'from_node_id': 'r2', 'to_node_id': 'm2', 'b1': None, 'b2': None}}},
        },
    }]
    blocks = [{
        'same_map_rigid': True,
        'reaction_ids': ['FRD7', 'SUCDi'],
        'same_map_snapshot': {
            'r:FRD7': [50.0, 50.0], 'r:SUCDi': [150.0, 50.0],
            'm:fum_c': [100.0, 100.0], 'm:succ_c': [200.0, 100.0],
        },
    }]
    before_met = {
        nid: (n['label_x'] - n['x'], n['label_y'] - n['y'])
        for nid, n in data[1]['nodes'].items() if 'label_x' in n
    }
    before_rx = {
        rx['bigg_id']: (rx['label_x'] - data[1]['nodes']['r1' if rx['bigg_id'] == 'FRD7' else 'r2']['x'],
                        rx['label_y'] - data[1]['nodes']['r1' if rx['bigg_id'] == 'FRD7' else 'r2']['y'])
        for rx in data[1]['reactions'].values()
    }
    _, rep = CA.restore_rigid_group_escher(data, blocks)
    assert rep['restored'] >= 4
    # Midmarkers realign onto a common translation of the snapshot → positions change.
    assert abs(data[1]['nodes']['r1']['y'] - data[1]['nodes']['r2']['y']) < 1e-6
    for nid, (ox, oy) in before_met.items():
        n = data[1]['nodes'][nid]
        assert abs((n['label_x'] - n['x']) - ox) < 1e-6
        assert abs((n['label_y'] - n['y']) - oy) < 1e-6
        assert math.hypot(n['label_x'] - n['x'], n['label_y'] - n['y']) < 20.0
    for rid, (ox, oy) in before_rx.items():
        rx = next(r for r in data[1]['reactions'].values() if r['bigg_id'] == rid)
        mid = 'r1' if rid == 'FRD7' else 'r2'
        mn = data[1]['nodes'][mid]
        assert abs((rx['label_x'] - mn['x']) - ox) < 1e-6
        assert abs((rx['label_y'] - mn['y']) - oy) < 1e-6


def test_clear_rigid_overlaps_escher_moves_labels():
    data = [{}, {
        'nodes': {
            'rigid': {'node_type': 'metabolite', 'bigg_id': 'met', 'x': 10.0, 'y': 0.0,
                      'label_x': 18.0, 'label_y': 4.0, 'node_is_primary': True},
            'other': {'node_type': 'metabolite', 'bigg_id': 'other', 'x': 10.5, 'y': 0.2,
                      'label_x': 18.5, 'label_y': 4.2, 'node_is_primary': True},
            'mid': {'node_type': 'midmarker', 'x': 0.0, 'y': 0.0},
        },
        'reactions': {
            '0': {'bigg_id': 'R1', 'label_x': 5.0, 'label_y': 5.0,
                  'segments': {'s0': {'from_node_id': 'mid', 'to_node_id': 'rigid', 'b1': None, 'b2': None}}},
            '1': {'bigg_id': 'ALG', 'label_x': 20.0, 'label_y': 5.0,
                  'segments': {'s0': {'from_node_id': 'mid', 'to_node_id': 'other', 'b1': None, 'b2': None}}},
        },
    }]
    blocks = [dict(id='b', same_map_rigid='M', reaction_ids=['R1'],
                   same_map_snapshot={'r:R1': [0.0, 0.0], 'm:met': [10.0, 0.0]})]
    before = {nid: (n['label_x'] - n['x'], n['label_y'] - n['y'])
              for nid, n in data[1]['nodes'].items() if 'label_x' in n}
    _, rep = CA.clear_rigid_node_overlaps_escher(data, blocks, min_dist=12.0)
    assert rep['moved'] >= 1
    for nid, (ox, oy) in before.items():
        n = data[1]['nodes'][nid]
        assert abs((n['label_x'] - n['x']) - ox) < 1e-6
        assert abs((n['label_y'] - n['y']) - oy) < 1e-6


def test_reanchor_distant_pulls_floating_labels():
    data = [{}, {
        'nodes': {
            'm1': {'node_type': 'metabolite', 'bigg_id': 'fum_c', 'x': 100.0, 'y': 100.0,
                   'label_x': 110.0, 'label_y': 2000.0, 'node_is_primary': True},
            'r1': {'node_type': 'midmarker', 'x': 50.0, 'y': 50.0},
        },
        'reactions': {
            '0': {'bigg_id': 'FRD7', 'label_x': 55.0, 'label_y': 1900.0,
                  'segments': {'s0': {'from_node_id': 'r1', 'to_node_id': 'm1', 'b1': None, 'b2': None}}},
        },
    }]
    rep = LP.reanchor_distant(data)
    assert rep['reanchored'] == 2
    assert max(_met_label_dists(data) + _rx_label_dists(data)) <= LP.MAX_ANCHOR_DIST


def _assert_composed_labels_anchored(folder, max_dist=LP.MAX_ANCHOR_DIST):
    from pathlib import Path
    from core import read
    import output_names
    path = output_names.map_json(Path(folder))
    assert path.exists(), path
    data = read(path)
    mets = _met_label_dists(data)
    rxs = _rx_label_dists(data)
    assert mets, 'no metabolite labels'
    assert max(mets) <= max_dist, (max(mets), sorted(mets, reverse=True)[:5])
    assert max(rxs) <= max_dist, (max(rxs), sorted(rxs, reverse=True)[:5])
    assert float(np.percentile(mets, 95)) <= max_dist


def test_e_coli_core_labels_anchored(tmp_path):
    """Compose e_coli_core and assert every node label stays near its glyph."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / 'e_coli_core'
    gem = root / 'examples' / 'e_coli_core.xml'
    if not gem.exists():
        import pytest
        pytest.skip('examples/e_coli_core.xml missing')
    r = subprocess.run(
        [sys.executable, str(root / 'project.py'), 'compose',
         '--gem', str(gem), '--out', str(out), '--source-paths'],
        cwd=str(root), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    _assert_composed_labels_anchored(out)


def test_iIT341_labels_anchored(tmp_path):
    """Compose iIT341 and assert every node label stays near its glyph."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / 'iIT341'
    gem = root / 'examples' / 'iIT341.xml'
    if not gem.exists():
        import pytest
        pytest.skip('examples/iIT341.xml missing')
    r = subprocess.run(
        [sys.executable, str(root / 'project.py'), 'compose',
         '--gem', str(gem), '--out', str(out), '--source-paths'],
        cwd=str(root), capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    _assert_composed_labels_anchored(out)
