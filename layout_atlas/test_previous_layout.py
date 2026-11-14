"""Build-on-previous anchors and graft helpers."""
import previous_layout as prev


def test_load_previous_anchors_filters_to_model(tmp_path):
    body = {
        'nodes': {
            '1': dict(node_type='metabolite', bigg_id='g6p_c', node_is_primary=True, x=10.0, y=20.0),
            '2': dict(node_type='metabolite', bigg_id='f6p_c', node_is_primary=True, x=30.0, y=40.0),
            '3': dict(node_type='midmarker', x=50.0, y=60.0),
            '4': dict(node_type='midmarker', x=70.0, y=80.0),
        },
        'reactions': {
            'r1': dict(bigg_id='PGI', segments={'s': dict(from_node_id='1', to_node_id='3')}),
            'r2': dict(bigg_id='PFK', segments={'s': dict(from_node_id='2', to_node_id='4')}),
        },
    }
    path = tmp_path / 'map.json'
    path.write_text(__import__('json').dumps([{'map_name': 't'}, body]))
    model = dict(
        reactions=[dict(id='PGI'), dict(id='OTHER')],
        metabolites=[dict(id='g6p_c'), dict(id='x_c')],
    )
    anchors = prev.load_previous_anchors(path, model)
    assert anchors['r:PGI'] == [50.0, 60.0]
    assert 'r:PFK' not in anchors
    assert anchors['m:g6p_c'] == [10.0, 20.0]
    assert 'm:f6p_c' not in anchors
    assert prev.reaction_anchor_share(anchors, model) == 0.5
    assert not prev.should_skip_packing(anchors, model)
    model2 = dict(reactions=[dict(id='PGI'), dict(id='PFK')], metabolites=[dict(id='g6p_c')])
    full = prev.load_previous_anchors(path)
    assert prev.should_skip_packing(full, model2)


def test_pin_scene_and_blocks():
    anchors = {'r:A': [1.0, 2.0], 'm:x': [3.0, 4.0]}
    scene = dict(positions={'r:A': [9.0, 9.0], 'm:x': [8.0, 8.0], 'r:B': [0.0, 0.0]})
    scene, n = prev.pin_scene(scene, anchors)
    assert n == 2
    assert scene['positions']['r:A'] == [1.0, 2.0]
    assert scene['positions']['r:B'] == [0.0, 0.0]
    # Matched glyphs: r:A → (+1,+2), m:x → (+3,+4); mean translation (+2,+3).
    blocks = [dict(positions={'r:A': [0.0, 0.0], 'm:x': [0.0, 0.0], 'r:C': [5.0, 5.0]})]
    blocks, n = prev.apply_block_positions(blocks, anchors)
    assert n == 3
    assert blocks[0]['positions']['r:A'] == [2.0, 3.0]
    assert blocks[0]['positions']['m:x'] == [2.0, 3.0]
    assert blocks[0]['positions']['r:C'] == [7.0, 8.0]


def test_pin_scene_translates_curated_structure_rigidly():
    """``--previous`` moves a curated structure by one translation, never per-copy."""
    import numpy as np
    # Structure S: two mets and one midmarker; free m: / r: outside it.
    scene = dict(
        positions={
            'r:R1': [0.0, 0.0],
            'copy:S:a': [-50.0, 10.0],
            'copy:S:b': [50.0, -10.0],
            'm:a': [100.0, 100.0],
            'm:c': [200.0, 200.0],
            'r:OUT': [300.0, 300.0],
            'side:R1:h': [0.0, 25.0],
            'side:OUT:h': [300.0, 320.0],
        },
        participants={
            ('R1', 'a'): 'copy:S:a',
            ('R1', 'b'): 'copy:S:b',
            ('R1', 'h'): 'side:R1:h',
        },
    )
    before = {k: list(v) for k, v in scene['positions'].items()}
    # Conflicting per-node previous anchors that would deform S if applied independently.
    anchors = {
        'r:R1': [100.0, 100.0],
        'm:a': [40.0, 120.0],
        'm:b': [160.0, 80.0],
        'm:c': [210.0, 210.0],
        'r:OUT': [310.0, 310.0],
    }
    scene, n = prev.pin_scene(scene, anchors)
    assert n >= 4
    pos = scene['positions']
    # Structure members share one translation (least-squares mean of matched offsets).
    deltas = [
        np.asarray(anchors['r:R1']) - before['r:R1'],
        np.asarray(anchors['m:a']) - before['copy:S:a'],
        np.asarray(anchors['m:b']) - before['copy:S:b'],
    ]
    expect = np.mean(deltas, axis=0)
    for k in ('r:R1', 'copy:S:a', 'copy:S:b'):
        got = np.asarray(pos[k]) - before[k]
        assert abs(got[0] - expect[0]) < 1e-9
        assert abs(got[1] - expect[1]) < 1e-9
    # Relative geometry inside S unchanged.
    ref = np.asarray(pos['r:R1'])
    for k in ('copy:S:a', 'copy:S:b'):
        rel_before = np.asarray(before[k]) - before['r:R1']
        rel_after = np.asarray(pos[k]) - ref
        assert abs(rel_before[0] - rel_after[0]) < 1e-9
        assert abs(rel_before[1] - rel_after[1]) < 1e-9
    # Outside-structure pins stay independent.
    assert pos['m:c'] == [210.0, 210.0]
    assert pos['r:OUT'] == [310.0, 310.0]
    assert pos['m:a'] == [40.0, 120.0]  # free primary restores independently
    # side: stubs follow their reaction midmarker.
    assert abs(pos['side:R1:h'][0] - (before['side:R1:h'][0] + expect[0])) < 1e-9
    assert abs(pos['side:R1:h'][1] - (before['side:R1:h'][1] + expect[1])) < 1e-9
    assert pos['side:OUT:h'] == [310.0, 330.0]


def test_pin_scene_uses_blocks_when_participants_missing():
    """Block reaction_ids keep midmarkers in the rigid body without participants."""
    import numpy as np
    scene = dict(
        positions={
            'r:R1': [0.0, 0.0],
            'copy:S:a': [-40.0, 0.0],
            'copy:S:b': [40.0, 0.0],
            'side:R1:h': [0.0, 20.0],
        },
        participants={},  # incomplete — membership comes from blocks
    )
    blocks = [dict(id='S', reaction_ids=['R1'])]
    before = {k: list(v) for k, v in scene['positions'].items()}
    anchors = {'r:R1': [100.0, 50.0], 'm:a': [50.0, 50.0], 'm:b': [150.0, 50.0]}
    scene, _n = prev.pin_scene(scene, anchors, blocks=blocks)
    deltas = [
        np.asarray(anchors['r:R1']) - before['r:R1'],
        np.asarray(anchors['m:a']) - before['copy:S:a'],
        np.asarray(anchors['m:b']) - before['copy:S:b'],
    ]
    expect = np.mean(deltas, axis=0)
    for k in before:
        got = np.asarray(scene['positions'][k]) - before[k]
        assert abs(got[0] - expect[0]) < 1e-9
        assert abs(got[1] - expect[1]) < 1e-9
    # Relative geometry unchanged.
    ref = np.asarray(scene['positions']['r:R1'])
    for k in ('copy:S:a', 'copy:S:b', 'side:R1:h'):
        rel_b = np.asarray(before[k]) - before['r:R1']
        rel_a = np.asarray(scene['positions'][k]) - ref
        assert abs(rel_b[0] - rel_a[0]) < 1e-9
        assert abs(rel_b[1] - rel_a[1]) < 1e-9


def test_filter_previous_map_drops_removed_reactions(tmp_path):
    body = {
        'nodes': {
            '1': dict(node_type='metabolite', bigg_id='a_c', node_is_primary=True, x=0.0, y=0.0),
            '2': dict(node_type='midmarker', x=10.0, y=0.0),
            '3': dict(node_type='midmarker', x=20.0, y=0.0),
            '4': dict(node_type='metabolite', bigg_id='b_c', node_is_primary=True, x=30.0, y=0.0),
        },
        'reactions': {
            'r1': dict(bigg_id='KEEP', segments={'s': dict(from_node_id='1', to_node_id='2')}),
            'r2': dict(bigg_id='DROP', segments={'s': dict(from_node_id='3', to_node_id='4')}),
        },
    }
    path = tmp_path / 'map.json'
    path.write_text(__import__('json').dumps([{'map_name': 't', 'metadata': {}}, body]))
    model = dict(reactions=[dict(id='KEEP'), dict(id='NEW')], metabolites=[dict(id='a_c')])
    data, report = prev.filter_previous_map(path, model)
    assert set(data[1]['reactions']) == {'r1'}
    assert '2' in data[1]['nodes'] and '3' not in data[1]['nodes']
    assert report['new_reactions'] == ['NEW']
    assert report['removed_reactions'] == 1


def test_place_leftover_reactions_never_drops_unanchored():
    model = dict(
        reactions=[
            dict(id='OLD', metabolites={'a_c': -1, 'b_c': 1}, lower_bound=-1000, upper_bound=1000),
            dict(id='NEW_EMBED', metabolites={'a_c': -1, 'b_c': 1}, lower_bound=-1000, upper_bound=1000),
            dict(id='NEW_SOLO', metabolites={'z_c': -1, 'y_c': 1}, lower_bound=-1000, upper_bound=1000),
        ],
        metabolites=[dict(id='a_c'), dict(id='b_c'), dict(id='z_c'), dict(id='y_c')],
    )
    anchors = {'r:OLD': [0.0, 0.0], 'm:a_c': [-20.0, 0.0], 'm:b_c': [20.0, 0.0]}
    scene = dict(
        positions={
            'r:OLD': [0.0, 0.0], 'm:a_c': [-20.0, 0.0], 'm:b_c': [20.0, 0.0],
            'r:NEW_EMBED': [999.0, 999.0], 'r:NEW_SOLO': [888.0, 888.0],
        },
        participants={
            ('OLD', 'a_c'): 'm:a_c', ('OLD', 'b_c'): 'm:b_c',
            ('NEW_EMBED', 'a_c'): 'm:a_c', ('NEW_EMBED', 'b_c'): 'm:b_c',
        },
        copies={'m:a_c': 'a_c', 'm:b_c': 'b_c'},
        labels={},
    )
    scene, report = prev.place_leftover_reactions(model, scene, anchors=anchors, cur=set(), min_embed_anchors=2)
    assert report['placed'] == 2
    assert report['embedded'] == 1
    assert report['new_components'] == 1
    assert 'r:NEW_EMBED' in scene['positions']
    assert 'r:NEW_SOLO' in scene['positions']
    assert ('NEW_SOLO', 'z_c') in scene['participants']
    assert ('NEW_SOLO', 'y_c') in scene['participants']
    assert scene['participants'][('NEW_SOLO', 'z_c')] in scene['positions']
    assert scene['positions']['r:OLD'] == [0.0, 0.0]
    assert prev.unanchored_reaction_ids(model, anchors) == ['NEW_EMBED', 'NEW_SOLO']


def test_keep_local_relayout_requires_substantial_crossing_drop():
    assert not prev.keep_local_relayout(10, 9)
    assert not prev.keep_local_relayout(10, 8)
    assert prev.keep_local_relayout(10, 7)
    assert not prev.keep_local_relayout(3, 2)
    assert prev.keep_local_relayout(8, 6)


def test_relax_grafted_midmarkers_keeps_labels_with_nodes():
    """Local graft relax must move labels via map_edit helpers."""
    body = {
        'nodes': {
            'm1': dict(node_type='metabolite', bigg_id='a_c', node_is_primary=True,
                       x=0.0, y=0.0, label_x=5.0, label_y=-5.0),
            'r1': dict(node_type='midmarker', x=50.0, y=0.0),
            'm2': dict(node_type='metabolite', bigg_id='b_c', node_is_primary=True,
                       x=100.0, y=0.0, label_x=105.0, label_y=-5.0),
            'side': dict(node_type='metabolite', bigg_id='h_c', node_is_primary=False,
                         x=50.0, y=20.0, label_x=55.0, label_y=15.0),
        },
        'reactions': {
            'rx': dict(
                bigg_id='NEW',
                label_x=60.0, label_y=-10.0,
                metabolites=[dict(bigg_id='a_c', coefficient=-1),
                             dict(bigg_id='b_c', coefficient=1),
                             dict(bigg_id='h_c', coefficient=1)],
                segments={
                    '0': dict(from_node_id='r1', to_node_id='m1', b1=None, b2=None),
                    '1': dict(from_node_id='r1', to_node_id='m2', b1=None, b2=None),
                    '2': dict(from_node_id='r1', to_node_id='side', b1=None, b2=None),
                },
            ),
        },
        'canvas': dict(x=0, y=0, width=200, height=100),
    }
    data = [{'map_name': 't', 'metadata': {}}, body]
    # Force a move by shifting base then relaxing toward origin-ish offsets.
    out, report = prev.relax_grafted_midmarkers(data, {'NEW': 'r1'}, step=10.0, radii=(1, 2))
    assert report['crossings_before'] is not None
    # Private side stub label offset preserved relative to node.
    side = out[1]['nodes']['side']
    assert abs((side['label_x'] - side['x']) - 5.0) < 1e-6
    assert abs((side['label_y'] - side['y']) - (-5.0)) < 1e-6
    rx = out[1]['reactions']['rx']
    mid = out[1]['nodes']['r1']
    # Reaction label moved by the same delta as the midmarker from the initial pose.
    assert abs((rx['label_x'] - mid['x']) - 10.0) < 1e-6
    assert abs((rx['label_y'] - mid['y']) - (-10.0)) < 1e-6


def test_graft_new_reactions_places_midmarker_between_sides():
    body = {
        'nodes': {
            '1': dict(node_type='metabolite', bigg_id='a_c', node_is_primary=True, x=0.0, y=0.0),
            '2': dict(node_type='midmarker', x=10.0, y=0.0),
            '3': dict(node_type='metabolite', bigg_id='b_c', node_is_primary=True, x=20.0, y=0.0),
        },
        'reactions': {
            'r1': dict(
                bigg_id='KEEP',
                metabolites=[dict(bigg_id='a_c', coefficient=-1), dict(bigg_id='b_c', coefficient=1)],
                segments={
                    '0': dict(from_node_id='2', to_node_id='1', b1=None, b2=None),
                    '1': dict(from_node_id='2', to_node_id='3', b1=None, b2=None),
                },
            ),
        },
        'canvas': dict(x=0, y=0, width=100, height=100),
    }
    data = [{'map_name': 't', 'metadata': {
        'source_reactions': {
            'KEEP': dict(stoichiometry={'a_c': -1, 'b_c': 1}, lower_bound=-1000, upper_bound=1000),
        },
    }}, body]
    model = dict(
        reactions=[
            dict(id='KEEP', metabolites={'a_c': -1, 'b_c': 1}, lower_bound=-1000, upper_bound=1000),
            dict(id='NEW', metabolites={'a_c': -1, 'b_c': 1}, lower_bound=-1000, upper_bound=1000),
            dict(id='SOLO', metabolites={'z_c': -1, 'y_c': 1}, lower_bound=-1000, upper_bound=1000),
        ],
        metabolites=[dict(id='a_c'), dict(id='b_c'), dict(id='z_c'), dict(id='y_c')],
    )
    out, report = prev.graft_new_reactions(data, model, ['NEW', 'SOLO'])
    assert report['placed'] == 2
    assert report['embedded'] >= 1
    bigg = {rx['bigg_id'] for rx in out[1]['reactions'].values()}
    assert bigg == {'KEEP', 'NEW', 'SOLO'}
    assert out[1]['nodes']['2']['x'] == 10.0
    new_rx = next(rx for rx in out[1]['reactions'].values() if rx['bigg_id'] == 'NEW')
    mid = out[1]['nodes'][next(iter(new_rx['segments'].values()))['from_node_id']]
    assert abs(mid['x'] - 10.0) < 1e-6
    assert 'NEW' in out[0]['metadata']['source_reactions']
    assert 'SOLO' in out[0]['metadata']['source_reactions']
