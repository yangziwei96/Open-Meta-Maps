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
    scene = dict(
        positions={
            'r:A': [9.0, 9.0], 'm:x': [8.0, 8.0], 'r:B': [0.0, 0.0],
            'inst:long:x:0': [50.0, 50.0],
        },
        copies={'m:x': 'x', 'inst:long:x:0': 'x'},
    )
    scene, n = prev.pin_scene(scene, anchors)
    assert n >= 3
    assert scene['positions']['r:A'] == [1.0, 2.0]
    assert scene['positions']['r:B'] == [0.0, 0.0]
    assert scene['positions']['inst:long:x:0'] == [3.0, 4.0]
    blocks = [dict(positions={'r:A': [0.0, 0.0], 'm:x': [0.0, 0.0], 'r:C': [5.0, 5.0]})]
    blocks, n = prev.apply_block_positions(blocks, anchors)
    assert n == 2
    assert blocks[0]['positions']['r:A'] == [1.0, 2.0]
    assert blocks[0]['positions']['r:C'] == [5.0, 5.0]


def test_pin_escher_restores_final_map_pins():
    # After export-frame drift, pin_escher must put previous pins back so
    # load_previous_anchors recovers the same relative geometry (fails if no-op).
    anchors = {
        'r:RA': [100.0, 200.0],
        'r:RB': [400.0, 200.0],
        'm:hub_c': [250.0, 100.0],
    }
    # Map frame is translated +50,+30 vs the previous anchors, and hub was nudged.
    data = [
        {'metadata': {}},
        {
            'nodes': {
                '1': dict(node_type='midmarker', x=150.0, y=230.0),
                '2': dict(node_type='midmarker', x=450.0, y=230.0),
                '3': dict(node_type='metabolite', bigg_id='hub_c', node_is_primary=True,
                          x=310.0, y=140.0, label_x=319.0, label_y=125.0),
                '4': dict(node_type='metabolite', bigg_id='hub_c', node_is_primary=True,
                          x=900.0, y=900.0, label_x=909.0, label_y=885.0),
            },
            'reactions': {
                '0': dict(bigg_id='RA', label_x=155.0, label_y=240.0,
                          segments={'s': dict(from_node_id='1', to_node_id='3')}),
                '1': dict(bigg_id='RB', label_x=455.0, label_y=240.0,
                          segments={'s': dict(from_node_id='2', to_node_id='3')}),
            },
            'canvas': dict(x=0, y=0, width=1000, height=1000),
        },
    ]
    out, n = prev.pin_escher(data, anchors)
    assert n >= 4
    nodes = out[1]['nodes']
    # Translation fit from the two midmarkers: +50, +30.
    assert abs(nodes['1']['x'] - 150.0) < 1e-6 and abs(nodes['1']['y'] - 230.0) < 1e-6
    assert abs(nodes['2']['x'] - 450.0) < 1e-6 and abs(nodes['2']['y'] - 230.0) < 1e-6
    # Every hub glyph sits on the translated previous primary — not left at a corner.
    assert abs(nodes['3']['x'] - 300.0) < 1e-6 and abs(nodes['3']['y'] - 130.0) < 1e-6
    assert abs(nodes['4']['x'] - 300.0) < 1e-6 and abs(nodes['4']['y'] - 130.0) < 1e-6
    # Pairwise distance between RA and RB matches the previous map.
    assert abs((nodes['2']['x'] - nodes['1']['x']) - 300.0) < 1e-6


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
