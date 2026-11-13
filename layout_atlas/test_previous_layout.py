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
    assert n >= 2
    assert scene['positions']['r:A'] == [1.0, 2.0]
    assert scene['positions']['r:B'] == [0.0, 0.0]
    # inst: glyphs are not scene-pinned; curated copy: keys follow the primary.
    assert scene['positions']['inst:long:x:0'] == [50.0, 50.0]
    scene2, n2 = prev.pin_scene(
        dict(positions={'r:A': [9.0, 9.0], 'm:x': [8.0, 8.0], 'copy:block:x': [50.0, 50.0]},
             copies={}),
        anchors)
    assert scene2['positions']['copy:block:x'] == [3.0, 4.0]
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
    # Uneven midmarker drift (not a pure frame translation).
    data = [
        {'metadata': {}},
        {
            'nodes': {
                '1': dict(node_type='midmarker', x=180.0, y=260.0),
                '2': dict(node_type='midmarker', x=430.0, y=210.0),
                '3': dict(node_type='metabolite', bigg_id='hub_c', node_is_primary=True,
                          x=310.0, y=140.0, label_x=319.0, label_y=125.0),
                '4': dict(node_type='metabolite', bigg_id='hub_c', node_is_primary=False,
                          x=900.0, y=900.0, label_x=909.0, label_y=885.0),
            },
            'reactions': {
                '0': dict(bigg_id='RA', label_x=185.0, label_y=270.0,
                          segments={'s': dict(
                              from_node_id='1', to_node_id='3',
                              b1=dict(x=200.0, y=250.0), b2=dict(x=280.0, y=160.0))}),
                '1': dict(bigg_id='RB', label_x=435.0, label_y=220.0,
                          segments={'s': dict(from_node_id='2', to_node_id='3')}),
            },
            'canvas': dict(x=0, y=0, width=1000, height=1000),
        },
    ]
    out, n = prev.pin_escher(data, anchors)
    assert n >= 3
    nodes = out[1]['nodes']
    # T = mean((80,60),(30,10)) = (55,35); midmarkers → previous+T.
    assert abs(nodes['1']['x'] - 155.0) < 1e-6 and abs(nodes['1']['y'] - 235.0) < 1e-6
    assert abs(nodes['2']['x'] - 455.0) < 1e-6 and abs(nodes['2']['y'] - 235.0) < 1e-6
    # Primary lands on previous+T; curated copy keeps its post-export position.
    assert abs(nodes['3']['x'] - 305.0) < 1e-6 and abs(nodes['3']['y'] - 135.0) < 1e-6
    assert abs(nodes['4']['x'] - 900.0) < 1e-6 and abs(nodes['4']['y'] - 900.0) < 1e-6
    assert (nodes['3']['x'], nodes['3']['y']) != (nodes['4']['x'], nodes['4']['y'])
    # Pairwise distance between RA and RB matches the previous map.
    assert abs((nodes['2']['x'] - nodes['1']['x']) - 300.0) < 1e-6
    # Bézier controls follow their endpoint node shifts.
    seg = out[1]['reactions']['0']['segments']['s']
    assert abs(seg['b1']['x'] - 175.0) < 1e-6 and abs(seg['b1']['y'] - 225.0) < 1e-6
    assert abs(seg['b2']['x'] - 275.0) < 1e-6 and abs(seg['b2']['y'] - 155.0) < 1e-6


def test_partial_previous_pin_preserves_copies_vs_plain():
    # Subset-model previous anchors (~partial --previous): pinning must not stack
    # every copy of a species, and must not raise coincident overlaps vs the
    # unpinned map (plain compose baseline for this synthetic fragment).
    anchors = {
        'r:PGI': [100.0, 100.0],
        'r:PFK': [300.0, 100.0],
        'm:g6p_c': [50.0, 100.0],
    }
    # Uneven midmarker drift; copies stay 200 px apart after the primary's shift.
    plain = [
        {'metadata': {}},
        {
            'nodes': {
                'm1': dict(node_type='midmarker', x=190.0, y=160.0),
                'm2': dict(node_type='midmarker', x=340.0, y=130.0),
                'g0': dict(node_type='metabolite', bigg_id='g6p_c', node_is_primary=True,
                           x=120.0, y=150.0),
                'g1': dict(node_type='metabolite', bigg_id='g6p_c', node_is_primary=False,
                           x=320.0, y=150.0),
                'f6': dict(node_type='metabolite', bigg_id='f6p_c', node_is_primary=True,
                           x=360.0, y=200.0),
            },
            'reactions': {
                '0': dict(bigg_id='PGI', segments={'s': dict(
                    from_node_id='m1', to_node_id='g0',
                    b1=dict(x=160.0, y=155.0), b2=dict(x=130.0, y=148.0))}),
                '1': dict(bigg_id='PFK', segments={'s': dict(
                    from_node_id='m2', to_node_id='g1',
                    b1=dict(x=320.0, y=135.0), b2=dict(x=330.0, y=148.0))}),
                '2': dict(bigg_id='NEW', segments={'s': dict(
                    from_node_id='m2', to_node_id='f6')}),
            },
            'canvas': dict(x=0, y=0, width=500, height=400),
        },
    ]

    def _coincident_met_pairs(body, tol=1e-6):
        mets = [(n['bigg_id'], float(n['x']), float(n['y']))
                for n in body['nodes'].values() if n.get('node_type') == 'metabolite']
        n = 0
        for i, (a, ax, ay) in enumerate(mets):
            for b, bx, by in mets[i + 1:]:
                if a == b and abs(ax - bx) <= tol and abs(ay - by) <= tol:
                    n += 1
        return n

    plain_pairs = _coincident_met_pairs(plain[1])
    assert plain_pairs == 0
    pinned, n = prev.pin_escher(__import__('copy').deepcopy(plain), anchors)
    assert n >= 3
    nodes = pinned[1]['nodes']
    # T = mean((90,60),(40,30)) = (65,45); primary → previous+T; copy stays put.
    assert abs(nodes['g0']['x'] - 115.0) < 1e-6 and abs(nodes['g0']['y'] - 145.0) < 1e-6
    assert abs(nodes['g1']['x'] - 320.0) < 1e-6 and abs(nodes['g1']['y'] - 150.0) < 1e-6
    assert (nodes['g0']['x'], nodes['g0']['y']) != (nodes['g1']['x'], nodes['g1']['y'])
    assert _coincident_met_pairs(pinned[1]) <= plain_pairs
    # Controls follow endpoint shifts (primary g0 shift -5,-5).
    seg = pinned[1]['reactions']['0']['segments']['s']
    assert abs(seg['b2']['x'] - 125.0) < 1e-6 and abs(seg['b2']['y'] - 143.0) < 1e-6


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
