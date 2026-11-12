"""Node-based cofactor hub counting matches the shared-glyph definition."""
from cofactor_hubs import cofactor_hubs


def test_single_glyph_multi_reaction_is_hub():
    model = dict(
        id='t',
        metabolites=[dict(id='o2_e', name='O2'), dict(id='o2_c', name='O2')],
        reactions=[],
    )
    # Force currency via chem id matching core.CURRENCY: o2 is currency.
    map_json = [
        {},
        {
            'nodes': {
                '1': dict(node_type='metabolite', bigg_id='o2_e', x=0, y=0),
                '2': dict(node_type='midmarker', x=10, y=0),
                '3': dict(node_type='midmarker', x=20, y=0),
            },
            'reactions': {
                'a': dict(bigg_id='EX_o2_e', segments={'0': dict(from_node_id='1', to_node_id='2')}),
                'b': dict(bigg_id='O2t', segments={'0': dict(from_node_id='1', to_node_id='3')}),
            },
        },
    ]
    report = cofactor_hubs(map_json, model, cur={'o2_e', 'o2_c'})
    assert report['hub_count'] == 1
    assert report['hubs'][0]['metabolite'] == 'o2_e'


def test_cloned_glyphs_are_not_hubs():
    model = dict(id='t', metabolites=[], reactions=[])
    map_json = [
        {},
        {
            'nodes': {
                '1': dict(node_type='metabolite', bigg_id='o2_e', x=0, y=0),
                '2': dict(node_type='metabolite', bigg_id='o2_e', x=5, y=0),
                '3': dict(node_type='midmarker', x=10, y=0),
                '4': dict(node_type='midmarker', x=20, y=0),
            },
            'reactions': {
                'a': dict(bigg_id='EX_o2_e', segments={'0': dict(from_node_id='1', to_node_id='3')}),
                'b': dict(bigg_id='O2t', segments={'0': dict(from_node_id='2', to_node_id='4')}),
            },
        },
    ]
    report = cofactor_hubs(map_json, model, cur={'o2_e'})
    assert report['hub_count'] == 0
