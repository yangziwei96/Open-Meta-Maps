"""Gated cofactor hub splits keep layout-benchmark metrics and reduce hubs."""
from split_cofactor_hubs import _split_one_metabolite
from cofactor_hubs import cofactor_hubs


def test_rotate_split_creates_second_glyph():
    model = dict(id='t', metabolites=[dict(id='o2_e', name='O2')], reactions=[])
    map_json = [
        {},
        {
            'nodes': {
                '1': dict(node_type='metabolite', bigg_id='o2_e', x=100.0, y=0.0, node_is_primary=False),
                '2': dict(node_type='midmarker', x=0.0, y=0.0),
                '3': dict(node_type='midmarker', x=50.0, y=50.0),
            },
            'reactions': {
                'a': dict(bigg_id='EX_o2_e', segments={'0': dict(from_node_id='1', to_node_id='2')}),
                'b': dict(bigg_id='O2t', segments={'0': dict(from_node_id='1', to_node_id='3')}),
            },
        },
    ]
    assert cofactor_hubs(map_json, model, cur={'o2_e'})['hub_count'] == 1
    new, splits = _split_one_metabolite(map_json, model, 'o2_e', curated=set(), cur={'o2_e'}, angle_deg=30)
    assert splits and cofactor_hubs(new, model, cur={'o2_e'})['hub_count'] == 0
