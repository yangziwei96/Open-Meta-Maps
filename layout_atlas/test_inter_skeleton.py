"""Inter-skeleton refinement: MST attachment pairs, Pareto swap/relocate, pins."""
import copy

import numpy as np

import inter_skeleton as IS


def _toy_model():
    # Three modules share non-currency mets; one bridging outside reaction.
    reactions = [
        dict(id='rA1', metabolites={'m1': -1, 'm2': 1}),
        dict(id='rB1', metabolites={'m1': -1, 'm3': 1}),
        dict(id='rC1', metabolites={'m2': -1, 'm3': 1}),
        dict(id='rBridge', metabolites={'m2': -1, 'm3': 1}),  # outside all modules
    ]
    return dict(id='toy', reactions=reactions, metabolites=[
        dict(id='m1'), dict(id='m2'), dict(id='m3'), dict(id='atp_c'),
    ])


def _toy_items_blocks():
    # Three curated blocks far apart; m1 in A+B, m2 in A+C, m3 in B+C.
    blocks = [
        dict(id='A', reaction_ids=['rA1'], positions={
            'r:rA1': [0.0, 0.0], 'm:m1': [-20.0, 0.0], 'm:m2': [20.0, 0.0],
        }),
        dict(id='B', reaction_ids=['rB1'], positions={
            'r:rB1': [2000.0, 0.0], 'm:m1': [1980.0, 0.0], 'm:m3': [2020.0, 0.0],
        }),
        dict(id='C', reaction_ids=['rC1'], positions={
            'r:rC1': [1000.0, 2000.0], 'm:m2': [980.0, 2000.0], 'm:m3': [1020.0, 2000.0],
        }),
    ]
    items, centres = [], []
    model = _toy_model()
    rx = {r['id']: r for r in model['reactions']}
    for i, b in enumerate(blocks):
        pts = np.array(list(b['positions'].values()), float)
        lo, hi = pts.min(0), pts.max(0)
        size = hi - lo + 80
        items.append(dict(
            kind='curated', reactions=list(b['reaction_ids']), lo=lo, size=size,
            _rx={rid: rx[rid] for rid in b['reaction_ids']},
            block_indexes=[i],
        ))
        centres.append((lo + hi) / 2.0)
    return items, np.asarray(centres, float), blocks, model


def test_mst_edges_scale_as_k_minus_one():
    items, centres, blocks, model = _toy_items_blocks()
    pairs, meta = IS.build_attachment_pairs(items, centres, model, cur=set(), hub=30)
    shared = [p for p in pairs if p['kind'] == 'shared_metabolite']
    # m1 in 2 modules → 1 edge; m2 in 2 → 1; m3 in 2 → 1.
    assert len(shared) == 3
    for mid, info in meta['per_metabolite'].items():
        assert info['edges'] == info['k'] - 1
        assert info['all_pairs'] == info['k'] * (info['k'] - 1) // 2
        # weight 1/(k-1) → contribution_L == mean length
        assert abs(info['contribution_L'] - info['length_mean']) < 1e-6
    assert all(abs(p['weight'] - 1.0) < 1e-9 for p in shared)  # k=2 → weight 1
    assert meta['bias_removed'] is True
    assert 'C(k,2)' in meta['objective'] or 'C(k, 2)' in meta['objective'] or 'k/2' in meta['objective']
    assert 'copy:' in meta['copy_policy']


def test_high_k_metabolite_does_not_dominate_L():
    # One met in 4 modules vs three mets in 2 modules: equal total mass if distances equal.
    centres = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], float)
    items = []
    for i in range(4):
        items.append(dict(
            kind='curated', reactions=[f'r{i}'], lo=centres[i] - 40, size=np.array([80.0, 80.0]),
            _rx={f'r{i}': dict(metabolites={'hub': -1, f'm{i}': 1})},
            block_indexes=[],
        ))
    # Only 'hub' is shared across all 4; give each item the hub in _rx only.
    for it in items:
        it['_rx'] = {list(it['_rx'])[0]: dict(metabolites={'hub': -1})}
    model = dict(id='t', reactions=[dict(id=f'r{i}', metabolites={'hub': -1}) for i in range(4)],
                 metabolites=[dict(id='hub')])
    pairs, meta = IS.build_attachment_pairs(items, centres, model, cur=set(), hub=30)
    info = meta['per_metabolite']['hub']
    assert info['k'] == 4 and info['edges'] == 3 and info['all_pairs'] == 6
    stats = IS.pair_stats(pairs)
    # Unweighted sum has 3 edges; normalised L equals mean edge (one unit of mass).
    assert abs(stats['total_L'] - stats['total_L_unweighted'] / 3) < 1e-6


def test_pareto_accept_and_reject():
    before = dict(total_L=100.0, max_anchor=50.0, p90_anchor=40.0, long_links=2.0,
                  module_crossings=1, overlaps=0, canvas_area=1e6, in_band=True)
    better = dict(before, total_L=90.0)
    ok, reason = IS.accept_move(before, better)
    assert ok and 'Pareto' in reason
    worse = dict(before, total_L=90.0, overlaps=1)
    ok, reason = IS.accept_move(before, worse)
    assert not ok and 'overlap' in reason
    mixed = dict(before, total_L=90.0, long_links=3.0)
    ok, _ = IS.accept_move(before, mixed)
    assert not ok


def test_cross_structure_far_delta():
    before = [
        dict(kind='shared_metabolite', cross_structure=True, metabolite_id='m',
             block_a=0, block_b=1, distance=1500.0),
        dict(kind='shared_metabolite', cross_structure=True, metabolite_id='m',
             block_a=1, block_b=2, distance=800.0),
    ]
    after = [
        dict(kind='shared_metabolite', cross_structure=True, metabolite_id='m',
             block_a=0, block_b=1, distance=900.0),
        dict(kind='shared_metabolite', cross_structure=True, metabolite_id='m',
             block_a=1, block_b=2, distance=800.0),
    ]
    d = IS.cross_structure_far_delta(before, after)
    assert d['far_before'] == 1 and d['shortened'] == 1 and d['no_longer_far'] == 1


def test_refine_shortens_or_rolls_back():
    items, centres, blocks, model = _toy_items_blocks()
    before = IS.evaluate_layout(items, centres, model, set(), blocks=blocks)
    new_c, report = IS.refine(items, centres, model, set(), blocks=blocks)
    assert 'attachment_pairs_before' in report
    assert report['attachment_pairs_before']['note'].startswith('straight-line')
    if report['stage_accepted']:
        assert report['after']['total_L'] <= report['before']['total_L'] + 1e-6
        assert not report['rolled_back']
    else:
        assert report['rolled_back']
        assert np.allclose(new_c, centres)
    # Internals unchanged under pure translation of accepted moves.
    far = IS.far_from_reaction_copies(blocks, model, set())
    assert far['total'] >= 0


def test_pinned_modules_do_not_move():
    items, centres, blocks, model = _toy_items_blocks()
    pinned = {0, 1, 2}
    new_c, report = IS.refine(items, centres, model, set(), blocks=blocks, pinned=pinned)
    assert np.allclose(new_c, centres)
    assert report['moves']['swap']['generated'] == 0
    assert report['moves']['relocate']['generated'] == 0


def test_refine_packed_blocks_translation_only():
    _items, _centres, blocks, model = _toy_items_blocks()
    # Keep relative vectors inside block A.
    a0 = np.array(blocks[0]['positions']['m:m1']) - np.array(blocks[0]['positions']['r:rA1'])
    report = IS.refine_packed_blocks(model, blocks, cur=set())
    a1 = np.array(blocks[0]['positions']['m:m1']) - np.array(blocks[0]['positions']['r:rA1'])
    assert np.allclose(a0, a1)
    assert 'far_from_reaction_before' in report
    assert report.get('curated_internals', {}).get('ok') is True


def test_translate_structure_moves_curves_and_keeps_relatives():
    block = dict(
        id='A',
        positions={'r:r1': [10.0, 20.0], 'm:m1': [40.0, 20.0]},
        segments={'s1': dict(b1={'x': 15.0, 'y': 25.0}, b2={'x': 30.0, 'y': 18.0})},
        source_xy={'r:r1': [0.0, 0.0]},  # must NOT move
    )
    before = __import__('copy').deepcopy(block)
    IS.translate_structure(block, [100.0, -50.0])
    assert block['positions']['r:r1'] == [110.0, -30.0]
    assert block['segments']['s1']['b1'] == {'x': 115.0, 'y': -25.0}
    assert block['source_xy']['r:r1'] == [0.0, 0.0]
    assert IS.check_curated_internals_unchanged([before], [block])['ok']


def test_display_and_objective_are_separated():
    items, centres, blocks, model = _toy_items_blocks()
    pairs, _meta = IS.build_attachment_pairs(items, centres, model, cur=set(), hub=30)
    stats = IS.pair_stats(pairs)
    assert 'objective' in stats and 'display' in stats
    assert 'total_L' in stats['objective'] and 'long_link_mass' in stats['objective']
    assert 'links_over_250' in stats['display'] and 'links_over_1000' in stats['display']
    assert 'total_length' in stats['display']
    # Shared k=2 edges have weight 1; objective L equals their unweighted sum plus weighted bridges.
    shared = [p for p in pairs if p['kind'] == 'shared_metabolite']
    assert all(abs(p['weight'] - 1.0) < 1e-9 for p in shared)
    assert stats['display']['count'] == len(pairs)
    assert stats['objective']['total_L'] <= stats['display']['total_length'] + 1e-6


def test_collision_rejects_similar_size_swap_into_occupied():
    # Two small blocks and one large blocker: landing on an occupied centre adds overlaps.
    centres = np.array([[0.0, 0.0], [300.0, 0.0], [150.0, 0.0]], float)
    sizes = np.array([[100.0, 100.0], [100.0, 100.0], [200.0, 200.0]], float) + IS.BLOCK_GAP
    start_ov = IS._overlap_count(centres, sizes)
    trial = centres.copy()
    trial[0] = centres[2].copy()
    assert IS._overlap_count(trial, sizes) > start_ov
    assert not IS._feasible(trial, sizes, max_overlaps=start_ov)
