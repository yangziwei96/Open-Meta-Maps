"""Structure-local metabolite instance assignment (curated reactions)."""
import copy

import numpy as np

import instances as I


def _scene(positions, participants, copies=None, labels=None, anchors=None):
    copies = copies or {k: k.split(':')[-1] for k in positions if not k.startswith('r:')}
    labels = labels or {k: copies.get(k, k[2:] if k.startswith('r:') else k) for k in positions}
    scene = dict(positions=positions, participants=participants, copies=copies, labels=labels)
    if anchors is not None:
        scene['anchors'] = anchors
    return scene


def test_prefers_own_structure_copy_over_foreign():
    # Reaction in block A is currently wired to block B's copy; own copy exists.
    model = {'reactions': [{'id': 'RA', 'metabolites': {'x_c': -1, 'y_c': 1}}], 'metabolites': []}
    blocks = [{'id': 'A', 'reaction_ids': ['RA'], 'positions': {}}]
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'copy:A:x_c': [40.0, 0.0],
            'copy:A:y_c': [-40.0, 0.0],
            'copy:B:x_c': [800.0, 0.0],
            'm:y_c': [0.0, 40.0],
        },
        {('RA', 'x_c'): 'copy:B:x_c', ('RA', 'y_c'): 'copy:A:y_c'},
    )
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur=set())
    assert out['participants'][('RA', 'x_c')] == 'copy:A:x_c'
    assert out['participants'][('RA', 'y_c')] == 'copy:A:y_c'
    assert rep['reassigned_to_own_copy'] == 1
    assert rep['added'] == 0
    # Curated copy geometry untouched.
    assert out['positions']['copy:A:x_c'] == [40.0, 0.0]
    assert out['positions']['copy:B:x_c'] == [800.0, 0.0]


def test_creates_local_inst_for_far_cross_structure_without_own_copy():
    model = {
        'reactions': [
            {'id': 'RA', 'metabolites': {'prpp_c': -1, 'ade_c': 1}},
            {'id': 'RB', 'metabolites': {'prpp_c': -1, 'gua_c': 1}},
        ],
        'metabolites': [],
    }
    blocks = [
        {'id': 'A', 'reaction_ids': ['RA'], 'positions': {}},
        {'id': 'B', 'reaction_ids': ['RB'], 'positions': {}},
    ]
    # RA has no own prpp copy; it currently uses B's copy far away.
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'r:RB': [1000.0, 0.0],
            'copy:B:prpp_c': [1000.0, 40.0],
            'copy:A:ade_c': [40.0, 0.0],
            'copy:B:gua_c': [1040.0, 0.0],
            'm:ade_c': [40.0, 0.0],
            'm:gua_c': [1040.0, 0.0],
        },
        {
            ('RA', 'prpp_c'): 'copy:B:prpp_c',
            ('RA', 'ade_c'): 'copy:A:ade_c',
            ('RB', 'prpp_c'): 'copy:B:prpp_c',
            ('RB', 'gua_c'): 'copy:B:gua_c',
        },
    )
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur=set())
    key = out['participants'][('RA', 'prpp_c')]
    assert key.startswith('inst:local:A:prpp_c')
    assert out['copies'][key] == 'prpp_c'
    assert rep['added'] == 1
    # Foreign curated copy and RB's attachment are unchanged.
    assert out['participants'][('RB', 'prpp_c')] == 'copy:B:prpp_c'
    assert out['positions']['copy:B:prpp_c'] == [1000.0, 40.0]
    # New instance sits on the old edge corridor, at ≥ separate()'s metabolite/reaction gap from RA.
    import inspect
    sep = inspect.signature(I.separate).parameters
    gap = max(float(sep['reaction_gap'].default), float(sep['metabolite_gap'].default))
    xy = out['positions'][key]
    d = float(np.hypot(xy[0] - 0.0, xy[1] - 0.0))
    assert d + 1e-9 >= gap
    assert xy[0] > 0.0  # toward the foreign copy at x=1000



def test_far_threshold_from_layout_not_a_new_constant():
    model = {
        'reactions': [
            {'id': f'R{i}', 'metabolites': {'a_c': -1, 'b_c': 1}} for i in range(12)
        ],
        'metabolites': [],
    }
    positions = {}
    parts = {}
    for i in range(12):
        positions[f'r:R{i}'] = [float(i * 10), 0.0]
        positions[f'm:a_c'] = [0.0, 20.0]
        positions[f'm:b_c'] = [110.0, 20.0]
        parts[(f'R{i}', 'a_c')] = 'm:a_c'
        parts[(f'R{i}', 'b_c')] = 'm:b_c'
    # One deliberately long edge so the fence exceeds NEAR.
    positions['r:R0'] = [0.0, 0.0]
    positions['m:a_c'] = [0.0, 20.0]
    positions['copy:X:a_c'] = [2000.0, 0.0]
    parts[('R0', 'a_c')] = 'copy:X:a_c'
    scene = _scene(positions, parts)
    thr = I._edge_far_threshold(model, scene, cur=set())
    # Fence equals Tukey fence on the same lengths the helper measured (not a hand-picked constant).
    lengths = []
    for r in model['reactions']:
        rp = np.array(scene['positions']['r:' + r['id']], float)
        for mid in r['metabolites']:
            key = scene['participants'][(r['id'], mid)]
            lengths.append(float(np.hypot(*(np.array(scene['positions'][key], float) - rp))))
    q1, q3 = np.percentile(lengths, [25, 75])
    assert abs(thr - float(q3 + 1.5 * (q3 - q1))) < 1e-9
    assert thr != I.FAR


def test_near_cross_structure_copy_left_for_packing():
    # A short foreign-copy link is kept (packing may still use it); only far ones are replaced.
    model = {'reactions': [{'id': 'RA', 'metabolites': {'x_c': -1}}], 'metabolites': []}
    blocks = [{'id': 'A', 'reaction_ids': ['RA'], 'positions': {}}]
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'copy:B:x_c': [80.0, 0.0],
            'm:x_c': [0.0, 40.0],
        },
        {('RA', 'x_c'): 'copy:B:x_c'},
    )
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur=set())
    assert out['participants'][('RA', 'x_c')] == 'copy:B:x_c'
    assert rep['added'] == 0


def test_within_structure_copy_untouched():
    model = {'reactions': [{'id': 'RA', 'metabolites': {'x_c': -1}}], 'metabolites': []}
    blocks = [{'id': 'A', 'reaction_ids': ['RA'], 'positions': {}}]
    scene = _scene(
        {'r:RA': [0.0, 0.0], 'copy:A:x_c': [500.0, 0.0]},
        {('RA', 'x_c'): 'copy:A:x_c'},
    )
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur=set())
    assert out['participants'][('RA', 'x_c')] == 'copy:A:x_c'
    assert out['positions']['copy:A:x_c'] == [500.0, 0.0]
    assert rep['added'] == 0
    assert rep['reassigned_to_own_copy'] == 0


def test_reattach_detaches_protected_group_member():
    model = {'reactions': [{'id': 'RA', 'metabolites': {'x_c': -1}},
                           {'id': 'RB', 'metabolites': {'x_c': -1}}], 'metabolites': []}
    blocks = [{'id': 'A', 'reaction_ids': ['RA'], 'positions': {}}]
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'r:RB': [1000.0, 0.0],
            'copy:B:x_c': [1000.0, 40.0],
        },
        {('RA', 'x_c'): 'copy:B:x_c', ('RB', 'x_c'): 'copy:B:x_c'},
    )
    scene['protected'] = [dict(
        canonical_metabolite_id='x_c', visual_instance_id='copy:B:x_c',
        attached_reaction_ids=['RA', 'RB'], anchor_region='B',
        protection_reason='expert', strict=True)]
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur=set())
    assert out['participants'][('RA', 'x_c')].startswith('inst:local:A:x_c')
    assert out['participants'][('RB', 'x_c')] == 'copy:B:x_c'
    # Remaining protected group is RB alone → dropped; no split left for verify_protected.
    assert all('RA' not in rec['attached_reaction_ids'] for rec in out.get('protected') or [])
    assert rep['added'] == 1


def test_pinned_and_side_glyphs_not_reassigned():
    model = {'reactions': [{'id': 'RA', 'metabolites': {'x_c': -1, 'h_c': 1}}], 'metabolites': []}
    blocks = [{'id': 'A', 'reaction_ids': ['RA'], 'positions': {}}]
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'copy:A:x_c': [40.0, 0.0],
            'copy:B:x_c': [900.0, 0.0],
            'side:RA:h_c': [0.0, 30.0],
        },
        {('RA', 'x_c'): 'copy:B:x_c', ('RA', 'h_c'): 'side:RA:h_c'},
        copies={'copy:A:x_c': 'x_c', 'copy:B:x_c': 'x_c', 'side:RA:h_c': 'h_c'},
        anchors={'copy:B:x_c': {'hard': True, 'xy': [900.0, 0.0]}},
    )
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur={'h_c'})
    # Pinned foreign copy stays the attachment; side glyph stays.
    assert out['participants'][('RA', 'x_c')] == 'copy:B:x_c'
    assert out['participants'][('RA', 'h_c')] == 'side:RA:h_c'
    assert rep['added'] == 0


def test_far_shared_m_reuses_nearby_local_only():
    # Far m:/inst: links reuse an existing movable instance when one is already nearby; no new glyph.
    model = {'reactions': [{'id': 'RA', 'metabolites': {'x_c': -1}}], 'metabolites': []}
    blocks = [{'id': 'A', 'reaction_ids': ['RA'], 'positions': {}}]
    scene = _scene(
        {'r:RA': [0.0, 0.0], 'm:x_c': [900.0, 0.0], 'inst:x_c:0': [60.0, 0.0]},
        {('RA', 'x_c'): 'm:x_c'},
        copies={'m:x_c': 'x_c', 'inst:x_c:0': 'x_c'},
    )
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur=set())
    assert out['participants'][('RA', 'x_c')] == 'inst:x_c:0'
    assert rep['reused_local'] == 1
    assert rep['added'] == 0
    assert out['positions']['m:x_c'] == [900.0, 0.0]


def test_does_not_reuse_other_structure_local_inst():
    # A nearby inst:local of another structure must not be reused (would recreate cross-structure links).
    model = {'reactions': [{'id': 'RA', 'metabolites': {'x_c': -1}}], 'metabolites': []}
    blocks = [{'id': 'A', 'reaction_ids': ['RA'], 'positions': {}}]
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'copy:B:x_c': [900.0, 0.0],
            'inst:local:B:x_c': [70.0, 0.0],
        },
        {('RA', 'x_c'): 'copy:B:x_c'},
        copies={'copy:B:x_c': 'x_c', 'inst:local:B:x_c': 'x_c'},
    )
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur=set())
    key = out['participants'][('RA', 'x_c')]
    assert key.startswith('inst:local:A:x_c')
    assert key != 'inst:local:B:x_c'
    assert rep['added'] == 1
    assert rep['reused_local'] == 0


def test_two_local_inst_keep_metabolite_gap():
    # Two metabolites of one structure whose foreign copies lie on the same ray still clear each other.
    model = {
        'reactions': [{'id': 'RA', 'metabolites': {'a_c': -1, 'b_c': 1}}],
        'metabolites': [],
    }
    blocks = [{'id': 'A', 'reaction_ids': ['RA'], 'positions': {}}]
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'copy:B:a_c': [1000.0, 0.0],
            'copy:B:b_c': [1100.0, 0.0],
        },
        {('RA', 'a_c'): 'copy:B:a_c', ('RA', 'b_c'): 'copy:B:b_c'},
    )
    out, rep = I.structure_local(model, copy.deepcopy(scene), blocks, cur=set())
    assert rep['added'] == 2
    ka = out['participants'][('RA', 'a_c')]
    kb = out['participants'][('RA', 'b_c')]
    import inspect
    gap = float(inspect.signature(I.separate).parameters['metabolite_gap'].default)
    d = float(np.hypot(*(np.array(out['positions'][ka], float) - np.array(out['positions'][kb], float))))
    assert d + 1e-9 >= gap
