"""Bridge-only share policy: share m: only when it joins otherwise-separate parts."""
import copy

import instances as I


def _scene(positions, participants, copies=None, labels=None):
    copies = copies or {k: k.split(':')[-1] for k in positions if not k.startswith('r:')}
    labels = labels or {k: copies.get(k, k[2:] if k.startswith('r:') else k) for k in positions}
    return dict(positions=positions, participants=participants, copies=copies, labels=labels)


def test_share_joins_bridges_separate_components():
    # R1—A—R2 and R3—B—R4 are separate; X is only used by R1 and R3 → both may share.
    model = {
        'reactions': [
            {'id': 'R1', 'metabolites': {'A_c': -1, 'X_c': 1}},
            {'id': 'R2', 'metabolites': {'A_c': -1, 'C_c': 1}},
            {'id': 'R3', 'metabolites': {'B_c': -1, 'X_c': 1}},
            {'id': 'R4', 'metabolites': {'B_c': -1, 'D_c': 1}},
        ],
        'metabolites': [],
    }
    parts = {
        ('R1', 'A_c'): 'm:A_c', ('R1', 'X_c'): 'm:X_c',
        ('R2', 'A_c'): 'm:A_c', ('R2', 'C_c'): 'm:C_c',
        ('R3', 'B_c'): 'm:B_c', ('R3', 'X_c'): 'm:X_c',
        ('R4', 'B_c'): 'm:B_c', ('R4', 'D_c'): 'm:D_c',
    }
    joins = I.share_joins(model, parts, 'X_c', set(), ['R1', 'R3'])
    assert joins == {'R1', 'R3'}


def test_share_joins_prefers_local_when_already_connected():
    # R1—A—R2 already connects R1 to R2 without sharing X; only one share.
    model = {
        'reactions': [
            {'id': 'R1', 'metabolites': {'A_c': -1, 'X_c': 1}},
            {'id': 'R2', 'metabolites': {'A_c': -1, 'X_c': 1}},
        ],
        'metabolites': [],
    }
    parts = {
        ('R1', 'A_c'): 'm:A_c', ('R1', 'X_c'): 'm:X_c',
        ('R2', 'A_c'): 'm:A_c', ('R2', 'X_c'): 'm:X_c',
    }
    joins = I.share_joins(model, parts, 'X_c', set(), ['R1', 'R2'])
    assert joins == {'R1'}
    assert 'R2' not in joins


def test_optimize_keeps_bridging_share_and_clones_non_bridge():
    # All reactions within NEAR of m:X so proximity keeps them on the shared glyph;
    # bridge policy then peels RC (already tied to RA via P) while RA+RB stay shared.
    model = {
        'reactions': [
            {'id': 'RA', 'metabolites': {'P_c': -1, 'X_c': 1}},
            {'id': 'RB', 'metabolites': {'Q_c': -1, 'X_c': 1}},
            {'id': 'RC', 'metabolites': {'P_c': -1, 'X_c': 1}},
        ],
        'metabolites': [],
    }
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'r:RB': [100.0, 0.0],
            'r:RC': [20.0, 30.0],
            'm:P_c': [10.0, 50.0],
            'm:Q_c': [90.0, 50.0],
            'm:X_c': [50.0, 0.0],
        },
        {
            ('RA', 'P_c'): 'm:P_c', ('RA', 'X_c'): 'm:X_c',
            ('RB', 'Q_c'): 'm:Q_c', ('RB', 'X_c'): 'm:X_c',
            ('RC', 'P_c'): 'm:P_c', ('RC', 'X_c'): 'm:X_c',
        },
    )
    out, rep = I.optimize(model, copy.deepcopy(scene), blocks=[], cur=set(), summary=False)
    x_of = {rid: out['participants'][(rid, 'X_c')] for rid in ('RA', 'RB', 'RC')}
    shared = {rid for rid, key in x_of.items() if key == 'm:X_c'}
    local = {rid for rid, key in x_of.items() if str(key).startswith('inst:')}
    # RB is otherwise separate → must remain a bridge share; exactly one of RA/RC stays shared.
    assert 'RB' in shared
    assert len(shared & {'RA', 'RC'}) == 1
    assert len(local & {'RA', 'RC'}) == 1
    peeled = next(iter(local & {'RA', 'RC'}))
    assert out['instance_reasons'][x_of[peeled]] == 'bridge_local'
    assert rep['bridge_local'] >= 1


def test_optimize_peels_far_bridge_joins_to_local():
    """A topologically bridging share that sits beyond FAR must still get a local inst:.

    Regression for iIT341 after #33: bridge-only share left edges_over_1000 ≫ 0.
    """
    model = {
        'reactions': [
            {'id': 'RA', 'metabolites': {'P_c': -1, 'X_c': 1}},
            {'id': 'RB', 'metabolites': {'Q_c': -1, 'X_c': 1}},
        ],
        'metabolites': [],
    }
    # RB is otherwise separate (bridge) but 5000 px from m:X — must peel locally.
    scene = _scene(
        {
            'r:RA': [0.0, 0.0],
            'r:RB': [5000.0, 0.0],
            'm:P_c': [10.0, 50.0],
            'm:Q_c': [5010.0, 50.0],
            'm:X_c': [50.0, 0.0],
        },
        {
            ('RA', 'P_c'): 'm:P_c', ('RA', 'X_c'): 'm:X_c',
            ('RB', 'Q_c'): 'm:Q_c', ('RB', 'X_c'): 'm:X_c',
        },
    )
    out, rep = I.optimize(model, copy.deepcopy(scene), blocks=[], cur=set(), summary=False)
    assert out['participants'][('RA', 'X_c')] == 'm:X_c'
    rb_key = out['participants'][('RB', 'X_c')]
    assert str(rb_key).startswith('inst:'), rb_key
    rb_xy = out['positions'][rb_key]
    # Local stub sits beside RB, not on the distant shared glyph.
    assert abs(rb_xy[0] - 5000.0) < 80.0
    # May peel via long_edge_split and/or bridge_local; either clears the spoke.
    assert out['instance_reasons'].get(rb_key) in ('bridge_local', 'long_edge_split')
    assert rep['edges_over_1000']['after'] == 0


def test_optimize_clones_when_all_users_already_connected():
    # Within NEAR: proximity would keep both on m:X; bridge peels the non-spanning one.
    model = {
        'reactions': [
            {'id': 'R1', 'metabolites': {'A_c': -1, 'X_c': 1}},
            {'id': 'R2', 'metabolites': {'A_c': -1, 'X_c': 1}},
        ],
        'metabolites': [],
    }
    scene = _scene(
        {
            'r:R1': [0.0, 0.0],
            'r:R2': [60.0, 0.0],
            'm:A_c': [30.0, 40.0],
            'm:X_c': [30.0, -40.0],
        },
        {
            ('R1', 'A_c'): 'm:A_c', ('R1', 'X_c'): 'm:X_c',
            ('R2', 'A_c'): 'm:A_c', ('R2', 'X_c'): 'm:X_c',
        },
    )
    out, rep = I.optimize(model, copy.deepcopy(scene), blocks=[], cur=set(), summary=False)
    shared_x = [rid for (rid, mid), key in out['participants'].items() if mid == 'X_c' and key == 'm:X_c']
    local_x = [rid for (rid, mid), key in out['participants'].items() if mid == 'X_c' and str(key).startswith('inst:')]
    assert len(shared_x) == 1
    assert len(local_x) == 1
    assert rep['bridge_local'] >= 1
    assert out['instance_reasons'][out['participants'][(local_x[0], 'X_c')]] == 'bridge_local'
