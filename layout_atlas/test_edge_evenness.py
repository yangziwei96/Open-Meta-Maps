"""Median-derived currency stub ceiling for algorithm-placed reactions."""
import edge_evenness as ee
import orientation as o


def _scene_with_core_edges(lengths):
    """Build a tiny model/scene whose core mid→met chords match ``lengths``."""
    n = len(lengths)
    metabolites = [dict(id=f'm{i}_c', compartment='c') for i in range(n)]
    metabolites += [dict(id='atp_c', compartment='c')]
    reactions = []
    positions = {}
    participants = {}
    for i, L in enumerate(lengths):
        rid = f'R{i}'
        mid = f'm{i}_c'
        reactions.append(dict(id=rid, metabolites={mid: -1, 'atp_c': -1},
                              lower_bound=0, upper_bound=1000))
        positions[f'r:{rid}'] = [0.0, float(i) * 200.0]
        positions[f'm:{mid}'] = [float(L), float(i) * 200.0]
        positions[f'side:{rid}:atp_c'] = [0.0, float(i) * 200.0 + 40.0]
        participants[(rid, mid)] = f'm:{mid}'
        participants[(rid, 'atp_c')] = f'side:{rid}:atp_c'
    model = dict(id='t', metabolites=metabolites, reactions=reactions)
    scene = dict(positions=positions, participants=participants, copies={}, labels={})
    return model, scene


def test_median_core_edge_ignores_side_stubs():
    model, scene = _scene_with_core_edges([100.0] * 10 + [400.0] * 2)
    cur = {'atp_c'}
    med = ee.median_core_edge(model, scene, cur, default=0.0)
    assert abs(med - 100.0) < 1e-6


def test_stub_ceiling_tracks_median():
    lo = o.stub_ceiling(80.0)
    hi = o.stub_ceiling(200.0)
    assert hi >= lo
    assert o.SIDE_MAX - 1e-6 <= lo <= o.SIDE_MAX + o.SIDE_MIN * o.SIDE_K + 1e-6
    assert abs(o.stub_ceiling(2.0 * o.SIDE_MAX + o.SIDE_SPREAD_MAX) - o.SIDE_MAX) < 1e-6
    assert abs(hi - (o.SIDE_MAX + o.SIDE_MIN * o.SIDE_K)) < 1e-6


def test_reaction_spoke_params_raise_only_non_curated():
    model, scene = _scene_with_core_edges([160.0] * 12)
    for i, r in enumerate(model['reactions']):
        scene['positions'][f'r:{r["id"]}'] = [0.0, float(i) * 500.0]
        scene['positions'][f'm:m{i}_c'] = [160.0, float(i) * 500.0]
        scene['positions'][f'side:{r["id"]}:atp_c'] = [0.0, float(i) * 500.0 + 40.0]
    cur = {'atp_c'}
    blocks = [dict(id='curated', reaction_ids=['R0'], positions={})]
    params = o.reaction_spoke_params(model, scene, cur, blocks=blocks)
    raised = o.stub_ceiling(160.0)
    assert abs(params['R0']['distance'] - o.SIDE_MAX) < 1e-6
    assert abs(params['R1']['distance'] - raised) < 1e-6
    assert params['R1']['distance'] > o.SIDE_MAX


def test_place_side_glyphs_keeps_side_max_when_raise_overlaps():
    model, scene = _scene_with_core_edges([160.0] * 12)
    for i, r in enumerate(model['reactions']):
        scene['positions'][f'r:{r["id"]}'] = [0.0, float(i) * 500.0]
        scene['positions'][f'm:m{i}_c'] = [160.0, float(i) * 500.0]
        scene['positions'][f'side:{r["id"]}:atp_c'] = [0.0, float(i) * 500.0 + 40.0]
    # Substrate-only R0 → axis from stub side toward +x; place blocker beyond SIDE_MAX on -x.
    scene['positions']['m:m0_c'] = [-160.0, 0.0]
    scene['positions']['m:blocker_c'] = [-65.0, 0.0]
    scene['copies']['m:blocker_c'] = 'blocker_c'
    model['metabolites'].append(dict(id='blocker_c', compartment='c'))
    cur = {'atp_c'}
    params = {r['id']: dict(distance=o.SIDE_MAX + o.SIDE_MIN * o.SIDE_K, spread=o.SIDE_SPREAD_MAX,
                            gap=o.side_gap_for(o.SIDE_MAX), clearance=500.0, floor=o.SIDE_MIN)
              for r in model['reactions']}
    moved, params2 = o.place_side_glyphs(model, scene['positions'], scene['participants'], cur,
                                         params=params, blocks=[], copies=scene['copies'])
    assert moved >= 1
    assert abs(params2['R0']['distance'] - o.SIDE_MAX) < 1e-6
    stub = scene['positions']['side:R0:atp_c']
    assert abs(abs(stub[0]) - o.SIDE_MAX) < 1.0
