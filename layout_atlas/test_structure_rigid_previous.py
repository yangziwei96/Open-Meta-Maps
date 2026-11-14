"""Curated structures stay rigid under ``--previous`` pins — including the FINAL map.

For every curated structure, relative coordinates of nodes, midmarkers, multimarkers,
b1/b2 curve controls, and metabolite / reaction labels must match the no-previous
drawing (translation only, float tolerance).  Rigidity is per curated structure
(spatial piece), not per whole module.  Shared reactions follow an explicit
ownership rule (most ``copy:`` participant hits → larger structure → smaller id).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import previous_layout as prev

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('e_coli_core', 'iIT341')
TOL = 1e-5  # set_node_xy round-trip at 6 decimals


def _read_map(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _make_partial_previous(data, out_path: Path, keep_fraction: float = 0.5) -> Path:
    """Drop later curated structures so ``--previous`` is a partial earlier map."""
    header, body = data[0], data[1]
    es = list((header.get('metadata') or {}).get('expert_structures') or [])
    assert es, 'expected curated structures in map metadata'
    n_keep = max(1, int(len(es) * keep_fraction))
    drop = set()
    for s in es[n_keep:]:
        drop.update(s.get('reactions') or [])
    assert drop, 'partial previous must drop at least one structure'
    reactions = {
        rid: rx for rid, rx in (body.get('reactions') or {}).items()
        if rx.get('bigg_id') not in drop
    }
    used = set()
    for rx in reactions.values():
        for seg in (rx.get('segments') or {}).values():
            for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                if nid is not None:
                    used.add(str(nid))
    nodes = {nid: n for nid, n in (body.get('nodes') or {}).items() if nid in used}
    meta = dict(header.get('metadata') or {})
    meta['expert_structures'] = es[:n_keep]
    out = [dict(header, metadata=meta), dict(body, nodes=nodes, reactions=reactions)]
    out_path.write_text(json.dumps(out), encoding='utf-8')
    return out_path


def _scene_from_result(data):
    """Rebuild a display-like scene from an exported map + glyph_identity."""
    meta, body = data[0]['metadata'], data[1]
    gi = meta.get('glyph_identity') or {}
    nodes, rxs = body['nodes'], body['reactions']
    es = meta.get('expert_structures') or []
    rid_to_module = {}
    for s in es:
        mod = (s.get('parent') or s['id']).split('#')[0]
        for rid in s.get('reactions') or []:
            rid_to_module[rid] = mod
    copy_keys = {k for k in gi if k.startswith('copy:') and k.count(':') >= 2}
    pos, parts = {}, {}
    blocks_by_id = {}
    for rx in rxs.values():
        rid = rx.get('bigg_id')
        if not rid:
            continue
        mid_nid = None
        for seg in (rx.get('segments') or {}).values():
            for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                if (nodes.get(nid) or {}).get('node_type') == 'midmarker':
                    mid_nid = nid
                    break
            if mid_nid is not None:
                break
        if mid_nid is None:
            continue
        pos['r:' + rid] = [float(nodes[mid_nid]['x']), float(nodes[mid_nid]['y'])]
        mod = rid_to_module.get(rid)
        if mod is not None:
            blocks_by_id.setdefault(mod, dict(id=mod, reaction_ids=[]))
            if rid not in blocks_by_id[mod]['reaction_ids']:
                blocks_by_id[mod]['reaction_ids'].append(rid)
        for seg in (rx.get('segments') or {}).values():
            for nid in (seg.get('from_node_id'), seg.get('to_node_id')):
                if nid == mid_nid:
                    continue
                n = nodes.get(nid) or {}
                if n.get('node_type') != 'metabolite' or not n.get('bigg_id'):
                    continue
                met = n['bigg_id']
                key = None
                if mod is not None:
                    ck = f'copy:{mod}:{met}'
                    if ck in copy_keys:
                        key = ck
                if key is None:
                    sk = f'side:{rid}:{met}'
                    key = sk if sk in gi else 'm:' + met
                pos[key] = [float(n['x']), float(n['y'])]
                parts[(rid, met)] = key
    blocks = [dict(id=bid, reaction_ids=list(b['reaction_ids']))
              for bid, b in sorted(blocks_by_id.items())]
    return dict(positions=pos, participants=parts, copies={}, labels={}), blocks


def _structure_relatives_from_scene(scene, blocks=None):
    """Relative coords of every structure member to that structure's reference midmarker."""
    pos = scene['positions']
    membership = prev._structure_membership(scene, pos, blocks=blocks)
    out = {}
    for bid, keys in membership.items():
        keys = sorted(k for k in keys if k in pos)
        if not keys:
            continue
        ref = next((k for k in keys if k.startswith('r:')), keys[0])
        ref_xy = np.asarray(pos[ref], float)
        out[bid] = {k: (np.asarray(pos[k], float) - ref_xy).tolist() for k in keys}
    return out


def _assert_relatives_equal(a, b, label):
    assert a, f'{label}: no curated structures'
    md = 0.0
    n = 0
    for bid, ra in a.items():
        rb = b.get(bid) or {}
        for k, va in ra.items():
            if k not in rb:
                continue
            d = float(np.hypot(*(np.asarray(va, float) - np.asarray(rb[k], float))))
            md = max(md, d)
            n += 1
    assert n > 0, f'{label}: no shared structure points'
    assert md <= TOL, f'{label}: max relative deviation {md} exceeds {TOL} ({n} points)'


def _escher_structure_relatives(data):
    """Relative geometry for the final drawn map (nodes, labels, b1/b2)."""
    snaps = prev.snapshot_structure_relatives_escher(data)
    return {sid: snap['relatives'] for sid, snap in snaps.items()}


def _assert_escher_relatives_equal(a, b, label):
    assert a, f'{label}: no curated structures on final map'
    md = 0.0
    n = 0
    missing = 0
    for sid, ra in a.items():
        rb = b.get(sid) or {}
        for k, va in ra.items():
            if k not in rb:
                missing += 1
                continue
            d = float(np.hypot(*(np.asarray(va, float) - np.asarray(rb[k], float))))
            md = max(md, d)
            n += 1
    assert n > 0, f'{label}: no shared final-map structure points'
    assert md <= TOL, (
        f'{label}: max relative deviation {md} exceeds {TOL} '
        f'({n} points compared, {missing} keys only on one side)')


def _compose_assets_ready():
    build_json = ROOT / 'layout_atlas' / 'results' / 'routing_build.json'
    if not build_json.is_file():
        # routing_build may live under layout_atlas/results or results/
        alt = ROOT / 'results' / 'routing_build.json'
        if not alt.is_file():
            return 'missing routing_build.json'
    gem = ROOT / 'examples' / 'e_coli_core.xml'
    if not gem.is_file():
        return f'missing {gem}'
    return None


def _compose(gem: Path, out: Path, previous: Path | None = None) -> Path:
    import compose_cli
    out.mkdir(parents=True, exist_ok=True)
    rc = compose_cli.run(
        gem.resolve(), out,
        source_paths=True, layout_style='curated',
        previous=previous.resolve() if previous is not None else None)
    assert rc == 0, f'compose failed rc={rc} out={out}'
    # Prefer OpenMetaMaps_*_result.json; fall back to map.json.
    matches = sorted(out.glob('OpenMetaMaps_*_result.json'))
    if matches:
        return matches[0]
    alt = out / 'map.json'
    assert alt.is_file(), f'no result map in {out}'
    return alt


@pytest.mark.parametrize('model_name', MODELS)
def test_pin_scene_preserves_curated_relatives(model_name, tmp_path):
    """Curated source relatives survive ``pin_scene`` with and without ``--previous``."""
    example = ROOT / 'example_results' / model_name / f'OpenMetaMaps_{model_name}_result.json'
    if not example.is_file():
        pytest.skip(f'missing {example}')
    data = _read_map(example)
    scene, blocks = _scene_from_result(data)
    curated = _structure_relatives_from_scene(scene, blocks=blocks)
    assert curated, model_name

    scene_noprev = dict(
        positions={k: list(v) for k, v in scene['positions'].items()},
        participants=dict(scene['participants']),
        copies={}, labels={},
    )
    prev.pin_scene(scene_noprev, anchors={}, blocks=blocks)
    _assert_relatives_equal(
        curated, _structure_relatives_from_scene(scene_noprev, blocks=blocks),
        f'{model_name} without --previous')

    partial = _make_partial_previous(data, tmp_path / f'{model_name}_partial.json')
    anchors = prev.load_previous_anchors(partial)
    scene_prev = dict(
        positions={k: list(v) for k, v in scene['positions'].items()},
        participants=dict(scene['participants']),
        copies={}, labels={},
    )
    prev.pin_scene(scene_prev, anchors, blocks=blocks)
    _assert_relatives_equal(
        curated, _structure_relatives_from_scene(scene_prev, blocks=blocks),
        f'{model_name} with --previous')


def test_shared_reaction_ownership_rule():
    """Shared reactions: most copy hits, then larger structure, then smaller id."""
    structures = [
        dict(id='A#cc0', parent='A', reactions=['R1', 'R2']),
        dict(id='B', parent='B', reactions=['R1', 'R3', 'R4']),
    ]
    # R1 has one copy glyph under module A and two under B → B wins despite larger id.
    scene = dict(
        participants={
            ('R1', 'm1'): 'copy:A:m1',
            ('R1', 'm2'): 'copy:B:m2',
            ('R1', 'm3'): 'copy:B:m3',
            ('R2', 'm1'): 'copy:A:m1',
            ('R3', 'x'): 'copy:B:x',
        },
        positions={
            'r:R1': [0, 0], 'r:R2': [1, 0], 'r:R3': [2, 0], 'r:R4': [3, 0],
            'copy:A:m1': [0, 1], 'copy:B:m2': [0, 2], 'copy:B:m3': [0, 3],
            'copy:B:x': [2, 1],
        },
    )
    owners = prev.assign_reaction_owners(structures, scene=scene, pos=scene['positions'])
    assert owners['R1'] == 'B'
    assert owners['R2'] == 'A#cc0'
    assert owners['R3'] == 'B'
    assert owners['R4'] == 'B'

    # Tie on copy hits: larger structure wins; still tied → smaller structure id.
    structures2 = [
        dict(id='Z', parent='Z', reactions=['R1']),
        dict(id='Y', parent='Y', reactions=['R1']),
    ]
    scene2 = dict(participants={}, positions={'r:R1': [0, 0]})
    owners2 = prev.assign_reaction_owners(structures2, scene=scene2, pos=scene2['positions'])
    assert owners2['R1'] == 'Y'  # same size (1), lexicographically smaller id


@pytest.mark.parametrize('model_name', MODELS)
def test_final_map_structure_relatives_with_partial_previous(model_name, tmp_path):
    """FINAL drawn maps: curated structure relatives match no-previous (tol 0)."""
    reason = _compose_assets_ready()
    if reason:
        pytest.skip(reason)
    gem = ROOT / 'examples' / f'{model_name}.xml'
    if not gem.is_file():
        pytest.skip(f'missing {gem}')
    env_seed = os.environ.get('PYTHONHASHSEED', '0')
    os.environ['PYTHONHASHSEED'] = '0'
    try:
        noprev_dir = tmp_path / f'{model_name}_noprev'
        prev_dir = tmp_path / f'{model_name}_prev'
        noprev_map = _compose(gem, noprev_dir)
        noprev = _read_map(noprev_map)
        partial = _make_partial_previous(noprev, tmp_path / f'{model_name}_partial.json')
        prev_map = _compose(gem, prev_dir, previous=partial)
        drawn = _read_map(prev_map)
    finally:
        os.environ['PYTHONHASHSEED'] = env_seed

    ref = _escher_structure_relatives(noprev)
    got = _escher_structure_relatives(drawn)
    # Only structures present on both sides (partial previous still draws all model rx).
    common = {sid: ref[sid] for sid in ref if sid in got}
    assert common, f'{model_name}: no common curated structures'
    _assert_escher_relatives_equal(common, got, f'{model_name} final map with --previous')


def test_angles_from_map_skips_missing_multimarker():
    """Dangling segment endpoints must not crash layout quality under --previous."""
    import orientation
    body = {
        'nodes': {
            'm0': dict(node_type='midmarker', x=0.0, y=0.0),
            'a': dict(node_type='metabolite', bigg_id='a_c', x=-10.0, y=0.0),
            'b': dict(node_type='metabolite', bigg_id='b_c', x=10.0, y=0.0),
        },
        'reactions': {
            'r1': dict(
                bigg_id='RX',
                metabolites=[
                    dict(bigg_id='a_c', coefficient=-1),
                    dict(bigg_id='b_c', coefficient=1),
                ],
                segments={
                    '0': dict(from_node_id='a', to_node_id='m0', b1=None, b2=None),
                    '1': dict(from_node_id='m0', to_node_id='es1', b1=None, b2=None),  # dangling
                    '2': dict(from_node_id='es1', to_node_id='b', b1=None, b2=None),
                },
            ),
        },
    }
    angles = orientation.angles_from_map([{}, body])
    assert 'RX' in angles
