"""Round-trip export tests: Escher result JSON → SBGN-ML, SBML Layout, CellDesigner SBML.

Uses e_coli_core and iIT341 only. Every model reaction and species must appear in
each export; SBGN-ML validates against the libSBGN 0.3 XSD; SBML and CellDesigner
validate with libSBML (no errors); coordinates match the result JSON; the
layout-benchmark SBGN reader matches every pathway node.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import EXAMPLES, read
import output_names
from model_io import load_model
import map_exports
from evaluation.pathway import read_sbml
from evaluation.reference_sbgn import match_glyphs, read_sbgn_glyphs, read_sbgn_positions

MODELS = [
    ('e_coli_core', EXAMPLES / 'e_coli_core.xml'),
    ('iIT341', EXAMPLES / 'iIT341.xml'),
]
CACHE = ROOT / 'layout_atlas' / 'results' / 'export_test_maps'

def _result(folder, ext):
    return output_names.map_export(folder, ext)




def collect_export_ids(data, model):
    body = map_exports._body(data)
    rx = {r.get('bigg_id') for r in body['reactions'].values() if r.get('bigg_id')}
    sp = {n.get('bigg_id') for n in body['nodes'].values()
          if n.get('node_type') == 'metabolite' and n.get('bigg_id')}
    return rx, sp


def sbgn_ids(path):
    """Reaction ids (process glyph ids) and species scheme ids / labels from SBGN-ML."""
    root = ET.parse(path).getroot()

    def local(tag):
        return tag.rsplit('}', 1)[-1]

    processes, entities = set(), set()
    for g in root.iter():
        if local(g.tag) != 'glyph':
            continue
        cls = (g.get('class') or '').strip()
        gid = g.get('id') or ''
        label = next((c.get('text') for c in g if local(c.tag) == 'label' and c.get('text')), None)
        if cls == 'process':
            if gid:
                processes.add(gid)
                if gid.startswith('re_'):
                    processes.add(gid[3:])
                if gid.startswith('R_'):
                    processes.add(gid[2:])
        elif cls in ('simple chemical', 'macromolecule', 'complex', 'unspecified entity'):
            if gid:
                entities.add(gid)
                if gid.startswith('M_'):
                    entities.add(gid[2:])
                if gid.startswith('ep_'):
                    entities.add(gid[3:])
            if label:
                entities.add(label)
    return processes, entities


def layout_ids(path):
    import libsbml
    doc = libsbml.readSBMLFromFile(str(path))
    model = doc.getModel()
    plugin = model.getPlugin('layout')
    layout = plugin.getLayout(0) if plugin and plugin.getNumLayouts() else None
    rx, sp = set(), set()
    if layout is None:
        return rx, sp
    for i in range(layout.getNumReactionGlyphs()):
        rid = layout.getReactionGlyph(i).getReactionId()
        if rid.startswith('R_'):
            rid = rid[2:]
        rx.add(rid)
    for i in range(layout.getNumSpeciesGlyphs()):
        sid = layout.getSpeciesGlyph(i).getSpeciesId()
        if sid.startswith('M_'):
            sid = sid[2:]
        sp.add(sid)
    return rx, sp


def celldesigner_ids(path):
    """Model reaction and species ids from CellDesigner SBML (ids only, never names)."""
    root = ET.parse(path).getroot()

    def local(tag):
        return tag.rsplit('}', 1)[-1]

    rx, sp = set(), set()
    for e in root.iter():
        if local(e.tag) == 'reaction' and 'sbml' in e.tag:
            rid = e.get('id') or ''
            stripped = rid[2:] if rid.startswith('r_') else rid
            if stripped:
                rx.add(stripped)
        if local(e.tag) == 'species' and 'sbml' in e.tag:
            sid = e.get('id') or ''
            if sid.startswith('s_degraded_'):
                continue
            stripped = sid[2:] if sid.startswith('s_') else sid
            if stripped:
                sp.add(stripped)
    return rx, sp


def coordinates_match(data, sbgn_path, tol=1.5):
    root = ET.parse(sbgn_path).getroot()

    def local(tag):
        return tag.rsplit('}', 1)[-1]

    centres = []
    for g in root.iter():
        if local(g.tag) != 'glyph':
            continue
        cls = (g.get('class') or '').strip()
        if cls in ('clone marker', 'unit of information', 'state variable'):
            continue
        bbox = next((c for c in g if local(c.tag) == 'bbox'), None)
        if bbox is None:
            continue
        x = float(bbox.get('x')) + float(bbox.get('w')) / 2.0
        y = float(bbox.get('y')) + float(bbox.get('h')) / 2.0
        centres.append((x, y))
    body = map_exports._body(data)
    for n in body['nodes'].values():
        if n.get('node_type') not in ('metabolite', 'midmarker'):
            continue
        x, y = float(n['x']), float(n['y'])
        if not any(math.hypot(x - cx, y - cy) <= tol for cx, cy in centres):
            return False
    return True


def _compose_if_needed(name: str, gem: Path) -> Path:
    out = CACHE / name
    stem = output_names.sanitize_stem(gem.name)
    if _result(out, 'json').is_file() and _result(out, 'sbgn').is_file() and _result(out, 'sbml').is_file():
        # Refresh exports with the current writer (compose layout kept).
        map_exports.write_all_exports(out, model_path=gem)
        return out
    out.mkdir(parents=True, exist_ok=True)
    for p in list(out.iterdir()):
        if p.is_file():
            p.unlink()
        elif p.is_dir() and p.name in ('upload', 'full', 'views'):
            shutil.rmtree(p, ignore_errors=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / 'project.py'), 'compose',
         '--gem', str(gem), '--out', str(out), '--source-paths'],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f'compose failed for {name} (code {proc.returncode})\n'
            f'stdout tail:\n{(proc.stdout or "")[-2000:]}\n'
            f'stderr tail:\n{(proc.stderr or "")[-2000:]}'
        )
    assert _result(out, 'json').is_file(), f'missing result JSON in {out}'
    assert _result(out, 'json').name == output_names.result_filename(stem, 'json')
    if not _result(out, 'sbgn').is_file():
        map_exports.write_all_exports(out, model_path=gem)
    return out


@pytest.fixture(scope='module', params=MODELS, ids=[m[0] for m in MODELS])
def composed(request):
    name, gem = request.param
    assert gem.is_file(), gem
    folder = _compose_if_needed(name, gem)
    model = load_model(gem)
    data = read(_result(folder, 'json'))
    return name, gem, folder, model, data


def test_export_files_exist(composed):
    name, gem, folder, model, data = composed
    for ext in ('json', 'sbgn', 'sbml', 'celldesigner.xml'):
        path = _result(folder, ext)
        assert path.is_file(), f'{name}: missing {path.name}'
        assert path.name == output_names.result_filename(output_names.sanitize_stem(gem.name), ext)


def test_sbgn_xsd_and_layout_benchmark_match(composed):
    name, gem, folder, model, data = composed
    ok, errors = map_exports.validate_sbgn_xsd(_result(folder, 'sbgn'))
    assert ok, f'{name}: SBGN XSD errors: {errors[:10]}'
    pathway = read_sbml(gem)
    matched = match_glyphs(pathway, _result(folder, 'sbgn'))
    missing = sorted(set(pathway.graph) - set(matched))
    assert not missing, (
        f'{name}: layout-benchmark matched {len(matched)}/{pathway.graph.number_of_nodes()}; '
        f'missing {missing[:12]}'
    )
    assert read_sbgn_glyphs(_result(folder, 'sbgn'))
    assert read_sbgn_positions(_result(folder, 'sbgn'))


def test_sbml_layout_validates(composed):
    name, gem, folder, model, data = composed
    ok, errors = map_exports.validate_sbml_libsbml(_result(folder, 'sbml'))
    assert ok, f'{name}: SBML errors: {errors[:10]}'
    import libsbml
    original = libsbml.readSBMLFromFile(str(gem))
    exported = libsbml.readSBMLFromFile(str(_result(folder, 'sbml')))
    assert original.getModel().getNumSpecies() == exported.getModel().getNumSpecies()
    assert original.getModel().getNumReactions() == exported.getModel().getNumReactions()
    plugin = exported.getModel().getPlugin('layout')
    assert plugin is not None and plugin.getNumLayouts() >= 1
    n_curved = sum(
        1
        for rx in data[1]['reactions'].values()
        for seg in rx['segments'].values()
        if seg.get('b1') and seg.get('b2')
    )
    layout = plugin.getLayout(0)
    n_bez = 0
    for i in range(layout.getNumReactionGlyphs()):
        rg = layout.getReactionGlyph(i)
        for j in range(rg.getNumSpeciesReferenceGlyphs()):
            srg = rg.getSpeciesReferenceGlyph(j)
            curve = srg.getCurve()
            if curve is None:
                continue
            for k in range(curve.getNumCurveSegments()):
                seg = curve.getCurveSegment(k)
                if seg.getTypeCode() == libsbml.SBML_LAYOUT_CUBICBEZIER:
                    n_bez += 1
    assert n_bez == n_curved, f'{name}: CubicBezier {n_bez} != result JSON curved segments {n_curved}'


def test_celldesigner_validates_and_has_identity(composed):
    name, gem, folder, model, data = composed
    ok, errors = map_exports.validate_sbml_libsbml(_result(folder, 'celldesigner.xml'))
    assert ok, f'{name}: CellDesigner libSBML errors ({len(errors)}): {errors[:8]}'
    root = ET.parse(_result(folder, 'celldesigner.xml')).getroot()

    def local(tag):
        return tag.rsplit('}', 1)[-1]

    species = [e for e in root.iter() if local(e.tag) == 'species' and 'sbml' in e.tag]
    with_ident = 0
    for sp in species:
        sid = sp.get('id') or ''
        if sid.startswith('s_degraded_'):
            continue
        text = ET.tostring(sp, encoding='unicode')
        assert 'speciesIdentity' in text, f'{name}: {sid} missing speciesIdentity'
        assert '<ns' not in text or 'class' in text
        with_ident += 1
    assert with_ident == len(model['metabolites'])
    # No empty participant lists
    for e in root.iter():
        if local(e.tag) in ('listOfReactants', 'listOfProducts') and 'sbml' in e.tag:
            kids = [c for c in e if local(c.tag) == 'speciesReference']
            assert kids, f'{name}: empty {local(e.tag)}'


def test_every_reaction_and_species_in_exports(composed):
    name, gem, folder, model, data = composed
    model_rx = {r['id'] for r in model['reactions']}
    model_sp = {m['id'] for m in model['metabolites']}

    map_rx, map_sp = collect_export_ids(data, model)
    assert model_rx <= map_rx, f'{name}: result JSON missing reactions {sorted(model_rx - map_rx)[:10]}'
    assert map_sp

    sbgn_rx, sbgn_sp = sbgn_ids(_result(folder, 'sbgn'))
    assert model_rx <= sbgn_rx, f'{name}: SBGN missing reactions {sorted(model_rx - sbgn_rx)[:10]}'
    assert model_sp <= sbgn_sp, f'{name}: SBGN missing species {sorted(model_sp - sbgn_sp)[:10]}'

    lay_rx, lay_sp = layout_ids(_result(folder, 'sbml'))
    assert model_rx <= lay_rx, f'{name}: SBML Layout missing reactions {sorted(model_rx - lay_rx)[:10]}'
    assert model_sp <= lay_sp, f'{name}: SBML Layout missing species {sorted(model_sp - lay_sp)[:10]}'

    cd_rx, cd_sp = celldesigner_ids(_result(folder, 'celldesigner.xml'))
    assert model_rx <= cd_rx, f'{name}: CellDesigner missing reactions {sorted(model_rx - cd_rx)[:10]}'
    assert model_sp <= cd_sp, f'{name}: CellDesigner missing species {sorted(model_sp - cd_sp)[:10]}'
    # Ids only — display names must not satisfy coverage by themselves
    names = {m.get('name') for m in model['metabolites'] if m.get('name') and m['name'] != m['id']}
    assert cd_sp & model_sp == model_sp
    assert not names or (cd_sp - model_sp - {f'degraded_{r}' for r in model_rx}).isdisjoint(names)


def test_coordinates_match_map_json(composed):
    name, gem, folder, model, data = composed
    assert coordinates_match(data, _result(folder, 'sbgn'), tol=1.5), (
        f'{name}: SBGN glyph centres do not match result JSON'
    )


def test_exports_do_not_alter_map_json(composed):
    name, gem, folder, model, data = composed
    before = (_result(folder, 'json')).read_bytes()
    map_exports.write_all_exports(folder, model_path=gem, model=model)
    after = (_result(folder, 'json')).read_bytes()
    assert before == after, f'{name}: result JSON changed during export'
    # The assertion must be able to fail: a deliberate mutation differs.
    mutated = bytearray(after)
    mutated[-2] = (mutated[-2] + 1) % 256
    assert bytes(mutated) != before


def test_map_edit_regenerates_exports(composed):
    name, gem, folder, model, data = composed
    import map_edit
    import line_render
    with tempfile.TemporaryDirectory(prefix='omm_edit_export_') as tmp:
        tmp = Path(tmp)
        stem = output_names.sanitize_stem(gem.name)
        map_name = output_names.result_filename(stem, 'json')
        svg_name = output_names.result_filename(stem, 'svg')
        export_names = [output_names.result_filename(stem, ext)
                        for ext in ('sbgn', 'sbml', 'celldesigner.xml')]
        output_names.write_meta(tmp, gem.name, size=gem.stat().st_size)
        shutil.copy(_result(folder, 'json'), tmp / map_name)
        (tmp / 'upload').mkdir()
        shutil.copy(gem, tmp / 'upload' / 'model.xml')
        (tmp / 'job.json').write_text(json.dumps({
            'id': 'j_editexport', 'status': 'done',
            'summary': {
                'result_stem': stem,
                'result_basename': output_names.result_basename(stem),
                'layout_quality': {},
            },
            'files': [
                {'name': map_name, 'bytes': 10, 'kind': 'json'},
                {'name': svg_name, 'bytes': 10, 'kind': 'svg'},
                {'name': 'report.json', 'bytes': 10, 'kind': 'json'},
            ],
        }))
        (tmp / 'report.json').write_text(json.dumps({
            'layout_quality': {'orientation_mode': 'off', 'final_geometry': {}},
        }))
        line_render.render_file(tmp / map_name, tmp / svg_name)
        map_exports.write_all_exports(tmp, model_path=tmp / 'upload' / 'model.xml')
        before = {n: (tmp / n).read_bytes() for n in export_names}
        body = data[1]
        nid = next(k for k, n in body['nodes'].items() if n.get('node_type') == 'metabolite')
        node = body['nodes'][nid]
        edit = {'nodes': {nid: {'x': float(node['x']) + 25.0, 'y': float(node['y']) + 15.0}}}
        map_edit.save_map_edit(tmp, edit)
        after = {n: (tmp / n).read_bytes() for n in export_names}
        assert after[export_names[0]] != before[export_names[0]]
        assert after[export_names[1]] != before[export_names[1]]
        assert after[export_names[2]] != before[export_names[2]]
        map_edit.reset_map(tmp)
        reset_sbgn = (tmp / export_names[0]).read_bytes()
        assert reset_sbgn != after[export_names[0]]
        ok, errs = map_exports.validate_sbgn_xsd(tmp / export_names[0])
        assert ok, errs[:5]


def test_cobra_json_writes_all_formats():
    """COBRA/BiGG JSON uploads get result SBML via an SBML document built from the model."""
    # Minimal COBRA-like JSON
    model_json = {
        'id': 'tiny',
        'version': '1',
        'compartments': {'c': 'cytosol'},
        'metabolites': [
            {'id': 'A_c', 'name': 'A', 'compartment': 'c', 'charge': 0, 'formula': 'C'},
            {'id': 'B_c', 'name': 'B', 'compartment': 'c', 'charge': 0, 'formula': 'C'},
        ],
        'reactions': [
            {'id': 'R1', 'name': 'R1', 'metabolites': {'A_c': -1, 'B_c': 1},
             'lower_bound': -1000, 'upper_bound': 1000, 'gene_reaction_rule': ''},
        ],
        'genes': [],
    }
    data = [
        dict(schema='https://escher.github.io/escher/jsonschema/1-0-0#', map_name='tiny',
             map_id='tiny', homepage='', map_description='', metadata={}),
        dict(
            nodes={
                '0': dict(node_type='metabolite', x=0, y=0, bigg_id='A_c', name='A',
                          node_is_primary=True, label_x=0, label_y=-20),
                '1': dict(node_type='metabolite', x=100, y=0, bigg_id='B_c', name='B',
                          node_is_primary=True, label_x=100, label_y=-20),
                'm': dict(node_type='midmarker', x=50, y=0),
            },
            reactions={
                '0': dict(
                    bigg_id='R1', name='R1', reversibility=True, label_x=50, label_y=20,
                    gene_reaction_rule='', genes=[],
                    metabolites=[dict(bigg_id='A_c', coefficient=-1), dict(bigg_id='B_c', coefficient=1)],
                    segments={
                        '0': dict(from_node_id='m', to_node_id='0', b1=None, b2=None),
                        '1': dict(from_node_id='m', to_node_id='1',
                                  b1=dict(x=60, y=-10), b2=dict(x=80, y=-10)),
                    },
                ),
            },
            text_labels={},
            canvas=dict(x=0, y=0, width=200, height=100),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'upload').mkdir()
        gem = tmp / 'upload' / 'model.json'
        gem.write_text(json.dumps(model_json))
        output_names.write_meta(tmp, 'model.json', size=gem.stat().st_size)
        stem = 'model'
        map_name = output_names.result_filename(stem, 'json')
        (tmp / map_name).write_text(json.dumps(data))
        written = map_exports.write_all_exports(tmp, model_path=gem, stem=stem)
        expect = [output_names.result_filename(stem, ext)
                  for ext in ('sbgn', 'sbml', 'celldesigner.xml')]
        assert written == expect
        assert (tmp / expect[1]).stat().st_size > 0
        # Temporary SBML for JSON uploads is written under the job folder (and removed after).
        assert not (tmp / 'export_model.xml').exists()
        ok, errs = map_exports.validate_sbml_libsbml(tmp / expect[1])
        assert ok, errs[:5]
        ok2, errs2 = map_exports.validate_sbml_libsbml(tmp / expect[2])
        assert ok2, errs2[:5]
        ok3, errs3 = map_exports.validate_sbgn_xsd(tmp / expect[0])
        assert ok3, errs3[:5]


def test_synthetic_bezier_clones_and_degraded():
    model = {
        'id': 'mini', 'name': 'mini', 'genes': [], 'annotations': [], 'notes': '',
        'metabolites': [
            dict(id='A_c', name='A', compartment='c', annotations=[]),
            dict(id='B_c', name='B', compartment='c', annotations=[]),
            dict(id='h_c', name='H+', compartment='c', annotations=[]),
            dict(id='ac_e', name='Acetate', compartment='e', annotations=[]),
        ],
        'reactions': [
            dict(id='R1', name='R1', metabolites={'A_c': -1, 'B_c': 1, 'h_c': 1},
                 lower_bound=-1000, upper_bound=1000, gene_reaction_rule='', annotations=[], notes=''),
            dict(id='EX_ac_e', name='EX_ac_e', metabolites={'ac_e': -1},
                 lower_bound=0, upper_bound=1000, gene_reaction_rule='', annotations=[], notes=''),
        ],
    }
    data = [
        dict(schema='https://escher.github.io/escher/jsonschema/1-0-0#', map_name='mini',
             map_id='mini', homepage='', map_description='', metadata={}),
        dict(
            nodes={
                '0': dict(node_type='metabolite', x=0, y=100, bigg_id='A_c', name='A',
                          node_is_primary=True, label_x=0, label_y=80),
                '1': dict(node_type='metabolite', x=200, y=100, bigg_id='B_c', name='B',
                          node_is_primary=True, label_x=200, label_y=80),
                '2': dict(node_type='metabolite', x=200, y=40, bigg_id='h_c', name='H+',
                          node_is_primary=False, label_x=200, label_y=20),
                '3': dict(node_type='metabolite', x=50, y=40, bigg_id='h_c', name='H+',
                          node_is_primary=False, label_x=50, label_y=20),
                '4': dict(node_type='metabolite', x=300, y=100, bigg_id='ac_e', name='Acetate',
                          node_is_primary=True, label_x=300, label_y=80),
                'm': dict(node_type='midmarker', x=100, y=100),
                'm2': dict(node_type='midmarker', x=300, y=140),
            },
            reactions={
                '0': dict(
                    bigg_id='R1', name='R1', reversibility=True, label_x=100, label_y=120,
                    gene_reaction_rule='', genes=[],
                    metabolites=[
                        dict(bigg_id='A_c', coefficient=-1),
                        dict(bigg_id='B_c', coefficient=1),
                        dict(bigg_id='h_c', coefficient=1),
                    ],
                    segments={
                        '0': dict(from_node_id='m', to_node_id='0', b1=None, b2=None),
                        '1': dict(from_node_id='m', to_node_id='1',
                                  b1=dict(x=130, y=80), b2=dict(x=170, y=80)),
                        '2': dict(from_node_id='m', to_node_id='2', b1=None, b2=None),
                    },
                ),
                '1': dict(
                    bigg_id='EX_ac_e', name='EX_ac_e', reversibility=False, label_x=300, label_y=160,
                    gene_reaction_rule='', genes=[],
                    metabolites=[dict(bigg_id='ac_e', coefficient=-1)],
                    segments={'0': dict(from_node_id='m2', to_node_id='4', b1=None, b2=None)},
                ),
            },
            text_labels={},
            canvas=dict(x=0, y=0, width=400, height=200),
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        map_exports.write_sbgn(tmp / 'map.sbgn', data, model)
        ok, errors = map_exports.validate_sbgn_xsd(tmp / 'map.sbgn')
        assert ok, errors
        root = ET.parse(tmp / 'map.sbgn').getroot()
        clones = root.findall('.//{http://sbgn.org/libsbgn/0.3}clone')
        assert len(clones) >= 2
        points = root.findall('.//{http://sbgn.org/libsbgn/0.3}point')
        assert len(points) >= 2, 'expected Bezier <point> control points on curved arcs'
        # Each control pair sits under next/end, not as sampled polyline next-points only
        controlled = [
            el for el in root.iter()
            if el.tag.endswith(('next', 'end')) and any(c.tag.endswith('point') for c in el)
        ]
        assert controlled, 'curved segments must carry <point> children on next/end'
        # Labels are metabolite names
        labels = [e.get('text') for e in root.iter() if e.tag.endswith('label') and e.get('text')]
        assert 'A' in labels and 'H+' in labels
        map_exports.write_celldesigner(tmp / 'map.celldesigner.xml', data, model)
        ok2, errs2 = map_exports.validate_sbml_libsbml(tmp / 'map.celldesigner.xml')
        assert ok2, errs2[:8]
        text = (tmp / 'map.celldesigner.xml').read_text()
        assert 'DEGRADED' in text
        assert 'speciesIdentity' in text
        assert '<listOfProducts/>' not in text.replace(' ', '')
