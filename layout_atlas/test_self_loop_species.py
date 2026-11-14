"""Self-loop / dual-role species: A -> A and A + B -> A + C must compose and draw."""
from __future__ import annotations

import json
from pathlib import Path

import compose_cli
import line_render
import model_io
import output_names
import semantic_transfer as st
from core import model_graph, participation_roles
from tool_pipeline import Failure, normalize_model


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests' / 'sbml_self_loop_species.xml'


class _Quiet:
    def warn(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


def test_load_keeps_gross_participation():
    model = model_io.load_model(FIXTURE)
    by_id = {r['id']: r for r in model['reactions']}
    assert by_id['r_self']['metabolites'] == {'A': 0.0}
    assert by_id['r_mixed']['metabolites'] == {'A': 0.0, 'B': -1.0, 'C': 1.0}
    assert list(by_id['r_mixed']['metabolites']) == ['A', 'B', 'C']
    assert participation_roles(0.0) == (-1.0, 1.0)
    assert participation_roles(-2.0) == (-2.0,)
    assert participation_roles(3.0) == (3.0,)
    normalize_model(model, _Quiet())
    g = model_graph(model)
    assert g.edges['r:r_self', 'm:A']['role'] == 'both'


def test_normalize_rejects_empty_not_dual_zero():
    model = dict(
        id='empty_rx',
        metabolites=[dict(id='A', compartment='c')],
        reactions=[dict(id='r_empty', metabolites={}, lower_bound=0.0, upper_bound=1000.0)],
    )
    try:
        normalize_model(model, _Quiet())
    except Failure as err:
        assert 'no stoichiometry' in str(err)
    else:
        raise AssertionError('expected Failure for a reaction with no participants')


def test_unique_equation_mapping_skips_factor_zero():
    """A one-participant curated reaction must not match target A -> A via factor 0."""
    curated = {'A_c': -1.0}
    target = dict(metabolites={'A_c': 0.0})
    options = {'A_c': ['A_c']}
    compartments = {'A_c': 'c'}
    assert st.unique_equation_mapping(curated, target, options, compartments) == []


def test_self_loop_compose_draws_every_reaction_and_species(tmp_path):
    out = tmp_path / 'self_loop'
    assert compose_cli.run(FIXTURE, out, no_library=True, source_paths=True) == 0
    data = json.loads(output_names.map_json(out).read_text(encoding='utf-8'))
    body = data[1]
    drawn_rx = {r['bigg_id'] for r in body['reactions'].values()}
    assert drawn_rx == {'r_self', 'r_mixed'}
    drawn_mets = {n['bigg_id'] for n in body['nodes'].values() if n.get('node_type') == 'metabolite'}
    assert drawn_mets == {'A', 'B', 'C'}
    midmarkers = [n for n in body['nodes'].values() if n.get('node_type') == 'midmarker']
    assert len(midmarkers) == 2
    by_id = {r['bigg_id']: r for r in body['reactions'].values()}
    # Net stoichiometry kept for identity; dual participation draws two segments to A.
    assert by_id['r_self']['metabolites'] == [dict(bigg_id='A', coefficient=0.0)]
    assert len(by_id['r_self']['segments']) == 2
    assert {m['bigg_id']: m['coefficient'] for m in by_id['r_mixed']['metabolites']} == {
        'A': 0.0, 'B': -1.0, 'C': 1.0,
    }
    assert len(by_id['r_mixed']['segments']) == 4  # A×2 + B + C
    svg = line_render.render(data)
    self_paths = [line for line in svg.split('<path ') if 'data-rx="r_self"' in line and 'data-met="A"' in line]
    assert len(self_paths) == 2
    assert sum('marker-end=' in line for line in self_paths) == 1
