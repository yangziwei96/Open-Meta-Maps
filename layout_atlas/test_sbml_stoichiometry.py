"""SBML stoichiometry reading: missing attributes, stoichiometryMath, assignment rules."""
from __future__ import annotations

from pathlib import Path

import compose_cli
import model_io
import output_names

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests'


def test_l2_stoichiometry_math_parse_and_compose(tmp_path):
    path = FIXTURES / 'sbml_l2_stoichiometry_math.xml'
    model = model_io.load_model(path)
    assert model['reactions'][0]['metabolites'] == {'A': -2.0, 'B': 1.0}
    assert 'E' not in model['reactions'][0]['metabolites']
    assert compose_cli.run(path, tmp_path / 'l2', no_library=True, source_paths=True) == 0
    assert output_names.map_json(tmp_path / 'l2').is_file()


def test_l3_missing_stoichiometry_parse_and_compose(tmp_path):
    path = FIXTURES / 'sbml_l3_missing_stoichiometry.xml'
    model = model_io.load_model(path)
    assert model['reactions'][0]['metabolites'] == {'A': -1.0, 'B': 1.0}
    assert compose_cli.run(path, tmp_path / 'l3_missing', no_library=True, source_paths=True) == 0
    assert output_names.map_json(tmp_path / 'l3_missing').is_file()


def test_l3_assignment_rule_stoichiometry_parse_and_compose(tmp_path):
    path = FIXTURES / 'sbml_l3_assignment_rule_stoichiometry.xml'
    model = model_io.load_model(path)
    assert model['reactions'][0]['metabolites'] == {'A': -2.0, 'B': 1.0}
    assert 'E' not in model['reactions'][0]['metabolites']
    assert compose_cli.run(path, tmp_path / 'l3_rule', no_library=True, source_paths=True) == 0
    assert output_names.map_json(tmp_path / 'l3_rule').is_file()


def test_unevaluable_variable_stoichiometry_defaults_to_one(tmp_path):
    path = tmp_path / 'l3_unevaluable.xml'
    path.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="l3_unevaluable">
    <listOfCompartments><compartment id="c" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="r1" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference id="srA" species="A" stoichiometry="9" constant="false"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference id="srB" species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
      </reaction>
    </listOfReactions>
    <listOfRules>
      <assignmentRule variable="srA">
        <math xmlns="http://www.w3.org/1998/MathML"><ci>A</ci></math>
      </assignmentRule>
    </listOfRules>
  </model>
</sbml>
''')
    model = model_io.load_model(path)
    assert model['reactions'][0]['metabolites'] == {'A': -1.0, 'B': 1.0}
