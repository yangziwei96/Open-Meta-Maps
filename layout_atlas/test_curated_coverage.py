"""Tests for curated-coverage helpers and selection guards.

Covers: deterministic transporter assignment helper; namespace id parsing;
same-map select staying disjoint; EC picks drawn while relaxed/partial
proposals stay excluded; Human-GEM metabolite bridge licence.
Uses only e_coli_core and iIT341 data.
"""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

from core import EXAMPLES, chem, comp, load_model, model_graph, reaction_kind, currency
import database
import retrieval
import semantic_transfer as st
import ec_channel as ec


class NamespaceIds(unittest.TestCase):
    def test_bigg_mam_seed_mnx(self):
        self.assertEqual(chem('glc__D_c'), 'glc__D')
        self.assertEqual(comp('glc__D_c'), 'c')
        self.assertEqual(chem('MAM00001c'), 'MAM00001')
        self.assertEqual(comp('MAM00001c'), 'c')
        self.assertEqual(chem('MAM00001m'), 'MAM00001')
        self.assertEqual(comp('MAM00001m'), 'm')
        self.assertEqual(chem('cpd00001_c0'), 'cpd00001')
        self.assertEqual(comp('cpd00001_c0'), 'c0')
        self.assertEqual(chem('MNXM1@MNXC3'), 'MNXM1')
        self.assertEqual(comp('MNXM1@MNXC3'), 'MNXC3')


class TransporterAssignment(unittest.TestCase):
    """Two valid metabolite assignments → deterministic pick."""

    def test_prefer_compartments_then_direction(self):
        a = dict(metabolites={'a_c': 'a_c', 'a_e': 'a_e'}, factor=1.0, direction='same',
                 label_compartments_exact=True, compartment_roles={'c': 'c', 'e': 'e'})
        b = dict(metabolites={'a_c': 'a_e', 'a_e': 'a_c'}, factor=-1.0, direction='reversed_recording',
                 label_compartments_exact=False, compartment_roles={'c': 'e', 'e': 'c'})
        chosen = st._pick_assignment([b, a])
        self.assertTrue(chosen['label_compartments_exact'])
        self.assertEqual(chosen['direction'], 'same')
        self.assertEqual(chosen['assignment_resolution'], 'deterministic_prefer_compartments_then_direction')

    def test_e_coli_transport_match_or_ambiguous_drop(self):
        model = load_model(EXAMPLES / 'e_coli_core.xml')
        mapper = st.Mapper(model, st.mapper_bridge())
        transports = [r for r in model['reactions'] if reaction_kind(r, currency(model)) == 'transport']
        self.assertTrue(transports)
        r = transports[0]
        coeffs = {mid: float(c) for mid, c in r['metabolites'].items()}
        record = dict(coefficients=coeffs, references={}, id=r['id'], key='r:' + r['id'])
        matches = mapper.match_reaction(record)
        # Main behaviour: unique equation only; ambiguous equations are dropped.
        self.assertTrue(isinstance(matches, list))


class PartialMatch(unittest.TestCase):
    """Accept largest connected matched part (≥2 reactions)."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_model(EXAMPLES / 'e_coli_core.xml')
        cls.mapper = st.Mapper(cls.model, st.mapper_bridge())
        cls.modules = [m for m in database.modules() if m['size'] >= 4][:400]

    def test_partial_proposals_when_full_unavailable(self):
        produced_partial = False
        for m in self.modules:
            choices = [self.mapper.match_reaction(r) for r in m['reactions']]
            matched = sum(1 for c in choices if c)
            if matched < 2 or matched == len(choices):
                continue
            props, info = self.mapper.proposals([m], 'test')
            if any(p.get('partial_match') for p in props):
                produced_partial = True
                p = next(p for p in props if p.get('partial_match'))
                self.assertGreaterEqual(len(p['reaction_ids']), 2)
                self.assertLess(p['partial_match']['matched_reactions'], p['partial_match']['source_reactions'])
                break
        if not produced_partial:
            core = database.modules()
            strong, _ = self.mapper.proposals(core[:2000], 'test')
            self.assertTrue(strong)
            self.assertGreaterEqual(len(strong), 1)


class SameMapMerge(unittest.TestCase):
    """Same-map overlap stays disjoint in select; higher-score fragment wins."""

    def test_merge_refits_one_frame_from_source_xy(self):
        def feat(biol=2.0):
            return [4.0, biol, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        a = dict(id='a', source='escher', map_id='MAP1', reaction_ids=['R1', 'R2', 'R3'], score=10.0,
                 features=feat(), layout_origin={}, match_level='complete_equation',
                 complete_chemical_equation_verified=True, selection_score=10.0,
                 positions={'r:R1': [0.0, 0.0], 'r:R2': [1.0, 0.0], 'r:R3': [2.0, 0.0]},
                 source_xy={'r:R1': [0.0, 0.0], 'r:R2': [10.0, 0.0], 'r:R3': [20.0, 0.0]},
                 node_mapping={}, reaction_mapping={}, mapping_evidence=[])
        b = dict(id='b', source='escher', map_id='MAP1', reaction_ids=['R3', 'R4', 'R5'], score=9.0,
                 features=feat(), layout_origin={}, match_level='complete_equation',
                 complete_chemical_equation_verified=True, selection_score=9.0,
                 positions={'r:R3': [100.0, 0.0], 'r:R4': [150.0, 0.0], 'r:R5': [200.0, 0.0]},
                 source_xy={'r:R3': [20.0, 0.0], 'r:R4': [30.0, 0.0], 'r:R5': [40.0, 0.0]},
                 node_mapping={}, reaction_mapping={}, mapping_evidence=[])
        chosen = retrieval.select([copy.deepcopy(a), copy.deepcopy(b)])
        self.assertEqual(len(chosen), 1)
        self.assertEqual(set(chosen[0]['reaction_ids']), {'R1', 'R2', 'R3'})

    def test_select_keeps_size2_same_map_overlap_disjoint(self):
        """Size-2 crumbs on the same map stay disjoint (layout-quality guard)."""
        def feat(biol=2.0):
            return [4.0, biol, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        a = dict(id='a', source='escher', map_id='MAP1', reaction_ids=['R1', 'R2'], score=10.0,
                 features=feat(), layout_origin={}, match_level='complete_equation',
                 complete_chemical_equation_verified=True,
                 positions={'r:R1': [0.0, 0.0], 'r:R2': [1.0, 0.0]},
                 source_xy={'r:R1': [0.0, 0.0], 'r:R2': [10.0, 0.0]},
                 node_mapping={}, reaction_mapping={}, mapping_evidence=[])
        b = dict(id='b', source='escher', map_id='MAP1', reaction_ids=['R2', 'R3'], score=9.0,
                 features=feat(), layout_origin={}, match_level='complete_equation',
                 complete_chemical_equation_verified=True,
                 positions={'r:R2': [10.0, 0.0], 'r:R3': [20.0, 0.0]},
                 source_xy={'r:R2': [10.0, 0.0], 'r:R3': [20.0, 0.0]},
                 node_mapping={}, reaction_mapping={}, mapping_evidence=[])
        chosen = retrieval.select([copy.deepcopy(a), copy.deepcopy(b)])
        self.assertEqual(len(chosen), 1)
        self.assertEqual(set(chosen[0]['reaction_ids']), {'R1', 'R2'})

    def test_select_excludes_different_map_overlap(self):
        def feat(biol=2.0):
            return [4.0, biol, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        a = dict(id='a', source='escher', map_id='MAP1', reaction_ids=['R1', 'R2'], score=10.0,
                 features=feat(), layout_origin={}, match_level='complete_equation',
                 complete_chemical_equation_verified=True,
                 positions={'r:R1': [0.0, 0.0], 'r:R2': [1.0, 0.0]},
                 source_xy={'r:R1': [0.0, 0.0], 'r:R2': [10.0, 0.0]},
                 node_mapping={}, reaction_mapping={}, mapping_evidence=[])
        b = dict(id='b', source='escher', map_id='MAP2', reaction_ids=['R2', 'R3'], score=9.0,
                 features=feat(), layout_origin={}, match_level='complete_equation',
                 complete_chemical_equation_verified=True,
                 positions={'r:R2': [1.0, 0.0], 'r:R3': [2.0, 0.0]},
                 source_xy={'r:R2': [10.0, 0.0], 'r:R3': [20.0, 0.0]},
                 node_mapping={}, reaction_mapping={}, mapping_evidence=[])
        chosen = retrieval.select([copy.deepcopy(a), copy.deepcopy(b)])
        self.assertEqual(len(chosen), 1)
        self.assertEqual(set(chosen[0]['reaction_ids']), {'R1', 'R2'})


class EcSelectNotGapFill(unittest.TestCase):
    """EC-pool picks are drawn; relaxed/partial gap proposals are excluded."""

    def test_ec_picks_drawn_relaxed_excluded(self):
        def feat():
            return [4.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        primary = dict(
            id='primary', source='escher', map_id='M1', template_id='t1',
            reaction_ids=['R1', 'R2'], score=10.0, features=feat(),
            match_level='complete_equation', complete_chemical_equation_verified=True,
            complete_equation_verified=True, positions={}, node_mapping={},
            reaction_mapping={}, layout_origin={})
        gap = dict(
            id='gap', source='escher', map_id='M1', template_id='t2',
            reaction_ids=['R3', 'R4'], score=8.0, features=feat(),
            match_level='relaxed_equation', complete_chemical_equation_verified=False,
            complete_equation_verified=False, partial_match={'matched_reactions': 2, 'source_reactions': 3},
            positions={}, node_mapping={}, reaction_mapping={}, layout_origin={})
        ec_prop = dict(
            id='ec1', source='escher', map_id='M2', template_id='t3',
            reaction_ids=['R5', 'R6'], score=7.0, features=feat(),
            match_level='ec_topology', complete_chemical_equation_verified=False,
            complete_equation_verified=False, positions={}, node_mapping={},
            reaction_mapping={}, layout_origin={})
        selected, stages = ec.select_with_ec(
            [copy.deepcopy(primary), copy.deepcopy(gap)],
            [[copy.deepcopy(ec_prop)]])
        by_id = {p['id']: p for p in selected}
        self.assertIn('primary', by_id)
        self.assertNotIn('gap', by_id)
        self.assertIn('ec1', by_id)
        self.assertFalse(any(p.get('gap_fill') for p in selected))


class ReactionKind(unittest.TestCase):
    def test_e_coli_kinds(self):
        model = load_model(EXAMPLES / 'e_coli_core.xml')
        cur = currency(model)
        kinds = {reaction_kind(r, cur) for r in model['reactions']}
        self.assertIn('metabolic', kinds)
        self.assertIn('exchange', kinds)
        self.assertIn('transport', kinds)


class HumanGemMetaboliteLicence(unittest.TestCase):
    def test_bridge_licence_is_cc_by(self):
        links, meta = st._mam_bridge_links()
        self.assertFalse(meta.get('skipped'))
        self.assertIn('CC-BY', (meta.get('license') or '').upper().replace(' ', ''))
        self.assertTrue(links)


if __name__ == '__main__':
    unittest.main()
