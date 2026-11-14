"""Curated coverage by reaction kind (report.covered.by_kind)."""
from __future__ import annotations

import unittest
from collections import Counter

from core import EXAMPLES, load_model, model_graph, reaction_kind
import curated_axes
import database
import ec_channel as ec
import retrieval
import semantic_transfer as st


def curated_reaction_ids(model):
    """Reaction ids covered by the default curated selection (no EC extension)."""
    g = model_graph(model)
    core = database.modules()
    mapper = st.Mapper(model, st.mapper_bridge())
    library = retrieval.Library(core)
    ps, _ = library.candidates(g, model=model, max_roots=120)
    strong, _ = mapper.proposals(core, 'open_core')
    selected, _ = ec.select_with_ec(ps + strong, [])
    selected, _ = curated_axes.reapply_same_map_layout(selected)
    return {r for p in selected for r in p['reaction_ids']}


class ReactionKind(unittest.TestCase):
    def test_e_coli_by_kind_totals_and_covered(self):
        model = load_model(EXAMPLES / 'e_coli_core.xml')
        kind_of = {r['id']: reaction_kind(r) for r in model['reactions']}
        totals = Counter(kind_of.values())
        self.assertEqual(totals['metabolic'], 50)
        self.assertEqual(totals['transport'], 25)
        self.assertEqual(totals['exchange'], 20)
        self.assertEqual(sum(totals.values()), 95)

        used = curated_reaction_ids(model)
        curated = Counter(kind_of[rid] for rid in used)
        self.assertEqual(curated['metabolic'], 45)
        self.assertEqual(curated['transport'], 15)
        self.assertEqual(curated['exchange'], 10)


if __name__ == '__main__':
    unittest.main()
