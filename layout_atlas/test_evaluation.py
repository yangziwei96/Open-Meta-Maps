"""Smoke tests for the unified layout evaluation path."""
from evaluation.pathway import PRODUCT, REACTANT, REACTION, SPECIES, Pathway
from evaluation import metrics
import networkx as nx


def _toy():
    g = nx.DiGraph()
    g.add_node("A", kind=SPECIES, label="A", compartment="c", width=60, height=24)
    g.add_node("B", kind=SPECIES, label="B", compartment="c", width=60, height=24)
    g.add_node("C", kind=SPECIES, label="C", compartment="e", width=60, height=24)
    g.add_node("r1", kind=REACTION, label="r1", compartment=None, width=10, height=10)
    g.add_edge("A", "r1", role=REACTANT)
    g.add_edge("r1", "B", role=PRODUCT)
    g.add_edge("r1", "C", role=PRODUCT)
    return Pathway(name="toy", graph=g)


def test_participant_consistency_opposite():
    p = _toy()
    pos = {"r1": (0.0, 0.0), "A": (-10.0, 0.0), "B": (10.0, 1.0), "C": (10.0, -1.0)}
    assert metrics.participant_consistency(p, pos) == 1.0


def test_participant_consistency_same_side():
    p = _toy()
    pos = {"r1": (0.0, 0.0), "A": (-10.0, 0.0), "B": (-10.0, 2.0), "C": (-8.0, -2.0)}
    assert metrics.participant_consistency(p, pos) == 0.0


def test_long_edge_share():
    p = _toy()
    pos = {"r1": (0.0, 0.0), "A": (-1.0, 0.0), "B": (1.0, 0.0), "C": (100.0, 0.0)}
    m = metrics.long_edge_metrics(p, pos, long_factor=3.0)
    assert m["long_edge_share"] is not None
    assert 0.0 < m["long_edge_share"] <= 1.0


def test_overlaps_no_touch():
    p = _toy()
    pos = {"r1": (0.0, 0.0), "A": (-100.0, 0.0), "B": (100.0, 0.0), "C": (0.0, 100.0)}
    m = metrics.overlaps(p, pos)
    assert m["overlapping_pairs"] == 0.0


def test_score_map_official_path():
    from evaluation import score_map
    import official_metrics as om

    data = [
        {"metadata": {}},
        {
            "nodes": {
                "m1": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
                "a": {"node_type": "metabolite", "x": -40.0, "y": 0.0, "bigg_id": "A_c", "node_is_primary": True},
                "b": {"node_type": "metabolite", "x": 40.0, "y": 0.0, "bigg_id": "B_c", "node_is_primary": True},
            },
            "reactions": {
                "1": {
                    "bigg_id": "R1",
                    "metabolites": [
                        {"bigg_id": "A_c", "coefficient": -1},
                        {"bigg_id": "B_c", "coefficient": 1},
                    ],
                    "segments": {
                        "0": {"from_node_id": "m1", "to_node_id": "a"},
                        "1": {"from_node_id": "m1", "to_node_id": "b"},
                    },
                }
            },
            "canvas": {"width": 100, "height": 40},
        },
    ]
    out = score_map.score_map("examples/e_coli_core.xml", escher_map=data, use_layout_benchmark=False)
    assert out["official"]["participant_consistency"] == 1.0
    assert out["official"]["crossings"] == om.score_escher_map(data)["crossings"]


def test_blank_space_empty_canvas_mostly_blank():
    # Four corners occupied; most of the middle is blank after clear radius.
    pts = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]
    m = metrics.blank_space(pts, segments=[], label_boxes=[], spacing=20.0)
    assert m["blank_space_share"] is not None
    assert m["blank_space_share"] > 0.2
    assert m["largest_empty_rect_share"] is not None
    assert m["largest_empty_rect_share"] > 0.0


def test_blank_space_dense_ink_low_blank():
    # Dense grid of points with small spacing → little blank space.
    pts = [(x, y) for x in range(0, 50, 5) for y in range(0, 50, 5)]
    m = metrics.blank_space(pts, spacing=10.0, cell_factor=0.6, clear_spacings=1.0)
    assert m["blank_space_share"] is not None
    assert m["blank_space_share"] < 0.5


def test_unrelated_overlaps_node_hit():
    # Two nodes from unrelated reactions almost on top of each other.
    node_xy = {"a": (0.0, 0.0), "b": (5.0, 0.0)}
    node_reaction = {"a": "R1", "b": "R2"}
    segs = []
    related = set()
    m = metrics.unrelated_overlaps(node_xy, node_reaction, segs, related, node_radius=7.0)
    assert m["unrelated_node_overlaps"] == 1.0


def test_structure_continuity_adjacent():
    from evaluation.continuity import structure_continuity

    # Minimal Escher-like map: two reactions sharing a non-currency metabolite, close together.
    data = [
        {"map_name": "toy", "map_id": "toy", "schema": "1.0.0", "metadata": {
            "reaction_provenance": {"r1": "expert_derived", "r2": "expert_derived"},
            "expert_structures": [{"id": "s1", "code": "U0001", "reactions": ["r1", "r2"]}],
        }},
        {"nodes": {
            "1": {"node_type": "midmarker", "x": 0, "y": 0},
            "2": {"node_type": "midmarker", "x": 50, "y": 0},
            "3": {"node_type": "metabolite", "x": 25, "y": 0, "bigg_id": "g6p_c", "node_is_primary": True},
        }, "reactions": {
            "a": {"bigg_id": "r1", "metabolites": [{"bigg_id": "g6p_c", "coefficient": -1}],
                  "segments": {"0": {"from_node_id": "1", "to_node_id": "3"}}},
            "b": {"bigg_id": "r2", "metabolites": [{"bigg_id": "g6p_c", "coefficient": 1}],
                  "segments": {"0": {"from_node_id": "2", "to_node_id": "3"}}},
        }, "canvas": {"x": 0, "y": 0, "width": 100, "height": 100}},
    ]
    c = structure_continuity(data)
    assert c["structures"] == 1
    assert c["pairs"] == 1
    assert c["contiguous_pair_share"] == 1.0
    assert c["split_structures"] == 0


if __name__ == "__main__":
    test_participant_consistency_opposite()
    test_participant_consistency_same_side()
    test_long_edge_share()
    test_overlaps_no_touch()
    test_blank_space_empty_canvas_mostly_blank()
    test_blank_space_dense_ink_low_blank()
    test_unrelated_overlaps_node_hit()
    test_structure_continuity_adjacent()
    print("ok")
