"""Fixture tests for official layout-benchmark metric definitions."""
import json
from pathlib import Path

import numpy as np
import official_metrics as om


FIX = Path(__file__).resolve().parent / "testdata" / "official_metrics"


def _toy_cross():
    """Two crossing chords → 1 official crossing; four non-overlapping glyphs."""
    nodes = {
        "m1": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
        "a": {"node_type": "metabolite", "x": 100.0, "y": 100.0, "bigg_id": "A_c", "node_is_primary": True,
              "label_x": 110.0, "label_y": 100.0},
        "m2": {"node_type": "midmarker", "x": 0.0, "y": 100.0},
        "b": {"node_type": "metabolite", "x": 100.0, "y": 0.0, "bigg_id": "B_c", "node_is_primary": True,
              "label_x": 110.0, "label_y": 0.0},
    }
    reactions = {
        "1": {
            "bigg_id": "R1", "label_x": 0, "label_y": -10,
            "metabolites": [{"bigg_id": "A_c", "coefficient": 1}],
            "segments": {"0": {"from_node_id": "m1", "to_node_id": "a"}},
        },
        "2": {
            "bigg_id": "R2", "label_x": 0, "label_y": 110,
            "metabolites": [{"bigg_id": "B_c", "coefficient": 1}],
            "segments": {"0": {"from_node_id": "m2", "to_node_id": "b"}},
        },
    }
    return [{"map_name": "toy", "metadata": {}}, {"nodes": nodes, "reactions": reactions,
                                                  "canvas": {"x": 0, "y": 0, "width": 200, "height": 200}}]


def _toy_overlap():
    """Two primary circles whose centres are 30 px apart → overlap (r+r=40)."""
    nodes = {
        "m1": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
        "a": {"node_type": "metabolite", "x": 50.0, "y": 0.0, "bigg_id": "A_c", "node_is_primary": True},
        "b": {"node_type": "metabolite", "x": 70.0, "y": 25.0, "bigg_id": "B_c", "node_is_primary": True},
    }
    reactions = {
        "1": {
            "bigg_id": "R1", "label_x": 0, "label_y": -10,
            "metabolites": [
                {"bigg_id": "A_c", "coefficient": -1},
                {"bigg_id": "B_c", "coefficient": 1},
            ],
            "segments": {
                "0": {"from_node_id": "m1", "to_node_id": "a"},
                "1": {"from_node_id": "m1", "to_node_id": "b"},
            },
        }
    }
    return [{"map_name": "ov", "metadata": {}}, {"nodes": nodes, "reactions": reactions,
                                                "canvas": {"x": 0, "y": 0, "width": 200, "height": 100}}]


def _toy_participant():
    """Reactant left, product right of mid → consistency 1.0 at 90°."""
    nodes = {
        "m1": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
        "a": {"node_type": "metabolite", "x": -40.0, "y": 0.0, "bigg_id": "A_c", "node_is_primary": True},
        "b": {"node_type": "metabolite", "x": 40.0, "y": 0.0, "bigg_id": "B_c", "node_is_primary": True},
    }
    reactions = {
        "1": {
            "bigg_id": "R1", "label_x": 0, "label_y": -10,
            "metabolites": [
                {"bigg_id": "A_c", "coefficient": -1},
                {"bigg_id": "B_c", "coefficient": 1},
            ],
            "segments": {
                "0": {"from_node_id": "m1", "to_node_id": "a"},
                "1": {"from_node_id": "m1", "to_node_id": "b"},
            },
        }
    }
    return [{"map_name": "pc", "metadata": {}}, {"nodes": nodes, "reactions": reactions,
                                                "canvas": {"x": 0, "y": 0, "width": 100, "height": 40}}]


def test_crossings_fixture():
    data = _toy_cross()
    s = om.score_escher_map(data)
    assert s["crossings"] == 1
    assert s["pieces"] == 2


def test_overlap_fixture():
    data = _toy_overlap()
    ov = om.glyph_overlaps(data, include_labels=False)
    assert ov["overlapping_pairs"] == 1
    # Far apart → no overlap
    data[1]["nodes"]["b"]["x"] = 200.0
    ov2 = om.glyph_overlaps(data, include_labels=False)
    assert ov2["overlapping_pairs"] == 0


def test_participant_fixture():
    data = _toy_participant()
    assert om.participant_consistency(data) == 1.0
    # Same side
    data[1]["nodes"]["b"]["x"] = -40.0
    data[1]["nodes"]["b"]["y"] = 20.0
    assert om.participant_consistency(data) == 0.0


def test_edge_metrics_orthogonal():
    data = _toy_participant()
    edges = om.escher_drawn_edges(data)
    segs = om.piece_segments(edges)
    assert abs(om.edge_orthogonality(segs) - 1.0) < 1e-9
    assert om.edge_length_deviation(edges) == 0.0


def test_aspect_band():
    data = _toy_cross()
    # content bbox ~100×100 → aspect 1.0, outside 4:3–16:9
    a = om.content_aspect(data)
    assert a["aspect_in_band"] is False
    data[1]["nodes"]["a"]["x"] = 160.0
    data[1]["nodes"]["b"]["x"] = 160.0
    a2 = om.content_aspect(data)
    # width 160, height 100 → 1.6 in band
    assert a2["aspect_in_band"] is True


def test_compartment_without_shapely():
    nodes = {
        "a": {"node_type": "metabolite", "x": 0.0, "y": 0.0, "bigg_id": "A_c"},
        "b": {"node_type": "metabolite", "x": 10.0, "y": 0.0, "bigg_id": "B_c"},
        "c": {"node_type": "metabolite", "x": 5.0, "y": 10.0, "bigg_id": "C_c"},
        "d": {"node_type": "metabolite", "x": 5.0, "y": 3.0, "bigg_id": "D_e"},  # inside c hull
        "e": {"node_type": "metabolite", "x": 100.0, "y": 0.0, "bigg_id": "E_e"},
        "f": {"node_type": "metabolite", "x": 110.0, "y": 0.0, "bigg_id": "F_e"},
        "g": {"node_type": "metabolite", "x": 105.0, "y": 10.0, "bigg_id": "G_e"},
    }
    data = [{}, {"nodes": nodes, "reactions": {}, "canvas": {"width": 120, "height": 20}}]
    v = om.species_in_other_compartment(data)
    assert v is not None
    assert v > 0  # D_e sits inside the c compartment triangle


def test_fixture_file_roundtrip(tmp_path=None):
    FIX.mkdir(parents=True, exist_ok=True)
    cases = {
        "cross.json": (_toy_cross(), {"crossings": 1, "overlapping_pairs": 0}),
        "overlap.json": (_toy_overlap(), {"crossings": 0, "overlapping_pairs": 1}),
        "participant.json": (_toy_participant(), {"participant_consistency": 1.0}),
    }
    for name, (data, expect) in cases.items():
        path = FIX / name
        path.write_text(json.dumps(data))
        loaded = json.loads(path.read_text())
        got = om.score_escher_map(loaded)
        for k, v in expect.items():
            if isinstance(v, float):
                assert abs(got[k] - v) < 1e-9, (name, k, got[k], v)
            else:
                assert got[k] == v, (name, k, got[k], v)


def test_layout_quality_uses_official():
    import layout_quality as lq

    data = _toy_cross()
    summary, _ = lq.evaluate(data, max_issues=10)
    assert summary["core"]["crossings"]["counted"] == 1
    assert summary["core"]["crossings"]["official"] == 1
    assert summary["core"]["participant_placement"]["threshold_deg"] == 90.0
    assert "edge_length_deviation" in summary["details"]
    assert "edge_orthogonality" in summary["details"]
    rows = {r["code"]: r for g in summary["details"]["panel"]["groups"] for r in g["rows"]}
    assert rows["aspect_ratio"]["target"] == "4:3–16:9"
    assert rows["edge_length_deviation"]["code"] == "edge_length_deviation"
    assert rows["edge_orthogonality"]["code"] == "edge_orthogonality"


if __name__ == "__main__":
    test_crossings_fixture()
    test_overlap_fixture()
    test_participant_fixture()
    test_edge_metrics_orthogonal()
    test_aspect_band()
    test_compartment_without_shapely()
    test_fixture_file_roundtrip()
    test_layout_quality_uses_official()
    print("ok")

def test_previous_pins_not_moved_by_official_clear():
    import curated_axes as ca
    data = [
        {"canvas": {"width": 200, "height": 200}, "metadata": {}},
        {
            "nodes": {
                "a": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "A_c",
                      "x": 0, "y": 0, "label_x": 0, "label_y": -20},
                "b": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "B_c",
                      "x": 10, "y": 0, "label_x": 10, "label_y": -20},
                "m": {"node_type": "midmarker", "x": 5, "y": 40},
            },
            "reactions": {
                "1": {
                    "bigg_id": "R1", "label_x": 5, "label_y": 50,
                    "metabolites": [{"bigg_id": "A_c", "coefficient": -1},
                                    {"bigg_id": "B_c", "coefficient": 1}],
                    "segments": {"s1": {"from_node_id": "a", "to_node_id": "m"},
                                 "s2": {"from_node_id": "m", "to_node_id": "b"}},
                }
            },
        },
    ]
    anchors = {"m:A_c": [0, 0], "r:R1": [5, 40]}
    data, rep = ca.clear_rigid_node_overlaps_escher(
        data, [], use_escher_radii=True, guard_official=True, anchors=anchors)
    nodes = data[1]["nodes"]
    assert nodes["a"]["x"] == 0 and nodes["a"]["y"] == 0
    assert nodes["m"]["x"] == 5 and nodes["m"]["y"] == 40
    assert rep.get("pinned") == 2


def test_shared_metabolite_curated_structures_translate_as_unit():
    """Two curated structures sharing a metabolite must not stretch when cleared."""
    import curated_axes as ca

    # Structure S1: R1–R2 via shared hinge H; structure S2: R3–R4 via H.
    # An intruder primary sits on H so the clear must translate the merged unit.
    data = [
        {"canvas": {"width": 400, "height": 300}, "metadata": {}},
        {
            "nodes": {
                "h": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "H_c",
                      "x": 100.0, "y": 100.0, "label_x": 100.0, "label_y": 80.0},
                "a": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "A_c",
                      "x": 40.0, "y": 100.0, "label_x": 40.0, "label_y": 80.0},
                "b": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "B_c",
                      "x": 160.0, "y": 100.0, "label_x": 160.0, "label_y": 80.0},
                "c": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "C_c",
                      "x": 100.0, "y": 40.0, "label_x": 100.0, "label_y": 20.0},
                "d": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "D_c",
                      "x": 100.0, "y": 160.0, "label_x": 100.0, "label_y": 140.0},
                "m1": {"node_type": "midmarker", "x": 70.0, "y": 100.0},
                "m2": {"node_type": "midmarker", "x": 130.0, "y": 100.0},
                "m3": {"node_type": "midmarker", "x": 100.0, "y": 70.0},
                "m4": {"node_type": "midmarker", "x": 100.0, "y": 130.0},
                "intruder": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "Z_c",
                             "x": 105.0, "y": 100.0, "label_x": 105.0, "label_y": 80.0},
                "mz": {"node_type": "midmarker", "x": 200.0, "y": 100.0},
                "z2": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "Y_c",
                       "x": 240.0, "y": 100.0, "label_x": 240.0, "label_y": 80.0},
            },
            "reactions": {
                "1": {"bigg_id": "R1", "label_x": 70, "label_y": 110,
                      "metabolites": [{"bigg_id": "A_c", "coefficient": -1},
                                      {"bigg_id": "H_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "a", "to_node_id": "m1"},
                                   "1": {"from_node_id": "m1", "to_node_id": "h"}}},
                "2": {"bigg_id": "R2", "label_x": 130, "label_y": 110,
                      "metabolites": [{"bigg_id": "H_c", "coefficient": -1},
                                      {"bigg_id": "B_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "h", "to_node_id": "m2"},
                                   "1": {"from_node_id": "m2", "to_node_id": "b"}}},
                "3": {"bigg_id": "R3", "label_x": 110, "label_y": 70,
                      "metabolites": [{"bigg_id": "C_c", "coefficient": -1},
                                      {"bigg_id": "H_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "c", "to_node_id": "m3"},
                                   "1": {"from_node_id": "m3", "to_node_id": "h"}}},
                "4": {"bigg_id": "R4", "label_x": 110, "label_y": 130,
                      "metabolites": [{"bigg_id": "H_c", "coefficient": -1},
                                      {"bigg_id": "D_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "h", "to_node_id": "m4"},
                                   "1": {"from_node_id": "m4", "to_node_id": "d"}}},
                "5": {"bigg_id": "RZ", "label_x": 200, "label_y": 110,
                      "metabolites": [{"bigg_id": "Z_c", "coefficient": -1},
                                      {"bigg_id": "Y_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "intruder", "to_node_id": "mz"},
                                   "1": {"from_node_id": "mz", "to_node_id": "z2"}}},
            },
        },
    ]
    blocks = [
        dict(id="s1", reaction_ids=["R1", "R2"]),
        dict(id="s2", reaction_ids=["R3", "R4"]),
    ]
    groups, nid_group, frozen = ca._curated_structure_groups(data, blocks)
    # Shared hinge → both structures frozen, neither translated as a unit.
    assert len(groups) == 0
    assert "h" in frozen and "a" in frozen and "c" in frozen
    assert nid_group.get("h") is None

    before = {nid: (float(n["x"]), float(n["y"])) for nid, n in data[1]["nodes"].items()}
    structure_nids = frozen

    def _pairwise(nids):
        pts = {nid: before[nid] for nid in nids}
        return {(u, v): (pts[v][0] - pts[u][0], pts[v][1] - pts[u][1])
                for i, u in enumerate(sorted(nids)) for v in sorted(nids)[i + 1:]}

    rel_before = _pairwise(structure_nids)
    data, rep = ca.clear_rigid_node_overlaps_escher(
        data, blocks, use_escher_radii=True, guard_official=True)
    after = {nid: (float(n["x"]), float(n["y"])) for nid, n in data[1]["nodes"].items()}
    # Relative geometry of both curated structures is unchanged (not stretched).
    for key, (dx, dy) in rel_before.items():
        u, v = key
        got = (after[v][0] - after[u][0], after[v][1] - after[u][1])
        assert abs(got[0] - dx) < 1e-9 and abs(got[1] - dy) < 1e-9, (key, dx, dy, got)
    # Contested curated nodes stay put; the free intruder may move.
    for nid in structure_nids:
        assert after[nid] == before[nid], nid


def test_curated_structure_translate_includes_shared_free_metabolite():
    """A curated structure sharing a metabolite with a free reaction translates as one unit."""
    import curated_axes as ca

    data = [
        {"canvas": {"width": 300, "height": 200}, "metadata": {}},
        {
            "nodes": {
                "h": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "H_c",
                      "x": 100.0, "y": 100.0, "label_x": 100.0, "label_y": 80.0},
                "a": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "A_c",
                      "x": 40.0, "y": 100.0, "label_x": 40.0, "label_y": 80.0},
                "b": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "B_c",
                      "x": 160.0, "y": 100.0, "label_x": 160.0, "label_y": 80.0},
                "m1": {"node_type": "midmarker", "x": 70.0, "y": 100.0},
                "m2": {"node_type": "midmarker", "x": 130.0, "y": 100.0},
                # Free reaction hanging off H (not curated).
                "mfree": {"node_type": "midmarker", "x": 100.0, "y": 160.0},
                "f": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "F_c",
                      "x": 100.0, "y": 200.0, "label_x": 100.0, "label_y": 180.0},
                "intruder": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "Z_c",
                             "x": 105.0, "y": 100.0, "label_x": 105.0, "label_y": 80.0},
                "mz": {"node_type": "midmarker", "x": 200.0, "y": 100.0},
                "z2": {"node_type": "metabolite", "node_is_primary": True, "bigg_id": "Y_c",
                       "x": 240.0, "y": 100.0, "label_x": 240.0, "label_y": 80.0},
            },
            "reactions": {
                "1": {"bigg_id": "R1", "label_x": 70, "label_y": 110,
                      "metabolites": [{"bigg_id": "A_c", "coefficient": -1},
                                      {"bigg_id": "H_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "a", "to_node_id": "m1"},
                                   "1": {"from_node_id": "m1", "to_node_id": "h"}}},
                "2": {"bigg_id": "R2", "label_x": 130, "label_y": 110,
                      "metabolites": [{"bigg_id": "H_c", "coefficient": -1},
                                      {"bigg_id": "B_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "h", "to_node_id": "m2"},
                                   "1": {"from_node_id": "m2", "to_node_id": "b"}}},
                "3": {"bigg_id": "RFREE", "label_x": 100, "label_y": 160,
                      "metabolites": [{"bigg_id": "H_c", "coefficient": -1},
                                      {"bigg_id": "F_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "h", "to_node_id": "mfree"},
                                   "1": {"from_node_id": "mfree", "to_node_id": "f"}}},
                "4": {"bigg_id": "RZ", "label_x": 200, "label_y": 110,
                      "metabolites": [{"bigg_id": "Z_c", "coefficient": -1},
                                      {"bigg_id": "Y_c", "coefficient": 1}],
                      "segments": {"0": {"from_node_id": "intruder", "to_node_id": "mz"},
                                   "1": {"from_node_id": "mz", "to_node_id": "z2"}}},
            },
        },
    ]
    blocks = [dict(id="s1", reaction_ids=["R1", "R2"])]
    groups, nid_group, frozen = ca._curated_structure_groups(data, blocks)
    assert len(groups) == 1
    assert "h" in groups[0] and nid_group.get("h") == 0
    before = {nid: (float(n["x"]), float(n["y"])) for nid, n in data[1]["nodes"].items()}
    data, rep = ca.clear_rigid_node_overlaps_escher(
        data, blocks, use_escher_radii=True, guard_official=False)
    after = {nid: (float(n["x"]), float(n["y"])) for nid, n in data[1]["nodes"].items()}
    # Structure relative geometry preserved (shared hinge included).
    for u, v in (("a", "h"), ("h", "b"), ("a", "b"), ("m1", "m2")):
        db = (before[v][0] - before[u][0], before[v][1] - before[u][1])
        da = (after[v][0] - after[u][0], after[v][1] - after[u][1])
        assert abs(da[0] - db[0]) < 1e-9 and abs(da[1] - db[1]) < 1e-9
    shifts = [(after[nid][0] - before[nid][0], after[nid][1] - before[nid][1]) for nid in groups[0]]
    assert all(abs(s[0] - shifts[0][0]) < 1e-9 and abs(s[1] - shifts[0][1]) < 1e-9 for s in shifts)
