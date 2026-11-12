"""Unit tests for final-geometry layout quality (python3 test_layout_quality.py)."""
import layout_quality as lq


def _rx(bigg, mid_id, met_id, met_bigg, label=(0, 0)):
    return {
        "bigg_id": bigg,
        "name": bigg,
        "label_x": label[0],
        "label_y": label[1],
        "metabolites": [{"bigg_id": met_bigg, "coefficient": -1}],
        "segments": {"0": {"from_node_id": mid_id, "to_node_id": met_id}},
    }


def _crossing_map():
    """Two algorithm×algorithm crossings: one near (150,150), one near (750,750)."""
    nodes = {
        "m1": {"node_type": "midmarker", "x": 100.0, "y": 100.0},
        "a1": {"node_type": "metabolite", "x": 200.0, "y": 200.0, "bigg_id": "A_c", "node_is_primary": True,
               "label_x": 210.0, "label_y": 210.0},
        "m2": {"node_type": "midmarker", "x": 100.0, "y": 200.0},
        "a2": {"node_type": "metabolite", "x": 200.0, "y": 100.0, "bigg_id": "B_c", "node_is_primary": True,
               "label_x": 210.0, "label_y": 90.0},
        "m3": {"node_type": "midmarker", "x": 700.0, "y": 700.0},
        "a3": {"node_type": "metabolite", "x": 800.0, "y": 800.0, "bigg_id": "C_c", "node_is_primary": True,
               "label_x": 810.0, "label_y": 810.0},
        "m4": {"node_type": "midmarker", "x": 700.0, "y": 800.0},
        "a4": {"node_type": "metabolite", "x": 800.0, "y": 700.0, "bigg_id": "D_c", "node_is_primary": True,
               "label_x": 810.0, "label_y": 690.0},
        # Anchor the window origin so region boxes are deterministic.
        "pad": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
    }
    reactions = {
        "1": _rx("R1", "m1", "a1", "A_c", (100, 90)),
        "2": _rx("R2", "m2", "a2", "B_c", (100, 210)),
        "3": _rx("R3", "m3", "a3", "C_c", (700, 690)),
        "4": _rx("R4", "m4", "a4", "D_c", (700, 810)),
    }
    return [
        {"map_name": "toy-cross", "map_id": "toy-cross", "schema": "1.0.0", "metadata": {}},
        {"nodes": nodes, "reactions": reactions, "canvas": {"x": 0, "y": 0, "width": 900, "height": 900}},
    ]


def test_region_window_keys_membership():
    # Event at x0+250 belongs to windows starting at x0+50 (k=floor(250/200)-1=0) and x0+200 (k=1)?
    # floor(250/200)=1 → k in {0,1}; windows [0,400) and [200,600).
    keys = lq._region_window_keys(250.0, 250.0, 0.0, 0.0)
    assert set(keys) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for kx, ky in keys:
        assert 0 + kx * 200 <= 250 < 0 + kx * 200 + 400
        assert 0 + ky * 200 <= 250 < 0 + ky * 200 + 400


def test_worst_region_count_matches_drawn_box():
    data = _crossing_map()
    summary, issues = lq.evaluate(data, max_issues=10)
    crossings = [c for c in issues["crossings"] if c["kind"] in ("algorithm_algorithm", "algorithm_curated", "algorithm_expert")]
    assert len(crossings) == 2, crossings
    # Uncapped list used for windows even when max_issues is tiny.
    assert summary["core"]["crossings"]["algorithm_algorithm"] == 2
    assert "main_metabolite" in summary["core"]["crossings"]
    assert "cofactor_clones" in summary["core"]
    assert "participant_placement" in summary["core"]
    assert "line_shape" in summary["details"]
    assert summary["core"]["crossings"]["algorithm_expert"] == summary["core"]["crossings"]["algorithm_curated"]

    near = [c for c in crossings if abs(c["x"] - 150) < 1 and abs(c["y"] - 150) < 1]
    far = [c for c in crossings if abs(c["x"] - 750) < 1 and abs(c["y"] - 750) < 1]
    assert len(near) == 1 and len(far) == 1

    windows = summary["worst_regions"]
    assert windows, "expected at least one worst region"
    for w in windows:
        inside = sum(
            1 for c in crossings
            if w["x"] <= c["x"] < w["x"] + w["w"] and w["y"] <= c["y"] < w["y"] + w["h"]
        )
        assert w["crossings"] == inside, (w, inside, crossings)
        # Reactions listed must correspond to crossings inside the drawn box.
        for rid in w["reactions"]:
            assert any(
                rid in c["reactions"] and w["x"] <= c["x"] < w["x"] + w["w"] and w["y"] <= c["y"] < w["y"] + w["h"]
                for c in crossings
            ), (rid, w, crossings)

    # The near and far crossings must not share a single 400×400 window count of 2.
    assert all(w["crossings"] <= 1 for w in windows), windows
    assert max(w["crossings"] for w in windows) == 1


def test_window_uses_uncapped_crossings():
    data = _crossing_map()
    summary, _ = lq.evaluate(data, max_issues=0)
    assert summary["core"]["crossings"]["algorithm_algorithm"] == 2
    assert summary["details"]["crossing_density"]["worst_window_crossings"] == 1
    assert any(w["crossings"] == 1 for w in summary["worst_regions"])


def test_detailed_panel_keeps_goals_only():
    data = _crossing_map()
    summary, _ = lq.evaluate(data, max_issues=10)
    d = summary["details"]
    assert "panel" in d
    groups = {g["id"]: g for g in d["panel"]["groups"]}
    labels = [r["label"] for g in d["panel"]["groups"] for r in g["rows"]]
    assert "Local crossing hotspot" in labels
    assert "Blank space / space utilisation" in labels
    # Cycle is a layout target owned by cycle layout · composition
    cycle_rows = [r for r in groups["structure_flow"]["rows"] if r["code"] == "cycle_circularity"]
    assert cycle_rows and cycle_rows[0]["stage"] == "cycle layout · composition"
    # Orderliness metrics are targets (not yet optimised)
    for code in ("main_segment_axis_share", "reaction_axis_share", "metabolite_grid_share", "chain_turn_median_deg"):
        rows = [r for r in groups["structure_flow"]["rows"] if r["code"] == code]
        assert rows and rows[0]["stage"] == lq.STAGE_NOT_YET, code
        assert rows[0].get("better") in ("higher", "lower")
    assert d.get("main_segment_axis_share") is not None or d.get("orderliness")
    # Flow direction is not a target → debug-only (collapsed), never a main Structure & flow row
    assert all(r["code"] != "flow_main_dir_share" for r in groups["structure_flow"]["rows"])
    if "debug" in groups:
        assert groups["debug"].get("debug") is True
        assert groups["debug"].get("collapsed") is True
        assert any(r["code"] == "flow_main_dir_share" for r in groups["debug"]["rows"])
    # Reproducibility is optional detail only (not a main card); absent without compare-map
    assert "reproducibility" not in groups
    # Removed duplicates / pure statistics / transfer-check extras
    assert "length" not in d
    assert "participant_consistency" not in d
    assert "area_per_node" not in d
    assert "subsystem_conventions" not in d
    assert "density" not in labels
    assert "catalyst" not in " ".join(labels).lower()
    # One content bbox: aspect and blank/utilisation agree
    if d.get("blank_space_share") is not None and d.get("space_utilisation") is not None:
        assert abs((d["blank_space_share"] + d["space_utilisation"]) - 1.0) < 1e-6
    if d.get("composition") and d.get("aspect_ratio") is not None:
        assert d["composition"]["aspect_ratio"] == d["aspect_ratio"]


def test_orderliness_metrics_axis_chain_and_grid():
    """Orderliness lives in map_rules and scores axis / chain / grid on final geometry."""
    from evaluation.map_rules import orderliness_metrics

    # Horizontal chain A -- R1 -- B -- R2 -- C, plus one vertical reaction D--R3--E.
    # All main segments and reaction axes are axis-aligned; B is collinear (turn 0°);
    # A/B/C share y and D/E share x with the spine → high grid share.
    nodes = {
        "m1": {"node_type": "midmarker", "x": 100.0, "y": 100.0},
        "a": {"node_type": "metabolite", "x": 0.0, "y": 100.0, "bigg_id": "A_c", "node_is_primary": True},
        "b": {"node_type": "metabolite", "x": 200.0, "y": 100.0, "bigg_id": "B_c", "node_is_primary": True},
        "m2": {"node_type": "midmarker", "x": 300.0, "y": 100.0},
        "c": {"node_type": "metabolite", "x": 400.0, "y": 100.0, "bigg_id": "C_c", "node_is_primary": True},
        "m3": {"node_type": "midmarker", "x": 200.0, "y": 200.0},
        "d": {"node_type": "metabolite", "x": 200.0, "y": 150.0, "bigg_id": "D_c", "node_is_primary": True},
        "e": {"node_type": "metabolite", "x": 200.0, "y": 250.0, "bigg_id": "E_c", "node_is_primary": True},
        # Cofactor spoke must be ignored for segment / axis / grid shares.
        "cof": {"node_type": "metabolite", "x": 140.0, "y": 160.0, "bigg_id": "atp_c", "node_is_primary": False},
    }
    reactions = {
        "1": {
            "bigg_id": "R1", "name": "R1", "label_x": 100, "label_y": 90,
            "metabolites": [
                {"bigg_id": "A_c", "coefficient": -1},
                {"bigg_id": "B_c", "coefficient": 1},
                {"bigg_id": "atp_c", "coefficient": -1},
            ],
            "segments": {
                "0": {"from_node_id": "m1", "to_node_id": "a"},
                "1": {"from_node_id": "m1", "to_node_id": "b"},
                "2": {"from_node_id": "m1", "to_node_id": "cof"},
            },
        },
        "2": {
            "bigg_id": "R2", "name": "R2", "label_x": 300, "label_y": 90,
            "metabolites": [
                {"bigg_id": "B_c", "coefficient": -1},
                {"bigg_id": "C_c", "coefficient": 1},
            ],
            "segments": {
                "0": {"from_node_id": "m2", "to_node_id": "b"},
                "1": {"from_node_id": "m2", "to_node_id": "c"},
            },
        },
        "3": {
            "bigg_id": "R3", "name": "R3", "label_x": 210, "label_y": 200,
            "metabolites": [
                {"bigg_id": "D_c", "coefficient": -1},
                {"bigg_id": "E_c", "coefficient": 1},
            ],
            "segments": {
                "0": {"from_node_id": "m3", "to_node_id": "d"},
                "1": {"from_node_id": "m3", "to_node_id": "e"},
            },
        },
    }
    data = [
        {"map_name": "toy-ord", "map_id": "toy-ord", "schema": "1.0.0", "metadata": {}},
        {"nodes": nodes, "reactions": reactions, "canvas": {"x": 0, "y": 0, "width": 500, "height": 300}},
    ]
    edges, _ = lq.build_edges(data)
    m = orderliness_metrics(data[1], edges, spacing=60.0)
    assert m["main_segment_axis_share"] == 1.0
    assert m["reaction_axis_share"] == 1.0
    assert m["chain_turn_median_deg"] == 0.0
    assert m["metabolite_grid_share"] == 1.0
    assert m["orderliness_segments"] == 6  # cofactor spoke excluded (one edge per spoke)
    assert m["orderliness_reactions"] == 3
    assert m["orderliness_chain_turns"] == 1
    # Evaluate must surface the same numbers on the panel as not-yet-optimised targets.
    summary, _ = lq.evaluate(data, max_issues=10)
    d = summary["details"]
    assert d["main_segment_axis_share"] == 1.0
    assert d["reaction_axis_share"] == 1.0
    assert d["chain_turn_median_deg"] == 0.0
    assert d["metabolite_grid_share"] == 1.0
    rows = {r["code"]: r for r in next(g for g in d["panel"]["groups"] if g["id"] == "structure_flow")["rows"]}
    for code in ("main_segment_axis_share", "reaction_axis_share", "metabolite_grid_share", "chain_turn_median_deg"):
        assert rows[code]["stage"] == lq.STAGE_NOT_YET
        assert rows[code]["miss"] is False


def test_orderliness_goals_match_curated_ecoli_core():
    """Panel goals are the curated BiGG e_coli_core scores (and beat our compose)."""
    import json
    from pathlib import Path
    from evaluation.map_rules import ORDERLINESS_GOALS, orderliness_metrics

    curated = Path(__file__).resolve().parents[1] / (
        "data/source_inputs/raw/escher/e_coli_core/e_coli_core.Core metabolism.json"
    )
    assert curated.is_file(), curated
    data = json.loads(curated.read_text())
    edges, _ = lq.build_edges(data)
    m = orderliness_metrics(data[1], edges, spacing=60.0)
    for key, goal in ORDERLINESS_GOALS.items():
        assert abs(float(m[key]) - float(goal)) < 1e-3, (key, m[key], goal)
    # Goals must sit on the panel as miss thresholds.
    summary, _ = lq.evaluate(data, max_issues=10)
    rows = {r["code"]: r for r in next(g for g in summary["details"]["panel"]["groups"] if g["id"] == "structure_flow")["rows"]}
    assert rows["main_segment_axis_share"]["miss"] is False
    assert rows["reaction_axis_share"]["miss"] is False
    assert rows["metabolite_grid_share"]["miss"] is False
    assert rows["chain_turn_median_deg"]["miss"] is False
    assert "curated e_coli_core" in rows["metabolite_grid_share"]["target"]
    # Curated grid must clearly beat a typical algorithmic layout (gate output if present).
    ours = Path("/tmp/omm-gate4/ecoli-after/map.json")
    if ours.is_file():
        od = json.loads(ours.read_text())
        om = orderliness_metrics(od[1], lq.build_edges(od)[0], spacing=60.0)
        assert m["metabolite_grid_share"] > om["metabolite_grid_share"] + 0.4
        assert m["main_segment_axis_share"] > om["main_segment_axis_share"]
        assert m["reaction_axis_share"] > om["reaction_axis_share"]
        assert m["chain_turn_median_deg"] < om["chain_turn_median_deg"]


def test_orderliness_metrics_invariant_to_resampling():
    """Count- and length-weighted axis shares and node-spacing grid tol ignore subdivision."""
    import numpy as np
    from evaluation.map_rules import orderliness_metrics

    # Diagonal + horizontal main edges so axis share is neither 0 nor 1.
    nodes = {
        "m1": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
        "a": {"node_type": "metabolite", "x": 100.0, "y": 0.0, "bigg_id": "A_c", "node_is_primary": True},
        "b": {"node_type": "metabolite", "x": 100.0, "y": 100.0, "bigg_id": "B_c", "node_is_primary": True},
        "m2": {"node_type": "midmarker", "x": 200.0, "y": 100.0},
        "c": {"node_type": "metabolite", "x": 300.0, "y": 100.0, "bigg_id": "C_c", "node_is_primary": True},
    }
    reactions = {
        "1": {
            "bigg_id": "R1", "name": "R1", "label_x": 0, "label_y": -10,
            "metabolites": [
                {"bigg_id": "A_c", "coefficient": -1},
                {"bigg_id": "B_c", "coefficient": 1},
            ],
            "segments": {
                "0": {"from_node_id": "m1", "to_node_id": "a"},
                "1": {"from_node_id": "m1", "to_node_id": "b"},
            },
        },
        "2": {
            "bigg_id": "R2", "name": "R2", "label_x": 200, "label_y": 90,
            "metabolites": [
                {"bigg_id": "B_c", "coefficient": -1},
                {"bigg_id": "C_c", "coefficient": 1},
            ],
            "segments": {
                "0": {"from_node_id": "m2", "to_node_id": "b"},
                "1": {"from_node_id": "m2", "to_node_id": "c"},
            },
        },
    }
    data = [
        {"map_name": "toy-resample", "map_id": "toy-resample", "schema": "1.0.0", "metadata": {}},
        {"nodes": nodes, "reactions": reactions, "canvas": {"x": 0, "y": 0, "width": 400, "height": 200}},
    ]
    edges, _ = lq.build_edges(data)
    base = orderliness_metrics(data[1], edges, spacing=60.0)
    assert base["main_segment_axis_share"] is not None
    assert base["metabolite_grid_share"] is not None

    def _subdivide(edge_list, factor=2):
        out = []
        for e in edge_list:
            pts = np.asarray(e["pts"], float)
            new = [pts[0]]
            for a, b in zip(pts[:-1], pts[1:]):
                for k in range(1, factor):
                    t = k / factor
                    new.append((1.0 - t) * a + t * b)
                new.append(b)
            out.append(dict(e, pts=np.asarray(new, float)))
        return out

    for factor in (2, 4):
        dense = orderliness_metrics(data[1], _subdivide(edges, factor), spacing=60.0)
        assert dense["main_segment_axis_share"] == base["main_segment_axis_share"], factor
        assert dense["main_segment_axis_share_length"] == base["main_segment_axis_share_length"], factor
        assert dense["metabolite_grid_share"] == base["metabolite_grid_share"], factor
        assert dense["orderliness_grid_tol_px"] == base["orderliness_grid_tol_px"], factor
        assert dense["orderliness_segments"] == base["orderliness_segments"], factor


def test_content_bbox_shared():
    from evaluation.map_rules import content_bbox

    data = _crossing_map()
    box = content_bbox(data[1], data[1].get("canvas"))
    assert box["aspect_ratio"] is not None
    assert box["width"] is not None and box["height"] is not None


def test_svg_has_no_provenance_legend():
    import ec_provenance as epv
    import line_render

    assert epv.LABELS["expert_derived"] == "Curated"
    assert not hasattr(epv, "legend_svg")
    data = _crossing_map()
    kinds = {r["bigg_id"]: "generated" for r in data[1]["reactions"].values()}
    svg = line_render.render(data, kinds)
    assert "width=\"330\"" not in svg
    assert "Expert-derived" not in svg
    assert "Curated:" not in svg  # legend text must not be painted into the SVG
    assert 'data-kind="generated"' in svg
    assert 'stroke="#222"' in svg
    for colour in epv.COLORS.values():
        assert f'stroke="{colour}"' not in svg
        assert f'fill="{colour}"' not in svg


def test_currency_spokes_marked():
    """Currency nodes use node_is_primary=False; their spokes, circles and labels are tagged."""
    import line_render

    data = [
        {},
        {
            "nodes": {
                "m1": {"node_type": "midmarker", "x": 100.0, "y": 100.0},
                "a1": {"node_type": "metabolite", "x": 200.0, "y": 100.0, "bigg_id": "g6p_c",
                       "node_is_primary": True, "label_x": 210.0, "label_y": 100.0},
                "c1": {"node_type": "metabolite", "x": 100.0, "y": 200.0, "bigg_id": "atp_c",
                       "node_is_primary": False, "label_x": 100.0, "label_y": 210.0},
            },
            "reactions": {
                "1": {
                    "bigg_id": "R1", "name": "R1", "label_x": 100, "label_y": 90,
                    "metabolites": [{"bigg_id": "g6p_c", "coefficient": -1},
                                    {"bigg_id": "atp_c", "coefficient": -1}],
                    "segments": {
                        "0": {"from_node_id": "m1", "to_node_id": "a1"},
                        "1": {"from_node_id": "m1", "to_node_id": "c1"},
                    },
                }
            },
            "canvas": {"x": 0, "y": 0, "width": 300, "height": 300},
        },
    ]
    svg = line_render.render(data)
    assert 'data-met="atp_c" data-currency="1"' in svg
    assert 'data-m="atp_c" data-currency="1"' in svg
    assert 'data-nl="c1" data-currency="1"' in svg
    assert 'data-met="g6p_c" data-currency' not in svg
    assert 'data-m="g6p_c" data-currency' not in svg
    assert 'data-nl="a1" data-currency' not in svg


if __name__ == "__main__":
    test_region_window_keys_membership()
    test_worst_region_count_matches_drawn_box()
    test_window_uses_uncapped_crossings()
    test_detailed_panel_keeps_goals_only()
    test_orderliness_metrics_axis_chain_and_grid()
    test_orderliness_goals_match_curated_ecoli_core()
    test_orderliness_metrics_invariant_to_resampling()
    test_content_bbox_shared()
    test_svg_has_no_provenance_legend()
    test_currency_spokes_marked()
    print("ok")
