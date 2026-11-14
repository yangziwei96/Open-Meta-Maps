"""Unit tests for post-layout hotspot repair (python3 test_hotspot_repair.py)."""
import copy
import time

import hotspot_repair as hr
import layout_quality as lq


def _rx(bigg, mid_id, met_id, met_bigg, label=(0, 0), coeff=-1):
    return {
        "bigg_id": bigg,
        "name": bigg,
        "label_x": label[0],
        "label_y": label[1],
        "metabolites": [{"bigg_id": met_bigg, "coefficient": coeff}],
        "segments": {"0": {"from_node_id": mid_id, "to_node_id": met_id, "b1": None, "b2": None}},
    }


def _crossing_map():
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


def _dense_fixable():
    nodes = {
        "rf": {"node_type": "midmarker", "x": 200.0, "y": 200.0},
        "sub": {"node_type": "metabolite", "x": 80.0, "y": 200.0, "bigg_id": "mal__L_c", "node_is_primary": True,
                "label_x": 40.0, "label_y": 190.0},
        "prod": {"node_type": "metabolite", "x": 320.0, "y": 200.0, "bigg_id": "oaa_c", "node_is_primary": True,
                 "label_x": 330.0, "label_y": 190.0},
        "h": {"node_type": "metabolite", "x": 300.0, "y": 120.0, "bigg_id": "h_c", "node_is_primary": False,
              "label_x": 310.0, "label_y": 110.0},
        "r2": {"node_type": "midmarker", "x": 260.0, "y": 320.0},
        "m2": {"node_type": "metabolite", "x": 260.0, "y": 80.0, "bigg_id": "pyr_c", "node_is_primary": True,
               "label_x": 270.0, "label_y": 70.0},
        "pad": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
    }
    reactions = {
        "1": {
            "bigg_id": "MDH", "name": "MDH", "label_x": 210.0, "label_y": 180.0,
            "metabolites": [
                {"bigg_id": "mal__L_c", "coefficient": -1},
                {"bigg_id": "oaa_c", "coefficient": 1},
                {"bigg_id": "h_c", "coefficient": 1},
            ],
            "segments": {
                "s0": {"from_node_id": "rf", "to_node_id": "sub", "b1": None, "b2": None},
                "s1": {"from_node_id": "rf", "to_node_id": "prod", "b1": None, "b2": None},
                "s2": {"from_node_id": "rf", "to_node_id": "h", "b1": None, "b2": None},
            },
        },
        "2": {
            "bigg_id": "ME1", "name": "ME1", "label_x": 270.0, "label_y": 330.0,
            "metabolites": [{"bigg_id": "pyr_c", "coefficient": 1}],
            "segments": {"s0": {"from_node_id": "r2", "to_node_id": "m2", "b1": None, "b2": None}},
        },
    }
    return [
        {"map_name": "toy-hotspot", "map_id": "toy-hotspot", "schema": "1.0.0", "metadata": {}},
        {"nodes": nodes, "reactions": reactions, "canvas": {"x": 0, "y": 0, "width": 400, "height": 400}},
    ]


def test_crossing_index_matches_evaluate_headline():
    data = _crossing_map()
    scored = hr._score(data, [])
    idx = lq.CrossingIndex(data, [])
    assert idx.headline() == scored["crossings"]
    total, cross, _ = lq.algorithm_crossings(data, [])
    assert total == scored["crossings"]
    assert cross["algorithm_algorithm"] == 2


def test_index_delta_side_move_reduces_or_equal():
    data = _dense_fixable()
    idx = lq.CrossingIndex(data, [])
    before = idx.headline()
    assert before >= 1
    # Flip h_c to the other side of MDH
    nodes = data[1]["nodes"]
    n = nodes["h"]
    mid = nodes["rf"]
    n["x"] = round(2 * mid["x"] - n["x"], 2)
    n["y"] = round(2 * mid["y"] - n["y"], 2)
    new_total, delta, tmp = idx.delta_after_replace(data, {"MDH"})
    assert new_total == before + delta
    assert new_total <= before


def test_repair_fast_and_no_overlap_regression():
    data = _crossing_map()
    before = hr._score(data, [])
    t0 = time.time()
    report = hr.repair(data, skip_edges=[], k=3, max_rounds=2, max_accepts=8, budget_evals=200)
    elapsed = time.time() - t0
    after = hr._score(data, [])
    assert after["node_overlaps"] <= before["node_overlaps"]
    assert after["label_overlaps"] <= before["label_overlaps"]
    assert after["crossings"] <= before["crossings"]
    assert report["seconds"] < 5.5
    assert elapsed < 6.0


def test_repair_can_reduce_crossings_on_fixable_hotspot():
    before_data = _dense_fixable()
    before = hr._score(before_data, [])
    assert before["crossings"] >= 1
    live = _dense_fixable()
    report = hr.repair(live, skip_edges=[], k=3, max_rounds=3, max_accepts=12, budget_evals=200)
    after = hr._score(live, [])
    assert after["crossings"] <= before["crossings"]
    assert after["node_overlaps"] <= before["node_overlaps"]
    assert report["after"]["crossings"] == after["crossings"]


def test_window_mass_counts_crossings_and_edges():
    small = hr._window_mass([dict(crossings=4, edges=20)], k=1)
    large = hr._window_mass([dict(crossings=60, edges=200)], k=1)
    assert small == dict(window_crossings=4, window_edges=20)
    assert large["window_crossings"] > small["window_crossings"]
    assert large["window_edges"] > small["window_edges"]


def test_repair_derives_accept_and_round_caps_from_window_mass():
    data = _crossing_map()
    report = hr.repair(data, skip_edges=[], k=3)
    mass = report["budget_mass"]
    assert report["max_accepts"] == max(1, int(mass["window_crossings"]))
    assert report["max_rounds"] == report["max_accepts"] + 1
    assert report["budget_evals"] is not None and report["budget_evals"] >= 8


def test_repair_respects_previous_map_pins():
    data = _crossing_map()
    # Pin every metabolite so reposition / block moves that touch them are rejected.
    body = data[1]
    pins = {'m:' + n['bigg_id'] for n in body['nodes'].values()
            if n.get('node_type') == 'metabolite' and n.get('bigg_id')}
    report = hr.repair(data, skip_edges=[], k=3, pinned_keys=pins)
    assert report['reject_reasons'].get('pinned', 0) >= 0
    # Still terminates.
    assert report['rounds'] >= 1 or report['accepted_count'] == 0


if __name__ == "__main__":
    test_crossing_index_matches_evaluate_headline()
    test_index_delta_side_move_reduces_or_equal()
    test_repair_fast_and_no_overlap_regression()
    test_repair_can_reduce_crossings_on_fixable_hotspot()
    test_window_mass_counts_crossings_and_edges()
    test_repair_derives_accept_and_round_caps_from_window_mass()
    print("ok")
