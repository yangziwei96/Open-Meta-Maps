"""Gated clear-through-nodes restyle (e_coli_core example_results)."""
from pathlib import Path

from evaluation.escher_bridge import load_escher
import edge_style
import layout_quality as lq

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example_results" / "e_coli_core" / "OpenMetaMaps_e_coli_core_result.json"


def test_curves_through_nodes_count_matches_evaluate():
    data = load_escher(EXAMPLE)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    n = lq.curves_through_nodes_count(data, skip)
    summary, _issues = lq.evaluate(data, skip)
    assert n == summary["core"]["overlaps"]["curves_through_nodes"]
    assert n >= 0


def test_official_lines_through_nodes_positive_on_example():
    data = load_escher(EXAMPLE)
    n = lq.official_lines_through_nodes_count(data)
    assert n >= 1
    report = lq.official_lines_through_nodes_report(data)
    assert report["count"] == n
    assert isinstance(report["reactions"], list)


def test_official_lines_through_nodes_without_shapely(monkeypatch):
    """Compose/clear-through must not hard-require eval-only shapely."""
    import sys

    class _Boom:
        def __getattr__(self, _name):
            raise ImportError("shapely blocked for compose path")

    monkeypatch.setitem(sys.modules, "shapely", _Boom())
    monkeypatch.setitem(sys.modules, "shapely.geometry", _Boom())
    data = load_escher(EXAMPLE)
    report = lq.official_lines_through_nodes_report(data)
    assert report["count"] >= 1
    assert isinstance(report["pairs"], list)


def test_clear_through_nodes_no_regression_on_e_coli_core():
    data = load_escher(EXAMPLE)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    before = lq.curves_through_nodes_count(data, skip)
    before_off = lq.official_lines_through_nodes_count(data, skip)
    before_cross = lq.CrossingIndex(data, skip).headline()
    report = edge_style.clear_through_nodes(data, skip)
    after = lq.curves_through_nodes_count(data, skip)
    after_off = lq.official_lines_through_nodes_count(data, skip)
    after_cross = lq.CrossingIndex(data, skip).headline()
    assert report["before"] == before
    assert report.get("before_official") == before_off
    assert after == report["after"]
    assert after_off == report["after_official"]
    assert after_cross == report["after_crossings"]
    if report["accepted"]:
        # Official pairs are the hard-rule target; OMM count may stay flat.
        assert after_off < before_off or after < before
        assert after_cross <= before_cross + 2
        assert after_off <= before_off
    else:
        # Reverted: live map unchanged.
        assert after == before
        assert after_off == before_off
        assert after_cross == before_cross


def test_clear_through_nodes_drives_e_coli_official_to_credit_band():
    """Official hard rule needs ≤4 pairs for any points (pass @0 / fail @5)."""
    data = load_escher(EXAMPLE)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    before_off = lq.official_lines_through_nodes_count(data, skip)
    assert before_off > 4
    report = edge_style.clear_through_nodes(data, skip)
    after_off = lq.official_lines_through_nodes_count(data, skip)
    assert report["accepted"]
    assert after_off <= 4
    assert after_off == report["after_official"]


def test_clear_through_nodes_skips_when_official_already_clean():
    """Post-hotspot / post-orthogonal clears must early-out once official is 0."""
    data = load_escher(EXAMPLE)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    edge_style.clear_through_nodes(data, skip)
    assert lq.official_lines_through_nodes_count(data, skip) == 0
    report = edge_style.clear_through_nodes(data, skip)
    assert report["reason"] == "official_clean"
    assert report["after_official"] == 0
    assert report["seconds"] < 1.0


def test_prepare_keeps_preserved_algorithm_corridor():
    """Preserved mid→multi→met corridors must not be dropped on prepare."""
    data = [{
        "metadata": {"preserved_edges": [["MDH", "mal__L_c"]]},
    }, {
        "nodes": {
            "mid": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
            "m1": {"node_type": "multimarker", "x": 0.0, "y": 50.0},
            "m2": {"node_type": "multimarker", "x": 0.0, "y": 100.0},
            "mal": {"node_type": "metabolite", "bigg_id": "mal__L_c", "x": 0.0, "y": 150.0},
            "oaa": {"node_type": "metabolite", "bigg_id": "oaa_c", "x": 50.0, "y": 0.0},
        },
        "reactions": {
            "r": {"bigg_id": "MDH", "segments": {
                "es_s1": {"from_node_id": "mid", "to_node_id": "m1", "b1": None, "b2": None},
                "es_s2": {"from_node_id": "m1", "to_node_id": "m2", "b1": None, "b2": None},
                "es_s3": {"from_node_id": "m2", "to_node_id": "mal", "b1": None, "b2": None},
                "s_oaa": {"from_node_id": "mid", "to_node_id": "oaa", "b1": None, "b2": None},
            }},
        },
    }]
    edge_style._prepare_algorithm_spokes_for_restyle(
        data, {"MDH"}, [["MDH", "mal__L_c"]],
    )
    segs = data[1]["reactions"]["r"]["segments"]
    assert "es_s1" in segs and "es_s3" in segs
    assert any(
        data[1]["nodes"].get(s["to_node_id"], {}).get("bigg_id") == "mal__L_c"
        or data[1]["nodes"].get(s["from_node_id"], {}).get("bigg_id") == "mal__L_c"
        for s in segs.values()
    )


def test_clear_through_rejects_preserved_path_deletion():
    """Surgical clear must not accept a preserved detour that severs mid→met."""
    data = [{
        "metadata": {"preserved_edges": [["MDH", "mal__L_c"]]},
    }, {
        "nodes": {
            "mid": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
            "mal": {"node_type": "metabolite", "bigg_id": "mal__L_c", "x": 0.0, "y": 100.0,
                    "node_is_primary": True},
            # Obstacle on the chord so official_clear may try a corridor.
            "obs": {"node_type": "metabolite", "bigg_id": "obs_c", "x": 0.0, "y": 50.0,
                    "node_is_primary": True},
        },
        "reactions": {
            "r": {"bigg_id": "MDH", "segments": {
                "s_mal": {"from_node_id": "mid", "to_node_id": "mal", "b1": None, "b2": None},
            }},
            "r2": {"bigg_id": "OBS", "segments": {
                "s": {"from_node_id": "mid", "to_node_id": "obs", "b1": None, "b2": None},
            }},
        },
    }]
    # Simulate a failed corridor accept candidate: dangling chain, no mal path.
    bad = [{
        "metadata": data[0]["metadata"],
    }, {
        "nodes": {
            "mid": {"node_type": "midmarker", "x": 0.0, "y": 0.0},
            "mal": {"node_type": "metabolite", "bigg_id": "mal__L_c", "x": 0.0, "y": 100.0,
                    "node_is_primary": True},
            "es1": {"node_type": "multimarker", "x": 40.0, "y": 40.0},
            "es2": {"node_type": "multimarker", "x": 80.0, "y": 80.0},
        },
        "reactions": {
            "r": {"bigg_id": "MDH", "segments": {
                "es_s1": {"from_node_id": "mid", "to_node_id": "es1", "b1": None, "b2": None},
                "es_s2": {"from_node_id": "es1", "to_node_id": "es2", "b1": None, "b2": None},
            }},
        },
    }]
    assert not edge_style._reaction_reaches_metabolite(bad, "MDH", "mal__L_c")
    rep = edge_style.repair_missing_preserved_spokes(bad, gate_official=False)
    assert rep["repaired"] >= 1
    assert edge_style._reaction_reaches_metabolite(bad, "MDH", "mal__L_c")
    # Live path still present on the intact map.
    assert edge_style._reaction_reaches_metabolite(data, "MDH", "mal__L_c")


def test_repair_fully_deleted_preserved_spoke():
    """Preserved MDH→mal with no segments at all is force-restored mid→met."""
    data = [{
        "metadata": {"preserved_edges": [["MDH", "mal__L_c"], ["MDH", "oaa_c"]]},
    }, {
        "nodes": {
            "mid": {"node_type": "midmarker", "x": 2433.5, "y": 3496.9},
            "mal": {"node_type": "metabolite", "bigg_id": "mal__L_c", "x": 2477.5, "y": 3284.9,
                    "node_is_primary": True},
            "oaa": {"node_type": "metabolite", "bigg_id": "oaa_c", "x": 2515.5, "y": 3816.9,
                    "node_is_primary": True},
        },
        "reactions": {
            "r": {"bigg_id": "MDH", "segments": {
                "1": {"from_node_id": "mid", "to_node_id": "oaa", "b1": None, "b2": None},
            }},
        },
    }]
    assert not edge_style._reaction_reaches_metabolite(data, "MDH", "mal__L_c")
    rep = edge_style.repair_missing_preserved_spokes(data, gate_official=False)
    assert rep["repaired"] >= 1
    assert edge_style._reaction_reaches_metabolite(data, "MDH", "mal__L_c")
    # New segment key is numeric (Escher schema).
    segs = data[1]["reactions"]["r"]["segments"]
    assert any(str(k).isdigit() and (
        segs[k]["to_node_id"] == "mal" or segs[k]["from_node_id"] == "mal"
    ) for k in segs)


def test_repair_missing_preserved_mdh_mal_spoke():
    """Dangling MDH multimarker chain with no path to mal__L_c is reconnected."""
    data = [{
        "metadata": {"preserved_edges": [["MDH", "mal__L_c"], ["MDH", "oaa_c"]]},
    }, {
        "nodes": {
            "mid": {"node_type": "midmarker", "x": 2433.5, "y": 3496.9},
            "es1": {"node_type": "multimarker", "x": 2290.0, "y": 3150.0},
            "es2": {"node_type": "multimarker", "x": 2245.0, "y": 2778.0},
            "es3": {"node_type": "multimarker", "x": 2199.0, "y": 2405.0},
            "mal": {"node_type": "metabolite", "bigg_id": "mal__L_c", "x": 2477.5, "y": 3284.9,
                    "node_is_primary": True},
            "oaa": {"node_type": "metabolite", "bigg_id": "oaa_c", "x": 2515.5, "y": 3816.9,
                    "node_is_primary": True},
        },
        "reactions": {
            "r": {"bigg_id": "MDH", "segments": {
                "es_s1": {"from_node_id": "mid", "to_node_id": "es1", "b1": None, "b2": None},
                "es_s2": {"from_node_id": "es1", "to_node_id": "es2", "b1": None, "b2": None},
                "es_s3": {"from_node_id": "es2", "to_node_id": "es3", "b1": None, "b2": None},
                "s_oaa": {"from_node_id": "mid", "to_node_id": "oaa", "b1": None, "b2": None},
            }},
        },
    }]
    rep = edge_style.repair_missing_preserved_spokes(data)
    assert rep["repaired"] >= 1
    # Path mid→mal exists again; dangling multimarkers removed.
    nodes = data[1]["nodes"]
    rx = data[1]["reactions"]["r"]
    assert "es1" not in nodes
    from collections import defaultdict
    adj = defaultdict(list)
    for s in rx["segments"].values():
        adj[s["from_node_id"]].append(s["to_node_id"])
        adj[s["to_node_id"]].append(s["from_node_id"])
    seen = {"mid"}
    stack = ["mid"]
    hit = False
    while stack:
        u = stack.pop()
        if nodes.get(u, {}).get("bigg_id") == "mal__L_c":
            hit = True
            break
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    assert hit
