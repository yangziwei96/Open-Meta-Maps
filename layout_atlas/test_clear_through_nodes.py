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
