"""Tests for the shared LayoutQualityEvaluator (e_coli_core / iIT341)."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import load_model
from evaluation.escher_bridge import load_escher
from hub_drawing import pathway_from_model
from layout_quality_evaluator import (
    FAR_EDGE_CLASSES,
    INSTANCE_DIRS,
    METRIC_EPS,
    SCORE_DIRS,
    SCORE_METRICS,
    TIP_DIRS,
    LayoutQualityEvaluator,
    accept_participant_consistency,
    coverage_metrics,
    curated_translation_only,
    dominates,
    far_edge_report,
    measure_scores,
    measure_scene_participant_consistency,
    metrics_not_worse,
    pipeline_crossings,
    previous_pinned_node_ids,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXAMPLE_RESULTS = ROOT / "example_results"


def _load(name: str):
    raw = load_escher(EXAMPLE_RESULTS / name / f"OpenMetaMaps_{name}_result.json")
    model = load_model(EXAMPLES / f"{name}.xml")
    pathway = pathway_from_model(model)
    return raw, model, pathway


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_pipeline_crossings_uses_skip(name):
    data, _model, _pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    n = pipeline_crossings(data, skip)
    assert isinstance(n, int) and n >= 0
    import layout_quality as lq

    assert n == lq.CrossingIndex(data, skip).headline()


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_measure_scores_keys(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    m = measure_scores(data, skip_edges=skip, model=model, pathway=pathway)
    for key in SCORE_METRICS:
        assert key in m
    assert m["species_drawn"] == pytest.approx(1.0)
    assert m["reactions_drawn"] == pytest.approx(1.0)
    assert m["crossings"] == pipeline_crossings(data, skip)


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_snapshot_evaluate_identical_accepted(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="metrics",
        require_local_improvement=False,
    )
    ev.snapshot()
    decision = ev.evaluate(copy.deepcopy(data))
    assert decision.hard_ok
    assert decision.metrics_ok
    assert decision.accepted
    assert ev.report()["accepted"] == 1


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_hard_reject_when_pin_moves(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    body = data[1]
    mid = next(
        nid for nid, n in body["nodes"].items() if n.get("node_type") == "midmarker"
    )
    rid = None
    for rx in body["reactions"].values():
        for seg in (rx.get("segments") or {}).values():
            if mid in (seg.get("from_node_id"), seg.get("to_node_id")):
                rid = rx.get("bigg_id")
                break
        if rid:
            break
    assert rid
    anchors = {f"r:{rid}": [0.0, 0.0]}
    pinned = previous_pinned_node_ids(data, anchors)
    assert mid in pinned

    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, previous_anchors=anchors,
        model=model, pathway=pathway, rule="metrics",
        require_local_improvement=False,
    )
    ev.snapshot()
    trial = copy.deepcopy(data)
    trial[1]["nodes"][mid]["x"] = float(trial[1]["nodes"][mid]["x"]) + 25.0
    decision = ev.evaluate(trial)
    assert not decision.accepted
    assert "pins_moved" in decision.reason
    assert ev.report()["rejected"] == 1


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_metric_regression_rejected(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="metrics",
        require_local_improvement=False,
    )
    baseline = ev.snapshot()
    worse = dict(baseline)
    worse["crossings"] = float(baseline["crossings"]) + 5
    decision = ev.evaluate(copy.deepcopy(data), metrics=worse)
    assert not decision.accepted
    assert "crossings" in decision.reason


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_local_improvement_required(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="metrics",
        require_local_improvement=True,
        local_objective=lambda d: float(pipeline_crossings(d, skip)),
    )
    ev.snapshot()
    decision = ev.evaluate(copy.deepcopy(data))
    assert not decision.accepted
    assert "no_local_improvement" in decision.reason


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_commit_and_rollback(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="metrics",
        require_local_improvement=False,
    )
    ev.snapshot()
    trial = copy.deepcopy(data)
    mets = [
        nid for nid, n in trial[1]["nodes"].items()
        if n.get("node_type") == "metabolite"
    ]
    assert mets
    nid = mets[0]
    x0 = float(data[1]["nodes"][nid]["x"])
    trial[1]["nodes"][nid]["x"] = x0 + 1.0
    before = copy.deepcopy(data)
    ev.commit(trial)
    assert float(ev.data[1]["nodes"][nid]["x"]) == pytest.approx(x0 + 1.0)
    ev.rollback()
    assert float(ev.data[1]["nodes"][nid]["x"]) == pytest.approx(x0 + 1.0)
    data[0], data[1] = before[0], before[1]


def test_pareto_prefer():
    dirs = SCORE_DIRS
    a = LayoutQualityEvaluator.__new__(LayoutQualityEvaluator)
    a.metric_dirs = dirs
    a.eps = METRIC_EPS
    from layout_quality_evaluator import Evaluation

    ea = Evaluation(True, "ok", True, True, True, dict(crossings=1, overlapping_pairs=1,
                    overlap_area_share=0.1, participant_consistency=0.9,
                    edge_length_deviation=0.2, edge_orthogonality=0.8,
                    angular_resolution=0.1, crossing_angular_resolution=0.7,
                    species_drawn=1.0, reactions_drawn=1.0))
    eb = Evaluation(True, "ok", True, True, True, dict(crossings=2, overlapping_pairs=1,
                    overlap_area_share=0.1, participant_consistency=0.9,
                    edge_length_deviation=0.2, edge_orthogonality=0.8,
                    angular_resolution=0.1, crossing_angular_resolution=0.7,
                    species_drawn=1.0, reactions_drawn=1.0))
    assert a.prefer(ea, eb) == -1
    assert a.prefer(eb, ea) == 1
    assert dominates(ea.metrics, eb.metrics, dirs)


def test_tip_evaluate_is_no_worse_including_aspect():
    data, model, pathway = _load("e_coli_core")
    ev = LayoutQualityEvaluator(
        data, model=model, pathway=pathway, rule="tip",
        require_local_improvement=False,
    )
    baseline = ev.snapshot(copy_data=False)
    assert "aspect_ratio" in baseline
    assert "curves_through_nodes" in baseline
    assert metrics_not_worse(baseline, baseline, TIP_DIRS)
    worse = dict(baseline, aspect_ratio=float(baseline["aspect_ratio"]) - 0.05)
    decision = ev.evaluate(copy.deepcopy(data), metrics=worse)
    assert not decision.accepted
    assert "aspect_ratio" in decision.reason
    worse_through = dict(baseline, curves_through_nodes=float(baseline["curves_through_nodes"]) + 1)
    decision2 = ev.evaluate(copy.deepcopy(data), metrics=worse_through)
    assert not decision2.accepted
    assert "curves_through_nodes" in decision2.reason


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_hard_reject_new_curves_through_nodes(name):
    """No-new hard gate: raising through-node hits vs snapshot is rejected."""
    import layout_quality as lq

    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="metrics",
        require_local_improvement=False,
    )
    ev.snapshot()
    # Identical candidate keeps residual through-nodes (not absolute zero).
    same = ev.evaluate(copy.deepcopy(data))
    assert same.accepted
    assert "new_curves_through_nodes" not in same.hard_failures

    real = lq.curves_through_nodes_count

    def bumped(d, skip_edges=None):
        return int(real(d, skip_edges)) + 1

    lq.curves_through_nodes_count = bumped
    try:
        decision = ev.evaluate(copy.deepcopy(data))
    finally:
        lq.curves_through_nodes_count = real
    assert not decision.accepted
    assert "new_curves_through_nodes" in decision.hard_failures
    assert "new_curves_through_nodes" in decision.reason


def test_instance_rule_new_only_side_consistency():
    """rule=instance rejects PC regressions; imperfect baselines may stay."""
    ev = LayoutQualityEvaluator([], rule="instance", require_local_improvement=False)
    ev.begin_report({"participant_consistency": 0.81})
    assert ev.evaluate(None, metrics={"participant_consistency": 0.81}).accepted
    assert ev.evaluate(None, metrics={"participant_consistency": 0.90}).accepted
    bad = ev.evaluate(None, metrics={"participant_consistency": 0.80})
    assert not bad.accepted
    assert "new_participant_inconsistency" in bad.reason
    assert INSTANCE_DIRS["participant_consistency"] == "higher"
    ok, why = accept_participant_consistency(0.5, 0.5)
    assert ok and why == "accepted"
    ok, why = accept_participant_consistency(0.5, 0.49)
    assert not ok


def test_measure_scene_participant_consistency_opposite_and_same():
    model = {"reactions": [{"id": "R1", "metabolites": {"a": -1, "b": 1}}]}
    opp = {
        "positions": {"r:R1": [0.0, 0.0], "m:a": [-10.0, 0.0], "m:b": [10.0, 0.0]},
        "participants": {("R1", "a"): "m:a", ("R1", "b"): "m:b"},
    }
    same = {
        "positions": {"r:R1": [0.0, 0.0], "m:a": [10.0, 0.0], "m:b": [10.0, 5.0]},
        "participants": {("R1", "a"): "m:a", ("R1", "b"): "m:b"},
    }
    assert measure_scene_participant_consistency(model, opp) == pytest.approx(1.0)
    assert measure_scene_participant_consistency(model, same) == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_curated_translation_only_identity(name):
    data, _model, _pathway = _load(name)
    assert curated_translation_only(data, copy.deepcopy(data))
    trial = copy.deepcopy(data)
    for n in trial[1]["nodes"].values():
        n["x"] = float(n["x"]) + 10.0
        n["y"] = float(n["y"]) + 5.0
    assert curated_translation_only(data, trial)
    curated = (data[0].get("metadata") or {}).get("expert_structures") or []
    assert curated, f"{name} example_results must include curated structures"
    rids = set(curated[0].get("reactions") or curated[0].get("reaction_ids") or [])
    mid = None
    for rx in trial[1]["reactions"].values():
        if rx.get("bigg_id") not in rids:
            continue
        for seg in (rx.get("segments") or {}).values():
            for end in ("from_node_id", "to_node_id"):
                nid = seg.get(end)
                if (trial[1]["nodes"].get(nid) or {}).get("node_type") == "midmarker":
                    mid = nid
                    break
            if mid:
                break
        if mid:
            break
    assert mid is not None, f"{name} curated structure must expose a midmarker"
    trial[1]["nodes"][mid]["x"] = float(trial[1]["nodes"][mid]["x"]) + 40.0
    assert not curated_translation_only(data, trial)


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_coverage_complete(name):
    data, model, _pathway = _load(name)
    cov = coverage_metrics(data, model)
    assert cov["species_drawn"] == pytest.approx(1.0)
    assert cov["reactions_drawn"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_far_edge_classes_partition(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    report = far_edge_report(data, skip_edges=skip)
    assert report["n_far"] >= 0
    assert sum(report["by_class"].values()) == report["n_far"]
    assert set(report["by_class"]) == set(FAR_EDGE_CLASSES)
    # Keys recovered when metabolite_instances is absent.
    if not (data[0].get("metadata") or {}).get("metabolite_instances"):
        assert any(row.get("instance_key") for row in report["top"]) or report["n_far"] == 0
    for row in report["top"]:
        assert row["far_class"] in FAR_EDGE_CLASSES
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="metrics",
        require_local_improvement=False,
    )
    assert ev.far_edges()["n_far"] == report["n_far"]
    assert ev.far_edges()["by_class"] == report["by_class"]


def test_far_edge_split_on_composed_iit341(tmp_path):
    """Compose iIT341 and assert the far-edge class split on that map."""
    gem = EXAMPLES / "iIT341.xml"
    assert gem.is_file()
    out = tmp_path / "iIT341"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "project.py"), "compose",
         "--gem", str(gem), "--out", str(out), "--source-paths"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1200,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    result = out / "OpenMetaMaps_iIT341_result.json"
    assert result.is_file(), proc.stdout[-2000:] + proc.stderr[-2000:]
    data = json.loads(result.read_text())
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    report = far_edge_report(data, skip_edges=skip)
    assert report["n_core"] == 1250
    by = report["by_class"]
    counts = [by[c] for c in FAR_EDGE_CLASSES]
    assert sum(counts) == report["n_far"]
    # Tukey fence on core mid↔met excl. side:; classes in FAR_EDGE_CLASSES order.
    assert counts == [4, 4, 12, 21, 16, 0, 0], (
        f"counts={counts} n_far={report['n_far']} thresh={report['length_threshold']}"
    )
    assert report["n_far"] == 57
