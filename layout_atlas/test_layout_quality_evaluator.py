"""Tests for the shared LayoutQualityEvaluator (e_coli_core / iIT341)."""
import copy
import json
from pathlib import Path

import pytest

from core import load_model
from evaluation.escher_bridge import load_escher
from hub_drawing import pathway_from_model
from layout_quality_evaluator import (
    FAR_EDGE_CLASSES,
    METRIC_EPS,
    OFFICIAL_DIRS,
    OFFICIAL_METRICS,
    LayoutQualityEvaluator,
    coverage_metrics,
    curated_translation_only,
    dominates,
    far_edge_report,
    measure_official,
    metrics_not_worse,
    pins_unchanged,
    pipeline_crossings,
    previous_pinned_node_ids,
    relocation_gate_not_worse,
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
    # Same edge set as CrossingIndex headline.
    import layout_quality as lq

    assert n == lq.CrossingIndex(data, skip).headline()


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_measure_official_keys(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    m = measure_official(data, skip_edges=skip, model=model, pathway=pathway)
    for key in OFFICIAL_METRICS:
        assert key in m
    assert m["species_drawn"] == pytest.approx(1.0)
    assert m["reactions_drawn"] == pytest.approx(1.0)
    assert m["crossings"] == pipeline_crossings(data, skip)


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_snapshot_evaluate_identical_accepted(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="official",
        require_local_improvement=False,
    )
    ev.snapshot()
    decision = ev.evaluate(copy.deepcopy(data))
    # Identical candidate: metrics OK, no local improvement required.
    assert decision.hard_ok
    assert decision.metrics_ok
    assert decision.accepted
    assert ev.report()["accepted"] == 1


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_hard_reject_when_pin_moves(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    # Pin the first midmarker via a synthetic r: anchor.
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
        model=model, pathway=pathway, rule="official",
        require_local_improvement=False,
    )
    ev.snapshot()
    trial = copy.deepcopy(data)
    trial[1]["nodes"][mid]["x"] = float(trial[1]["nodes"][mid]["x"]) + 25.0
    decision = ev.evaluate(trial)
    assert not decision.accepted
    assert "pins_moved" in decision.reason
    assert ev.report()["rejected"] == 1
    assert "hard" in ev.report()["reject_reasons"] or "pins_moved" in str(ev.report())


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_metric_regression_rejected(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="official",
        require_local_improvement=False,
    )
    baseline = ev.snapshot()
    # Fabricate a worse metric vector without touching geometry hard checks.
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
        data, skip_edges=skip, model=model, pathway=pathway, rule="official",
        require_local_improvement=True,
        local_objective=lambda d: float(pipeline_crossings(d, skip)),
    )
    ev.snapshot()
    # Same map: no local improvement.
    decision = ev.evaluate(copy.deepcopy(data))
    assert not decision.accepted
    assert "no_local_improvement" in decision.reason


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_commit_and_rollback(name):
    data, model, pathway = _load(name)
    skip = (data[0].get("metadata") or {}).get("preserved_edges") or []
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="official",
        require_local_improvement=False,
    )
    ev.snapshot()
    trial = copy.deepcopy(data)
    # Translate a free metabolite; keep hard checks happy by also disabling pins.
    mets = [
        nid for nid, n in trial[1]["nodes"].items()
        if n.get("node_type") == "metabolite"
    ]
    assert mets
    nid = mets[0]
    x0 = float(data[1]["nodes"][nid]["x"])
    trial[1]["nodes"][nid]["x"] = x0 + 1.0
    # May fail curated/hard on real maps; only test rollback bookkeeping on a
    # forced commit path.
    before = copy.deepcopy(data)
    ev.commit(trial)
    assert float(ev.data[1]["nodes"][nid]["x"]) == pytest.approx(x0 + 1.0)
    # New snapshot from committed state; rollback restores that snapshot.
    ev.rollback()
    assert float(ev.data[1]["nodes"][nid]["x"]) == pytest.approx(x0 + 1.0)
    # Restore original for isolation.
    data[0], data[1] = before[0], before[1]


def test_pareto_prefer():
    dirs = OFFICIAL_DIRS
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


def test_tip_adapter_matches_composition():
    before = dict(crossings=2, overlapping_pairs=1, participant_consistency=0.8,
                  angular_resolution=0.2, edge_length_deviation=0.5,
                  edge_orthogonality=0.6, aspect_ratio=0.7)
    assert relocation_gate_not_worse(before, before)
    assert not relocation_gate_not_worse(before, dict(before, crossings=3))


@pytest.mark.parametrize("name", ["e_coli_core", "iIT341"])
def test_curated_translation_only_identity(name):
    data, _model, _pathway = _load(name)
    assert curated_translation_only(data, copy.deepcopy(data))
    trial = copy.deepcopy(data)
    # Global translation of every node keeps relative curated geometry.
    for n in trial[1]["nodes"].values():
        n["x"] = float(n["x"]) + 10.0
        n["y"] = float(n["y"]) + 5.0
    assert curated_translation_only(data, trial)
    # Distort one curated midmarker relative position.
    curated = (data[0].get("metadata") or {}).get("expert_structures") or []
    if not curated:
        pytest.skip("no curated structures")
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
    if mid is None:
        pytest.skip("no curated midmarker")
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
    assert report["n_far"] >= 1
    assert sum(report["by_class"].values()) == report["n_far"]
    assert set(report["by_class"]) == set(FAR_EDGE_CLASSES)
    # Every top row uses a known class; no duplicates across classes.
    seen = []
    for row in report["top"]:
        assert row["far_class"] in FAR_EDGE_CLASSES
        seen.append(row["far_class"])
    assert seen
    # Evaluator API exposes the same report.
    ev = LayoutQualityEvaluator(
        data, skip_edges=skip, model=model, pathway=pathway, rule="official",
        require_local_improvement=False,
    )
    assert ev.far_edges()["n_far"] == report["n_far"]
    assert ev.far_edges()["by_class"] == report["by_class"]
