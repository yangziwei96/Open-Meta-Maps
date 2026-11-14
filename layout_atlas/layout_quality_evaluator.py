"""Shared layout-quality evaluator for compose-stage accept / reject.

Wired into compose today:
  - tip nudge (``composition.nudge_frontier_escher``): ``rule="tip"`` with
    ``snapshot`` / ``prefilter`` / ``evaluate`` (aspect is part of ``TIP_DIRS``)
  - hub split (``split_cofactor_hubs``): ``HUB_DIRS`` + ``metrics_not_worse``
  - peripheral / block relocation (``quality_gate.peripheral_pass``,
    ``block_relocation._accept`` when not balanced): ``rule="peripheral"``
  - orthogonal routing (``orthogonal_route._gate_ok`` and length-deviation
    caps): ``ORTHO_DIRS`` + ``metrics_not_worse``
  - new ``inst:`` / structure moves (``instances.structure_local``,
    ``instances.optimize``, relocation / peripheral): ``rule="instance"``
    with ``INSTANCE_DIRS`` (participant side-consistency, new-only)

Shared tolerance: ``METRIC_EPS``.

API: ``snapshot()`` → ``evaluate()`` → ``commit()`` / ``rollback()`` /
``prefer()``.  Callers may cheap-reject with ``prefilter()`` first; final
acceptance uses ``evaluate()``.

Accept rule (three tiers, no weighted sum):
  1. Hard constraints reject immediately (skipped for tip / peripheral;
     ``rule="instance"`` promotes side-consistency regressions to hard).
     Includes no-new ``curves_through_nodes`` (edge-through-node) vs snapshot —
     curated residuals are allowed; new algorithm geometry must not add hits.
  2. None of the directed metrics may worsen by more than ``METRIC_EPS``.
  3. The candidate must improve its local objective (Pareto on the gated
     metrics when no scalar objective is set), unless the rule opts out.
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

METRIC_EPS = 1e-4

# Hub ``score_map`` columns plus pipeline crossings and coverage ratios.
# Values come from ``layout_gate.score_map`` / ``orthogonal_route.gate_metrics``
# and ``pipeline_crossings`` — not from an external official scorer.
SCORE_DIRS: dict[str, str] = {
    "crossings": "lower",
    "overlapping_pairs": "lower",
    "overlap_area_share": "lower",
    "participant_consistency": "higher",
    "edge_length_deviation": "lower",
    "edge_orthogonality": "higher",
    "angular_resolution": "lower",
    "crossing_angular_resolution": "higher",
    "species_drawn": "higher",
    "reactions_drawn": "higher",
    "curves_through_nodes": "lower",
    # drawn_per_node is undirected in the benchmark; still reported, not gated.
}

SCORE_METRICS: tuple[str, ...] = tuple(SCORE_DIRS) + ("drawn_per_node",)

# Tip-nudge metric directions (subset + aspect_ratio).
TIP_DIRS: dict[str, str] = {
    "crossings": "lower",
    "overlapping_pairs": "lower",
    "angular_resolution": "lower",
    "edge_length_deviation": "lower",
    "participant_consistency": "higher",
    "edge_orthogonality": "higher",
    "aspect_ratio": "higher",
    "curves_through_nodes": "lower",
}

# Hub-split gate columns (single copy; ``layout_gate`` imports this).
HUB_DIRS: dict[str, str] = {
    "crossings": "lower",
    "crossing_density": "higher",
    "crossing_angular_resolution": "higher",
    "edge_length_deviation": "lower",
    "angular_resolution": "lower",
    "edge_orthogonality": "higher",
    "aspect_ratio": "higher",
    "homogeneity": "lower",
    "overlapping_pairs": "lower",
    "overlap_area_share": "lower",
    "participant_consistency": "higher",
    "curves_through_nodes": "lower",
}

ORTHO_DIRS: dict[str, str] = {
    "crossings": "lower",
    "edge_length_deviation": "lower",
    "angular_resolution": "lower",
    "participant_consistency": "higher",
    "curves_through_nodes": "lower",
}

# New ``inst:`` creation and structure moves: relative (new-only) side-consistency
# gate. Existing curated residuals below 1.0 are allowed; regressions are not.
INSTANCE_DIRS: dict[str, str] = {
    "participant_consistency": "higher",
}


def metrics_not_worse(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    dirs: Mapping[str, str],
    eps: float = METRIC_EPS,
) -> bool:
    """True when no directed metric in ``dirs`` worsens beyond ``eps``."""
    for key, better in dirs.items():
        bv, av = before.get(key), after.get(key)
        if bv is None or av is None:
            continue
        if better == "lower" and float(av) > float(bv) + eps:
            return False
        if better == "higher" and float(av) < float(bv) - eps:
            return False
    return True


def metric_regressions(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    dirs: Mapping[str, str],
    eps: float = METRIC_EPS,
) -> list[str]:
    """Names of directed metrics that worsen beyond ``eps``."""
    bad = []
    for key, better in dirs.items():
        bv, av = before.get(key), after.get(key)
        if bv is None or av is None:
            continue
        if better == "lower" and float(av) > float(bv) + eps:
            bad.append(key)
        elif better == "higher" and float(av) < float(bv) - eps:
            bad.append(key)
    return bad


def dominates(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    dirs: Mapping[str, str],
    eps: float = METRIC_EPS,
) -> bool:
    """Pareto: ``a`` is at least as good as ``b`` on every directed metric and better on one."""
    saw_better = False
    for key, better in dirs.items():
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None:
            continue
        af, bf = float(av), float(bv)
        if better == "lower":
            if af > bf + eps:
                return False
            if af < bf - eps:
                saw_better = True
        else:
            if af < bf - eps:
                return False
            if af > bf + eps:
                saw_better = True
    return saw_better


def _skip_from_data(data, skip_edges=None):
    meta = (data[0].get("metadata") or {}) if isinstance(data[0], dict) else {}
    if skip_edges is not None:
        return list(skip_edges)
    return list(meta.get("preserved_edges") or [])


def pipeline_crossings(data, skip_edges=None) -> int:
    """Headline crossing count on the pipeline edge set (with skip filters)."""
    import layout_quality as lq

    skip = _skip_from_data(data, skip_edges)
    return int(lq.CrossingIndex(data, skip).headline())


def _content_bbox(body):
    xs, ys = [], []
    for n in (body.get("nodes") or {}).values():
        if n.get("node_type") in ("metabolite", "midmarker", "multimarker"):
            xs.append(float(n["x"]))
            ys.append(float(n["y"]))
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs), max(ys)


def canvas_area(data) -> float:
    c = (data[1].get("canvas") or {}) if isinstance(data, list) else {}
    return float(c.get("width") or 0.0) * float(c.get("height") or 0.0)


def content_area(data) -> float:
    x0, y0, x1, y1 = _content_bbox(data[1])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def cropped_label_count(data) -> int:
    """Labels whose anchor falls outside the map canvas."""
    body = data[1]
    c = body.get("canvas") or {}
    x0 = float(c.get("x") or 0.0)
    y0 = float(c.get("y") or 0.0)
    x1 = x0 + float(c.get("width") or 0.0)
    y1 = y0 + float(c.get("height") or 0.0)
    if x1 <= x0 or y1 <= y0:
        return 0
    n = 0
    for node in (body.get("nodes") or {}).values():
        if "label_x" not in node or "label_y" not in node:
            continue
        lx, ly = float(node["label_x"]), float(node["label_y"])
        if lx < x0 or lx > x1 or ly < y0 or ly > y1:
            n += 1
    for rx in (body.get("reactions") or {}).values():
        if "label_x" not in rx or "label_y" not in rx:
            continue
        lx, ly = float(rx["label_x"]), float(rx["label_y"])
        if lx < x0 or lx > x1 or ly < y0 or ly > y1:
            n += 1
    return n


def previous_pinned_node_ids(data, previous_anchors) -> set:
    """Escher node ids locked by ``--previous`` ``r:`` / ``m:`` pins."""
    if not previous_anchors:
        return set()
    body = data[1]
    nodes = body.get("nodes") or {}
    out = set()
    mid_of_rx = {}
    mets_of = defaultdict(list)
    for nid, n in nodes.items():
        nt = n.get("node_type")
        if nt == "metabolite" and n.get("bigg_id"):
            mets_of[n["bigg_id"]].append(nid)
    for rxn in (body.get("reactions") or {}).values():
        rid = rxn.get("bigg_id")
        if not rid:
            continue
        for seg in (rxn.get("segments") or {}).values():
            for end in ("from_node_id", "to_node_id"):
                nid = seg.get(end)
                n = nodes.get(nid) or {}
                if n.get("node_type") == "midmarker":
                    mid_of_rx[rid] = nid
                    break
    for key in previous_anchors:
        sk = str(key)
        if sk.startswith("r:"):
            nid = mid_of_rx.get(sk[2:])
            if nid is not None:
                out.add(nid)
        elif sk.startswith("m:"):
            out.update(mets_of.get(sk[2:]) or [])
    return out


def _structure_reaction_sets(data) -> list[tuple[str, frozenset]]:
    rows = []
    for s in (data[0].get("metadata") or {}).get("expert_structures") or []:
        rids = frozenset(s.get("reactions") or s.get("reaction_ids") or [])
        if rids:
            rows.append((str(s.get("id") or s.get("code") or len(rows)), rids))
    return rows


def _nodes_for_reactions(data, rids: frozenset) -> list[str]:
    body = data[1]
    nodes = body.get("nodes") or {}
    out = []
    for rxn in (body.get("reactions") or {}).values():
        if rxn.get("bigg_id") not in rids:
            continue
        for seg in (rxn.get("segments") or {}).values():
            for end in ("from_node_id", "to_node_id"):
                nid = seg.get(end)
                if nid in nodes and nid not in out:
                    out.append(nid)
    return out


def _relative_xy(data, nids: Sequence[str]) -> dict[str, tuple[float, float]] | None:
    nodes = data[1].get("nodes") or {}
    pts = []
    for nid in nids:
        n = nodes.get(nid)
        if not n:
            return None
        pts.append((nid, float(n["x"]), float(n["y"])))
    if not pts:
        return {}
    cx = sum(p[1] for p in pts) / len(pts)
    cy = sum(p[2] for p in pts) / len(pts)
    return {nid: (x - cx, y - cy) for nid, x, y in pts}


def curated_translation_only(before, after, eps: float = METRIC_EPS) -> bool:
    """True when every curated structure keeps relative geometry (translate only)."""
    for _sid, rids in _structure_reaction_sets(before):
        nids = _nodes_for_reactions(before, rids)
        if not nids:
            continue
        rb = _relative_xy(before, nids)
        ra = _relative_xy(after, nids)
        if rb is None or ra is None or set(rb) != set(ra):
            return False
        for nid, (x0, y0) in rb.items():
            x1, y1 = ra[nid]
            if abs(x1 - x0) > eps or abs(y1 - y0) > eps:
                return False
    return True


def pins_unchanged(before, after, pinned: Iterable[str], eps: float = METRIC_EPS) -> bool:
    """True when every pinned node keeps its absolute position."""
    nb = before[1].get("nodes") or {}
    na = after[1].get("nodes") or {}
    for nid in pinned:
        b, a = nb.get(nid), na.get(nid)
        if b is None or a is None:
            return False
        if abs(float(a["x"]) - float(b["x"])) > eps or abs(float(a["y"]) - float(b["y"])) > eps:
            return False
    return True


def coverage_metrics(data, model) -> dict[str, float | None]:
    """species_drawn / reactions_drawn / drawn_per_node from Escher glyphs vs the model."""
    if not model:
        return dict(species_drawn=None, reactions_drawn=None, drawn_per_node=None)
    body = data[1]
    drawn_mets = []
    for n in (body.get("nodes") or {}).values():
        if n.get("node_type") == "metabolite" and n.get("bigg_id"):
            drawn_mets.append(n["bigg_id"])
    drawn_rx = []
    for rx in (body.get("reactions") or {}).values():
        if rx.get("bigg_id"):
            drawn_rx.append(rx["bigg_id"])
    model_mets = {m["id"] for m in (model.get("metabolites") or [])}
    model_rx = {r["id"] for r in (model.get("reactions") or [])}
    uniq_m = set(drawn_mets) & model_mets if model_mets else set(drawn_mets)
    uniq_r = set(drawn_rx) & model_rx if model_rx else set(drawn_rx)
    species_drawn = (len(uniq_m) / len(model_mets)) if model_mets else None
    reactions_drawn = (len(uniq_r) / len(model_rx)) if model_rx else None
    drawn_nodes = len(drawn_mets) + len(drawn_rx)
    unique_drawn = len(set(drawn_mets) | set(drawn_rx))
    drawn_per = (drawn_nodes / unique_drawn) if unique_drawn else None
    return dict(
        species_drawn=species_drawn,
        reactions_drawn=reactions_drawn,
        drawn_per_node=drawn_per,
    )


def measure_scores(
    data,
    *,
    skip_edges=None,
    model=None,
    pathway=None,
) -> dict[str, Any]:
    """Score-map + pipeline-crossing metric vector for the three-tier accept rule.

    Failures propagate; callers that need a soft path must catch themselves.
    """
    import layout_gate as lg
    import layout_quality as lq
    import orthogonal_route as ortho

    skip = _skip_from_data(data, skip_edges)
    out: dict[str, Any] = {
        "crossings": float(pipeline_crossings(data, skip)),
        # Panel / quality-gate geometry (not drawing-graph score_map).
        "curves_through_nodes": float(lq.curves_through_nodes_count(data, skip)),
    }
    if pathway is not None:
        scored = lg.score_map(pathway, data)
        for k in (
            "overlapping_pairs",
            "overlap_area_share",
            "participant_consistency",
            "edge_length_deviation",
            "edge_orthogonality",
            "angular_resolution",
            "crossing_angular_resolution",
        ):
            out[k] = scored.get(k)
    else:
        node_ov, _lab = lq.overlap_counts(data)
        out["overlapping_pairs"] = float(node_ov)
        out["overlap_area_share"] = None
        gm = ortho.gate_metrics(data, pathway=None)
        out["participant_consistency"] = gm.get("participant_consistency")
        out["edge_length_deviation"] = gm.get("edge_length_deviation")
        out["angular_resolution"] = gm.get("angular_resolution")
        out["edge_orthogonality"] = ortho.piece_axis_alignment(data, pathway=None)
        out["crossing_angular_resolution"] = None
    out.update(coverage_metrics(data, model))
    return out


def measure_tip(data, *, pathway=None) -> dict[str, Any]:
    """Tip-nudge gate metrics; crossings match compose ``algorithm_crossings`` (no skip)."""
    import layout_quality as lq
    import orthogonal_route as ortho

    gm = ortho.gate_metrics(data, pathway=pathway)
    return dict(
        crossings=int(lq.algorithm_crossings(data)[0]),
        participant_consistency=gm.get("participant_consistency"),
        angular_resolution=gm.get("angular_resolution"),
        edge_length_deviation=gm.get("edge_length_deviation"),
        edge_orthogonality=ortho.piece_axis_alignment(data, pathway=pathway),
        aspect_ratio=_content_aspect(data[1]),
        overlapping_pairs=int(_glyph_overlaps(data)),
        curves_through_nodes=float(lq.curves_through_nodes_count(data)),
    )


def measure_scene_participant_consistency(model, scene) -> float | None:
    """Scene-level opposite-side share (reactant×product leave vectors, >90°).

    Same geometry as ``layout_gate._through_reaction_share`` / official
    ``participant_consistency``, but on compose ``scene`` keys (``r:`` /
    ``copy:`` / ``inst:`` / …) so ``inst:`` creation can be gated before export.
    """
    from core import participation_roles

    pos = scene.get("positions") or {}
    parts = scene.get("participants") or {}
    good = total = 0
    for r in model.get("reactions") or []:
        rid = r.get("id")
        if rid is None:
            continue
        rk = "r:" + rid
        if rk not in pos:
            continue
        origin = np.asarray(pos[rk], float)
        inward: list[np.ndarray] = []
        outward: list[np.ndarray] = []
        for mid, coeff in (r.get("metabolites") or {}).items():
            key = parts.get((rid, mid))
            if key is None or key not in pos:
                continue
            vec = np.asarray(pos[key], float) - origin
            for role in participation_roles(coeff):
                (inward if role < 0 else outward).append(vec)
        for vin in inward:
            for vout in outward:
                total += 1
                if float(np.dot(vin, vout)) < 0.0:
                    good += 1
    if total == 0:
        return None
    return good / total


def accept_participant_consistency(
    before: float | None,
    after: float | None,
    *,
    eps: float = METRIC_EPS,
) -> tuple[bool, str]:
    """New-only side-consistency gate via ``rule="instance"``.

    Returns ``(ok, reason)``. Absolute values below 1.0 are fine; only a
    regression beyond ``eps`` is rejected.
    """
    gate = LayoutQualityEvaluator(
        [],
        rule="instance",
        require_local_improvement=False,
        eps=eps,
    )
    gate.begin_report({"participant_consistency": before})
    decision = gate.evaluate(
        None, metrics={"participant_consistency": after},
    )
    return bool(decision.accepted), decision.reason


@dataclass
class Evaluation:
    """Result of evaluating one candidate against the snapshot baseline."""

    accepted: bool
    reason: str
    hard_ok: bool = True
    metrics_ok: bool = True
    local_ok: bool = True
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    hard_failures: list[str] = field(default_factory=list)
    metric_failures: list[str] = field(default_factory=list)
    local_delta: float | None = None


@dataclass
class _Snapshot:
    data: Any
    metrics: dict[str, Any]
    hard: dict[str, Any]
    local: Any = None


class LayoutQualityEvaluator:
    """Snapshot → evaluate → commit / rollback accept gate."""

    def __init__(
        self,
        data,
        *,
        skip_edges=None,
        previous_anchors=None,
        model=None,
        pathway=None,
        local_objective: Callable[[Any], float] | None = None,
        rule: str = "metrics",
        metric_dirs: Mapping[str, str] | None = None,
        require_local_improvement: bool = True,
        eps: float = METRIC_EPS,
    ):
        self.data = data
        self.skip_edges = _skip_from_data(data, skip_edges) if data else list(skip_edges or [])
        self.previous_anchors = previous_anchors or {}
        self.model = model
        self.pathway = pathway
        self.local_objective = local_objective
        self.rule = rule
        self.eps = float(eps)
        self.require_local_improvement = bool(require_local_improvement)
        if metric_dirs is not None:
            self.metric_dirs = dict(metric_dirs)
        elif rule == "tip":
            self.metric_dirs = dict(TIP_DIRS)
        elif rule == "instance":
            self.metric_dirs = dict(INSTANCE_DIRS)
        else:
            self.metric_dirs = dict(SCORE_DIRS)
        self._snap: _Snapshot | None = None
        self.accepted = 0
        self.rejected = 0
        self.reject_reasons: dict[str, int] = {}

    # --- measurement -----------------------------------------------------

    def measure(self, data) -> dict[str, Any]:
        if self.rule == "peripheral":
            return self._measure_peripheral(data)
        if self.rule == "tip":
            return measure_tip(data, pathway=self.pathway)
        return measure_scores(
            data,
            skip_edges=self.skip_edges,
            model=self.model,
            pathway=self.pathway,
        )

    def _measure_peripheral(self, data) -> dict[str, Any]:
        import quality_gate as qg

        row, _summary, _issues = qg.evaluate_map(data, self.skip_edges)
        return row

    def _hard_state(self, data) -> dict[str, Any]:
        import layout_quality as lq

        return dict(
            canvas=canvas_area(data),
            cropped=cropped_label_count(data),
            coverage=coverage_metrics(data, self.model),
            # No-new edge-through-node: absolute zero is not required of curated
            # residuals; algorithm accepts must not raise the count.
            curves_through_nodes=int(
                lq.curves_through_nodes_count(data, self.skip_edges)
            ),
        )

    def hard_failures(self, candidate) -> list[str]:
        """Hard-constraint failures of ``candidate`` vs the snapshot."""
        if self._snap is None:
            raise RuntimeError("snapshot() before evaluate()")
        import layout_quality as lq

        base = self._snap.data
        fails = []
        # Instance / structure-move gate may evaluate metrics-only (no map body).
        if self.rule == "instance" and (
            not isinstance(base, list) or not isinstance(candidate, list)
        ):
            return fails
        pinned = previous_pinned_node_ids(base, self.previous_anchors)
        if pinned and not pins_unchanged(base, candidate, pinned, self.eps):
            fails.append("pins_moved")
        if not curated_translation_only(base, candidate, self.eps):
            fails.append("curated_not_translation_only")
        cov_b = self._snap.hard.get("coverage") or {}
        if cov_b and isinstance(candidate, list):
            cov_a = coverage_metrics(candidate, self.model)
            for key in ("species_drawn", "reactions_drawn"):
                bv, av = cov_b.get(key), cov_a.get(key)
                if bv is None or av is None:
                    continue
                if float(av) + self.eps < float(bv) or float(av) + self.eps < 1.0:
                    if float(bv) >= 1.0 - self.eps and float(av) < 1.0 - self.eps:
                        fails.append("not_everything_drawn")
                        break
                    if float(av) + self.eps < float(bv):
                        fails.append("not_everything_drawn")
                        break
        if self._snap.hard and isinstance(candidate, list):
            if canvas_area(candidate) > float(self._snap.hard.get("canvas") or 0.0) + self.eps:
                fails.append("canvas_grew")
            if cropped_label_count(candidate) > int(self._snap.hard.get("cropped") or 0):
                fails.append("new_cropped_labels")
            through_b = int(self._snap.hard.get("curves_through_nodes") or 0)
            through_a = int(
                lq.curves_through_nodes_count(candidate, self.skip_edges)
            )
            if through_a > through_b:
                fails.append("new_curves_through_nodes")
        return fails

    # --- snapshot / evaluate / commit ------------------------------------

    def snapshot(self, local=None, *, copy_data: bool = True) -> dict[str, Any]:
        """Freeze baseline geometry + metrics. Returns the measured metrics.

        ``copy_data=False`` records metrics only (no map deepcopy) when hard
        constraint rollback is not needed.
        """
        metrics = self.measure(self.data)
        local_val = local
        if local_val is None and self.local_objective is not None:
            local_val = self.local_objective(self.data)
        self._snap = _Snapshot(
            data=copy.deepcopy(self.data) if copy_data else self.data,
            metrics=dict(metrics),
            hard=self._hard_state(self.data) if copy_data else {},
            local=local_val,
        )
        return dict(metrics)

    def begin_report(self, metrics: Mapping[str, Any] | None = None) -> None:
        """Start accept/reject bookkeeping without deep-copying the map."""
        self._snap = _Snapshot(
            data=self.data,
            metrics=dict(metrics or {}),
            hard={},
            local=None,
        )

    def prefilter(
        self,
        candidate,
        *,
        max_crossings: int | None = None,
        max_overlaps: int | None = None,
    ) -> Evaluation | None:
        """Cheap local reject. Returns an Evaluation when rejected, else None.

        Final acceptance must still go through ``evaluate()``.  For ``rule="tip"``
        also rejects when aspect_ratio worsens (same direction as ``TIP_DIRS``).
        """
        if self._snap is None:
            raise RuntimeError("snapshot() before prefilter()")
        if max_overlaps is not None and _glyph_overlaps(candidate) > max_overlaps:
            return self._reject("prefilter_overlaps")
        if self.rule == "tip":
            asp0 = float(self._snap.metrics.get("aspect_ratio") or 0.0)
            asp1 = _content_aspect(candidate[1])
            if asp1 + self.eps < asp0:
                return self._reject("prefilter_aspect")
        if max_crossings is not None:
            import layout_quality as lq

            # Tip cheap-reject uses algorithm_crossings (no skip), matching measure_tip.
            cross = (
                int(lq.algorithm_crossings(candidate)[0])
                if self.rule == "tip"
                else pipeline_crossings(candidate, self.skip_edges)
            )
            if cross > max_crossings:
                return self._reject("prefilter_crossings")
        return None

    def evaluate(
        self,
        candidate,
        *,
        metrics: Mapping[str, Any] | None = None,
        local=None,
        local_before=None,
        local_after=None,
        disconnected=None,
    ) -> Evaluation:
        """Full three-tier (or stage-rule) evaluation vs the snapshot."""
        if self._snap is None:
            raise RuntimeError("snapshot() before evaluate()")

        if self.rule == "peripheral":
            return self._evaluate_peripheral(candidate, metrics=metrics, disconnected=disconnected)
        if self.rule == "tip":
            # Tip: no-worse on TIP_DIRS (includes aspect); content shrink is the caller's gate.
            after = dict(metrics) if metrics is not None else self.measure(candidate)
            before = self._snap.metrics
            metric_fails = metric_regressions(before, after, self.metric_dirs, self.eps)
            ok = not metric_fails
            reason = "accepted" if ok else "metrics:" + ",".join(metric_fails)
            ev = Evaluation(ok, reason, True, ok, True, after, before, [], metric_fails)
            self._record(ev)
            return ev
        if self.rule == "instance":
            # New-only side consistency: regressions are hard; absolute <1.0 is fine.
            after = dict(metrics) if metrics is not None else self.measure(candidate)
            before = self._snap.metrics
            metric_fails = metric_regressions(before, after, self.metric_dirs, self.eps)
            hard_fails = []
            if "participant_consistency" in metric_fails:
                hard_fails.append("new_participant_inconsistency")
            # Keep pin / curated rigidity when a full map candidate is supplied.
            if isinstance(candidate, list) and isinstance(self._snap.data, list):
                hard_fails.extend(
                    f for f in self.hard_failures(candidate)
                    if f not in hard_fails
                )
            hard_ok = not hard_fails
            metrics_ok = not metric_fails
            ok = hard_ok and metrics_ok
            if not hard_ok:
                reason = "hard:" + ",".join(hard_fails)
            elif not metrics_ok:
                reason = "metrics:" + ",".join(metric_fails)
            else:
                reason = "accepted"
            ev = Evaluation(
                ok, reason, hard_ok, metrics_ok, True, after, before, hard_fails, metric_fails,
            )
            self._record(ev)
            return ev

        hard_fails = self.hard_failures(candidate)
        after = dict(metrics) if metrics is not None else self.measure(candidate)
        before = self._snap.metrics
        metric_fails = metric_regressions(before, after, self.metric_dirs, self.eps)

        local_ok, local_delta, local_reason = self._local_ok(
            candidate, local=local, local_before=local_before, local_after=local_after, after=after
        )

        hard_ok = not hard_fails
        metrics_ok = not metric_fails
        if not hard_ok:
            reason = "hard:" + ",".join(hard_fails)
            ev = Evaluation(
                False, reason, False, metrics_ok, local_ok, after, before, hard_fails, metric_fails, local_delta
            )
            self._record(ev)
            return ev
        if not metrics_ok:
            reason = "metrics:" + ",".join(metric_fails)
            ev = Evaluation(
                False, reason, True, False, local_ok, after, before, hard_fails, metric_fails, local_delta
            )
            self._record(ev)
            return ev
        if self.require_local_improvement and not local_ok:
            reason = local_reason or "no_local_improvement"
            ev = Evaluation(
                False, reason, True, True, False, after, before, hard_fails, metric_fails, local_delta
            )
            self._record(ev)
            return ev
        ev = Evaluation(
            True, "accepted", True, True, True, after, before, hard_fails, metric_fails, local_delta
        )
        self._record(ev)
        return ev

    def _evaluate_peripheral(self, candidate, *, metrics=None, disconnected=None) -> Evaluation:
        import quality_gate as qg

        if metrics is not None:
            new = dict(metrics)
        elif isinstance(candidate, dict) and "hard" in candidate and "crossings" in candidate:
            new = dict(candidate)
        else:
            new, _, _ = qg.evaluate_map(
                candidate, self.skip_edges, disconnected=disconnected if disconnected is not None else 0
            )
        old = self._snap.metrics
        ok, why = qg.accept(old, new)
        ev = Evaluation(
            bool(ok),
            why,
            hard_ok=not (new.get("hard", 0) > old.get("hard", 0)),
            metrics_ok=bool(ok) or "hard" not in why,
            local_ok=bool(ok),
            metrics=new,
            baseline_metrics=old,
        )
        self._record(ev)
        return ev

    def _local_ok(self, candidate, *, local, local_before, local_after, after) -> tuple[bool, float | None, str]:
        if local_before is not None and local_after is not None:
            delta = float(local_after) - float(local_before)
            return delta < -self.eps, delta, ("no_local_improvement" if delta >= -self.eps else "")
        if self.local_objective is not None:
            before = self._snap.local
            if before is None:
                before = self.local_objective(self._snap.data)
            after_v = local if local is not None else self.local_objective(candidate)
            delta = float(after_v) - float(before)
            return delta < -self.eps, delta, ("no_local_improvement" if delta >= -self.eps else "")
        if local is not None and self._snap.local is not None:
            delta = float(local) - float(self._snap.local)
            return delta < -self.eps, delta, ("no_local_improvement" if delta >= -self.eps else "")
        if dominates(after, self._snap.metrics, self.metric_dirs, self.eps):
            return True, None, ""
        if not self.require_local_improvement:
            return True, None, ""
        return False, None, "no_local_improvement"

    def prefer(self, a: Evaluation, b: Evaluation) -> int:
        """Constraint-first + Pareto. -1 if ``a`` better, 1 if ``b`` better, 0 else."""
        if a.hard_ok and not b.hard_ok:
            return -1
        if b.hard_ok and not a.hard_ok:
            return 1
        if a.accepted and not b.accepted:
            return -1
        if b.accepted and not a.accepted:
            return 1
        if dominates(a.metrics, b.metrics, self.metric_dirs, self.eps):
            return -1
        if dominates(b.metrics, a.metrics, self.metric_dirs, self.eps):
            return 1
        return 0

    def commit(self, candidate) -> None:
        """Install an accepted candidate as the new working map and baseline."""
        if isinstance(candidate, list) and len(candidate) >= 2:
            self.data[0], self.data[1] = candidate[0], candidate[1]
        self.snapshot()

    def rollback(self) -> None:
        """Restore working data from the last snapshot."""
        if self._snap is None:
            raise RuntimeError("snapshot() before rollback()")
        self.data[0], self.data[1] = copy.deepcopy(self._snap.data[0]), copy.deepcopy(self._snap.data[1])

    def report(self) -> dict[str, Any]:
        return dict(
            accepted=self.accepted,
            rejected=self.rejected,
            reject_reasons=dict(self.reject_reasons),
            rule=self.rule,
        )

    def far_edges(self, *, length_threshold: float | None = None, top: int = 20) -> dict[str, Any]:
        """Far-edge report on the working map (mutually exclusive classes)."""
        return far_edge_report(
            self.data, skip_edges=self.skip_edges, length_threshold=length_threshold, top=top
        )

    def _reject(self, reason: str) -> Evaluation:
        ev = Evaluation(False, reason, True, True, False, {}, self._snap.metrics if self._snap else {})
        self._record(ev)
        return ev

    def _record(self, ev: Evaluation) -> None:
        if ev.accepted:
            self.accepted += 1
        else:
            self.rejected += 1
            key = ev.reason.split(":")[0] if ev.reason else "rejected"
            self.reject_reasons[key] = self.reject_reasons.get(key, 0) + 1


def _content_aspect(body) -> float:
    x0, y0, x1, y1 = _content_bbox(body)
    w, h = abs(x1 - x0), abs(y1 - y0)
    if w <= 1e-12 or h <= 1e-12:
        return 1.0
    return float(min(w, h) / max(w, h))


def _glyph_overlaps(data) -> int:
    from composition import glyph_circle_overlapping_pairs

    return int(glyph_circle_overlapping_pairs(data))


# --- far-edge classification (mutually exclusive + exhaustive) -----------

# Priority order: first match wins. Every far edge gets exactly one label.
FAR_EDGE_CLASSES: tuple[str, ...] = (
    "within_structure_curated",
    "copy_reused_by_noncurated",
    "copy_reused_by_curated",
    "inst",
    "m",
    "copy_home_unknown",
    "side_other",
)


def _norm_structure_id(sid: str | None) -> str | None:
    """Strip ``#…`` suffixes so reaction-side and copy-side structure ids compare equal."""
    if not sid:
        return None
    return str(sid).split("#", 1)[0]


def _metabolite_node_keys(data) -> dict[str, str]:
    """Escher metabolite node id → scene instance key (copy:/inst:/m:/side:).

    Prefers ``metadata.metabolite_instances``. When that table is missing,
    rebuilds the same mapping ``drawing.export`` assigned: node id ``i`` is the
    ``i``-th key of ``sorted(glyph_identity keys ∪ {r:rid})``, verified against
    each metabolite's ``bigg_id``. No topology guessing.
    """
    out = {}
    table = (data[0].get("metadata") or {}).get("metabolite_instances") or {}
    for row in table.values():
        for inst in row.get("instances") or []:
            nid = inst.get("node")
            key = inst.get("key")
            if nid is not None and key:
                out[str(nid)] = str(key)
    if out:
        return out
    return _metabolite_node_keys_from_glyph_identity(data)


def _metabolite_node_keys_from_glyph_identity(data) -> dict[str, str]:
    """Exact export-id reconstruction from ``glyph_identity`` (no fallbacks)."""
    meta = (data[0].get("metadata") or {}) if isinstance(data[0], dict) else {}
    gi = meta.get("glyph_identity") or {}
    if not gi:
        return {}
    body = data[1]
    nodes = body.get("nodes") or {}
    rids = sorted(
        {rx.get("bigg_id") for rx in (body.get("reactions") or {}).values() if rx.get("bigg_id")}
    )
    all_keys = sorted(list(gi.keys()) + [f"r:{rid}" for rid in rids])
    out = {}
    for i, key in enumerate(all_keys):
        if key not in gi:
            continue
        nid = str(i)
        n = nodes.get(nid)
        if not n or n.get("node_type") != "metabolite":
            continue
        if n.get("bigg_id") != gi[key]:
            continue
        out[nid] = str(key)
    return out


def _reaction_structure_ids(data) -> dict[str, str]:
    """Reaction bigg_id → curated structure id (raw; compare via ``_norm_structure_id``)."""
    out = {}
    for s in (data[0].get("metadata") or {}).get("expert_structures") or []:
        sid = str(s.get("id") or s.get("code") or "")
        if not sid:
            continue
        for rid in s.get("reactions") or s.get("reaction_ids") or []:
            out[str(rid)] = sid
    return out


def _copy_structure_id(key: str) -> str | None:
    if not key.startswith("copy:"):
        return None
    parts = key.split(":")
    return parts[1] if len(parts) >= 3 else None


def _chord_length_px(edge, nodes) -> float | None:
    """Reaction-centre (midmarker) → metabolite Euclidean length."""
    mid = met = None
    for nid in edge.get("ends") or ():
        n = nodes.get(nid) or {}
        nt = n.get("node_type")
        if nt == "midmarker":
            mid = n
        elif nt == "metabolite":
            met = n
    if mid is None or met is None:
        return None
    return float(math.hypot(float(mid["x"]) - float(met["x"]), float(mid["y"]) - float(met["y"])))


def _met_end_node(edge, nodes) -> str | None:
    """Metabolite endpoint node id of a pipeline edge record."""
    for nid in edge.get("ends") or ():
        n = nodes.get(nid) or {}
        if n.get("node_type") == "metabolite":
            return nid
    return None


def classify_far_edge(
    edge,
    nodes,
    *,
    node_key: Mapping[str, str],
    rid_struct: Mapping[str, str],
) -> str:
    """Assign exactly one ``FAR_EDGE_CLASSES`` label to a pipeline edge.

    Copy reuse: a ``copy:`` glyph drawn for structure S is used by a reaction
    that is not a member of S — either a non-curated reaction
    (``copy_reused_by_noncurated``) or a reaction belonging to another curated
    structure (``copy_reused_by_curated``). Structure ids compare after stripping
    a ``#…`` suffix.
    """
    rid = str(edge.get("rid") or "")
    rx_struct = _norm_structure_id(rid_struct.get(rid))
    met_nid = _met_end_node(edge, nodes)
    key = node_key.get(str(met_nid), "") if met_nid is not None else ""
    copy_sid = _norm_structure_id(_copy_structure_id(key))

    if key.startswith("side:"):
        return "side_other"

    # Within-structure curated: reaction and its copy: glyph share structure S.
    if rx_struct and copy_sid and rx_struct == copy_sid:
        return "within_structure_curated"

    if key.startswith("copy:"):
        if not copy_sid:
            return "copy_home_unknown"
        if not rx_struct:
            return "copy_reused_by_noncurated"
        if copy_sid != rx_struct:
            return "copy_reused_by_curated"
        return "copy_home_unknown"

    if key.startswith("inst:"):
        return "inst"
    if key.startswith("m:"):
        return "m"

    return "side_other"


def _tukey_upper_fence(lengths: Sequence[float]) -> float:
    """Q3 + 1.5·IQR upper fence (linear percentiles)."""
    arr = np.asarray(lengths, float)
    q1, q3 = np.percentile(arr, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


def far_edge_report(
    data,
    *,
    skip_edges=None,
    length_threshold: float | None = None,
    top: int = 20,
) -> dict[str, Any]:
    """Far edges on core mid↔met connections (excluding ``side:`` endpoints).

    Default threshold is the Tukey upper fence (Q3 + 1.5·IQR) on chord lengths
    from reaction centre to metabolite instance. Pass ``length_threshold`` to
    override. Classes are mutually exclusive; counts sum to ``n_far``.
    """
    import layout_quality as lq

    skip = _skip_from_data(data, skip_edges)
    edges, nodes = lq.build_edges(data, skip)
    node_key = _metabolite_node_keys(data)
    rid_struct = _reaction_structure_ids(data)

    core = []
    lengths = []
    for e in edges:
        met_nid = _met_end_node(e, nodes)
        key = node_key.get(str(met_nid), "") if met_nid is not None else ""
        if key.startswith("side:"):
            continue
        L = _chord_length_px(e, nodes)
        if L is None:
            continue
        core.append(e)
        lengths.append(L)

    empty = {c: 0 for c in FAR_EDGE_CLASSES}
    if not lengths:
        return dict(
            n_edges=len(edges),
            n_core=0,
            n_far=0,
            length_threshold=0.0,
            by_class=empty,
            top=[],
            classes=list(FAR_EDGE_CLASSES),
            note="core mid↔met edges excluding side:; classes sum to n_far",
        )

    thresh = float(length_threshold) if length_threshold is not None else _tukey_upper_fence(lengths)
    far_rows = []
    for e, length in zip(core, lengths):
        if length < thresh:
            continue
        klass = classify_far_edge(e, nodes, node_key=node_key, rid_struct=rid_struct)
        met_nid = _met_end_node(e, nodes)
        far_rows.append(
            dict(
                reaction=e.get("rid"),
                metabolite=e.get("met"),
                length_px=round(float(length), 1),
                far_class=klass,
                instance_key=node_key.get(str(met_nid), "") if met_nid is not None else "",
                curated=bool(e.get("curated")),
            )
        )
    far_rows.sort(key=lambda r: -r["length_px"])
    by_class = {c: 0 for c in FAR_EDGE_CLASSES}
    for row in far_rows:
        by_class[row["far_class"]] = by_class.get(row["far_class"], 0) + 1
    return dict(
        n_edges=len(edges),
        n_core=len(core),
        n_far=len(far_rows),
        length_threshold=round(thresh, 1),
        by_class=by_class,
        top=far_rows[:top],
        classes=list(FAR_EDGE_CLASSES),
        note="core mid↔met edges excluding side:; classes sum to n_far",
    )


__all__ = [
    "METRIC_EPS",
    "SCORE_DIRS",
    "SCORE_METRICS",
    "TIP_DIRS",
    "HUB_DIRS",
    "ORTHO_DIRS",
    "INSTANCE_DIRS",
    "FAR_EDGE_CLASSES",
    "LayoutQualityEvaluator",
    "Evaluation",
    "metrics_not_worse",
    "metric_regressions",
    "dominates",
    "pipeline_crossings",
    "measure_scores",
    "measure_tip",
    "measure_scene_participant_consistency",
    "accept_participant_consistency",
    "coverage_metrics",
    "pins_unchanged",
    "curated_translation_only",
    "previous_pinned_node_ids",
    "cropped_label_count",
    "canvas_area",
    "content_area",
    "classify_far_edge",
    "far_edge_report",
]
