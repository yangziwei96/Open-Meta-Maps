"""Shared layout-quality evaluator for compose-stage accept / reject.

One implementation for the pipeline quality gate, gap-packing ``_GATE_EPS``
checks, tip nudge, hub split, hotspot repair, and orthogonal routing.

Canonical measurements:
  - crossings: ``layout_quality.CrossingIndex(data, skip_edges).headline()``
    (pipeline edge set, filters and geometry)
  - pins: both ``r:`` and ``m:`` ``--previous`` anchors unchanged vs snapshot
  - curated structures: translation-only vs snapshot

API: ``snapshot()`` a baseline, ``evaluate()`` a candidate, then ``commit()``
or ``rollback()``.  Final acceptance always uses the full evaluation; callers
may cheap-reject with ``prefilter()`` first.

Official rule (three tiers, no weighted sum):
  1. Hard constraints reject immediately.
  2. None of the 11 official metrics may worsen by more than ``METRIC_EPS``.
  3. The candidate must improve its local objective.
Candidates are compared constraint-first, then by Pareto dominance on the
official metrics.
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

METRIC_EPS = 1e-4

# Official layout-benchmark directions used by the compose accept rule.
OFFICIAL_DIRS: dict[str, str] = {
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
    # drawn_per_node is undirected in the benchmark; still reported, not gated.
}

OFFICIAL_METRICS: tuple[str, ...] = tuple(OFFICIAL_DIRS) + ("drawn_per_node",)

# Gap-packing / tip-nudge metric directions (subset + aspect_ratio).
TIP_DIRS: dict[str, str] = {
    "crossings": "lower",
    "overlapping_pairs": "lower",
    "angular_resolution": "lower",
    "edge_length_deviation": "lower",
    "participant_consistency": "higher",
    "edge_orthogonality": "higher",
    "aspect_ratio": "higher",
}

# Hub-split gate columns (layout_gate._GATE_DIRS).
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
}

ORTHO_DIRS: dict[str, str] = {
    "crossings": "lower",
    "edge_length_deviation": "lower",
    "angular_resolution": "lower",
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


def curated_structure_node_ids(data) -> set:
    """Escher node ids that belong to curated-structure reactions."""
    curated_rx = set()
    for s in (data[0].get("metadata") or {}).get("expert_structures") or []:
        curated_rx.update(s.get("reactions") or s.get("reaction_ids") or [])
    if not curated_rx:
        return set()
    body = data[1]
    nodes = body.get("nodes") or {}
    out = set()
    for rxn in (body.get("reactions") or {}).values():
        if rxn.get("bigg_id") not in curated_rx:
            continue
        for seg in (rxn.get("segments") or {}).values():
            for end in ("from_node_id", "to_node_id"):
                nid = seg.get(end)
                if nid is not None and nid in nodes:
                    out.add(nid)
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


def measure_official(
    data,
    *,
    skip_edges=None,
    model=None,
    pathway=None,
) -> dict[str, Any]:
    """Full official metric vector used by the three-tier accept rule."""
    import layout_gate as lg
    import layout_quality as lq

    skip = _skip_from_data(data, skip_edges)
    out: dict[str, Any] = {
        "crossings": float(pipeline_crossings(data, skip)),
    }
    # Overlaps + geometric metrics from the hub drawing graph when a pathway is available.
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
        try:
            import orthogonal_route as ortho

            gm = ortho.gate_metrics(data, pathway=None)
            out["participant_consistency"] = gm.get("participant_consistency")
            out["edge_length_deviation"] = gm.get("edge_length_deviation")
            out["angular_resolution"] = gm.get("angular_resolution")
            out["edge_orthogonality"] = ortho.piece_axis_alignment(data, pathway=None)
        except Exception:
            out.setdefault("participant_consistency", None)
            out.setdefault("edge_length_deviation", None)
            out.setdefault("angular_resolution", None)
            out.setdefault("edge_orthogonality", None)
        out["crossing_angular_resolution"] = None
    out.update(coverage_metrics(data, model))
    return out


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
        rule: str = "official",
        metric_dirs: Mapping[str, str] | None = None,
        cross_tol: float = 0.0,
        require_local_improvement: bool = True,
        eps: float = METRIC_EPS,
    ):
        self.data = data
        self.skip_edges = _skip_from_data(data, skip_edges)
        self.previous_anchors = previous_anchors or {}
        self.model = model
        self.pathway = pathway
        self.local_objective = local_objective
        self.rule = rule
        self.eps = float(eps)
        self.cross_tol = float(cross_tol)
        self.require_local_improvement = bool(require_local_improvement)
        if metric_dirs is not None:
            self.metric_dirs = dict(metric_dirs)
        elif rule == "tip":
            self.metric_dirs = dict(TIP_DIRS)
        elif rule == "hub":
            self.metric_dirs = dict(HUB_DIRS)
        elif rule == "ortho":
            self.metric_dirs = dict(ORTHO_DIRS)
        else:
            self.metric_dirs = dict(OFFICIAL_DIRS)
        self._snap: _Snapshot | None = None
        self.accepted = 0
        self.rejected = 0
        self.reject_reasons: dict[str, int] = {}

    # --- measurement -----------------------------------------------------

    def measure(self, data) -> dict[str, Any]:
        if self.rule == "peripheral":
            return self._measure_peripheral(data)
        if self.rule == "tip":
            return self._measure_tip(data)
        if self.rule in ("hub", "hotspot"):
            # Hub/hotspot callers usually pass precomputed metric dicts; still
            # expose official crossings for hard checks / reports.
            if self.pathway is not None and self.rule == "hub":
                import layout_gate as lg

                m = lg.score_map(self.pathway, data)
                m["crossings"] = float(pipeline_crossings(data, self.skip_edges))
                return m
            return dict(crossings=float(pipeline_crossings(data, self.skip_edges)))
        return measure_official(
            data,
            skip_edges=self.skip_edges,
            model=self.model,
            pathway=self.pathway,
        )

    def _measure_peripheral(self, data) -> dict[str, Any]:
        import quality_gate as qg

        row, _summary, _issues = qg.evaluate_map(data, self.skip_edges)
        return row

    def _measure_tip(self, data) -> dict[str, Any]:
        import orthogonal_route as ortho

        gm = ortho.gate_metrics(data, pathway=self.pathway)
        # Tip historically used algorithm_crossings without skip; consolidate
        # onto the pipeline CrossingIndex with skip_edges.
        return dict(
            crossings=float(pipeline_crossings(data, self.skip_edges)),
            participant_consistency=gm.get("participant_consistency"),
            angular_resolution=gm.get("angular_resolution"),
            edge_length_deviation=gm.get("edge_length_deviation"),
            edge_orthogonality=ortho.piece_axis_alignment(data, pathway=self.pathway),
            aspect_ratio=_content_aspect(data[1]),
            overlapping_pairs=float(_glyph_overlaps(data)),
        )

    def _hard_state(self, data) -> dict[str, Any]:
        pinned = previous_pinned_node_ids(data, self.previous_anchors)
        return dict(
            pinned=sorted(pinned),
            pin_xy={
                nid: (
                    float((data[1]["nodes"][nid])["x"]),
                    float((data[1]["nodes"][nid])["y"]),
                )
                for nid in pinned
                if nid in (data[1].get("nodes") or {})
            },
            canvas=canvas_area(data),
            content=content_area(data),
            cropped=cropped_label_count(data),
            coverage=coverage_metrics(data, self.model),
        )

    def hard_failures(self, candidate) -> list[str]:
        """Hard-constraint failures of ``candidate`` vs the snapshot."""
        if self._snap is None:
            raise RuntimeError("snapshot() before evaluate()")
        base = self._snap.data
        fails = []
        pinned = previous_pinned_node_ids(base, self.previous_anchors)
        if pinned and not pins_unchanged(base, candidate, pinned, self.eps):
            fails.append("pins_moved")
        if not curated_translation_only(base, candidate, self.eps):
            fails.append("curated_not_translation_only")
        cov_b = self._snap.hard["coverage"]
        cov_a = coverage_metrics(candidate, self.model)
        for key in ("species_drawn", "reactions_drawn"):
            bv, av = cov_b.get(key), cov_a.get(key)
            if bv is None or av is None:
                continue
            if float(av) + self.eps < float(bv) or float(av) + self.eps < 1.0:
                # Everything drawn: coverage must stay complete when it was complete.
                if float(bv) >= 1.0 - self.eps and float(av) < 1.0 - self.eps:
                    fails.append("not_everything_drawn")
                    break
                if float(av) + self.eps < float(bv):
                    fails.append("not_everything_drawn")
                    break
        if canvas_area(candidate) > self._snap.hard["canvas"] + self.eps:
            fails.append("canvas_grew")
        if cropped_label_count(candidate) > int(self._snap.hard["cropped"]):
            fails.append("new_cropped_labels")
        if self.rule == "peripheral":
            # Peripheral hard flag from evaluate_map (bezier coverage).
            row = candidate if isinstance(candidate, dict) and "hard" in candidate else None
            # checked in evaluate via metrics path
        return fails

    # --- snapshot / evaluate / commit ------------------------------------

    def snapshot(self, local=None, *, copy_data: bool = True) -> dict[str, Any]:
        """Freeze baseline geometry + metrics. Returns the measured metrics.

        ``copy_data=False`` records metrics only (no map deepcopy). Use that for
        stage bookkeeping when hard-constraint rollback is not needed, so
        budget-bound compose stages stay within their try counts.
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
        max_content_area: float | None = None,
        max_aspect_drop: float | None = None,
    ) -> Evaluation | None:
        """Cheap local reject. Returns an Evaluation when rejected, else None.

        Final acceptance must still go through ``evaluate()``.
        """
        if self._snap is None:
            raise RuntimeError("snapshot() before prefilter()")
        if max_content_area is not None and content_area(candidate) >= max_content_area:
            return self._reject("prefilter_content_area", candidate)
        if max_overlaps is not None and _glyph_overlaps(candidate) > max_overlaps:
            return self._reject("prefilter_overlaps", candidate)
        if max_aspect_drop is not None:
            asp0 = float(self._snap.metrics.get("aspect_ratio") or 0.0)
            asp1 = _content_aspect(candidate[1])
            if asp1 + self.eps < asp0 - float(max_aspect_drop):
                return self._reject("prefilter_aspect", candidate)
        if max_crossings is not None:
            if pipeline_crossings(candidate, self.skip_edges) > max_crossings:
                return self._reject("prefilter_crossings", candidate)
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
        if self.rule == "hub":
            return self._evaluate_hub(candidate, metrics=metrics)
        if self.rule == "hotspot":
            return self._evaluate_hotspot(
                candidate, metrics=metrics, local_before=local_before, local_after=local_after
            )
        if self.rule == "ortho":
            return self._evaluate_dirs(candidate, metrics=metrics, require_improve=False)

        # Official / tip: hard → official/tip metrics → local improvement.
        hard_fails = self.hard_failures(candidate)
        # Tip also forbids content growth (canvas content bbox).
        if self.rule == "tip":
            if content_area(candidate) >= content_area(self._snap.data) - 0.0:
                # Tip requires a strict content shrink; recorded separately when used
                # as a prefilter. Here treat non-shrink as local failure below.
                pass
        after = dict(metrics) if metrics is not None else self.measure(candidate)
        before = self._snap.metrics
        metric_fails = metric_regressions(before, after, self.metric_dirs, self.eps)
        if self.cross_tol and "crossings" in metric_fails:
            bv, av = before.get("crossings"), after.get("crossings")
            if bv is not None and av is not None and float(av) <= float(bv) + self.cross_tol:
                metric_fails = [k for k in metric_fails if k != "crossings"]

        local_ok, local_delta, local_reason = self._local_ok(
            candidate, local=local, local_before=local_before, local_after=local_after, after=after
        )
        if self.rule == "tip":
            # Tip gate is no-worse only (improvement is the content-area shrink
            # checked by the caller / prefilter); do not demand metric improvement.
            local_ok = True
            local_reason = ""

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

    def _evaluate_dirs(self, candidate, *, metrics=None, require_improve=False) -> Evaluation:
        after = dict(metrics) if metrics is not None else self.measure(candidate)
        before = self._snap.metrics
        metric_fails = metric_regressions(before, after, self.metric_dirs, self.eps)
        ok = not metric_fails
        if require_improve and ok:
            ok = dominates(after, before, self.metric_dirs, self.eps) or any(
                after.get(k) != before.get(k) for k in self.metric_dirs
            )
        reason = "accepted" if ok else ("metrics:" + ",".join(metric_fails) if metric_fails else "no_improvement")
        ev = Evaluation(ok, reason, True, not metric_fails, True, after, before, [], metric_fails)
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

    def _evaluate_hub(self, candidate, *, metrics=None) -> Evaluation:
        after = dict(metrics) if metrics is not None else self.measure(candidate)
        before = self._snap.metrics
        # Hub split: no regression vs original snapshot AND (caller may also
        # compare against a moving current score separately).
        ok = metrics_not_worse(before, after, self.metric_dirs, self.eps)
        fails = metric_regressions(before, after, self.metric_dirs, self.eps)
        reason = "accepted" if ok else "metrics:" + ",".join(fails)
        ev = Evaluation(ok, reason, True, ok, True, after, before, [], fails)
        self._record(ev)
        return ev

    def _evaluate_hotspot(self, candidate, *, metrics=None, local_before=None, local_after=None) -> Evaluation:
        # Hotspot local objective: headline crossings must drop; overlaps not up.
        import layout_quality as lq

        before_cross = (
            float(local_before)
            if local_before is not None
            else float(self._snap.metrics.get("crossings") or pipeline_crossings(self._snap.data, self.skip_edges))
        )
        if local_after is not None:
            after_cross = float(local_after)
        elif metrics and "crossings" in metrics:
            after_cross = float(metrics["crossings"])
        else:
            after_cross = float(pipeline_crossings(candidate, self.skip_edges))
        n_ov, l_ov = lq.overlap_counts(candidate)
        n0, l0 = lq.overlap_counts(self._snap.data)
        hard_fails = []
        if n_ov > n0:
            hard_fails.append("node_overlap")
        if l_ov > l0:
            hard_fails.append("label_overlap")
        local_ok = after_cross < before_cross
        if hard_fails:
            reason = hard_fails[0]
            ev = Evaluation(False, reason, False, True, local_ok, {"crossings": after_cross}, self._snap.metrics, hard_fails, [], after_cross - before_cross)
            self._record(ev)
            return ev
        if not local_ok:
            ev = Evaluation(False, "no_crossing_drop", True, True, False, {"crossings": after_cross}, self._snap.metrics, [], [], after_cross - before_cross)
            self._record(ev)
            return ev
        ev = Evaluation(True, "accepted", True, True, True, {"crossings": after_cross}, self._snap.metrics, [], [], after_cross - before_cross)
        self._record(ev)
        return ev

    def _local_ok(self, candidate, *, local, local_before, local_after, after) -> tuple[bool, float | None, str]:
        if local_before is not None and local_after is not None:
            delta = float(local_after) - float(local_before)
            # Lower-is-better local scalars (crossings, area, …).
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
        # Default local objective: Pareto improvement on the gated metrics.
        if dominates(after, self._snap.metrics, self.metric_dirs, self.eps):
            return True, None, ""
        if not self.require_local_improvement:
            return True, None, ""
        return False, None, "no_local_improvement"

    def accept(self, evaluation: Evaluation) -> bool:
        return bool(evaluation.accepted)

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

    def far_edges(self, *, p90: float | None = None, top: int = 20) -> dict[str, Any]:
        """Far-edge report on the working map (mutually exclusive classes)."""
        return far_edge_report(
            self.data, skip_edges=self.skip_edges, p90=p90, top=top
        )

    def _reject(self, reason: str, candidate) -> Evaluation:
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
    try:
        from composition import glyph_circle_overlapping_pairs

        return int(glyph_circle_overlapping_pairs(data))
    except Exception:
        import layout_quality as lq

        return int(lq.overlap_counts(data)[0])


# --- far-edge classification (mutually exclusive + exhaustive) -----------

# Priority order: first match wins. Every far edge gets exactly one label.
FAR_EDGE_CLASSES: tuple[str, ...] = (
    "within_structure_curated",  # curated geometry inside one structure
    "copy_reused_by_curated",  # copy: glyph shared with another curated structure
    "copy_reused_by_noncurated",  # copy: glyph used by a non-curated reaction
    "movable_inst_or_m",  # edge endpoint is a movable inst: or m: glyph
    "other",  # side: stubs and remaining algorithm edges
)


def _metabolite_node_keys(data) -> dict[str, str]:
    """Escher metabolite node id → scene instance key (copy:/inst:/m:/side:)."""
    out = {}
    table = (data[0].get("metadata") or {}).get("metabolite_instances") or {}
    for row in table.values():
        for inst in row.get("instances") or []:
            nid = inst.get("node")
            key = inst.get("key")
            if nid is not None and key:
                out[str(nid)] = str(key)
    return out


def _reaction_structure_ids(data) -> dict[str, str]:
    """Reaction bigg_id → curated structure id."""
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


def _edge_length_px(edge) -> float:
    pts = edge.get("pts")
    if pts is None:
        return 0.0
    arr = np.asarray(pts, float)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(arr, axis=0).T)))


def _met_end_node(edge, nodes) -> str | None:
    """Metabolite endpoint node id of a pipeline edge record."""
    for nid in edge.get("ends") or ():
        n = nodes.get(nid) or {}
        if n.get("node_type") == "metabolite":
            return nid
    ends = edge.get("ends") or ()
    return ends[1] if len(ends) > 1 else (ends[0] if ends else None)


def classify_far_edge(
    edge,
    nodes,
    *,
    node_key: Mapping[str, str],
    rid_struct: Mapping[str, str],
) -> str:
    """Assign exactly one ``FAR_EDGE_CLASSES`` label to a pipeline edge."""
    rid = str(edge.get("rid") or "")
    rx_struct = rid_struct.get(rid)
    met_nid = _met_end_node(edge, nodes)
    key = node_key.get(str(met_nid), "") if met_nid is not None else ""
    copy_sid = _copy_structure_id(key)

    # 1. Within-structure curated geometry (reaction + its copy: glyph share S).
    if rx_struct and copy_sid == rx_struct:
        return "within_structure_curated"
    if rx_struct and edge.get("curated") and not key.startswith(("copy:", "inst:", "m:", "side:")):
        return "within_structure_curated"

    # 2 / 3. copy: reused outside its home structure.
    if key.startswith("copy:"):
        if rx_struct and copy_sid and copy_sid != rx_struct:
            # Edge reaction is curated but belongs to a different structure.
            return "copy_reused_by_curated"
        if not rx_struct:
            return "copy_reused_by_noncurated"
        # Curated reaction in same map but copy has no / unknown home: treat as curated reuse.
        return "copy_reused_by_curated"

    # 4. Movable open instances.
    if key.startswith("inst:") or key.startswith("m:"):
        return "movable_inst_or_m"
    # Also flag when either end maps to inst:/m: (defensive).
    for nid in edge.get("ends") or ():
        k2 = node_key.get(str(nid), "")
        if k2.startswith("inst:") or k2.startswith("m:"):
            return "movable_inst_or_m"

    return "other"


def far_edge_report(
    data,
    *,
    skip_edges=None,
    p90: float | None = None,
    top: int = 20,
) -> dict[str, Any]:
    """Far edges (≥ p90 length) with a mutually exclusive class per edge.

    Counts by class sum to ``n_far`` (exhaustive). Categories do not overlap.
    """
    import layout_quality as lq

    skip = _skip_from_data(data, skip_edges)
    edges, nodes = lq.build_edges(data, skip)
    lengths = [_edge_length_px(e) for e in edges]
    if not lengths:
        empty = {c: 0 for c in FAR_EDGE_CLASSES}
        return dict(
            n_edges=0,
            n_far=0,
            p90=0.0,
            by_class=empty,
            top=[],
            classes=list(FAR_EDGE_CLASSES),
            note="mutually exclusive far-edge classes; counts sum to n_far",
        )
    thresh = float(p90) if p90 is not None else float(np.percentile(lengths, 90))
    node_key = _metabolite_node_keys(data)
    rid_struct = _reaction_structure_ids(data)
    far_rows = []
    for e, length in zip(edges, lengths):
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
        n_far=len(far_rows),
        p90=round(thresh, 1),
        by_class=by_class,
        top=far_rows[:top],
        classes=list(FAR_EDGE_CLASSES),
        note="mutually exclusive far-edge classes; counts sum to n_far",
    )


# --- legacy adapters (behaviour-identical wrappers) ----------------------

def relocation_gate_not_worse(before, after, eps=METRIC_EPS) -> bool:
    """Tip-nudge no-worse check (same directions as composition._GATE_EPS)."""
    return metrics_not_worse(before, after, TIP_DIRS, eps)


def no_regression(before, after, tol=METRIC_EPS) -> bool:
    """Hub-split no-regression check (same columns as layout_gate._GATE_DIRS)."""
    return metrics_not_worse(before, after, HUB_DIRS, tol)


def peripheral_accept(old, new):
    """Pipeline peripheral quality-gate accept (delegates to quality_gate.accept)."""
    import quality_gate as qg

    return qg.accept(old, new)


__all__ = [
    "METRIC_EPS",
    "OFFICIAL_DIRS",
    "OFFICIAL_METRICS",
    "TIP_DIRS",
    "HUB_DIRS",
    "ORTHO_DIRS",
    "FAR_EDGE_CLASSES",
    "LayoutQualityEvaluator",
    "Evaluation",
    "metrics_not_worse",
    "metric_regressions",
    "dominates",
    "pipeline_crossings",
    "measure_official",
    "coverage_metrics",
    "pins_unchanged",
    "curated_translation_only",
    "previous_pinned_node_ids",
    "curated_structure_node_ids",
    "cropped_label_count",
    "canvas_area",
    "content_area",
    "classify_far_edge",
    "far_edge_report",
    "relocation_gate_not_worse",
    "no_regression",
    "peripheral_accept",
]
