"""Single evaluation path: benchmark-aligned metrics + final-geometry (layout_quality)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import escher_bridge, findings, metrics, score_map
from .pathway import read_sbml
from .stability import position_displacement, reproducibility

Positions = dict[str, tuple[float, float]]


def _read_json(path: Path):
    return json.loads(path.read_text())


def _structures_near_map(map_path: Path | None, escher_map) -> list[dict] | None:
    """Load reused structures from map metadata or sibling modules.json."""
    meta = (escher_map[0].get("metadata") if isinstance(escher_map, list) else {}) or {}
    if meta.get("expert_structures"):
        return meta["expert_structures"]
    if map_path is None:
        return None
    modules = Path(map_path).with_name("modules.json")
    if not modules.exists():
        return None
    try:
        rows = _read_json(modules)
    except Exception:
        return None
    out = []
    for m in rows if isinstance(rows, list) else []:
        rids = m.get("reactions") or m.get("reaction_ids") or []
        if len(rids) >= 2:
            out.append(dict(id=m.get("module_id") or m.get("id") or m.get("code"), code=m.get("code"), reactions=list(rids)))
    return out or None


def _optional_layout_quality(escher_map, structures=None):
    """Reuse the existing final-geometry evaluator (quality gate); never duplicate it."""
    try:
        import layout_quality
    except ImportError:
        return None, None
    try:
        summary, issues = layout_quality.evaluate(escher_map, structures=structures)
        return summary, issues
    except Exception as ex:  # noqa: BLE001 — evaluation aid must not fail the whole score
        return dict(error=f"{type(ex).__name__}: {ex}"), None


def score_escher_map(
    gem_path: str | Path,
    map_path: str | Path | None = None,
    escher_map=None,
    reference: Positions | None = None,
    reference_path: str | Path | None = None,
    include_geometry: bool = True,
    normalise: bool = True,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Score one Escher map against its SBML model.

    Returns a single report dict with:
      - ``benchmark``: Project 22 metrics (pathway-specific always; gdMetriX when available)
      - ``geometry``: existing ``layout_quality.evaluate`` summary (curve crossings, collisions, …)
      - ``examples``: reaction ids / regions for the weakest spots
      - ``coverage`` / ``runtime_seconds``
    """
    gem_path = Path(gem_path)
    pathway = read_sbml(gem_path)
    if escher_map is None:
        if map_path is None:
            raise ValueError("map_path or escher_map is required")
        escher_map = escher_bridge.load_escher(map_path)
    t0 = time.perf_counter()
    pos = escher_bridge.escher_positions(pathway, escher_map, y_up=True)
    widths = {n: float(pathway.graph.nodes[n]["width"]) for n in pos if n in pathway.graph}
    if normalise:
        pos_metric = escher_bridge.normalise_positions(pos, widths)
    else:
        pos_metric = pos

    ref_pos = reference
    if reference_path is not None and ref_pos is None:
        ref_pos = _load_reference(pathway, reference_path)

    # Restrict pathway to placed nodes for fair graph metrics
    placed = [n for n in pathway.graph if n in pos_metric]
    sub = pathway.graph.subgraph(placed).copy()
    from .pathway import Pathway

    sub_pathway = Pathway(name=pathway.name, graph=sub, compartments=pathway.compartments, source=pathway.source)

    benchmark = metrics.all_benchmark_metrics(sub_pathway, pos_metric, reference=ref_pos)
    # Prefer shared official_metrics on the final Escher geometry (one implementation).
    official = score_map.score_official(escher_map)
    for key in (
        "crossings",
        "overlapping_pairs",
        "overlap_area_share",
        "participant_consistency",
        "edge_length_deviation",
        "edge_orthogonality",
        "species_in_other_compartment",
        "aspect_ratio",
    ):
        if official.get(key) is not None:
            benchmark[key] = official[key]
    benchmark["official"] = official
    if score_map.layout_benchmark_available():
        benchmark["layout_benchmark_external"] = score_map.score_with_layout_benchmark(gem_path, escher_map)
    geom_summary, geom_issues = (None, None)
    structures = _structures_near_map(Path(map_path) if map_path else None, escher_map)
    # Final-geometry (curve sampling) is O(segments²); skip automatically on large GEMs.
    # Continuity lives in layout_quality.evaluate — call it once there, or alone when geometry is skipped.
    n_rxn = len(pathway.reactions)
    continuity = None
    if include_geometry and n_rxn > 800:
        continuity = _structure_continuity_only(escher_map, structures)
        cont_core = None
        if isinstance(continuity, dict) and "error" not in continuity:
            cont_core = {k: continuity.get(k) for k in (
                "contiguous_pair_share", "pairs", "contiguous_pairs", "structures", "split_structures",
                "split_structure_share", "reaction_spacing_px", "adjacency_factor", "note")}
        geom_summary = dict(
            skipped=True,
            reason=f"{n_rxn} reactions: use layout_issues.json from compose, or run evaluate without skipping geometry on a smaller map",
            core=dict(structure_continuity=cont_core),
        )
    elif include_geometry:
        geom_summary, geom_issues = _optional_layout_quality(escher_map, structures=structures)
        if isinstance(geom_summary, dict) and "error" not in geom_summary:
            continuity = ((geom_summary.get("details") or {}).get("structure_continuity")
                          or (geom_summary.get("core") or {}).get("structure_continuity"))
        else:
            continuity = _structure_continuity_only(escher_map, structures)
    else:
        continuity = _structure_continuity_only(escher_map, structures)

    examples = findings.weak_spots(sub_pathway, pos_metric, issues=geom_issues)
    cov = escher_bridge.coverage(pathway, pos)
    seconds = runtime_seconds
    if seconds is None and map_path is not None:
        # Prefer compose report timing when present next to the map
        report = Path(map_path).with_name("report.json")
        if report.exists():
            try:
                rj = _read_json(report)
                fg = ((rj.get("layout_quality") or {}).get("final_geometry") or {})
                seconds = (
                    rj.get("seconds")
                    or (rj.get("timings") or {}).get("total")
                    or fg.get("seconds")
                )
            except Exception:
                seconds = None

    if isinstance(geom_summary, dict) and "error" not in geom_summary:
        details = geom_summary.setdefault("details", {})
        if seconds is not None:
            details.setdefault("runtime_seconds", seconds)
        # Prefer map-native aspect_ratio (W/H) in the shared geometry block.
        if details.get("aspect_ratio") is not None:
            benchmark["aspect_ratio"] = details["aspect_ratio"]
        if details.get("species_in_other_compartment") is not None and benchmark.get("species_in_other_compartment") is None:
            benchmark["species_in_other_compartment"] = details["species_in_other_compartment"]
        # area_per_node / participant_consistency stay on the evaluate CLI via
        # metrics.all_benchmark_metrics (not the Layout quality panel).

    return dict(
        model=pathway.name,
        gem=str(gem_path.resolve()),
        map=str(Path(map_path).resolve()) if map_path else None,
        coverage=cov,
        runtime_seconds=seconds,
        metric_seconds=round(time.perf_counter() - t0, 3),
        dependencies=dict(
            gdmetrix=metrics.gdmetrix_available(),
            shapely=metrics.shapely_available(),
            layout_benchmark=score_map.layout_benchmark_available(),
        ),
        benchmark=benchmark,
        geometry=geom_summary,
        structure_continuity=continuity,
        examples=examples,
        note=(
            "Development evaluation (not a fair tool comparison). "
            "Atlas sources are never excluded for evaluation overlap. "
            "Official figures from official_metrics (layout-benchmark definitions, MIT; "
            "gdMetriX never vendored); optional layout-benchmark import via score_map; "
            "transfer-check field names for cofactor/cycle/flow rules; "
            "participant consistency uses >90°."
        ),
    )


def _structure_continuity_only(escher_map, structures=None):
    """Continuity without the heavy curve-geometry pass (single implementation in evaluation.continuity)."""
    try:
        from evaluation.continuity import structure_continuity

        return structure_continuity(escher_map, structures=structures)
    except Exception as ex:  # noqa: BLE001
        return dict(error=f"{type(ex).__name__}: {ex}")


def score_paths(
    gem: str | Path,
    map_json: str | Path,
    reference: str | Path | None = None,
    compare_map: str | Path | None = None,
    include_geometry: bool = True,
) -> dict[str, Any]:
    """CLI-friendly entry: paths in, report dict out. Optional compare_map → stability block."""
    report = score_escher_map(
        gem,
        map_json,
        reference_path=reference,
        include_geometry=include_geometry,
    )
    if compare_map:
        pathway = read_sbml(gem)
        a = escher_bridge.escher_positions(pathway, escher_bridge.load_escher(map_json))
        b = escher_bridge.escher_positions(pathway, escher_bridge.load_escher(compare_map))
        report["stability_vs_compare"] = position_displacement(a, b)
        report["reproducibility"] = reproducibility([a, b])
        geom = report.get("geometry")
        if isinstance(geom, dict) and "error" not in geom:
            geom.setdefault("details", {})["reproducibility"] = report["stability_vs_compare"]
    return report


def _load_reference(pathway, path: str | Path) -> Positions | None:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = _read_json(path)
        # Escher map or {node: [x,y]}
        if isinstance(data, list) or (isinstance(data, dict) and "nodes" in data):
            return escher_bridge.escher_positions(pathway, data)
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    out[k] = (float(v[0]), float(v[1]))
            return out or None
    if path.suffix.lower() == ".sbgn":
        try:
            from .reference_sbgn import gallery_reference

            return gallery_reference(pathway, path)
        except Exception:
            return None
    return None
