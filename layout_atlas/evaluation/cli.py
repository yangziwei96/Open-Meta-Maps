"""CLI: score Open Meta Maps Escher output with the unified evaluation path.

  python3 project.py evaluate --gem MODEL.xml --map DIR/OpenMetaMaps_<stem>_result.json [--out DIR/evaluation.json]
  python3 project.py evaluate --gem MODEL.xml --map DIR/OpenMetaMaps_<stem>_result.json --reference REF.sbgn
  python3 project.py evaluate --batch MANIFEST.json --out results/evaluation/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# layout_atlas is on sys.path when launched via project.py / cli.py
from evaluation.score import score_paths


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _flat_row(report: dict) -> dict:
    b = report.get("benchmark") or {}
    geom = report.get("geometry") or {}
    g = geom.get("core") or {}
    d = geom.get("details") or {}
    cross = g.get("crossings") or {}
    coll = g.get("overlaps") or g.get("collisions") or {}
    long_lines = g.get("long_lines") or {}
    place = g.get("participant_placement") or {}
    cof = g.get("cofactor_clones") or {}
    stab = report.get("reproducibility") or report.get("stability_vs_compare") or {}
    return {
        "model": report.get("model"),
        "placement_share": (report.get("coverage") or {}).get("placement_share"),
        "runtime_seconds": report.get("runtime_seconds"),
        "crossings": b.get("crossings"),
        "crossing_density": b.get("crossing_density"),
        "overlapping_pairs": b.get("overlapping_pairs"),
        "overlap_area_share": b.get("overlap_area_share"),
        "participant_consistency": b.get("participant_consistency"),
        "inout_opposite_share": place.get("inout_opposite_share"),
        "inout_opposite_share_core": place.get("inout_opposite_share_core"),
        "species_in_other_compartment": b.get("species_in_other_compartment") if b.get("species_in_other_compartment") is not None else d.get("species_in_other_compartment"),
        "long_edge_share": b.get("long_edge_share"),
        "edge_len_max_rel": long_lines.get("edge_len_max_rel"),
        "edge_length_deviation": b.get("edge_length_deviation"),
        "aspect_ratio": (
            d.get("aspect_ratio")
            if d.get("aspect_ratio") is not None
            else (b.get("aspect_ratio") if b.get("aspect_ratio") is not None else b.get("aspect_ratio_bbox"))
        ),
        "area_per_node": b.get("area_per_node") if b.get("area_per_node") is not None else b.get("area_per_node_bbox"),
        "blank_space_share": (
            d.get("blank_space_share")
            if d.get("blank_space_share") is not None
            else ((d.get("blank_space") or {}).get("blank_space_share") if isinstance(d.get("blank_space"), dict) else b.get("blank_space_share"))
        ),
        "space_utilisation": (
            d.get("space_utilisation")
            if d.get("space_utilisation") is not None
            else ((d.get("composition") or {}).get("space_utilisation") if isinstance(d.get("composition"), dict) else None)
        ),
        "currency_hub_violations": cof.get("currency_hub_violations"),
        "currency_long_arc_share": cof.get("currency_long_arc_share"),
        "bends_per_arc": (d.get("line_shape") or {}).get("bends_per_arc"),
        "cycle_circularity": (d.get("cycles") or {}).get("cycle_circularity"),
        "cycle_direction_consistency": (d.get("cycles") or {}).get("cycle_direction_consistency"),
        "cycle_order_consistency": (d.get("cycles") or {}).get("cycle_order_consistency"),
        "main_segment_axis_share": d.get("main_segment_axis_share") or (d.get("orderliness") or {}).get("main_segment_axis_share"),
        "main_segment_axis_share_length": d.get("main_segment_axis_share_length") or (d.get("orderliness") or {}).get("main_segment_axis_share_length"),
        "reaction_axis_share": d.get("reaction_axis_share") or (d.get("orderliness") or {}).get("reaction_axis_share"),
        "chain_turn_median_deg": d.get("chain_turn_median_deg") or (d.get("orderliness") or {}).get("chain_turn_median_deg"),
        "metabolite_grid_share": d.get("metabolite_grid_share") or (d.get("orderliness") or {}).get("metabolite_grid_share"),
        "flow_main_dir_share": d.get("flow_main_dir_share"),
        "exchange_periphery_share": d.get("exchange_periphery_share"),
        "largest_empty_rect_share": (
            d.get("largest_empty_rect_share")
            if d.get("largest_empty_rect_share") is not None
            else b.get("largest_empty_rect_share")
        ),
        "unrelated_overlaps_total": (d.get("unrelated_overlaps") or {}).get("total"),
        "unrelated_node_overlaps": (d.get("unrelated_overlaps") or {}).get("node_overlaps"),
        "unrelated_edge_through_node": (d.get("unrelated_overlaps") or {}).get("edge_through_node"),
        "unrelated_collinear_edge_overlaps": (d.get("unrelated_overlaps") or {}).get("collinear_edge_overlaps"),
        "geom_crossings_counted": cross.get("counted", cross.get("total")),
        "geom_crossings_main": cross.get("main_metabolite"),
        "geom_crossings_currency": cross.get("currency_involved"),
        "geom_overlapping_pairs": coll.get("overlapping_pairs"),
        "geom_node_overlap_pairs": coll.get("node_overlap_pairs"),
        "geom_label_overlap_pairs": coll.get("label_overlap_pairs"),
        "structure_continuity": (
            (report.get("structure_continuity") or {}).get("contiguous_pair_share")
            if isinstance(report.get("structure_continuity"), dict)
            else (g.get("structure_continuity") or {}).get("contiguous_pair_share")
        ),
        "split_structures": (
            (report.get("structure_continuity") or {}).get("split_structures")
            if isinstance(report.get("structure_continuity"), dict)
            else (g.get("structure_continuity") or {}).get("split_structures")
        ),
        "procrustes": b.get("procrustes") if b.get("procrustes") is not None else stab.get("procrustes_disparity") or stab.get("procrustes_mean"),
        "distance_correlation": b.get("distance_correlation"),
        "knn_preservation": b.get("knn_preservation"),
        "reproducibility_mean_displacement": stab.get("mean_displacement") or stab.get("mean_displacement_mean"),
        "stress": b.get("stress"),
        "gdmetrix": (report.get("dependencies") or {}).get("gdmetrix"),
    }


def run_one(args) -> int:
    if not args.gem or not args.map:
        print("evaluate needs --gem and --map", file=sys.stderr)
        return 2
    report = score_paths(
        args.gem,
        args.map,
        reference=args.reference,
        compare_map=args.compare_map,
        include_geometry=not args.no_geometry,
    )
    out = Path(args.out) if args.out else Path(args.map).with_name("evaluation.json")
    _dump(out, report)
    row = _flat_row(report)
    print(json.dumps(row, indent=2))
    print(f"wrote {out}")
    return 0


def run_batch(args) -> int:
    """MANIFEST.json: list of {name, gem, map, reference?} or a directory of compose results."""
    manifest_path = Path(args.batch)
    out_dir = Path(args.out or "results/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_dir():
        jobs = []
        for map_json in sorted(manifest_path.glob("*/OpenMetaMaps_*_result.json")):
            # gem may be recorded in report.json
            gem = None
            report = map_json.with_name("report.json")
            if report.exists():
                rj = json.loads(report.read_text())
                gem = (rj.get("model") or {}).get("path") or rj.get("gem")
            upload = map_json.parent / "upload"
            if gem is None and upload.exists():
                files = list(upload.glob("model.*"))
                gem = str(files[0]) if files else None
            if gem:
                jobs.append(dict(name=map_json.parent.name, gem=gem, map=str(map_json)))
    else:
        jobs = json.loads(manifest_path.read_text())
    rows = []
    reports = []
    for job in jobs:
        name = job.get("name") or Path(job["gem"]).stem
        print(f"== {name}", flush=True)
        try:
            report = score_paths(
                job["gem"],
                job["map"],
                reference=job.get("reference"),
                compare_map=job.get("compare_map"),
                include_geometry=not args.no_geometry,
            )
        except Exception as ex:  # noqa: BLE001
            report = dict(model=name, error=f"{type(ex).__name__}: {ex}")
            print(f"  FAILED: {report['error']}", file=sys.stderr)
        _dump(out_dir / f"{name}.json", report)
        reports.append(report)
        if "error" not in report:
            rows.append(_flat_row(report))
    if rows:
        fields = list(rows[0])
        with (out_dir / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"summary: {out_dir / 'summary.csv'}")
    _dump(out_dir / "all.json", reports)
    return 0 if all("error" not in r for r in reports) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Score an Open Meta Maps Escher layout: Project 22 layout-benchmark "
            "fields plus the same final-geometry block used by compose "
            "(crossings, overlaps, continuity, blank/empty space)."
        ))
    p.add_argument("--gem", help="Input GEM: SBML (.xml/.sbml) or COBRA/BiGG JSON")
    p.add_argument("--map", help="Composed Escher OpenMetaMaps_<stem>_result.json to score")
    p.add_argument("--reference", help="Optional reference layout: SBGN, Escher JSON, or {id:[x,y]} JSON")
    p.add_argument("--compare-map", help="Optional second result map JSON for reproducibility / stability metrics")
    p.add_argument("--batch", help="Batch manifest JSON, or a folder of compose result directories")
    p.add_argument("--out", help="Write JSON report here (file for one map, directory for --batch)")
    p.add_argument("--no-geometry", action="store_true",
                   help="Skip layout_quality final-geometry (curve crossings, collisions, continuity)")
    args = p.parse_args(argv)
    if args.batch:
        return run_batch(args)
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
