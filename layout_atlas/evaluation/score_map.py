"""Optional external evaluator: import layout-benchmark when installed.

Primary scoring uses ``official_metrics`` / ``layout_quality`` (this repository).
When the ``layout_benchmark`` package is importable, ``score_with_layout_benchmark``
runs its ``all_metrics`` on an Escher drawing for cross-checks. Never vendors
gdMetriX.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import official_metrics as om


def layout_benchmark_available() -> bool:
    try:
        import layout_benchmark  # noqa: F401
        return True
    except ImportError:
        return False


def score_official(escher_map) -> dict[str, Any]:
    """In-repo official figures (shared with the Layout quality panel / gate)."""
    return om.score_escher_map(escher_map)


def score_with_layout_benchmark(
    gem_path: str | Path,
    escher_map,
    *,
    layout: str | None = "escher",
) -> dict[str, Any] | None:
    """Run the external layout-benchmark scorer when the package is installed.

    Returns None when layout-benchmark (or its hard deps) cannot be imported.
    """
    try:
        from layout_benchmark.external import escher_drawing
        from layout_benchmark.graph import read_sbml
        from layout_benchmark.metrics import all_metrics
    except ImportError:
        return None
    try:
        pathway = read_sbml(str(gem_path))
        drawing = escher_drawing(pathway, escher_map)
        return dict(all_metrics(drawing, pathway, layout=layout))
    except Exception as ex:  # noqa: BLE001 — optional external path must not break evaluate
        return dict(error=f"{type(ex).__name__}: {ex}")


def score_map(
    gem_path: str | Path,
    map_path: str | Path | None = None,
    escher_map=None,
    *,
    use_layout_benchmark: bool = True,
) -> dict[str, Any]:
    """Score one map: always compute in-repo official metrics; optionally compare.

    ``use_layout_benchmark`` tries the external package when True (default) and
    leaves ``external`` as None when it is missing.
    """
    if escher_map is None:
        if map_path is None:
            raise ValueError("map_path or escher_map is required")
        escher_map = json.loads(Path(map_path).read_text())
    official = score_official(escher_map)
    external = None
    if use_layout_benchmark:
        external = score_with_layout_benchmark(gem_path, escher_map)
    return dict(
        official=official,
        external=external,
        layout_benchmark_available=layout_benchmark_available(),
        note="Primary figures from official_metrics; layout-benchmark is optional",
    )
