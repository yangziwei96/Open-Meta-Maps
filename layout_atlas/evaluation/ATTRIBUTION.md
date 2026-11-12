# Attribution

Pathway-specific layout metrics in this package (`evaluation/metrics.py` overlaps,
participant consistency, compartment metrics, reference similarity; `pathway.py`
SBML graph reading; `escher_bridge.py` Escher map → drawing positions) are adapted
from [layout-benchmark](https://github.com/elixir2026-project22/layout-benchmark)
(`src/layout_benchmark/metrics.py`, `graph.py`, `external.py` `escher_drawing`,
`reference.py`), Copyright 2026 the layout-benchmark copyright holders, licensed
under the MIT License (see that repository's `LICENSE.md`).

Official layout-benchmark metric *definitions* for Escher maps are also
reimplemented in `layout_atlas/official_metrics.py` (clean-room; MIT definitions
from the same project at commit `837067d`). That shared module is the single
implementation used by the quality gate, the Layout quality panel, and
`evaluation.score` / `evaluation.score_map`.

Transfer-check field names (`currency_hub_violations`, `currency_long_arc_share`,
`inout_opposite_share`, `edge_len_max_rel`, `bends_per_arc`, `cycle_circularity`,
`cycle_direction`, `cycle_order_consistency`, `flow_main_dir_share`, …) follow
[layout-transfer-check](https://github.com/elixir2026-project22/layout-transfer-check)
and are reimplemented in `map_rules.py` / `layout_quality.py` (no GPL code copied).

`blank_space` and `unrelated_overlaps` are clean-room reimplementations of ideas from
[MetaCarto](https://github.com/forxhunter/MetaCarto) (`src/layout/metrics.py`,
CC BY 4.0). MetaCarto code is not copied; see `docs/layout_evaluation.md`.

**Optional imports** (never vendored into this repository):

- **shapely** — used by `evaluation/metrics.py` for pathway box-overlap and
  compartment metrics when installed; without it those fields are `None`.
  `official_metrics.py` also uses shapely's `Point.buffer(..., quad_segs=16)`
  when installed (same definition as layout-benchmark); a pure 64-gon fallback
  matches those areas when shapely is absent.
- **gdMetriX** (GPL-3.0) — imported by `evaluation/metrics.py` `general_metrics`
  when installed; without it general graph-drawing fields are `None`. Official
  Escher figures in `official_metrics.py` do not import gdMetriX.

Installing layout-benchmark separately also enables the optional external scorer
in `evaluation.score_map`.
