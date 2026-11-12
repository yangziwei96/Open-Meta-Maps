# Attribution

Official layout-benchmark metric *definitions* are reimplemented in
`layout_atlas/official_metrics.py` (clean-room; MIT definitions from
[layout-benchmark](https://github.com/elixir2026-project22/layout-benchmark)
at commit `837067d`). That shared module is the single implementation used by
the quality gate, the Layout quality panel, and `evaluation.score` /
`evaluation.score_map`.

Pathway graph reading (`pathway.py`), Escher position helpers
(`escher_bridge.py`), and optional external scoring (`score_map.py` importing
`layout_benchmark` when installed) remain adapted from the same project
(Copyright 2026 the layout-benchmark copyright holders, MIT — see that
repository's `LICENSE.md`).

Transfer-check field names (`currency_hub_violations`, `currency_long_arc_share`,
`inout_opposite_share`, `edge_len_max_rel`, `bends_per_arc`, `cycle_circularity`,
`cycle_direction`, `cycle_order_consistency`, `flow_main_dir_share`, …) follow
[layout-transfer-check](https://github.com/elixir2026-project22/layout-transfer-check)
and are reimplemented in `map_rules.py` / `layout_quality.py` (no GPL code copied).

`blank_space` and `unrelated_overlaps` are clean-room reimplementations of ideas from
[MetaCarto](https://github.com/forxhunter/MetaCarto) (`src/layout/metrics.py`,
CC BY 4.0). MetaCarto code is not copied; see `docs/layout_evaluation.md`.

gdMetriX is **not** included here and is never imported on the default evaluate
path (GPL-3.0). Install layout-benchmark separately if you want its external
scorer via `evaluation.score_map`.
