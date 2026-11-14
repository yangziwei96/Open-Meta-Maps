# Layout evaluation

Scores Open Meta Maps layouts against the
[layout-benchmark](https://github.com/elixir2026-project22/layout-benchmark) metrics and the
[layout-transfer-check](https://github.com/elixir2026-project22/layout-transfer-check) field
names. All default Atlas sources remain available during evaluation.

CLI (optional deps: `pip install -r layout_atlas/requirements-eval.txt`):

```sh
python3 project.py compose --gem examples/e_coli_core.xml --out results/eval_runs/e_coli_core --source-paths
python3 project.py evaluate --gem examples/e_coli_core.xml --map results/eval_runs/e_coli_core/OpenMetaMaps_e_coli_core_result.json --out results/evaluation/e_coli_core.json
```

One computation path:

1. **`layout_quality.evaluate`** — final-geometry block shared by the quality gate, the tool
   **Layout quality** panel, and `evaluate` (curve crossings with main/currency split, overlaps,
   long lines, participant placement ≥120°, cofactor clones, curated-structure continuity, plus
   a curated **Detailed metrics** table aligned with layout goals).
2. **`layout_atlas/evaluation/`** — pathway metrics aligned with layout-benchmark (MIT; see
   `ATTRIBUTION.md`) plus optional gdMetriX (GPL-3.0, **import only**, never vendored; CLI extras).

User-facing panel strings use **curated** (not “expert”). The 120° opposite-side threshold is the
transfer-check first guess. Benchmark `participant_consistency` (>90°) stays on the evaluate CLI
only (not duplicated on the panel).

gdMetriX straight-line `crossings` are skipped when the undirected graph has more than 1200
edges. For maps with more than 800 reactions the curve geometry block is skipped in `evaluate`
and should be read from compose’s `layout_issues.json` / `report.json`.

## Metric table (e_coli_core · iIT341)

Maps composed with `--source-paths` (Bézier / curated layout default). Geometry columns come from
`layout_quality` on the final `OpenMetaMaps_<stem>_result.json`.

Baseline (atlas-extraction-1 + compactness-aware select) vs +iCH360 (`--source-paths`):

| metric | e_coli_core main | e_coli_core +iCH360 | iIT341 main | iIT341 +iCH360 |
|---|---:|---:|---:|---:|
| curated_coverage_share | 0.7368 | 0.7368 | 0.3069 | 0.2924 |
| crossings (counted) | 46 | 46 | 299 | 239 |
| overlapping_pairs | 5 | 5 | 18 | 16 |
| unrelated_overlaps_total | 5 | 5 | 17 | 8 |
| blank_space_share | 0.5826 | 0.5826 | 0.9401 | 0.8605 |
| edge_len_max_rel | 12.187 | 12.187 | 27.975 | 27.871 |
| aspect_ratio (W/H) | 1.1726 | 1.1726 | 0.5394 | 0.8079 |
| runtime_s | 77.9 | 74.7 | 457.3 | 454.2 |
| iCH360 modules used | — | 0 | — | 7 |

Refresh notes (2026-11-11): rebased iCH360 onto atlas-extraction-1 + compactness-aware select and re-composed e_coli_core / iIT341. e_coli_core unchanged with iCH360 loaded. iIT341 selects 4×`full_map` + 3×`ions_metals_uptake` size-2 geometry windows; crossings/blank/aspect improve while coverage dips slightly. Historical evaluate CSV (not a copy of the table above): [`docs/evaluation/summary.csv`](evaluation/summary.csv).

**Test scope for this refresh:** only `e_coli_core` and `iIT341` (not iMM904 / iML1515).

## Panel cards

| Card | Definition |
|---|---|
| Connection crossings | algorithm×algorithm + algorithm×curated; tooltip splits main-metabolite vs currency-involved; curated×curated not counted |
| Overlaps | node + label overlap pairs; curves through nodes/labels in the subtitle |
| Long lines | length p90 in median reaction spacings; `edge_len_max_rel` = max/median |
| Participant placement | share of reactant/product pairs ≥120° apart (all and core-only) |
| Cofactor clones | `currency_hub_violations` (currency instance linking reactions farther than 4× median spacing); `currency_long_arc_share` (currency lines >3× spacing) |
| Curated-structure continuity | contiguous pair share of reused curated structures (3× median reaction spacing) |

Bézier coverage is a **warning** when below 100% (still a quality-gate hard check), not a card.

## Detailed metrics (panel)

Only metrics the layout optimises or should optimise (layout **targets**, including backlog).
Non-target diagnostics are hidden under Debug. Each row has a plain-language English label,
a one-sentence tip (higher/lower + target), and the responsible pipeline stage
(or **not yet optimised**). Values that clearly miss the target use a subtle
amber highlight.

| Group | Rows | Stage |
|---|---|---|
| Connections | Local crossing hotspot; Unrelated overlaps | Bézier routing · quality gate; spacing / relaxation · Bézier routing |
| Line shape | Detour; Hooks / sharp turns / self-intersections; Bends per connection | Bézier routing |
| Structure & flow | Exchange reactions on the periphery; Axis-aligned main segments; Axis-aligned reaction axes; Grid-aligned metabolites; Median chain turn angle; Cycle circularity / direction / order; Species in another compartment | composition & peripherals; **not yet optimised** (orderliness / octilinear snap; ring/cycle placement; compartment packing) |
| Space & shape | Map size; Aspect ratio vs target; Blank space / space utilisation; Largest empty rectangle; Isolated reactions; Most reactions in one cell | composition & packing |
| Reproducibility | Procrustes / mean displacement (when `evaluate --compare-map`) | **not yet optimised** (detail row) |
| Debug diagnostics (hidden by default) | Main flow-direction share | Debug diagnostic |
| Pipeline diagnostics (collapsed) | Composition-pass counts; Peripheral-module counts | composition & peripherals |

**Orderliness (not yet optimised):** length-weighted main-segment and reaction-axis shares within ±10° of
horizontal/vertical (cofactors excluded; resampling a curve does not change the segment score);
metabolite grid share (tolerance 0.02 × median nearest-neighbour main-metabolite distance, floor
1 px — small enough that the curated BiGG e_coli_core map stays clearly above our compose);
median turn angle at degree-2 main metabolites along linear chains. Panel goals cite the curated
BiGG e_coli_core ("Core metabolism") scores for these four metrics.

**One bounding box:** map size, aspect ratio (W/H) and blank / utilisation all use the
content-node bbox (metabolites + midmarkers). `space_utilisation = 1 − blank_space_share`
on the same blank-space raster.

## Structure continuity

| model | contiguous_pair_share | structures | split_structures | pairs |
|---|---:|---:|---:|---:|
| e_coli_core | 0.856 | 11 | 0 | 146 |
| iIT341 | 0.950 | 90 | 3 | 238 |

## Reproducibility / stability

`evaluate --compare-map` compares two maps with Procrustes residual displacement. Compose does
not yet expose a seed flag.

## blank_space and unrelated_overlaps (MetaCarto ideas)

Clean-room reimplementations of ideas from
[MetaCarto](https://github.com/forxhunter/MetaCarto) `src/layout/metrics.py` (CC BY 4.0).
No MetaCarto code is copied. Single implementation: `evaluation.metrics.blank_space`
(also exposed as `blank_space_from_positions` for component packing).

- **blank_space** — rasterise nodes, segments and labels at cell size ≈ 0.6 × median
  reaction spacing; a cell is blank if no ink lies within ≈ 1 × spacing. Report
  `blank_space_share`, `space_utilisation = 1 − blank`, and `largest_empty_rect_share`.
- **unrelated_overlaps** — node-on-node, edge-through-node, and collinear edge-on-edge
  overlaps between reactions that share no non-currency metabolite.

## Licence note

Pathway-specific metric ports: MIT (layout-benchmark). MetaCarto metric *ideas*
credited above (CC BY 4.0). gdMetriX remains optional GPL-3.0 — imported when
installed, never vendored into this repository.
