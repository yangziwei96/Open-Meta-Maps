# layout_atlas: the Composer and the Atlas

This directory is the Python package behind Open Meta Maps. It builds a metabolic map from a genome-scale model by matching its reactions to curated community structures (the Atlas), assembling those structures into a skeleton, laying out the remaining reactions algorithmically and drawing the result with Bézier connections. Installation and the quick start are in the [root README](../README.md); method notes are in [`docs/`](../docs/).

## Entry points

| Entry | What it does |
|---|---|
| `python3 project.py tool` | local web tool (`tool_server.py`, page in `results/tool/index.html`): upload a model, run the pipeline, preview, download |
| `python3 project.py compose --gem MODEL.xml --out DIR --source-paths` | the same pipeline from the command line (`compose_cli.py` → `tool_pipeline.py`; Bézier/`curated` layout is the default) |
| `python3 project.py data list` / `fetch` | optional data packages, see [docs/data_packages.md](../docs/data_packages.md) |
| `python3 project.py setup` | one-time build of the edge routers (needs `g++` or `clang++`, see below) |
| `python3 project.py catalog` | rebuild the module catalog pages from the Atlas |

A run writes `OpenMetaMaps_<name>_result.json` (Escher), matching `.svg` / `.sbgn` / `.sbml` / `.celldesigner.xml`, plus `report.json`, `reaction_provenance.json`, `layout_issues.json`, `licenses.json`, `skeleton.json` and `modules.json` into the output folder (`<name>` is the model file stem). `reaction_provenance.json` labels every reaction as curated (community layout), topology-only suggestion, functional (EC) suggestion, bridge or generated. Report curated, topology-only and EC suggestions as separate rates.

## Pipeline and main modules

| Stage | Modules |
|---|---|
| Model input and identifiers | `model_io.py` (SBML via libSBML, COBRA / BiGG JSON), `identifier_bridge.py` (BiGG → ChEBI / EC tables from `data/identifier_bridge.json`), `../tools/build_identifier_bridge.py` (rebuilds them from the public models) |
| Matching and selection | `retrieval.py`, `semantic_transfer.py`, `ec_channel.py`, `ec_provenance.py`, `database.py` (reads `data/atlas.sqlite`) |
| Skeleton and global layout | `drawing.py`, `global_layout.py`, `pathway_backbone.py`, `completion.py` |
| Context and anchors for unmatched reactions | `context.py`, `completion_context.py`, `instances.py` (metabolite instances and protected shared connections), `orientation.py` |
| Whole-map composition | `composition.py` (recovers stray components, arranges independent modules in rows), `peripheral.py` (exchange and transport modules), `block_relocation.py` (moves curated blocks that sit far from the reactions they connect to) |
| Connections and labels | `edge_style.py` (Bézier connections from curated templates, joint routing per reaction), `reaction_templates.py`, `label_placement.py`, `line_render.py`, `routing.py` (libavoid) |
| Local repair | `hotspot_repair.py` (post-layout pass: worst_regions windows via shared `CrossingIndex`, incremental delta recount; accept/round caps from window crossing mass, stop after an idle pass; default on) |
| Quality | `layout_quality.py` (final-geometry for the quality gate: curve crossings, overlaps, …), `quality_gate.py` (accepts a change only if the final geometry is not worse), `evaluation/` (layout-benchmark metrics + the same final-geometry block; CLI `python3 project.py evaluate`) |
| Pages | `ui.py`, `navigation.py`, `gallery*.py` (tool UI is the committed `results/tool/index.html`) |

Files whose hashes are recorded by saved results (`routing.py`, `drawing.py`, `geometry.py`, `retrieval.py`, `semantic_transfer.py`, `celldesigner.py`) should not be edited casually.

## Switches

Maps are drawn with Bézier connections by default (`layout_style=curated`). `--layout-style legacy` remains an internal CLI option for orthogonal libavoid routing. The environment variables below switch individual steps; defaults are what a normal run uses.

| Variable | Default | Meaning |
|---|---|---|
| `OPM_COMPOSE`, `OPM_COMPOSE_LONG` | on | whole-map composition; relocation threshold normalised by reaction spacing |
| `OPM_PERIPHERAL` | on | attach exchange + transport modules to the pathway they belong to |
| `OPM_BLOCK_RELOC`, `OPM_RELOC_ACCEPT` | off, `balanced` | relocate curated blocks with a long connection (opt-in); `strict` uses the strict quality gate |
| `OPM_EXPERT_SUBMODULE` | `branch` | Relocate exchange + transport modules (including coupled transporters) that sit far from the rest of a large **curated** block next to it (legacy env name). Boundary-crossing connections are re-drawn (metadata key `regenerated_expert_edges`). `on` = simple transports only, `off` = disabled |
| `OPM_RELOC_CLEAR` | off | experimental local space-making for relocations |
| `OPM_LOWCONF` | off | experimental: split topology-only structures whose reactions have unrelated neighbourhoods |
| `OPM_REGIONS`, `OPM_REGION_PLACE` | off | experimental region-level arrangement (did not improve the example maps) |
| `OPM_DROP_TOPO_BIG`, `OPM_SUMMARY_LAYOUT` | off, `ring` | experimental handling of very large reactions (biomass) |
| `OPM_HOTSPOT_REPAIR` | on | after routing and labels, greedily repair the top-k (default 3) 400 px crossing hotspots with cheap local moves (`hotspot_repair.py`); set `off` to skip |

## System Requirements

### C++ Compiler for libavoid Router

The edge router (`routing.py`) compiles a native libavoid binary. A C++11-capable compiler is required:

- **Linux**: Install `g++` (usually via `build-essential` on Debian/Ubuntu: `sudo apt-get install build-essential`)
- **macOS**: Install Xcode command-line tools: `xcode-select --install` (provides `clang++`)
- **Windows**: Not tested; consider WSL with Linux tools

The build prefers `g++` when available, falling back to `clang++`.

## Tests

```sh
cd layout_atlas
python3 test_orientation.py
python3 test_ec_provenance.py
python3 test_reaction_map_matching.py
python3 map_matching_eval.py          # BRENDA / Atlas match rates for e_coli_core, iIT341
python3 test_evaluation.py
python3 test_layout_quality.py
python3 test_hotspot_repair.py
python3 test_tool_lifetime.py         # tool-server bye/GRACE/STALE auto-exit
```


## Data

See [`data/README.md`](../data/README.md). Sources, licences and attribution: [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and [`docs/licensing_audit.md`](../docs/licensing_audit.md).
