# Open Meta Maps

Open Meta Maps turns a genome-scale metabolic model (GEM) into a complete, editable metabolic map. It has two parts: a curated-structure **Atlas** built from public community maps, and a **layout optimisation** pipeline that reuses matching Atlas geometry and completes the rest with Bézier connections.

ELIXIR BioHackathon Europe 2026 **[Project 22](https://github.com/elixir2026-project22)** (Open Pathway Maps).

## Workflow

<img src="docs/images/workflow.svg" alt="Workflow: an input model is matched against the Atlas of curated pathway structures; the Layout Composer keeps a curated skeleton, completes the rest with Bézier connections, and reports layout quality." width="700">

*Diagram source: `python3 tools/generate_workflow_svg.py` (optional `--check` writes a 700 px PNG preview).*

Every reaction and metabolite of the model appears in the map. Stoichiometry and compartments are unchanged.

## Quick start

Requirements: Python 3.11+, Node.js with npm, a C++ compiler (`clang++`) and network access for the first setup. SBML via libSBML; COBRA / BiGG JSON also supported.

```sh
git clone https://github.com/elixir2026-project22/Open-Meta-Maps.git
cd Open-Meta-Maps

python3 -m pip install -r layout_atlas/requirements.txt
npm ci --prefix layout_atlas/runtime
python3 project.py setup        # one-time: build the edge routers
```

### Map Generator

```sh
python3 project.py tool
# or: ./start_tool.command
```

Open <http://127.0.0.1:8765/tool/index.html>. Upload an SBML or COBRA JSON model, or pick an example (`e_coli_core`, `iIT341`, `iMM904`, `iML1515`). The viewer supports pan, wheel/pinch zoom and a minimap. Download the finished map as SVG, PNG, Escher JSON, SBGN-ML, SBML Layout or CellDesigner SBML. The server exits when the last browser tab closes (~8 s grace); pass `--keep-running` to disable that. The same server also serves **Atlas Browser** (`/atlas_browser/index.html`) to search curated modules and download Escher JSON with a provenance/licence manifest (search index built on first use under `data/cache/`).

### Command line

```sh
python3 project.py compose --gem examples/e_coli_core.xml --out results/e_coli_core/ --source-paths
```

Outputs use `OpenMetaMaps_<name>_result.<ext>`, where `<name>` is the model file stem (sanitised). Formats: `.json` (Escher), `.sbgn` (SBGN-ML Process Description), `.sbml` (input SBML with Layout package), `.celldesigner.xml` (CellDesigner 4.x annotations), plus `.svg` / `.png`. The Map Generator download row offers the same formats.

Rebuild after a small model edit while keeping the previous layout: pass `--previous` with the prior result JSON. Matched reaction midmarkers and primary metabolites are pinned by stable id; when at least 80% of reactions are already placed, module packing is skipped and the previous Escher geometry is reused. New reactions are grafted on the chord between drawn substrate/product participants (opposite sides); leftovers without anchors are still parked so coverage stays complete. A local nudge of grafted midmarkers is kept when crossings do not worsen.

```sh
python3 project.py compose --gem examples/e_coli_core_edited.xml --out results/e_coli_core_edited/ --source-paths --previous results/e_coli_core/OpenMetaMaps_e_coli_core_result.json
```

Layout evaluation: `pip install -r layout_atlas/requirements-eval.txt`, then `python3 project.py evaluate --gem MODEL.xml --map DIR/OpenMetaMaps_<stem>_result.json`. See [docs/layout_evaluation.md](docs/layout_evaluation.md).

## 1. Data construction — the curated-structure Atlas

The Atlas stores reusable pathway modules (topology, reaction identity, geometry, licence) cut from public community maps. Default load: `data/open_core.sqlite` plus in-tree extensions. Licences and extraction provenance from [`data/atlas_registry.json`](data/atlas_registry.json); counts from [`docs/images/atlas/atlas_overlap.json`](docs/images/atlas/atlas_overlap.json) (About uses the same files).

<img src="docs/images/atlas/atlas_overview.png" alt="Overview of the Atlas: one panel per source, sized by its unique reaction sets and filled with every unique module, with the largest distinct shapes drawn as framed tiles." width="700">

*One panel per source; panel area scales with its unique reaction sets, and every unique set is one tile. Framed tiles show the largest distinct shapes. Titles give the source maps, cut windows and distinct structures of each source. Regenerate: `PYTHONPATH=layout_atlas python3 tools/atlas_module_figures.py overview`.*

### Sources and licences

| Source | Source maps | Cut windows | Distinct structures | Licence |
|---|---:|---:|---:|---|
| Escher (community maps) | 22 | 13,047 | 970 | CC0-1.0 |
| Reactome (release 97 diagrams) | 86 | 1,811 | 315 | CC0 (data) |
| WikiPathways semantic (20261110) | 65 | 669 | 97 | CC0-1.0 |
| WikiPathways geometry-only (20261110) | 869 | 15,611 | 2,592 | CC0-1.0 |
| Human-maps (official Escher JSON) | 130 | 4,690 | 908 | CC BY 4.0 |
| BRENDA (pathway maps; layout possibly automatic) | 193 | 11,703 | 281 | CC BY 4.0 |
| Metabolism Regulation (SBGN-ML maps) | 22 | 229 | 31 | CC BY 4.0 |
| PANTHER (3.4/CD4.1 CellDesigner maps) | 8 | 138 | 27 | CC BY 4.0 |
| iCH360 (custom Escher maps) | 10 | 436 | 92 | MIT |
| **Total (default load)** | **1,405** | **48,334** | **5,310** | |

**1,405 source maps → 48,334 cut windows → 5,310 distinct structures.**

- **Source maps** are the maps that yield at least one module.
- **Cut windows** are every stored module: each map is cut into breadth-first reaction windows of increasing size (up to 24 reactions, 40 for Escher) grown from many root reactions, so most windows nest inside larger windows of the same map. Small windows are kept intentionally because they match small models.
- **Distinct structures** are windows that are not contained in a larger window of the same source map (reaction set; node set for geometry-only windows) and have at least 3 reactions, merged across sources by dedup key. The per-source sum is 5,313; 3 structures are identical across sources.
- Dedup key: canonical reaction-id set (geometry fingerprint for WikiPathways geometry-only windows). Exact dedup alone leaves 36,103 unique windows.

Stats: [`docs/images/atlas/atlas_overlap.json`](docs/images/atlas/atlas_overlap.json), computed by `compute_overlap()` in `tools/atlas_module_figures.py` (`PYTHONPATH=layout_atlas python3 tools/atlas_module_figures.py stats`).

Attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Source notes: [docs/data_sources.md](docs/data_sources.md).

### Example modules

<img src="docs/images/atlas/atlas_examples.svg" alt="Grid of example Atlas modules with source, map name and reaction count captions." width="700">

*Two–three representative modules per source. Regenerate: `PYTHONPATH=layout_atlas python3 tools/atlas_module_figures.py examples`.*

### Extraction, identity and selection

- **Cut** — documented extractors: in-repo Escher community cutter (`escher_community_windows`, parity-tested); Stage B `diagram_modules` / `geometry_modules` (size-scaled roots, near-nested dedupe) for Reactome, WikiPathways, Human-maps, Metabolism Regulation, PANTHER and iCH360; BRENDA’s own extractor.
- **Deduplicate** — near-duplicate windows grouped into core families and geometric variants; Stage B also drops near-nested modules (`nest_factor=2`).
- **Canonical ids** — unique crosswalks map reaction references to BiGG, MetaNetX and/or Rhea where available (Human-maps via Human-GEM; Metabolism Regulation via participant ChEBI; Escher / iCH360 already BiGG).
- **Provenance** — each module records source, map, licence and an `extraction` block (method, version, parameters, counts).
- **Selection** — coverage ranks first; among same-source biochemical matches with the same reaction set, the denser drawing is kept.

Worked examples (cut boundaries, ID remap, cross-source dedup, provenance, compact selection): [docs/atlas_data_management.md](docs/atlas_data_management.md#worked-examples). On-disk layout: [data/README.md](data/README.md).

### Download the Atlas

```sh
python3 project.py data export                                       # unique windows after exact dedup -> dist/openmetamaps-atlas-<date>.tar.gz
python3 project.py data export --source panther --licence CC-BY-4.0  # one source, one licence (both repeatable)
python3 project.py data export --all-windows --out atlas_all.tar.gz  # every cut window instead of unique reaction sets
```

The archive holds one Escher JSON per module (`modules/<source>/<module_id>.json`), `manifest.json` (source map, licence, attribution text, reaction count, canonical reaction ids, extraction provenance and `also_in` for identical reaction sets in other sources), a short README and `THIRD_PARTY_NOTICES.md`. The default export has 36,103 windows (one per unique reaction set, from 48,334 cut windows); re-running it gives a byte-identical archive. To pick individual modules, search the Atlas Browser and download the selection as the same Escher JSON with the same manifest fields. Details: [docs/data_packages.md](docs/data_packages.md).

---

## 2. Layout optimisation algorithm

### Matching and curated reuse

Reactions of the input model are matched to Atlas modules (EC currentisation, Rhea metabolite scoring, name normalisation, weighted Jaccard, vicinity propagation). Report curated, topology-only and EC suggestions as separate rates in `reaction_provenance.json`. EC similarity extension is **off by default** (`--ec`). See [docs/reaction_map_matching.md](docs/reaction_map_matching.md) · [docs/methodology.md](docs/methodology.md).

Matched geometry is kept as drawn (translated only). Everything else is placed by the Composer (`layout_style=curated` by default; `--layout-style legacy` is CLI-only for orthogonal libavoid).

### Layout stages

`tool_pipeline.draw_map` / `draw_global_map`. Small models (≤150 reactions) take the full path; mid-size (≤3 000) use a global plan then the same curated Bézier pass; larger models keep a skeleton and draw ~50-reaction views on demand.

| Stage | What it does | Optimises | Main code |
|---|---|---|---|
| **Rigid curated skeleton** | Scale matched structures; pack rigid blocks with shared metabolites as junctions; network bridges as anchors | Curated-structure continuity (with matching) | `drawing.py`, `geometry.py` |
| **Complete uncovered reactions** | Place free nodes around fixed anchors; optional context for currency-only reactions | — | `completion.py`, `context.py`, `completion_context.py` |
| **Metabolite instances** | Extra visual copies of non-currency metabolites to shorten long edges; currency stays private `side:` glyphs | Long lines; cofactor clones | `instances.py`, `drawing.py` |
| **Orientation** | Put free reactions between substrates and products; currency copies on the correct side | Participant placement | `orientation.py` |
| **Composition & peripherals** | Pull related components in; shelf independents; attach exchange/transport; optional relocation inside large curated blocks | Long lines; cofactor clones; exchange on periphery; aspect; blank space | `composition.py`, `peripheral.py`, `block_relocation.py` |
| **Spacing / relaxation** | Soft-preserve curated shape; shorten edges; minimum node spacing | Connection crossings; unrelated overlaps | `geometry.py`, `global_layout.py` |
| **Bézier connections** | Cubic Béziers from reaction templates; optional `--source-paths` for original Escher curves | Connection crossings; unrelated overlaps; detours, hooks, bends; quality gate | `edge_style.py`, `source_paths.py`, `routing.py` |
| **Labels** | Move labels among candidates to cut collisions | — | `label_placement.py` |
| **Integrity** | Model hash, every reaction drawn, visual stoichiometry; protected shared glyphs must not split | — | `tool_pipeline.py` |

**Determinism:** fixed layout seed `20261105` (`core.SEEDS`), ELK/fCoSE seeds, `PYTHONHASHSEED=0` on re-exec. Displacement measured by `evaluate --compare-map` only.

`--previous OpenMetaMaps_<stem>_result.json` pins matched nodes and grafts leftovers.

**Known gaps** ([docs/limitations.md](docs/limitations.md)): 2D component packing, harder orientation (incl. transporters), compartment-aware packing, stronger currency/biomass handling, crossing-aware composition acceptance, compose seed flag. Experimental off by default: `OPM_REGIONS`, `OPM_LOWCONF`, whole-block `OPM_BLOCK_RELOC`.

### Example output

Maps composed from the bundled example models. Every reaction and metabolite is drawn.

<p align="center">
<img src="docs/images/examples/e_coli_core.png" alt="Escherichia coli core map (e_coli_core, 95 reactions)" width="700"><br>
<em>Escherichia coli</em> core (<code>e_coli_core</code>, 95 reactions)
</p>

<p align="center">
<img src="docs/images/examples/iIT341.png" alt="Helicobacter pylori iIT341 map (554 reactions)" width="700"><br>
<em>Helicobacter pylori</em> iIT341 (554 reactions)
</p>

*Previews are cropped to content and draw every reaction and metabolite label of the result SVG. Zoom in to read the labels. Regenerate: `PYTHONPATH=layout_atlas python3 tools/render_map_preview.py DIR/OpenMetaMaps_NAME_result.json docs/images/examples/NAME.png`.*

**Download.** Result files of both maps are in [`example_results/`](example_results/) (file list: [example_results/README.md](example_results/README.md)).

| Model | Escher JSON | SVG | PNG | SBGN-ML | SBML Layout | CellDesigner | Licences | Provenance | Report |
|---|---|---|---|---|---|---|---|---|---|
| <code>e_coli_core</code> | [OpenMetaMaps_e_coli_core_result.json](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/OpenMetaMaps_e_coli_core_result.json) | […_result.svg](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/OpenMetaMaps_e_coli_core_result.svg) | […_result.png](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/OpenMetaMaps_e_coli_core_result.png) | […_result.sbgn](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/OpenMetaMaps_e_coli_core_result.sbgn) | […_result.sbml](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/OpenMetaMaps_e_coli_core_result.sbml) | […_result.celldesigner.xml](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/OpenMetaMaps_e_coli_core_result.celldesigner.xml) | [licenses.json](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/licenses.json) | [reaction_provenance.json](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/reaction_provenance.json) | [report.json](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/e_coli_core/report.json) |
| <code>iIT341</code> | [OpenMetaMaps_iIT341_result.json](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/OpenMetaMaps_iIT341_result.json) | […_result.svg](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/OpenMetaMaps_iIT341_result.svg) | […_result.png](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/OpenMetaMaps_iIT341_result.png) | […_result.sbgn](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/OpenMetaMaps_iIT341_result.sbgn) | […_result.sbml](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/OpenMetaMaps_iIT341_result.sbml) | […_result.celldesigner.xml](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/OpenMetaMaps_iIT341_result.celldesigner.xml) | [licenses.json](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/licenses.json) | [reaction_provenance.json](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/reaction_provenance.json) | [report.json](https://github.com/elixir2026-project22/Open-Meta-Maps/raw/main/example_results/iIT341/report.json) |

---

## 3. Evaluation criteria

Compose writes `layout_quality.evaluate` into `layout_issues.json` / `report.json`. `python3 project.py evaluate` adds layout-rules metrics via `evaluation/score.py`. See [docs/layout_evaluation.md](docs/layout_evaluation.md).

### Metrics

| Group | Metrics | Code |
|---|---|---|
| **Layout-benchmark metrics** | Crossings and crossing density, crossing angular resolution, overlapping pairs and overlap area share, participant consistency, edge length deviation, edge orthogonality, angular resolution, aspect ratio, area per node, homogeneity, stress, coverage | `metrics.py`, `escher_bridge.py` |
| **Pathway-scorer rules** | Longest line over median edge length, share of lines longer than 2.5× the median, bends per line, cofactor clones (currency hubs and long currency arcs) | `map_rules.py` |
| **Map rules** | Cycle circularity, direction and order; axis-aligned main segments and reaction axes; metabolites on a grid; chain turns; main flow direction; exchange reactions on the periphery; blank space and largest empty rectangle; unrelated overlaps (node overlaps, lines through nodes, collinear lines); label overlaps; species outside their compartment | `map_rules.py` |
| **Curated-structure integrity** | Structure continuity and split structures: curated structures stay whole and are only translated | `continuity.py` |
| **Stability** | Reproducibility across runs; displacement and Procrustes disparity against a previous map (`--previous`) | `stability.py` |
| **Reference similarity** | Procrustes, distance correlation and kNN preservation against a reference SBGN map | `reference_sbgn.py` |

Layout targets (including backlog) stay on Detailed metrics; non-targets under Debug (including main flow-direction share).

### Layout rules (status)

| Rule | Status | Notes |
|---|---|---|
| Avoid crossings | Partial | Minimised in Bézier routing / quality gate; GEMs still show many algo crossings |
| Avoid overlaps | Partial | Spacing + label pass + gate; overlaps grow with map size |
| Avoid long lines | Partial | Instances, composition, peripherals help; currency/biomass spokes remain |
| Place participants consistently | Partial | Orientation on by default; share still below curated reference maps |
| Respect compartments | Not yet | Used in packing/shelves; cross-compartment species metric still high |
| Compartment packing | Not yet | |
| Aspect / compactness | Partial | Shelf aspect target; independent components still linear shelves |
| Axis-aligned segments / axes | Not yet | |
| Grid metabolites | Not yet | |
| Chain turns | Not yet | |
| Cycle placement | Not yet | |
| Reproducibility | Partial | Fixed seeds; no compose seed CLI |
| Build on previous layout | Partial | `--previous OpenMetaMaps_<stem>_result.json` reuses matched geometry and grafts leftovers |
| Scale to GEMs | Partial | Global ≤3 000 + tiled views; large maps start routing-degraded |

---

## Outputs

Final map downloads are named `OpenMetaMaps_<name>_result.<ext>`, where `<name>` is the uploaded (or CLI) model file stem, sanitised to letters, digits, `-`, `_` and `.`.

| Output | Description |
|---|---|
| `OpenMetaMaps_<name>_result.json` | Editable Escher map |
| `OpenMetaMaps_<name>_result.sbgn` | SBGN-ML 0.3 Process Description (species, process glyphs with ports, arcs, compartments, clone markers) |
| `OpenMetaMaps_<name>_result.sbml` | Input SBML with Layout package v1 (and Render when available); model ids and stoichiometry unchanged |
| `OpenMetaMaps_<name>_result.celldesigner.xml` | CellDesigner 4.x SBML (`speciesAlias` bounds, base reactants/products, edit points) |
| `OpenMetaMaps_<name>_result.svg` | Vector visualisation |
| `OpenMetaMaps_<name>_result.png` | Raster preview (Map Generator download) |
| `report.json` | Coverage and layout-quality measures |
| `reaction_provenance.json` | Per-reaction provenance and matching evidence |
| `layout_issues.json` | Crossings, collisions and other layout problems |
| `licenses.json` | Licensing metadata of Atlas structures used |

## Licensing

**Code** is [MIT](LICENSE). **Atlas data and generated maps** mix CC0, CC BY 4.0 and MIT sources; keep the attributions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Documentation

- [Installation](docs/installation.md)
- [Data sources](docs/data_sources.md) · [Atlas data management](docs/atlas_data_management.md) · [Atlas layout](data/README.md) · [iCH360 import](docs/ich360_import.md) · [Optional packages](docs/data_packages.md)
- [Reaction–map matching](docs/reaction_map_matching.md) · [EC extension](docs/ec_extension_recommendation.md)
- [Methodology](docs/methodology.md) · [Limitations](docs/limitations.md) · [Layout evaluation](docs/layout_evaluation.md)
- [Code overview](layout_atlas/README.md) · [About / Provenance](layout_atlas/results/about/index.html) (regenerate with `python3 layout_atlas/navigation.py`)

## Project status

Under active development. Focus: global pathway organisation, less visual overlap, consistent layouts across GEMs.

## Citation and acknowledgements

A citation will be added with the first release. Until then, cite this repository and the sources in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Developed during ELIXIR BioHackathon Europe 2026, building on the Escher, WikiPathways, Reactome and wider metabolic-modelling communities.
