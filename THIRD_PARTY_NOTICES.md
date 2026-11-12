# Third-party sources and attribution

Open Meta Maps publishes **processed structures**: the module graphs, reaction identities, geometric layouts and layout constraints that were extracted from existing community-drawn pathway maps and pathway databases. Each source below is credited, and the licence as recorded in this project is stated. Where the recorded licence could not be confirmed, that is said explicitly. This file records provenance; it is not legal advice. See [docs/licensing_audit.md](docs/licensing_audit.md) for the evidence behind each entry.

## Curated layout sources (the Atlas)

| Source | What is used | Where it comes from | Licence as recorded |
|---|---|---|---|
| **Human-maps** | modules in `data/extensions/human_maps/` from official `escher/` JSON | <https://github.com/SysBioChalmers/Human-maps> | CC BY 4.0 for the imported JSON; companion SVGs are CC BY-SA and are not imported |
| **PANTHER Pathway** | CellDesigner modules in `data/extensions/panther/`; raw XML in `data/source_inputs/raw/panther/` | <https://data.pantherdb.org/ftp/pathway/3.4/CD4.1/>; terms <https://pantherdb.org/tou.jsp> | CC BY 4.0 |
| **Metabolism Regulation gallery** | SBGN-ML modules in `data/extensions/metabolismregulation/`; raw `.sbgn` in `data/source_inputs/raw/metabolismregulation/` | <https://metabolismregulation.github.io/gallery/>; about/licence <https://metabolismregulation.github.io/about/> | CC BY 4.0 (hand-drawn layouts; attribution required) |
| **iCH360 Visualisation** | custom Escher modules in `data/extensions/ich360/`; raw JSON in `data/source_inputs/raw/ich360/` (10 maps; `compressed_map.json` excluded) | <https://github.com/marco-corrao/iCH360>; Corrao et al. doi:10.1371/journal.pcbi.1013564; Zenodo doi:10.5281/zenodo.15048160 | MIT (repo LICENSE); Zenodo deposit of the same maps is CC BY 4.0 |
| **BRENDA pathway maps** | pathway modules in `data/extensions/brenda/` | <https://www.brenda.dsmz.de/>; Chang et al. NAR 2026 doi:10.1093/nar/gkaf1113 | CC BY 4.0; layout possibly automatic |
| **Escher maps** (RECON1, e_coli_core, iAF1260, iJO1366, iMM904; 22 maps) | module structures and layouts extracted from the maps; the original map files are kept as input in `data/source_inputs/raw/escher/` | <https://github.com/escher/community-maps> (commit `6aa92abded6d0f9c4502ff681e14e40c5c00d61f`); repository `LICENSE` is CC0 | CC0-1.0 (Atlas source label `escher`) |
| **WikiPathways** | GPML pathway maps and structures extracted from them | <https://www.wikipathways.org>; terms snapshot `data/source_inputs/terms/wikipathways.html` (accessed 2026-11-05) | CC0 |
| **Reactome** (release 97) | pathway diagram and graph data, structures extracted from them | <https://reactome.org>; terms snapshot `data/source_inputs/terms/reactome.html` (accessed 2026-11-05) | CC0 for data; pathway illustrations are CC BY 4.0 and are not used |

## Reference data

| Source | What is used | Licence as recorded |
|---|---|---|
| **Rhea** (release 143) | reaction identities, directions and mappings (`data/rhea/`) | CC BY 4.0; the Rhea copyright statement is kept in `data/rhea/LICENSE.txt` |
| **BiGG Models** | example genome-scale models in `examples/` (e_coli_core, iML1515, iMM904) and `data/models/iYO844.xml` | BiGG licence; **non-commercial / educational terms apply and are to be reviewed**; the source of `iIT341.xml` is **not recorded** |
| **AraCore** (Arabidopsis core model v2.1, P. Wendering et al.) | `data/models/AraCore_v2_1.xml` | MIT (`data/models/LICENSE.AraCore`) |
| **Human-GEM** (SysBioChalmers) | its model annotations are the source of part of `data/identifier_bridge.json` (the model file itself is not included); `model/reactions.tsv` and `model/metabolites.tsv` are kept under `data/source_inputs/crosswalks/` and compiled into `human_gem_reaction_crosswalk.json` / `human_gem_metabolite_crosswalk.json` | CC BY 4.0 according to the project page; please cite Human-GEM when using it |
| **Identifier tables** (`data/identifier_bridge.json`) | BiGG metabolite → ChEBI and BiGG reaction → EC, extracted from the annotations of e_coli_core, iML1515, iMM904 (BiGG) and Human-GEM; every entry lists the models it comes from | the terms of those models apply (BiGG licence; CC BY 4.0 for Human-GEM) |
| **Human-GEM metabolites.tsv** | MAM → BiGG / ChEBI / MetaNetX bridge used by the mapper (`canonical_ids.build_human_gem_metabolite_crosswalk`) | CC BY 4.0; see `data/source_inputs/crosswalks/Human-GEM-metabolites.SOURCE.json` |
| **Currency / cofactor list** (`data/currency_metabolites.json`) | 65 BiGG metabolite ids (ions, small inorganics, coenzymes); each checked against the BiGG models | identifiers only; provenance in the file |

## Software and schemas

| Component | Licence as recorded |
|---|---|
| **Escher** map schema (`data/escher_schema/`, used unchanged) | MIT (licence kept next to the file) |
| **libSBGN SBGN-ML 0.3 XSD** (`data/sbgn_schema/SBGN.xsd`, `render.xsd`; used to validate exported SBGN-ML) | LGPL 2.1 or Apache 2.0 (dual); licence kept in `data/sbgn_schema/LICENSE.txt`; upstream <https://github.com/sbgn/libsbgn> |
| **libSBML** (`python-libsbml`, SBML reader/writer and Layout package; installed as a dependency, not redistributed) | LGPL 2.1 |
| **lxml** (SBGN-ML XSD validation; installed as a dependency, not redistributed) | BSD-3-Clause |
| **adaptagrams / libavoid** (edge routing, `data/routing/`) | LGPL; exact terms **to be confirmed**, notice to be added |
| **layout-benchmark** pathway metrics (adapted into `layout_atlas/evaluation/`, not a runtime dependency) | MIT; see `layout_atlas/evaluation/ATTRIBUTION.md` and <https://github.com/elixir2026-project22/layout-benchmark> |
| **gdMetriX** (optional import for general graph-drawing metrics in `evaluate`) | GPL-3.0; **not vendored**; install via `layout_atlas/requirements-eval.txt` if needed |

## Not included in this repository

- Yeast-maps (GPL-3.0) and MicroMap (NC-SA) are excluded; `layout_atlas/celldesigner.py` remains a generic CellDesigner importer.
- Optional **evaluation stand-in models** (iCH360 MIT, BioModels iPS189 CC0, Reactome SBML CC0, metabolismregulation gallery F001 CC BY 4.0) are not committed; see `examples/eval_models/manifest.json` and `fetch_eval_models.py`.

## Related work

MetaCarto (<https://github.com/forxhunter/MetaCarto>) informed the design of some layout steps; its published methods were re-implemented independently, and no MetaCarto code or maps are included.
