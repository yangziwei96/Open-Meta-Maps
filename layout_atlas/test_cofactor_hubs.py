"""Internal hub counts on bundled example_results (e_coli_core / iIT341).

``cofactor_hubs`` counts currency metabolites drawn as one glyph linked to ≥2
reactions. Those internal counts are not the Pathway Scorer hub total (iIT341
bundled map: internal 9 vs Pathway Scorer 5).
"""
from pathlib import Path

from cofactor_hubs import cofactor_hubs
from core import currency, load_model
from evaluation.escher_bridge import load_escher

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
EXAMPLE_RESULTS = ROOT / "example_results"


def test_currency_includes_layout_carriers():
    e_model = load_model(EXAMPLES / "e_coli_core.xml")
    i_model = load_model(EXAMPLES / "iIT341.xml")
    assert "atp_c" in currency(e_model)
    assert "co2_e" in currency(i_model)
    assert "gtp_c" in currency(i_model)
    assert "fad_c" in currency(i_model)


def test_e_coli_core_hubs():
    raw = load_escher(EXAMPLE_RESULTS / "e_coli_core" / "OpenMetaMaps_e_coli_core_result.json")
    model = load_model(EXAMPLES / "e_coli_core.xml")
    report = cofactor_hubs(raw, model)
    assert report["hub_count"] == 6
    assert [h["metabolite"] for h in report["hubs"]] == [
        "co2_e", "h2o_e", "nh4_e", "o2_c", "o2_e", "pi_e",
    ]


def test_iIT341_hubs():
    raw = load_escher(EXAMPLE_RESULTS / "iIT341" / "OpenMetaMaps_iIT341_result.json")
    model = load_model(EXAMPLES / "iIT341.xml")
    report = cofactor_hubs(raw, model)
    assert report["hub_count"] == 9
    assert [h["metabolite"] for h in report["hubs"]] == [
        "co2_e", "fe2_e", "h2o_e", "nh4_e", "no2_e", "no3_e", "o2_e", "pi_e", "so4_e",
    ]
