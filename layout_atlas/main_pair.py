"""Choose one main substrate→product pair for each de-novo reaction (backbone edge).

Curated Atlas structures stay translation-only; this module only informs how algorithm-placed
reactions pick their through-axis and which currency participants are short side stubs.

Selection is graph/annotation based (chemical formula overlap, non-currency tier, metabolite
degree, same compartment, connectivity to already drawn neighbours). No model-specific hacks.
Ideas follow MetaCarto's primary-compound reduction (CC BY 4.0; doi:10.64898/2026.09.19.752882);
no MetaCarto code is copied.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from core import chem, currency

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str | None) -> dict[str, int]:
    if not formula:
        return {}
    counts: dict[str, int] = {}
    for element, digits in _TOKEN.findall(str(formula)):
        counts[element] = counts.get(element, 0) + (int(digits) if digits else 1)
    return counts


def _non_h_size(counts: dict[str, int]) -> float:
    return float(sum(n for el, n in counts.items() if el != "H"))


def moiety_overlap(a: dict[str, int], b: dict[str, int]) -> float:
    """Shared non-hydrogen atom count times relative overlap (Jaccard-style on atom bags)."""
    if not a or not b:
        return 0.0
    keys = {el for el in set(a) | set(b) if el != "H"}
    if not keys:
        return 0.0
    overlap = float(sum(min(a.get(el, 0), b.get(el, 0)) for el in keys))
    if overlap <= 0.0:
        return 0.0
    denom = max(_non_h_size(a), _non_h_size(b), 1e-9)
    return overlap * (overlap / denom)


def _trigrams(text: str) -> set[str]:
    text = f" {text} "
    return {text[i : i + 3] for i in range(len(text) - 2)}


def name_similarity(a: str, b: str) -> float:
    a = re.sub(r"\[[^\]]*\]", "", a or "").strip().lower()
    b = re.sub(r"\[[^\]]*\]", "", b or "").strip().lower()
    if not a or not b:
        return 0.0
    ta, tb = _trigrams(a), _trigrams(b)
    return len(ta & tb) / float(len(ta | tb))


def metabolite_degrees(model) -> dict[str, int]:
    deg: Counter = Counter()
    for r in model["reactions"]:
        for mid in r["metabolites"]:
            deg[mid] += 1
    return dict(deg)


def formula_index(model) -> dict[str, dict[str, int]]:
    out = {}
    for m in model["metabolites"]:
        formula = m.get("chemicalFormula") or m.get("formula")
        out[m["id"]] = parse_formula(formula)
    return out


def name_index(model) -> dict[str, str]:
    return {m["id"]: m.get("name") or m["id"] for m in model["metabolites"]}


def compartment_of(mid: str, model_compartments: dict[str, str] | None = None) -> str:
    if model_compartments and mid in model_compartments:
        return model_compartments[mid]
    if "@" in mid:
        return mid.split("@")[-1]
    m = re.search(r"_([a-z][0-9]?)$", mid)
    return m.group(1) if m else "unknown"


def _currency_tier(mid: str, cur: set[str]) -> int:
    """0 = ordinary carrier, 1 = currency/cofactor. Currency never wins while a non-currency alternative exists."""
    return 1 if mid in cur else 0


def _chemistry_score(sub, prod, formulas, names, degrees, comps, degree_cap: float) -> float:
    value = moiety_overlap(formulas.get(sub, {}), formulas.get(prod, {}))
    if value <= 0.0 and chem(sub) == chem(prod):
        # Transport of the same compound: small positive so the reaction still gets a pair.
        value = 1.0 if chem(sub) not in ("h", "h2o") else 0.1
    elif value <= 0.0:
        value = 0.05 + 0.9 * name_similarity(names.get(sub, sub), names.get(prod, prod))
    # Soft degree damping relative to the model's own max degree (unit-free).
    connectivity = min(degrees.get(sub, 0) + degrees.get(prod, 0), degree_cap)
    value *= 1.0 - 0.1 * connectivity / float(degree_cap)
    if compartment_of(sub, comps) == compartment_of(prod, comps):
        value *= 1.02
    return value


def _holders_from_scene(scene) -> dict[str, set[str]]:
    """metabolite id -> set of reaction ids already attached via a non-side glyph."""
    copies = scene.get("copies") or {}
    parts = scene.get("participants") or {}
    out = defaultdict(set)
    for (rid, mid), key in parts.items():
        if key.startswith("side:"):
            continue
        mid2 = copies.get(key, mid)
        out[mid2].add(rid)
    return out


def choose_main_pair(reaction, *, cur, formulas, names, degrees, comps=None, holders=None, degree_cap=None):
    """Return (main_sub, main_prod) or (None, None) when no substrate/product pair exists.

    Ranking:
      1. currency tier sum (lower is better: prefer non-currency on both sides)
      2. chemistry / formula overlap (higher)
      3. more already-drawn neighbour reactions on the pair ends
      4. deterministic id order
    """
    mets = reaction["metabolites"]
    subs = sorted(m for m, c in mets.items() if c < 0)
    prods = sorted(m for m, c in mets.items() if c > 0)
    if not subs or not prods:
        return None, None

    holders = holders or {}
    cap = float(degree_cap if degree_cap is not None else max(sum(degrees.values()), 1))
    candidates = []
    for s in subs:
        for p in prods:
            if s == p:
                continue
            tier = _currency_tier(s, cur) + _currency_tier(p, cur)
            score = _chemistry_score(s, p, formulas, names, degrees, comps, cap)
            neigh = len(holders.get(s, set()) - {reaction["id"]}) + len(holders.get(p, set()) - {reaction["id"]})
            candidates.append((tier, -score, -neigh, s, p))

    if not candidates:
        return None, None

    candidates.sort()
    return candidates[0][3], candidates[0][4]


def choose_all(model, scene=None, cur=None):
    """{reaction_id: (main_sub, main_prod)} for every reaction that has both roles."""
    cur = currency(model) if cur is None else cur
    formulas = formula_index(model)
    names = name_index(model)
    degrees = metabolite_degrees(model)
    degree_cap = float(max((degrees.get(m, 0) for m in degrees), default=1))
    # Cap is twice the busiest single metabolite so a pair's summed degrees stay in [0, 1] after clamp.
    degree_cap = max(2.0 * degree_cap, 1.0)
    comps = {m["id"]: m.get("compartment") or compartment_of(m["id"]) for m in model["metabolites"]}
    holders = _holders_from_scene(scene) if scene else {}
    out = {}
    for r in model["reactions"]:
        sub, prod = choose_main_pair(
            r,
            cur=cur,
            formulas=formulas,
            names=names,
            degrees=degrees,
            comps=comps,
            holders=holders,
            degree_cap=degree_cap,
        )
        if sub is not None and prod is not None:
            out[r["id"]] = (sub, prod)
    return out


def side_metabolites(reaction, main_sub, main_prod):
    """Participants that are not the chosen backbone pair, split by stoichiometric sign."""
    consumed, produced = [], []
    for mid, coeff in sorted(reaction["metabolites"].items()):
        if mid in (main_sub, main_prod):
            continue
        (consumed if coeff < 0 else produced).append(mid)
    return consumed, produced


def unit_axis(sub_xy, prod_xy, fallback=(1.0, 0.0)):
    u = np.asarray(prod_xy, float) - np.asarray(sub_xy, float)
    n = float(np.hypot(*u))
    if n < 1e-6:
        return np.asarray(fallback, float)
    return u / n
