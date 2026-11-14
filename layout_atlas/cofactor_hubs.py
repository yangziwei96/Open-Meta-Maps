"""Pathway Scorer–style cofactor hub count (node-based).

The scorer's clones rule flags a currency metabolite drawn once but connected to
several reactions. Shortening spokes does not change that count; each reaction
needs its own clone of the currency species.

Currency ids are the model's layout ``currency()`` set.
"""
from __future__ import annotations

from collections import defaultdict

from core import currency


def glyph_reaction_links(map_json) -> dict[str, set[str]]:
    """Metabolite node id -> set of reaction bigg_ids that touch it."""
    nodes = map_json[1]["nodes"]
    out: dict[str, set[str]] = defaultdict(set)
    for rx in map_json[1]["reactions"].values():
        rid = rx.get("bigg_id")
        if not rid:
            continue
        for seg in (rx.get("segments") or {}).values():
            for nid in (seg.get("from_node_id"), seg.get("to_node_id")):
                if nid in nodes and nodes[nid].get("node_type") == "metabolite":
                    out[nid].add(rid)
    return out


def cofactor_hubs(map_json, model, cur=None) -> dict:
    """Return hubs: currency species with exactly one multi-reaction glyph."""
    cur = currency(model) if cur is None else set(cur)
    nodes = map_json[1]["nodes"]
    links = glyph_reaction_links(map_json)
    by_bigg: dict[str, list[str]] = defaultdict(list)
    for nid, node in nodes.items():
        if node.get("node_type") != "metabolite":
            continue
        bigg = node.get("bigg_id")
        if bigg in cur:
            by_bigg[bigg].append(nid)
    hubs = []
    for bigg, glyphs in sorted(by_bigg.items()):
        if len(glyphs) != 1:
            continue
        nid = glyphs[0]
        rids = links.get(nid, set())
        if len(rids) >= 2:
            hubs.append(
                dict(
                    metabolite=bigg,
                    glyph=nid,
                    reactions=sorted(rids),
                    n_reactions=len(rids),
                )
            )
    return dict(
        hub_count=len(hubs),
        hubs=hubs,
    )
