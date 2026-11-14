"""Clone-aware Escher reading for the hub-split accept gate (compose path).

Returns only what the gate scores: a directed participant graph, glyph
positions (with primary flags), and bend waypoints along each drawn edge.
Does not import the evaluation package.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import networkx as nx

SPECIES = "species"
REACTION = "reaction"

Positions = dict[str, tuple[float, float]]
Routes = dict[tuple[Any, Any], list[tuple[float, float]]]


class HubPathway:
    """Directed species↔reaction incidence graph (BiGG ids)."""

    __slots__ = ("graph",)

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph


def pathway_from_model(model: dict) -> HubPathway:
    """Incidence graph from an Open Meta Maps model dict."""
    g = nx.DiGraph()
    for met in model.get("metabolites") or []:
        g.add_node(met["id"], kind=SPECIES)
    for rx in model.get("reactions") or []:
        rid = rx["id"]
        g.add_node(rid, kind=REACTION)
        for mid, coeff in (rx.get("metabolites") or {}).items():
            if mid not in g:
                continue
            c = float(coeff)
            if c < 0:
                g.add_edge(mid, rid, role="reactant")
            elif c > 0:
                g.add_edge(rid, mid, role="product")
    g.remove_nodes_from([n for n in list(g) if g.degree(n) == 0])
    return HubPathway(graph=g)


def _map_body(escher_map: list | dict) -> dict:
    return escher_map[1] if isinstance(escher_map, list) else escher_map


def _resolve(bigg: str | None, known: set[str]) -> str | None:
    """Map an Escher BiGG id onto a pathway node id when the two naming schemes differ."""
    if not bigg:
        return None
    if bigg in known:
        return bigg
    for prefix in ("M_", "R_", "S_", "s_", "r_"):
        if bigg.startswith(prefix) and bigg[len(prefix) :] in known:
            return bigg[len(prefix) :]
        tagged = prefix + bigg
        if tagged in known:
            return tagged
    return None


def _point(node: dict, y_up: bool) -> tuple[float, float]:
    x, y = float(node["x"]), float(node["y"])
    return (x, -y) if y_up else (x, y)


def _glyph_node(escher_nid: str) -> str:
    """Stable drawing-graph id for one metabolite glyph (clone-aware)."""
    return f"g:{escher_nid}"


def _segment_adjacency(reaction: dict) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for seg in (reaction.get("segments") or {}).values():
        a, b = seg.get("from_node_id"), seg.get("to_node_id")
        if a is None or b is None:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _interior_bends(adj: dict[str, set[str]], start: str, goal: str, nodes: dict, y_up: bool):
    """Interior marker coordinates along the segment path from ``start`` to ``goal``."""
    if start == goal:
        return []
    parent = {start: None}
    q = deque([start])
    found = False
    while q:
        u = q.popleft()
        if u == goal:
            found = True
            break
        for v in adj.get(u, ()):
            if v in parent:
                continue
            parent[v] = u
            q.append(v)
    if not found:
        return []
    trail = []
    cur = goal
    while cur is not None:
        trail.append(cur)
        cur = parent[cur]
    trail.reverse()
    bends = []
    for nid in trail[1:-1]:
        node = nodes.get(nid)
        if node is not None:
            bends.append(_point(node, y_up))
    return bends


def drawing_graph(pathway: HubPathway, escher_map: list | dict, y_up: bool = True):
    """Clone-aware graph, positions, and polyline bends for gate metrics.

    Metabolite clones are distinct nodes (one per Escher glyph). Each reaction sits
    at its midmarker. Bend lists follow multimarker chains between glyph and midmarker.
    """
    body = _map_body(escher_map)
    nodes = body["nodes"]
    kinds = {n: d.get("kind") for n, d in pathway.graph.nodes(data=True)}
    species_ids = {n for n, k in kinds.items() if k == SPECIES}
    reaction_ids = {n for n, k in kinds.items() if k == REACTION}

    g = nx.DiGraph()
    pos: Positions = {}
    routes: Routes = {}
    glyph_of: dict[str, str] = {}  # escher metabolite nid → drawing node

    for nid, node in nodes.items():
        if node.get("node_type") != "metabolite":
            continue
        sid = _resolve(node.get("bigg_id"), species_ids)
        if sid is None:
            continue
        gid = _glyph_node(nid)
        g.add_node(gid, kind=SPECIES, primary=bool(node.get("node_is_primary")))
        pos[gid] = _point(node, y_up)
        glyph_of[nid] = gid

    for reaction in body["reactions"].values():
        rid = _resolve(reaction.get("bigg_id"), reaction_ids)
        if rid is None:
            continue
        adj = _segment_adjacency(reaction)
        mid = next(
            (n for n in adj if nodes.get(n, {}).get("node_type") == "midmarker"),
            None,
        )
        if mid is None or mid not in nodes:
            continue
        g.add_node(rid, kind=REACTION, primary=True)
        pos[rid] = _point(nodes[mid], y_up)

        for met_nid, gid in glyph_of.items():
            if met_nid not in adj:
                continue
            sid = _resolve(nodes[met_nid].get("bigg_id"), species_ids)
            if sid is None:
                continue
            bends_to_mid = _interior_bends(adj, met_nid, mid, nodes, y_up)
            if pathway.graph.has_edge(rid, sid):
                g.add_edge(rid, gid, role="product")
                routes[(rid, gid)] = list(reversed(bends_to_mid))
            elif pathway.graph.has_edge(sid, rid):
                g.add_edge(gid, rid, role="reactant")
                routes[(gid, rid)] = bends_to_mid
    return g, pos, routes


def stub_endpoints(pathway: HubPathway, map_json, stubs: list[tuple[str, str]]):
    """Map Escher (midmarker id, metabolite glyph id) pairs to drawing-graph ends."""
    body = _map_body(map_json)
    nodes = body["nodes"]
    reaction_ids = {
        n for n, d in pathway.graph.nodes(data=True) if d.get("kind") == REACTION
    }
    species_ids = {
        n for n, d in pathway.graph.nodes(data=True) if d.get("kind") == SPECIES
    }
    mid_to_rid: dict[str, str] = {}
    for rx in body["reactions"].values():
        rid = _resolve(rx.get("bigg_id"), reaction_ids)
        if rid is None:
            continue
        for seg in (rx.get("segments") or {}).values():
            for nid in (seg.get("from_node_id"), seg.get("to_node_id")):
                if nodes.get(nid, {}).get("node_type") == "midmarker":
                    mid_to_rid[nid] = rid
    edges = []
    for mid_nid, met_nid in stubs:
        rid = mid_to_rid.get(mid_nid)
        sid = _resolve((nodes.get(met_nid) or {}).get("bigg_id"), species_ids)
        if rid is None or sid is None:
            continue
        edges.append((rid, _glyph_node(met_nid)))
    return edges
