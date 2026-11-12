"""One global map for large models: scalable replacements for the steps of `tool_pipeline.draw_map` that do not scale past ~150 reactions.

The small-model path is already global: all selected expert structures are packed together (ELK, shared metabolites become junctions),
the core graph is completed around them, and one map is exported.  Large models used to be cut into ~50-reaction views that were drawn
separately and tiled, which loses every relation between views and draws shared metabolites once per view.  The same global steps stop
scaling for three reasons that sit in hashed files (not modified here):

  * `geometry.relax`      all-pairs collision term over every display glyph (np.triu_indices): ~18 million pairs for iMM904
  * `drawing.complete`    fCoSE subprocess with a fixed 90 s timeout
  * `drawing.export`      routing with libavoid and per-glyph scans (routing is switched off here, see `routing_status`)

so this module provides `complete_global`, `scalable_scene_relax` and `deadline`, and `instance_table` (which visual node is which instance of
which GEM metabolite).  `tool_pipeline.draw_map(..., scalable=True)` uses them; the other steps (pack_modules, display_scene, export,
integrity checks, source paths, provenance) are the unchanged ones.
"""
import contextlib
import copy
import json
import signal
import subprocess
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from core import BASE, SEEDS, currency
import drawing
import geometry


@contextlib.contextmanager
def deadline(seconds):
    """Raise TimeoutError in the main thread after `seconds` (the pipeline runs one job per process, so SIGALRM is safe)."""
    def handler(signum, frame):
        raise TimeoutError(f'global layout budget of {seconds:.0f} s exceeded')
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def complete_global(g, anchors, seed=SEEDS[0], iterations=350, timeout=900):
    """Same as `drawing.complete` for plain anchors (fCoSE, anchors fixed) with a timeout that fits thousands of nodes."""
    xy = drawing.initial(g, anchors, seed)
    elements = [dict(data=dict(id=k), position=dict(x=xy[k][0], y=xy[k][1])) for k in sorted(g)]
    elements.extend(dict(data=dict(id='edge' + str(i), source=a, target=b)) for i, (a, b) in enumerate(sorted(g.edges())))
    payload = dict(backend='fcose', elements=elements, anchors=[dict(nodeId=k, position=dict(x=v[0], y=v[1])) for k, v in anchors.items() if k in g],
                   seed=seed, iterations=iterations)
    run = subprocess.run(['node', str(BASE / 'runtime/layout_backend.cjs')], input=json.dumps(payload), text=True, capture_output=True, timeout=timeout, check=True)
    result = json.loads(run.stdout)
    out = {k: [v['x'], v['y']] for k, v in result.items()}
    assert all(np.linalg.norm(np.array(out[k]) - v) < 1e-5 for k, v in anchors.items() if k in g)
    return out


# ------------------------------------------------------------------------------------------------ scene relaxation without all-pairs terms
def _pairs(xy, radius):
    return cKDTree(xy).query_pairs(radius, output_type='ndarray').reshape(-1, 2)


def _project_spacing(xy, hard, low, high, distance, steps=80):
    """Push glyphs that are closer than `distance` apart (vectorised over neighbour pairs; hard anchors do not move)."""
    xy = xy.copy()
    for _ in range(steps):
        pairs = _pairs(xy, distance)
        if not len(pairs):
            break
        i, j = pairs[:, 0], pairs[:, 1]
        delta = xy[i] - xy[j]
        norm = np.linalg.norm(delta, axis=1)
        angle = (i * 17 + j * 31) * 2.399963
        zero = norm < 1e-9
        delta[zero] = np.stack([np.cos(angle[zero]), np.sin(angle[zero])], axis=1)
        norm[zero] = 1.0
        shift = delta / norm[:, None] * (distance - norm + .01)[:, None]
        move_i = (~hard[i]).astype(float)
        move_j = (~hard[j]).astype(float)
        share = np.where((move_i + move_j) > 0, 1.0 / np.maximum(move_i + move_j, 1.0), 0.0)
        push = np.zeros_like(xy)
        np.add.at(push, i, shift * (move_i * share)[:, None])
        np.add.at(push, j, -shift * (move_j * share)[:, None])
        xy = np.maximum(low, np.minimum(high, xy + push))
    return xy


def relax_sparse(g, positions, hints, shape_edges=(), iterations=200, rounds=8):
    """`geometry.relax` with the same energy (soft anchors, edge length 65, module shape, minimum spacing) but neighbour-list collisions."""
    cfg = geometry.CONFIG
    keys = sorted(positions)
    ids = {k: i for i, k in enumerate(keys)}
    initial = np.array([positions[k] for k in keys], float)
    hard = np.array([bool(hints.get(k, {}).get('hard', False)) for k in keys])
    weights = np.array([.008 + .04 * hints[k].get('confidence', .5) if k in hints else .002 for k in keys])
    low = np.full(initial.shape, -np.inf)
    high = np.full(initial.shape, np.inf)
    for k, hint in hints.items():
        i = ids[k]
        reference = np.array(hint['xy'], float)
        radius = 0. if hint.get('hard') else float(hint.get('allowed_motion', cfg['soft_motion']))
        initial[i] = reference
        low[i], high[i] = reference - radius, reference + radius
    ea = np.array([[ids[a], ids[b]] for a, b in sorted(tuple(sorted(e)) for e in g.edges())], dtype=int).reshape(-1, 2)
    se = np.array([[ids[a], ids[b]] for a, b in shape_edges if a in ids and b in ids], dtype=int).reshape(-1, 2)
    original_vectors = initial[se[:, 0]] - initial[se[:, 1]]
    spacing = cfg['min_node_spacing']
    x = initial.copy()
    converged = []

    def make_energy(pa, pb):
        def energy(flat):
            xy = flat.reshape(-1, 2)
            grad = 2 * weights[:, None] * (xy - initial)
            value = float(np.sum(weights[:, None] * (xy - initial) ** 2))
            if len(ea):
                delta = xy[ea[:, 0]] - xy[ea[:, 1]]
                lengths = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
                error = lengths - 65.
                value += .015 * np.sum(error ** 2)
                force = .03 * error[:, None] * delta / lengths[:, None]
                np.add.at(grad, ea[:, 0], force)
                np.add.at(grad, ea[:, 1], -force)
            if len(se):
                error = xy[se[:, 0]] - xy[se[:, 1]] - original_vectors
                value += cfg['shape_weight'] * np.sum(error ** 2)
                force = 2 * cfg['shape_weight'] * error
                np.add.at(grad, se[:, 0], force)
                np.add.at(grad, se[:, 1], -force)
            if len(pa):
                delta = xy[pa] - xy[pb]
                lengths = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
                mask = lengths < spacing
                error = lengths[mask] - spacing
                value += cfg['collision_weight'] * np.sum(error ** 2)
                force = 2 * cfg['collision_weight'] * error[:, None] * delta[mask] / lengths[mask, None]
                np.add.at(grad, pa[mask], force)
                np.add.at(grad, pb[mask], -force)
            return float(value), grad.ravel()
        return energy

    bounds = list(zip(low.ravel(), high.ravel()))
    per_round = max(10, iterations // rounds)
    for _ in range(rounds):
        pairs = _pairs(x, spacing * 2.5)  # neighbours that may collide within the next few iterations
        result = minimize(make_energy(pairs[:, 0], pairs[:, 1]), x.ravel(), jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=per_round, ftol=1e-9, maxls=30))
        x = result.x.reshape(-1, 2)
        converged.append(bool(result.success))
        if result.success:
            break
    final = _project_spacing(x, hard, low, high, spacing)
    assert np.isfinite(final).all()
    remaining = _pairs(final, spacing)
    distances = np.linalg.norm(final[remaining[:, 0]] - final[remaining[:, 1]], axis=1) if len(remaining) else np.array([])
    err = final[se[:, 0]] - final[se[:, 1]] - original_vectors
    audit = dict(optimizer_converged=bool(converged and converged[-1]), rounds=len(converged), nodes=len(keys), hard_anchors=int(hard.sum()),
                 remaining_spacing_violations=int(np.sum(distances < spacing - 1e-5)), shape_vector_rmse=float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.,
                 hint_max_drift=float(max((np.linalg.norm(final[ids[k]] - h['xy']) for k, h in hints.items() if k in ids), default=0.)),
                 motion_bound='axis-aligned box in display pixels', method='sparse neighbour-list relaxation (geometry.relax energy)')
    return {k: final[i].tolist() for i, k in enumerate(keys)}, audit


def scalable_scene_relax(model, scene, blocks, iterations=200, extra_groups=()):
    """`geometry.complete_scene` for large scenes: same hints (soft anchors on the structures' glyphs) and shape edges, sparse relaxation."""
    scene = copy.deepcopy(scene)
    g = geometry.scene_graph(model, scene)
    reactions = {r['id']: r for r in model['reactions']}
    hints, shape_edges = {}, []
    for b in blocks:
        node_keys = set()
        for rid in b['reaction_ids']:
            rk = 'r:' + rid
            if rk not in scene['positions']:
                continue
            node_keys.add(rk)
            for mid in reactions[rid]['metabolites']:
                if 'm:' + mid not in b['positions']:
                    continue
                mk = scene['participants'][rid, mid]
                if not mk.startswith('side:'):
                    node_keys.add(mk)
        for key in node_keys:
            hints.setdefault(key, dict(xy=scene['positions'][key], hard=False, confidence=b.get('anchor_confidence', .5), allowed_motion=geometry.CONFIG['soft_motion'],
                                       module_id=b['id'], origin='retrieved_template'))
        shape_edges.extend(sorted(tuple(sorted(e)) for e in g.subgraph(node_keys).edges()))
    for grp in extra_groups:  # automatic modules: softer anchors (they were laid out by the algorithm, not drawn by an expert)
        node_keys = set()
        for rid in grp['reactions']:
            rk = 'r:' + rid
            if rk not in scene['positions']:
                continue
            node_keys.add(rk)
            for mid in reactions[rid]['metabolites']:
                if 'm:' + mid in grp.get('positions', {}):
                    mk = scene['participants'][rid, mid]
                    if not mk.startswith('side:'):
                        node_keys.add(mk)
        for key in node_keys:
            hints.setdefault(key, dict(xy=scene['positions'][key], hard=False, confidence=.25, allowed_motion=60.0, module_id=grp['id'], origin='automatic_module'))
        shape_edges.extend(sorted(tuple(sorted(e)) for e in g.subgraph(node_keys).edges()))
    final, audit = relax_sparse(g, scene['positions'], hints, shape_edges, iterations)
    scene['positions'] = final
    scene['anchors'] = hints
    return scene, audit


# ------------------------------------------------------------------------------------------------ instances of GEM metabolites on the map
def instance_table(model, scene):
    """Visual instances of every GEM metabolite: `copy:` (taken from an expert structure), `m:` (the shared node of reactions outside structures),
    `side:` (a currency metabolite drawn next to each reaction).  Node ids follow drawing.export (index in the sorted scene keys)."""
    ids = {k: str(i) for i, k in enumerate(sorted(scene['positions']))}
    cur = currency(model)
    table = defaultdict(list)
    for key, mid in scene['copies'].items():
        role = key.split(':', 1)[0]
        row = dict(node=ids[key], role={'copy': 'expert_copy', 'm': 'shared', 'side': 'side', 'inst': 'local_instance'}[role], key=key)
        if key in scene.get('instance_reasons', {}):
            row['reason'] = scene['instance_reasons'][key]
        table[mid].append(row)
    return {mid: dict(currency=mid in cur, instances=v) for mid, v in table.items()}


def instance_summary(table):
    c = Counter()
    for mid, row in table.items():
        c['metabolites'] += 1
        c['instances'] += len(row['instances'])
        if row['currency']:
            c['currency_metabolites'] += 1
            c['currency_instances'] += len(row['instances'])
        else:
            c['non_currency_instances'] += len(row['instances'])
            if len(row['instances']) > 1:
                c['non_currency_metabolites_with_duplicates'] += 1
                c['non_currency_duplicate_instances'] += len(row['instances']) - 1
    return dict(c)


# ------------------------------------------------------------------------------------------------ block-level layout (2-D, driven by shared metabolites)
BLOCK_GAP = 90.0


def block_weights(model, blocks, g, cur):
    """Attraction between expert blocks: metabolites they share (non-currency), reactions outside the blocks that touch both, and a weak
    same-compartment-group term so organelles cluster."""
    owner = defaultdict(set)
    for i, b in enumerate(blocks):
        for k in b['positions']:
            if k.startswith('m:') and k[2:] not in cur:
                owner[k[2:]].add(i)
    weights = defaultdict(float)
    for mid, bs in owner.items():
        bs = sorted(bs)
        if len(bs) > 1:
            w = 1.0 / (len(bs) - 1)
            for a in range(len(bs)):
                for c in range(a + 1, len(bs)):
                    weights[bs[a], bs[c]] += w
    in_block = {rid for b in blocks for rid in b['reaction_ids']}
    for r in model['reactions']:
        if r['id'] in in_block:
            continue
        touched = sorted({i for mid in r['metabolites'] if mid not in cur for i in owner.get(mid, ())})
        if len(touched) > 1:
            w = .5 / (len(touched) - 1)
            for a in range(len(touched)):
                for c in range(a + 1, len(touched)):
                    weights[touched[a], touched[c]] += w
    by_container = defaultdict(list)
    for i, b in enumerate(blocks):
        by_container[b.get('container')].append(i)
    for members in by_container.values():
        for a in range(len(members)):
            for c in range(a + 1, len(members)):
                weights[members[a], members[c]] += .05
    return weights


def layout_blocks(sizes, weights, seed=SEEDS[0], iterations=500, aspect=1.4, init=None):
    """Rigid rectangles on a plane: weighted springs between blocks, overlap removal, mild gravity that pulls the layout towards `aspect`
    (width / height).  Deterministic.  Returns centres."""
    import networkx as nx
    n = len(sizes)
    sizes = np.asarray(sizes, float) + BLOCK_GAP
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (a, b), w in weights.items():
        G.add_edge(a, b, weight=w)
    area = float(np.sum(sizes[:, 0] * sizes[:, 1]))
    radius = np.sqrt(area * 2.0) / 2
    if init is None:
        start = nx.spring_layout(G, weight='weight', seed=seed, iterations=200, scale=1.0)
        xy = np.array([start[i] for i in range(n)]) * radius
        xy[:, 0] *= aspect ** .5
        xy[:, 1] /= aspect ** .5
    else:
        xy = np.array(init, float)  # a given arrangement (block_candidates); the same spring / overlap refinement follows
    edges = [(a, b, w) for (a, b), w in sorted(weights.items())]
    ea = np.array([[a, b] for a, b, _ in edges], dtype=int).reshape(-1, 2)
    ew = np.array([w for *_, w in edges], float)
    half = sizes / 2
    wsum = np.zeros(n)
    if len(ea):
        np.add.at(wsum, ea[:, 0], ew)
        np.add.at(wsum, ea[:, 1], ew)
    for it in range(iterations):
        step = .35 * (1 - it / iterations) + .05
        push = np.zeros_like(xy)
        if len(ea):
            d = xy[ea[:, 1]] - xy[ea[:, 0]]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            direction = d / dist[:, None]
            touch = np.abs(direction[:, 0]) * (half[ea[:, 0], 0] + half[ea[:, 1], 0]) + np.abs(direction[:, 1]) * (half[ea[:, 0], 1] + half[ea[:, 1], 1])
            pull = np.clip(dist - touch, 0, None) * ew
            np.add.at(push, ea[:, 0], direction * pull[:, None])
            np.add.at(push, ea[:, 1], -direction * pull[:, None])
        push /= (wsum[:, None] + 1.0)  # weighted-mean step: stable however many neighbours a hub block has
        centre = xy.mean(0)
        push -= (xy - centre) * np.array([.015 / aspect ** .5, .015 * aspect ** .5])  # stronger gravity
        xy += push * min(1.0, step * 2)
        xy = _separate(xy, half, passes=3)
    xy = _separate(xy, half, passes=400)
    # Final compaction pass to squeeze out empty space
    xy = _compact(xy, half, weights, iterations=40)
    return xy


def _separate(xy, half, passes=3):
    """Remove rectangle overlaps by pushing apart along the axis of smaller penetration (in place)."""
    n = len(xy)
    P, H = xy.tolist(), half.tolist()  # plain floats: same arithmetic, no per-element array access
    for _ in range(passes):
        moved = False
        order = np.argsort(np.array([p[0] for p in P])).tolist()
        for ii in range(n):
            pi, hi = P[order[ii]], H[order[ii]]
            for jj in range(ii + 1, n):
                pj, hj = P[order[jj]], H[order[jj]]
                dx = pj[0] - pi[0]
                if dx > hi[0] + hj[0]:
                    break
                dy = pj[1] - pi[1]
                px = hi[0] + hj[0] - abs(dx)
                py = hi[1] + hj[1] - abs(dy)
                if px > 0 and py > 0:
                    moved = True
                    if px < py:
                        sgn = 1.0 if dx >= 0 else -1.0
                        pi[0] -= sgn * px / 2
                        pj[0] += sgn * px / 2
                    else:
                        sgn = 1.0 if dy >= 0 else -1.0
                        pi[1] -= sgn * py / 2
                        pj[1] += sgn * py / 2
        if not moved:
            break
    xy[:] = P
    return xy


def _compact(xy, half, weights, iterations=50):
    """Pull blocks toward centre while respecting overlaps, weighted by connectivity.
    Blocks with more connections have more pull toward the dense core."""
    n = len(xy)
    if n < 2:
        return xy
    xy = xy.copy()
    # Compute weight sum for each block (how connected it is)
    wsum = np.zeros(n)
    for (a, b), w in weights.items():
        wsum[a] += w
        wsum[b] += w
    wsum = np.maximum(wsum, 0.1)
    
    for it in range(iterations):
        # Weighted centre (more connected blocks pull harder)
        weighted_centre = np.average(xy, axis=0, weights=wsum)
        # Pull toward centre, stronger for less connected blocks (they should move more)
        strength = 0.15 * (1.0 - it / iterations)
        for i in range(n):
            direction = weighted_centre - xy[i]
            dist = np.linalg.norm(direction)
            if dist > 1.0:
                # Less connected blocks move more
                move = direction / dist * min(dist * strength / (wsum[i] + 0.1), 20.0)
                xy[i] += move
        # Resolve overlaps
        xy = _separate(xy, half, passes=3)
    return xy


# ------------------------------------------------------------------------------------------------ G1.5: automatic modules for reactions no expert structure covers
MAX_AUTO = 90        # soft size limit of an automatic module (reactions); larger communities are split
MIN_AUTO = 4         # communities smaller than this are merged into their best neighbour
LOCAL_SPACING = (110.0, 75.0, 60.0)  # minimum centre distances inside automatic modules (reaction-reaction, metabolite-metabolite, reaction-metabolite)
BACKBONE = False        # V7: fixed backbone chain inside automatic modules (set by tool_pipeline for --layout-style expert)
backbone_stats = {}
BLOCK_LAYOUT = 'auto'  # 'auto' (stress/spring candidates scored for compactness + long links) | 'single' (one spring arrangement)
HUB = 30             # a metabolite that touches more modules than this does not glue modules together


def _louvain(graph, resolution, seed):
    from networkx.algorithms.community import louvain_communities
    return [sorted(c, key=str) for c in louvain_communities(graph, weight='weight', resolution=resolution, seed=seed)]


def reaction_communities(model, g, blocks, cur):
    """Communities of the reactions outside the expert blocks, from the real network.

    Each expert block is contracted to one node so the structures stay whole; free reactions are linked by the (non-currency) metabolites
    they share, each shared metabolite contributing 1/(n-1) (hubs contribute nothing), reduced when the compartment groups differ.  Louvain on
    this graph; communities above MAX_AUTO free reactions are split again with a higher resolution, communities below MIN_AUTO are merged into
    the neighbour they are most strongly linked to.  Reactions without any non-currency metabolite go to compartment groups of at most 25.
    Returns (groups, stats); a group is dict(id, reactions, compartment).
    """
    import networkx as nx
    in_block = {rid: i for i, b in enumerate(blocks) for rid in b['reaction_ids']}
    free = [r['id'] for r in model['reactions'] if r['id'] not in in_block]
    comp = {rid: g.nodes['r:' + rid].get('compartment', 'unknown') for rid in free}
    node_of = lambda rid: ('b', in_block[rid]) if rid in in_block else ('r', rid)
    users = defaultdict(set)
    for r in model['reactions']:
        for mid in r['metabolites']:
            if mid not in cur:
                users[mid].add(node_of(r['id']))
    H = nx.Graph()
    H.add_nodes_from(('r', rid) for rid in free)
    H.add_nodes_from(('b', i) for i in range(len(blocks)))
    block_comp = {('b', i): b.get('container') for i, b in enumerate(blocks)}
    for mid, nodes in users.items():
        nodes = sorted(nodes, key=str)
        if len(nodes) < 2 or len(nodes) > HUB:
            continue
        w = 1.0 / (len(nodes) - 1)
        for a in range(len(nodes)):
            for c in range(a + 1, len(nodes)):
                na, nc = nodes[a], nodes[c]
                same = comp.get(na[1]) == comp.get(nc[1]) if na[0] == 'r' and nc[0] == 'r' else True
                weight = w * (1.0 if same else .4)
                if H.has_edge(na, nc):
                    H[na][nc]['weight'] += weight
                else:
                    H.add_edge(na, nc, weight=weight)
    isolated = [n for n in H if H.degree(n) == 0 and n[0] == 'r']
    work = H.subgraph([n for n in H if n not in set(isolated)]).copy()
    seed = SEEDS[0]
    communities = _louvain(work, 1.0, seed) if work.number_of_nodes() else []

    def free_count(c):
        return sum(1 for n in c if n[0] == 'r')

    final, queue = [], list(communities)
    while queue:
        c = queue.pop()
        if free_count(c) > MAX_AUTO and len(c) > 2:
            parts = _louvain(work.subgraph(c).copy(), 2.5, seed)
            if len(parts) > 1:
                queue.extend(parts)
                continue
        final.append(c)
    # merge tiny communities into the neighbour with the strongest link
    index = {n: i for i, c in enumerate(final) for n in c}
    sizes = [free_count(c) for c in final]
    for i in sorted(range(len(final)), key=lambda i: sizes[i]):
        if sizes[i] == 0 or sizes[i] >= MIN_AUTO:
            continue
        link = Counter()
        for n in final[i]:
            for nb in work[n]:
                if index[nb] != i:
                    link[index[nb]] += work[n][nb]['weight']
        target = max(link, key=link.get) if link else None
        if target is not None and sizes[target] + sizes[i] <= MAX_AUTO * 1.3:
            final[target] = final[target] + final[i]
            for n in final[i]:
                index[n] = target
            sizes[target] += sizes[i]
            final[i], sizes[i] = [], 0
    groups = []
    for c in final:
        rids = sorted(n[1] for n in c if n[0] == 'r')
        if rids:
            groups.append(dict(reactions=rids, expert_blocks=sorted(n[1] for n in c if n[0] == 'b')))
    by_comp = defaultdict(list)
    for n in sorted(isolated):
        by_comp[comp[n[1]]].append(n[1])
    for roles, rids in sorted(by_comp.items()):
        for i in range(0, len(rids), 25):
            groups.append(dict(reactions=rids[i:i + 25], expert_blocks=[], isolated=True))
    for k, grp in enumerate(groups):
        roles = Counter(comp[r] for r in grp['reactions'])
        grp['id'] = 'auto%d' % k
        grp['compartment'] = roles.most_common(1)[0][0]
    stats = dict(auto_modules=len(groups), free_reactions=len(free), sizes=sorted((len(x['reactions']) for x in groups), reverse=True),
                 isolated_reactions=len(isolated), contracted_nodes=H.number_of_nodes(), contracted_edges=H.number_of_edges())
    return groups, stats


def local_layouts(model, g, groups, cur, timeout=900):
    """Layout of every automatic module on its own (fCoSE, nothing fixed).  All modules go to one backend call: each starts in its own far-away
    cell, so they do not interact.  Returns {group id: {node key: [x, y]}} relative to the module's own top-left corner, and sizes."""
    import networkx as nx
    reactions = {r['id']: r for r in model['reactions']}
    elements, edges, cells = [], [], {}
    anchors_fixed, fixed_here = [], {}
    backbone_stats.clear()
    backbone_stats.update(chains=0, reactions=0, lengths=[], direction_known=0)
    side = int(np.ceil(np.sqrt(max(len(groups), 1))))
    members = {}
    for k, grp in enumerate(groups):
        nodes = set()
        for rid in grp['reactions']:
            nodes.add('r:' + rid)
            for mid in reactions[rid]['metabolites']:
                if mid not in cur:
                    nodes.add('m:' + mid)
        sub = g.subgraph(nodes)
        members[grp['id']] = (sorted(sub.nodes()), sorted(tuple(sorted(e)) for e in sub.edges()))
        scale = max(100.0, 38.0 * np.sqrt(max(len(sub), 1)))
        pos = nx.spring_layout(sub, seed=SEEDS[0], iterations=60, scale=scale) if len(sub) > 1 else {n: np.zeros(2) for n in sub}
        ox, oy = (k % side) * 20000.0, (k // side) * 20000.0
        cells[grp['id']] = (ox, oy)
        chain_pos = {}
        if BACKBONE:   # V7: the longest chain of the module is a fixed backbone (straight, evenly spaced, folded into rows), the rest is laid out around it
            import pathway_backbone
            chain = pathway_backbone.find_chain(reactions, grp['reactions'], cur)
            if chain:
                chain_pos = {n: p for n, p in pathway_backbone.chain_positions(chain).items() if n in sub}
                backbone_stats['chains'] += 1
                backbone_stats['reactions'] += len(chain['reactions'])
                backbone_stats['lengths'].append(len(chain['reactions']))
                backbone_stats['direction_known'] += int(chain['direction'] != 0)
        fixed_here[grp['id']] = set(chain_pos)
        for n in sub.nodes():
            xy = chain_pos.get(n, pos[n])
            elements.append(dict(data=dict(id=grp['id'] + '|' + n), position=dict(x=float(xy[0] + ox), y=float(xy[1] + oy))))
            if n in chain_pos:
                anchors_fixed.append(dict(nodeId=grp['id'] + '|' + n, position=dict(x=float(xy[0] + ox), y=float(xy[1] + oy))))
        for i, (a, b) in enumerate(members[grp['id']][1]):
            edges.append(dict(data=dict(id=f"e{grp['id']}|{a}|{b}", source=grp['id'] + '|' + a, target=grp['id'] + '|' + b)))
    payload = dict(backend='fcose', elements=elements + edges, anchors=anchors_fixed, seed=SEEDS[0], iterations=350)
    run = subprocess.run(['node', str(BASE / 'runtime/layout_backend.cjs')], input=json.dumps(payload), text=True, capture_output=True, timeout=timeout, check=True)
    result = json.loads(run.stdout)
    import completion
    out, sizes = {}, {}
    for grp in groups:
        gid = grp['id']
        ox, oy = cells[gid]
        pts = {n: [result[gid + '|' + n]['x'] - ox, result[gid + '|' + n]['y'] - oy] for n in members[gid][0]}
        sub = g.subgraph(members[gid][0])
        completion.spread(pts, sub, {n: 1 for n in fixed_here.get(gid, ())}, spacing=LOCAL_SPACING)  # keep labels and glyphs apart; the backbone stays
        arr = np.array(list(pts.values()))
        lo = arr.min(0)
        out[gid] = {n: (np.array(p) - lo).tolist() for n, p in pts.items()}
        sizes[gid] = (arr.max(0) - lo).tolist()
    return out, sizes, members


def group_weights(model, groups, cur):
    """Attraction between every pair of modules (expert blocks first, then automatic ones): shared non-currency metabolites, hubs excluded,
    plus a weak same-compartment term."""
    owner = defaultdict(set)
    for gi, grp in enumerate(groups):
        for rid in grp['reactions']:
            for mid in grp['_rx'][rid]['metabolites']:
                if mid not in cur:
                    owner[mid].add(gi)
    weights = defaultdict(float)
    for mid, gs in owner.items():
        gs = sorted(gs)
        if 1 < len(gs) <= HUB:
            w = 1.0 / (len(gs) - 1)
            for a in range(len(gs)):
                for c in range(a + 1, len(gs)):
                    weights[gs[a], gs[c]] += w
    by_comp = defaultdict(list)
    for gi, grp in enumerate(groups):
        by_comp[grp.get('compartment')].append(gi)
    for members in by_comp.values():
        if len(members) <= 60:
            for a in range(len(members)):
                for c in range(a + 1, len(members)):
                    weights[members[a], members[c]] += .05
    return weights


def _shared_mets(rx_a, rx_b, cur):
    """Non-currency metabolites shared by two reaction-id sets."""
    ma = {m for r in rx_a.values() for m in r['metabolites'] if m not in cur}
    mb = {m for r in rx_b.values() for m in r['metabolites'] if m not in cur}
    return ma & mb


def _local_met_offset(block_pos, shared, lo, size):
    """Mean of shared-met glyphs in block coords, as offset from block centre (or None)."""
    pts = []
    for k, v in (block_pos or {}).items():
        if not k.startswith('m:'):
            continue
        mid = k[2:]
        if mid in shared:
            pts.append(v)
    if not pts:
        return None
    mean = np.mean(np.asarray(pts, float), axis=0)
    centre = np.asarray(lo, float) + np.asarray(size, float) / 2.0
    return mean - centre


def _face_shared_metabolites(items, centres, weights, n_expert, blocks=None, cur=None,
                             max_nudge=80.0, face_axis=False):
    """Nudge curated block centres so linked pairs sit closer (shared-met sides optional).

    Translation only.  ``face_axis`` reorients pairs onto the shared-metabolite side; leave
    it off when the packing already satisfies aspect — axis flips raise curated×curated
    crossings.  Returns (centres, report).
    """
    centres = np.asarray(centres, float).copy()
    if n_expert < 2:
        return centres, dict(nudged=0)
    cur = cur or set()
    blocks = blocks or []
    nudged = 0
    faced = 0
    pairs = sorted(((w, a, b) for (a, b), w in weights.items()
                    if a < n_expert and b < n_expert and w >= 0.25), reverse=True)
    half = np.array([np.asarray(items[i]['size'], float) / 2.0 for i in range(n_expert)])
    for w, a, b in pairs[:60]:
        delta = centres[b] - centres[a]
        dist = float(np.hypot(*delta))
        if dist < 1e-6:
            continue
        axis = None
        if face_axis:
            shared = _shared_mets(items[a].get('_rx') or {}, items[b].get('_rx') or {}, cur)
            if shared and a < len(blocks) and b < len(blocks):
                oa = _local_met_offset(blocks[a].get('positions'), shared, items[a]['lo'], items[a]['size'])
                if oa is not None and float(np.hypot(*oa)) > 1.0:
                    axis = oa / (np.linalg.norm(oa) + 1e-9)
        if axis is not None:
            mid = 0.5 * (centres[a] + centres[b])
            sep = float(np.abs(axis[0]) * (half[a, 0] + half[b, 0]) +
                        np.abs(axis[1]) * (half[a, 1] + half[b, 1])) + 50.0
            for idx, tgt in ((a, mid - axis * (sep / 2.0)), (b, mid + axis * (sep / 2.0))):
                step = tgt - centres[idx]
                n = float(np.hypot(*step))
                if n > max_nudge:
                    step = step * (max_nudge / n)
                centres[idx] = centres[idx] + step
            faced += 1
            nudged += 1
            continue
        touch = float(np.abs(delta[0]) / dist * (half[a, 0] + half[b, 0]) +
                      np.abs(delta[1]) / dist * (half[a, 1] + half[b, 1]))
        gap = dist - touch
        if gap <= 40.0:
            continue
        pull = min(max_nudge, gap - 40.0) * min(1.0, w / 2.0)
        step = (delta / dist) * (pull / 2.0)
        centres[a] = centres[a] + step
        centres[b] = centres[b] - step
        nudged += 1
    if nudged and n_expert >= 2:
        sizes = np.array([items[i]['size'] for i in range(len(centres))], float) + BLOCK_GAP
        centres = _separate(centres, sizes / 2.0, passes=60)
    return centres, dict(nudged=nudged, faced=faced, pairs_considered=len(pairs), face_axis=bool(face_axis))


def plan_global(model, selected, bridges, g, log=None):
    """Complete placement of the whole model: expert blocks (rigid, as packed by pack_modules) and automatic modules (rigid, locally laid out) are
    placed together on one plane by their shared metabolites.  Returns dict(anchors, blocks, containers, routes, groups, stats)."""
    import time
    timings, stamp = {}, [time.time()]

    def lap(name):
        timings[name] = round(time.time() - stamp[0], 1)
        stamp[0] = time.time()
        if log:
            log(f'  {name}: {timings[name]} s')
    cur = currency(model)
    anchors, blocks, containers, routes = drawing.pack_modules(model, copy.deepcopy(selected), bridges, physical_scale=True)
    import curated_axes
    # Same-source fragments → one rigid group at original relative positions (absolute map coords).
    blocks, same_map_bake = curated_axes.bake_same_map_positions(blocks, model_n_reactions=len(model['reactions']))
    rigid_groups = curated_axes.same_map_packing_groups(blocks)
    if log and same_map_bake.get('blocks'):
        log(f"Same-map rigid: {same_map_bake['blocks']} block(s) in {same_map_bake['groups']} group(s) "
            f"mode={same_map_bake.get('mode')}")
        pad = same_map_bake.get('padding_compression') or {}
        if pad.get('applied'):
            log(f"  padding compress: {pad.get('segments_compressed', 0)} segment(s) "
                f"(x={pad.get('gaps_x', 0)}/−{pad.get('removed_x', 0)}px, "
                f"y={pad.get('gaps_y', 0)}/−{pad.get('removed_y', 0)}px) "
                f"max_gap={pad.get('max_gap')}")
    lap('curated blocks (ELK)')
    reactions = {r['id']: r for r in model['reactions']}
    groups, stats = reaction_communities(model, g, blocks, cur)
    lap('communities of uncovered reactions')
    layouts, sizes, members = local_layouts(model, g, groups, cur)
    lap('local layout of automatic modules')
    # Packing items: each same-map rigid group is ONE composite tile; other expert blocks stay individuals.
    member_of = {}
    for mid, idxs in rigid_groups.items():
        for i in idxs:
            member_of[i] = mid
    items = []
    expert_plan = []  # parallel to centres[:n_pack_expert]: ('block', i) or ('rigid', mid, idxs)
    seen_rigid = set()
    for i, b in enumerate(blocks):
        mid = member_of.get(i)
        if mid is not None:
            if mid in seen_rigid:
                continue
            seen_rigid.add(mid)
            idxs = rigid_groups[mid]
            pts = np.vstack([np.array(list(blocks[j]['positions'].values()), float) for j in idxs])
            lo, hi = pts.min(0), pts.max(0)
            rx_all = [rid for j in idxs for rid in blocks[j]['reaction_ids']]
            items.append(dict(kind='expert_rigid', reactions=list(rx_all), lo=lo, size=hi - lo + 80,
                              compartment=blocks[idxs[0]].get('container'),
                              _rx={rid: reactions[rid] for rid in rx_all if rid in reactions},
                              map_id=blocks[idxs[0]].get('map_id'), same_map_rigid=mid,
                              block_indexes=list(idxs)))
            expert_plan.append(('rigid', mid, list(idxs)))
        else:
            pts = np.array(list(b['positions'].values()), float)
            lo, hi = pts.min(0), pts.max(0)
            items.append(dict(kind='expert', reactions=list(b['reaction_ids']), lo=lo, size=hi - lo + 80,
                              compartment=b.get('container'), _rx={rid: reactions[rid] for rid in b['reaction_ids']},
                              map_id=b.get('map_id'), block_indexes=[i]))
            expert_plan.append(('block', i, [i]))
    n_expert_items = len(items)
    for grp in groups:
        sz = np.array(sizes[grp['id']]) + 80
        items.append(dict(kind='auto', reactions=grp['reactions'], size=sz, compartment=grp['compartment'],
                          id=grp['id'], _rx={rid: reactions[rid] for rid in grp['reactions']}))
    for it in items:
        it['compartment'] = it.get('compartment')
        # Cap pathological local-layout / ELK extents so one runaway tile cannot
        # inflate skyline / spring canvases into tens of thousands of pixels.
        # Rigid same-map backbones keep their true AABB (must match lo for the
        # later centre→delta translation); only soft-cap absurd outliers.
        sz = np.asarray(it['size'], float)
        nrx = max(len(it.get('reactions') or []), 1)
        if it.get('kind') == 'expert_rigid':
            cap = max(float(sz.max()), 80.0 + 120.0 * max(8.0, float(np.sqrt(nrx) * 6.0)))
            it['size'] = np.minimum(sz, cap)
        else:
            cap = 80.0 + 140.0 * max(8.0, float(nrx))
            it['size'] = np.minimum(sz, cap)
    weights = group_weights(model, [dict(reactions=i['reactions'], compartment=i['compartment'], _rx=i['_rx']) for i in items], cur)
    # Soft same-map attraction only for expert tiles that were NOT merged into a rigid composite.
    SAME_MAP_W = 0.14
    for i in range(n_expert_items):
        for j in range(i + 1, n_expert_items):
            if items[i].get('same_map_rigid') or items[j].get('same_map_rigid'):
                continue
            if items[i].get('map_id') and items[i].get('map_id') == items[j].get('map_id'):
                weights[i, j] = weights.get((i, j), 0.0) + SAME_MAP_W
    # Amplify chemical neighbour weights so connected blocks sit adjacent (long-line gate).
    for key in list(weights):
        weights[key] = weights[key] * 1.30
    lap('module weights')
    if BLOCK_LAYOUT == 'auto':
        import block_candidates
        _, links = block_candidates.module_links([dict(reactions=i['reactions'], _rx=i['_rx']) for i in items], cur, HUB)
        centres, placement = block_candidates.place(items, weights, links, layout_blocks, log=log)
    else:
        centres, placement = layout_blocks([i['size'] for i in items], weights, aspect=1.5), dict(selection=dict(chosen='spring', rule='single arrangement (block_layout=single)'))
    centres = np.asarray(centres, float)
    # Gentle pull-together only (no axis flip): keep strongly linked curated pairs close
    # without reshuffling the aspect-feasible packing enough to raise curated×curated crossings.
    centres, face_report = _face_shared_metabolites(
        items, centres, weights, n_expert_items, blocks=blocks, cur=cur, max_nudge=80.0, face_axis=False)
    placement = dict(placement or {}, same_map_weights=SAME_MAP_W, link_weight_scale=1.30,
                     face_shared=face_report, same_map_bake=same_map_bake,
                     rigid_groups={str(k)[:24]: v for k, v in rigid_groups.items()})
    lap('placement of all modules')
    new_anchors = {}
    # Expert tiles (individuals or rigid composites): one shared translation per tile.
    for plan, it, centre in zip(expert_plan, items[:n_expert_items], centres[:n_expert_items]):
        delta = centre - it['size'] / 2 + 40 - it['lo']
        for bi in plan[2]:
            b = blocks[bi]
            b['positions'] = {k: (np.array(v) + delta).tolist() for k, v in b['positions'].items()}
            for k, v in b['positions'].items():
                if k.startswith('r:'):
                    anchors[k] = v
                else:
                    new_anchors.setdefault(k, []).append(v)
    expert_met = {k for k in new_anchors}
    for k, vals in new_anchors.items():
        anchors[k] = np.mean(vals, axis=0).tolist()
    # automatic modules: reactions and the metabolites they are "home" for
    home = defaultdict(Counter)
    for grp in groups:
        for rid in grp['reactions']:
            for mid in reactions[rid]['metabolites']:
                if mid not in cur and 'm:' + mid not in expert_met:
                    home[mid][grp['id']] += 1
    home_of = {mid: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for mid, c in home.items()}
    for grp, it, centre in zip(groups, items[n_expert_items:], centres[n_expert_items:]):
        local = layouts[grp['id']]
        offset = centre - it['size'] / 2 + 40
        grp['positions'] = {}
        for n, p in local.items():
            xy = (np.array(p) + offset).tolist()
            if n.startswith('r:'):
                anchors[n] = xy
                grp['positions'][n] = xy
            elif home_of.get(n[2:]) == grp['id']:
                anchors[n] = xy
                grp['positions'][n] = xy
        grp['size'] = it['size'].tolist()
    lap('anchors')
    missing = [k for k in g if k not in anchors]
    # expert-side bookkeeping used by the callers
    contain = defaultdict(list)
    for b in blocks:
        pts = np.array(list(b['positions'].values()), float)
        contain[b['container']].append((pts.min(0) - 30, pts.max(0) + 30))
    roles = {c['id']: c['roles'] for c in containers}
    containers = [dict(id=cid, roles=roles.get(cid, []), x=float(min(lo[0] for lo, _ in bx)), y=float(min(lo[1] for lo, _ in bx)),
                       width=float(max(hi[0] for _, hi in bx) - min(lo[0] for lo, _ in bx)), height=float(max(hi[1] for _, hi in bx) - min(lo[1] for lo, _ in bx)))
                  for cid, bx in contain.items()]
    # how the automatic modules attach to the expert blocks (modules that share a non-currency metabolite)
    block_mets = defaultdict(set)
    for i, b in enumerate(blocks):
        for rid in b['reaction_ids']:
            for mid in reactions[rid]['metabolites']:
                if mid not in cur:
                    block_mets[mid].add(i)
    attached = 0
    for grp in groups:
        linked = Counter()
        for rid in grp['reactions']:
            for mid in reactions[rid]['metabolites']:
                for i in block_mets.get(mid, ()):
                    linked[i] += 1
        grp['linked_expert_blocks'] = len(linked)
        attached += bool(linked)
    stats.update(auto_modules_linked_to_expert=attached, auto_modules_not_linked=len(groups) - attached)
    stats.update(missing_anchors=len(missing), timings=timings, expert_blocks=len(blocks), placement=placement, backbone=dict(backbone_stats, lengths=sorted(backbone_stats.get('lengths', []), reverse=True)[:30]))
    return dict(anchors=anchors, blocks=blocks, containers=containers, routes=routes, groups=groups, stats=stats, missing=missing)
