"""EC-number evidence channel for module transfer (opt-in, additive).

A source module reaction is linked to a target-model reaction when both carry the same *fully specified*
EC number (a.b.c.d with four numeric levels).  EC is a weak identity: one EC can cover several reactions and many
reactions (transport, exchange, biomass, spontaneous) have no EC.  The channel therefore

* never replaces the strong channel (complete-equation matching in `semantic_transfer`);
* requires the module's reaction/metabolite incidence to fit the target network (graph check), not just EC equality;
* maps a core metabolite only when it is uniquely determined (identity candidates or structure) and leaves the rest
  unmapped instead of guessing;
* labels every proposal `match_level='ec_topology'`, biochemical level 1, with the shared EC numbers as evidence.

Transferred and deleted EC numbers are brought up to date with ExPASy ENZYME (`enzyme.dat`); when several
candidate ECs remain, Rhea reaction participants (ChEBI) score them (open data only — KEGG is not used).
For Atlas modules that only name their metabolites (BRENDA, some CellDesigner maps), a weighted Jaccard of
normalised names can link a source reaction to a target reaction and filter EC hits.  Method ideas from
ecmapper (GPL-3.0; independent rewrite).

Existing retrieval, ranking and saved results are untouched; this module only adds candidates.
"""
import csv
import itertools
import re
import urllib.request
from functools import lru_cache

from core import *
import identifier_bridge as ib
import semantic_transfer as st

CONFIG = dict(module_combinations=16, proposal_budget=6000, min_module_size=2, max_unmapped_fraction=.5)
FULL_EC = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
_EC_TOKEN = re.compile(r'\d+\.\d+\.\d+\.n?\d+')
_PARTIAL_EC = re.compile(r'^\d+(\.\d+){0,3}(\.-)?$')
ENZYME_URL = 'https://ftp.expasy.org/databases/enzyme/enzyme.dat'
ENZYME_CACHE = BASE / 'runtime' / 'cache' / 'enzyme.dat'


def full_ecs(values):
    return {v for v in values if isinstance(v, str) and FULL_EC.match(v)}


def normalize_ec(value):
    """Strip `EC` prefixes; pad incomplete numbers with `.-` (e.g. `2.7.1` -> `2.7.1.-`)."""
    if not value:
        return None
    s = re.sub(r'^(ec[\s:_-]*)+', '', str(value).strip(), flags=re.I)
    s = s.replace(' ', '')
    if FULL_EC.match(s) or re.match(r'^\d+\.\d+\.\d+\.n\d+$', s) or re.match(r'^\d+\.\d+\.\d+\.[Bb]\d+$', s):
        return s
    parts = s.split('.')
    if not parts or not all(p.isdigit() or p == '-' for p in parts):
        return None
    while len(parts) < 4:
        parts.append('-')
    return '.'.join(parts[:4]) if _PARTIAL_EC.match('.'.join(parts[:4])) else None


def is_complete(ec):
    return bool(ec and FULL_EC.match(ec))


def covers(general, specific):
    """Whether incomplete `general` includes `specific` (e.g. 1.1.1.- covers 1.1.1.1)."""
    if not general or not specific or general == specific:
        return False
    return all(g in ('-', s) for g, s in zip(general.split('.'), specific.split('.')))


def drop_covered(ecs):
    ecs = set(ecs)
    return sorted(ec for ec in ecs if is_complete(ec) or not any(covers(ec, other) for other in ecs))


# --- ExPASy ENZYME (transferred / deleted) -------------------------------------------------------

@lru_cache(maxsize=1)
def enzyme_table():
    """Parse `enzyme.dat` -> {ec: dict(transferred_to=tuple, deleted=bool)}."""
    path = ENZYME_CACHE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ENZYME_URL, timeout=60) as r, path.open('wb') as w:
            w.write(r.read())
    entries = {}
    ec, description = None, []
    with path.open(encoding='utf-8') as f:
        for line in f:
            code, value = line[:2], line[5:].strip()
            if code == 'ID':
                ec, description = value, []
            elif code == 'DE':
                description.append(value)
            elif code == '//':
                if ec is not None:
                    text = ' '.join(description)
                    if text.startswith('Transferred entry:'):
                        entries[ec] = dict(transferred_to=tuple(_EC_TOKEN.findall(text)), deleted=False)
                    elif text.startswith('Deleted entry'):
                        entries[ec] = dict(transferred_to=(), deleted=True)
                    else:
                        entries[ec] = dict(transferred_to=(), deleted=False)
                ec = None
    return entries


def current_ecs(ecs):
    """Replace transferred EC numbers by their successors; drop deleted ones."""
    table = enzyme_table()
    result, todo, seen = set(), list(ecs), set()
    while todo:
        ec = todo.pop()
        if not ec or ec in seen:
            continue
        seen.add(ec)
        entry = table.get(ec)
        if entry is None or not (entry['transferred_to'] or entry['deleted']):
            result.add(ec)
        elif entry['transferred_to']:
            todo.extend(entry['transferred_to'])
    return drop_covered(result)


# --- Rhea participants per EC (open substitute for KEGG compound lists) --------------------------

@lru_cache(maxsize=1)
def rhea_to_ec():
    table = defaultdict(set)
    with (DATA / 'rhea/rhea2xrefs.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['DB'] == 'EC':
                for key in (row['RHEA_ID'], row['MASTER_ID']):
                    table[key].add(row['ID'])
    return table


@lru_cache(maxsize=1)
def ec_to_rhea():
    table = defaultdict(set)
    with (DATA / 'rhea/rhea2xrefs.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['DB'] == 'EC':
                table[row['ID']].add(row['MASTER_ID'] or row['RHEA_ID'])
    return table


@lru_cache(maxsize=1)
def rhea_ec_chebi():
    """EC -> list of frozensets of ChEBI ids (one frozenset per Rhea master reaction)."""
    from semantics import rhea_index
    ri = rhea_index()
    out = defaultdict(list)
    for ec, rheas in ec_to_rhea().items():
        for rid in rheas:
            reaction = ri['reactions'].get(rid) or ri['reactions'].get(ri['master'].get(rid, rid))
            if not reaction:
                continue
            chebis = frozenset(p['chebi'] for p in reaction['participants'] if p.get('chebi'))
            if chebis and chebis not in out[ec]:
                out[ec].append(chebis)
    return dict(out)


@lru_cache(maxsize=1)
def bigg_to_ec():
    """BiGG reaction id -> fully specified EC numbers, from `data/identifier_bridge.json`."""
    return ib.bigg_to_ec()


def score_ec_by_metabolites(reaction_chebi_sets, ec):
    """Best weighted Jaccard of the model metabolites (sets of ChEBI) vs one Rhea reaction of `ec`."""
    refs = rhea_ec_chebi().get(ec) or []
    if not reaction_chebi_sets or not refs:
        return None
    left = [frozenset(f'chebi:{c}' for c in s) for s in reaction_chebi_sets]
    return max(ib.weighted_jaccard(left, [frozenset(f'chebi:{c}' for c in ref)]) for ref in refs)


def select_ecs(candidates, reaction_chebi_sets=None, min_score=None, min_margin=None):
    """Disambiguate several EC numbers: currentise, drop covered incompletes, score via Rhea metabolites.

    Returns (selected_list, selection_tag, scores_dict).  No BKMS / KEGG / organism tie-breakers
    in the default path (open data only; organism tie-breakers are opt-in elsewhere and disagreed
    often against curated models).
    """
    min_score = ib.THRESHOLDS['ec_min_score'] if min_score is None else min_score
    min_margin = ib.THRESHOLDS['ec_min_margin'] if min_margin is None else min_margin
    candidates = current_ecs(candidates)
    if not candidates:
        return [], 'none', {}
    if len(candidates) == 1:
        return candidates, 'single', {}
    scores = {}
    if reaction_chebi_sets:
        for ec in candidates:
            scores[ec] = score_ec_by_metabolites(reaction_chebi_sets, ec)
        known = {ec: s for ec, s in scores.items() if s is not None}
        best = max(known.values(), default=0.0)
        if best >= min_score:
            kept = sorted(ec for ec, s in known.items() if s >= best - min_margin)
            if len(kept) == 1:
                return kept, 'metabolites', scores
            if kept:
                return kept, 'ambiguous', scores
    return candidates, 'ambiguous', scores


def source_ecs(reaction):
    """Fully specified *current* EC numbers of a source module reaction, with their origin."""
    origin = defaultdict(set)
    for v in reaction.get('references', {}).get('ec-code', []):
        n = normalize_ec(str(v))
        if n:
            origin['source_annotation'].add(n)
    if reaction.get('ec'):
        n = normalize_ec(str(reaction['ec']))
        if n:
            origin['source_annotation'].add(n)
    for v in reaction.get('references', {}).get('rhea', []):
        origin['rhea2xrefs'] |= rhea_to_ec().get(str(v), set())
    for v in reaction.get('references', {}).get('bigg.reaction', []):
        origin['bigg_annotation'] |= bigg_to_ec().get(str(v), set())
    raw = set().union(*origin.values()) if origin else set()
    merged = set(current_ecs(full_ecs(raw) | {e for e in raw if is_complete(e)}))
    # keep provisional / incomplete only when nothing complete remains
    if not merged:
        merged = set(current_ecs(raw)) if raw else set()
    return full_ecs(merged) or set(drop_covered(merged)), {k: sorted(full_ecs(current_ecs(v))) for k, v in origin.items() if full_ecs(current_ecs(v))}


def attach_map_metabolites(modules):
    """Attach normalised metabolite key lists to reactions that only have map labels (BRENDA / CD)."""
    for m in modules:
        labels = {n['key']: (n.get('label') or n.get('name') or '') for n in m.get('nodes', [])}
        for r in m.get('reactions', []):
            mains, cofactors = [], []
            for mk in r.get('core', []):
                name = labels.get(mk) or ''
                if name:
                    mains.append(name)
            for a in m.get('attachments', []):
                # open-core modules store attachment glyph ids as strings; BRENDA/CD use dicts with names
                if isinstance(a, dict) and a.get('reaction') == r.get('key') and a.get('name'):
                    cofactors.append(a['name'])
            keys = []
            for name in mains + cofactors:
                k = ib.metabolite_keys(names=[name])
                if k and k not in keys:
                    keys.append(k)
            r['map_metabolite_names'] = dict(main=mains, cofactor=cofactors)
            r['map_metabolite_keys'] = keys


def enrich(modules):
    """Attach `ec_numbers` / `ec_origin` and map metabolite keys to module reactions in memory."""
    attach_map_metabolites(modules)
    have = total = 0
    for m in modules:
        for r in m.get('reactions', []):
            ecs, origin = source_ecs(r)
            r['ec_numbers'], r['ec_origin'] = sorted(ecs), origin
            total += 1
            have += bool(ecs)
    return dict(reactions=total, with_full_ec=have)


def target_ecs(model):
    """Current full EC numbers per target reaction (annotation + BiGG bridge)."""
    bridge = bigg_to_ec()
    out = {}
    for r in model['reactions']:
        ann = r.get('annotations', [])
        ecs = {re.search(r'ec-code/([^/?#]+)', a.get('uri', ''))[1] for a in ann if 'ec-code/' in a.get('uri', '')}
        ecs |= set(bridge.get(r['id'], ()))
        for a in ann:
            m = re.search(r'bigg\.reaction/([^/?#]+)', a.get('uri', ''))
            if m:
                ecs |= set(bridge.get(m.group(1), ()))
        out[r['id']] = set(current_ecs(full_ecs(ecs)))
    return out


def model_metabolite_keys(model):
    """Reaction id -> list of metabolite key frozensets (names + BiGG + ChEBI)."""
    chebi_bridge = ib.metabolite_chebi()
    names_by_bigg = ib.bigg_names()
    out = {}
    meta = {m['id']: m for m in model['metabolites']}
    for r in model['reactions']:
        keys, seen = [], set()
        for mid in r['metabolites']:
            m = meta[mid]
            refs = xrefs(m.get('annotations', []))
            bigg = set(refs.get('bigg.metabolite', [])) | {chem(mid)}
            chebis = set(refs.get('chebi', []))
            for b in bigg:
                chebis |= chebi_bridge.get(b, set())
            names = {m.get('name') or ''}
            for b in bigg:
                names |= names_by_bigg.get(b, set())
            k = ib.metabolite_keys(names=names, compounds={f'chebi:{c}' for c in chebis}, bigg_ids=bigg)
            if k and k not in seen:
                keys.append(k)
                seen.add(k)
        out[r['id']] = keys
    return out


def model_reaction_chebi(model):
    """Reaction id -> list of frozensets of ChEBI ids (for EC disambiguation via Rhea)."""
    chebi_bridge = ib.metabolite_chebi()
    out = {}
    meta = {m['id']: m for m in model['metabolites']}
    for r in model['reactions']:
        sets = []
        for mid in r['metabolites']:
            m = meta[mid]
            refs = xrefs(m.get('annotations', []))
            chebis = set(refs.get('chebi', []))
            for b in set(refs.get('bigg.metabolite', [])) | {chem(mid)}:
                chebis |= chebi_bridge.get(b, set())
            chebis.discard('15378')  # proton
            if chebis:
                sets.append(frozenset(chebis))
        out[r['id']] = sets
    return out


class ECMapper(st.Mapper):
    """Subclass of the equation mapper so metabolite identity candidates can be reused as a filter."""

    def __init__(self, model, bridge):
        super().__init__(model, bridge)
        self.ec = target_ecs(model)
        self.by_ec = defaultdict(set)
        for rid, ecs in self.ec.items():
            for e in ecs:
                self.by_ec[e].add(rid)
        self.graph = model_graph(model)
        self.audit_ec = Counter()
        self.met_keys = model_metabolite_keys(model)
        self.chebi_sets = model_reaction_chebi(model)
        self._by_main = None

    def _main_index(self):
        if self._by_main is None:
            idx = defaultdict(set)
            for rid, keys in self.met_keys.items():
                if self._is_transport(rid):
                    continue
                for k in keys:
                    if not ib.is_currency_keys(k):
                        for token in k:
                            idx[token].add(rid)
            self._by_main = idx
        return self._by_main

    def _is_transport(self, rid):
        keys = self.met_keys.get(rid) or []
        # same non-proton metabolite key appearing twice (two compartments) → transport-like
        counts = Counter(keys)
        return any(v > 1 and not ib.is_currency_keys(k) for k, v in counts.items())

    def reaction_matches(self, reaction):
        """Target reactions linked by current EC and/or scored metabolite overlap (name-only sources)."""
        ecs = set(reaction.get('ec_numbers') or ())
        # disambiguate source ECs against... we don't have target yet; keep all current ECs for indexing
        out = defaultdict(lambda: dict(shared_ec=set(), score=None, via=set()))
        for e in ecs:
            for rid in self.by_ec.get(e, ()):
                out[rid]['shared_ec'].add(e)
                out[rid]['via'].add('ec')
        # incomplete source ECs (1.1.1.-) include complete target ECs
        for e in reaction.get('ec_numbers') or ():
            if is_complete(e):
                continue
            for te, rids in self.by_ec.items():
                if covers(e, te):
                    for rid in rids:
                        out[rid]['shared_ec'].add(te)
                        out[rid]['via'].add('ec')

        src_keys = reaction.get('map_metabolite_keys') or []
        if src_keys and not self._is_transport_source(src_keys):
            shared = Counter()
            for k in src_keys:
                if ib.is_currency_keys(k):
                    continue
                hit = set().union(*(self._main_index().get(token, ()) for token in k))
                for rid in hit:
                    shared[rid] += 1
            for rid, n in shared.items():
                if n >= 2 or (n >= 1 and out[rid]['via']):
                    score = ib.weighted_jaccard(src_keys, self.met_keys.get(rid) or [])
                    out[rid]['score'] = score
                    out[rid]['via'].add('metabolites')

        # score EC-linked candidates that have no metabolite score yet
        for rid, info in out.items():
            if info['score'] is None and src_keys:
                info['score'] = ib.weighted_jaccard(src_keys, self.met_keys.get(rid) or [])

        results = []
        scores = [info['score'] for info in out.values() if info['score'] is not None]
        best = max(scores) if scores else None
        for rid, info in sorted(out.items()):
            linked = bool(info['via'] & {'ec'})
            score = info['score']
            if score is None:
                # EC-only hit without map metabolites (open-core modules with identifiers)
                if linked:
                    results.append(dict(target_reaction=rid, shared_ec=sorted(info['shared_ec']), score=None, via=sorted(info['via'])))
                continue
            if ib.accept_score(score, linked=linked, best=best):
                results.append(dict(target_reaction=rid, shared_ec=sorted(info['shared_ec']), score=round(score, 4), via=sorted(info['via'])))
            elif linked and score >= ib.THRESHOLDS['min_score'] - 1e-9:
                # keep EC link even when below best-gap if it still clears the floor (multi-subsystem)
                if best is None or score >= best - ib.THRESHOLDS['max_gap']:
                    results.append(dict(target_reaction=rid, shared_ec=sorted(info['shared_ec']), score=round(score, 4), via=sorted(info['via'])))
        return results

    @staticmethod
    def _is_transport_source(src_keys):
        counts = Counter(src_keys)
        return any(v > 1 and not ib.is_currency_keys(k) for k, v in counts.items())

    def metabolite_options(self, node, parent_targets):
        """Target metabolites for a module metabolite: neighbours shared by all parent reactions, filtered by identity if known."""
        domain = None
        for rid in parent_targets:
            nb = {k for k in self.graph['r:' + rid]}
            domain = nb if domain is None else domain & nb
        if not domain:
            return [], 'no_common_target_metabolite'
        identity = {'m:' + t for t in self.candidates(node.get('entity'))} if isinstance(node.get('entity'), str) else set()
        # name-only entities (lid… / plain labels): try normalised name against target metabolite names
        if not identity and isinstance(node.get('label'), str) and node['label']:
            want = ib.name_keys(node['label'])
            named = set()
            for mid in domain:
                tid = mid[2:] if mid.startswith('m:') else mid
                meta = self.met.get(tid)
                if meta and ib.name_keys(meta.get('name') or '') & want:
                    named.add(mid)
            if named:
                return sorted(named), 'name_filtered'
        if identity & domain:
            return sorted(identity & domain), 'identity_filtered'
        return sorted(domain), 'structure_only'

    def proposals(self, modules, source_label):
        g = self.graph
        skips, out, matched = Counter(), [], set()
        support = defaultdict(set)
        for m in modules:
            support[m['core_family']].add(m['lineage'])

        @lru_cache(maxsize=4096)
        def initial(rids):
            keys = {'r:' + r for r in rids}
            for r in rids:
                keys.update(g['r:' + r])
            sub = nx.Graph()
            sub.add_nodes_from((k, dict(g.nodes[k])) for k in sorted(keys))
            sub.add_edges_from((a, b, dict(g.edges[a, b])) for a, b in sorted(tuple(sorted(e)) for e in g.subgraph(keys).edges()))
            return sub, nx.spring_layout(sub, seed=SEEDS[0], iterations=35, scale=1.)

        examined = 0
        for m in sorted(modules, key=lambda m: (-m['size'], m['id'])):
            if m['size'] < CONFIG['min_module_size'] or not m.get('reactions') or len(m['reactions']) != m['size']:
                continue
            examined += 1
            choices = [self.reaction_matches(r) for r in m['reactions']]
            if any(not c for c in choices):
                skips['source_reaction_without_ec_match'] += 1
                continue
            keys = {r['key'] for r in m['reactions']}
            parents = defaultdict(set)
            for e in m['edges']:
                if e['a'] in keys:
                    parents[e['b']].add(e['a'])
                if e['b'] in keys:
                    parents[e['a']].add(e['b'])
            for combo in itertools.islice(itertools.product(*choices), CONFIG['module_combinations']):
                rids = tuple(sorted(c['target_reaction'] for c in combo))
                if len(set(rids)) != len(rids):
                    skips['duplicate_target_reaction'] += 1
                    continue
                rmap = {r['key']: c['target_reaction'] for r, c in zip(m['reactions'], combo)}
                node_map = {k: 'r:' + v for k, v in rmap.items()}
                evidence, ambiguous, used = [], [], set()
                fit = True
                for n in m['nodes']:
                    if n['role'] == 'reaction':
                        continue
                    pk = [rmap[p] for p in parents[n['key']] if p in rmap]
                    options, how = self.metabolite_options(n, pk)
                    if how == 'no_common_target_metabolite':
                        fit = False
                        break
                    options = [o for o in options if o not in used]
                    if len(options) == 1:
                        node_map[n['key']] = options[0]
                        used.add(options[0])
                        evidence.append(dict(node=n['key'], target=options[0], how=how))
                    else:
                        ambiguous.append(n['key'])
                if not fit:
                    skips['core_incidence_does_not_fit_target'] += 1
                    continue
                if any(not g.has_edge(node_map[e['a']], node_map[e['b']]) for e in m['edges'] if e['a'] in node_map and e['b'] in node_map):
                    skips['mapped_edge_missing_in_target'] += 1
                    continue
                if len(node_map) < 3:
                    skips['too_few_mapped_nodes'] += 1
                    continue
                tg, generic = initial(rids)
                source = {n['key']: n for n in m['nodes']}
                ks = sorted(node_map)
                frame = fit_frame([source[k]['xy'] for k in ks], [generic[node_map[k]] for k in ks])
                positions = transform({node_map[k]: source[k]['xy'] for k in ks}, frame)
                source_xy = {node_map[k]: list(source[k]['xy']) for k in ks}
                targetports = {k: len(set(g[k]) - {'r:' + rid for rid in rids}) for k in tg if g.nodes[k]['role'] == 'metabolite'}
                srcports, ports = len(m.get('ports', [])), sum(bool(v) for v in targetports.values())
                initial_fit = float(np.sqrt(np.mean([np.sum((positions[k] - generic[k]) ** 2) for k in positions])))
                xy = np.array(list(positions.values()))
                extent = np.ptp(xy, axis=0)
                area = max(float(np.prod(extent)), .05)
                spread = float(np.std(xy[:, 0]) + np.std(xy[:, 1]))
                vec = [np.array(positions[node_map[e['a']]]) - positions[node_map[e['b']]] for e in m['edges'] if e['a'] in node_map and e['b'] in node_map]
                axis = float(np.mean([min(abs(v[0]), abs(v[1])) / max(np.linalg.norm(v), 1e-8) < .2 for v in vec])) if vec else 0.
                features = [len(rids), 1, abs(ports - srcports), 0., float(extent[0] / max(extent[1], .05)), len(tg) / area, axis,
                            len(support[m['core_family']]), initial_fit, sum(targetports.values()) * float(extent[0] / max(extent[1], .05)),
                            ports / max(len(tg), 1) * spread]
                mean_score = float(np.mean([c['score'] for c in combo if c.get('score') is not None] or [0.6]))
                checks = [dict(source_reaction=r['id'], source_key=r['key'], target_reaction=c['target_reaction'], shared_ec=c['shared_ec'],
                               score=c.get('score'), via=c.get('via'), source_ec_origin=r.get('ec_origin', {}),
                               target_ec=sorted(self.ec[c['target_reaction']]))
                          for r, c in zip(m['reactions'], combo)]
                p = dict(id=uid(['ec-topology', m['id'], rids, sorted(node_map.items())]), template_id=m['id'], map_id=m['map_id'], lineage=m['lineage'],
                         source=m['source'], species=m['species'], license=m['license'], source_citation=m.get('source_citation'),
                         partition=m.get('partition', 'open_core'), geometry_fingerprint=m['geometry_fingerprint'], reaction_ids=list(rids),
                         node_mapping=node_map, reaction_mapping=rmap, positions={k: np.array(v).tolist() for k, v in positions.items()},
                         source_xy=source_xy, position_origins={node_map[k]: source[k]['position_origin'] for k in ks}, source_to_initial_frame=frame,
                         match_level='ec_topology', complete_chemical_equation_verified=False, complete_equation_verified=False,
                         label_compartments_verified=False, mapping_evidence=checks, metabolite_evidence=evidence,
                         unmapped_source_nodes=sorted(ambiguous), source_node_count=len(m['nodes']), features=features,
                         score=2 * len(rids) + .6 + .2 * mean_score - .1 * features[2] - .2 * initial_fit - .3 * len(ambiguous) / max(1, len(m['nodes'])),
                         expert_query_coordinates_used=False, template_quality=m['quality'], match_mode='ec_number_plus_incidence_fit',
                         layout_origin=m.get('layout_origin'), counted_as_expert_design=m['source'] != 'brenda',
                         unmapped_target_nodes=sorted(set(tg) - set(node_map.values())),
                         ec_policy='current full EC equality and/or scored name-only metabolites; weak identity; strong equation channel outranks it')
                out.append(p)
                matched.update(rids)
                if len(out) >= CONFIG['proposal_budget']:
                    break
            if len(out) >= CONFIG['proposal_budget']:
                break
        return out, dict(library=source_label, examined_multi_modules=examined, candidates=len(out), proposal_budget=CONFIG['proposal_budget'],
                         budget_reached=len(out) >= CONFIG['proposal_budget'], skips=dict(skips), candidate_reaction_ids=sorted(matched),
                         candidate_reactions=len(matched), config=CONFIG,
                         interpretation='EC-supported candidates with an incidence-fit check; not a certification of chemical equivalence')


def usable(proposal):
    """EC proposals whose source module is mostly mapped; heavily unmapped modules carry little layout information.

    Non-EC proposals have no ``unmapped_source_nodes`` field and are usable.
    """
    if 'unmapped_source_nodes' not in proposal:
        return True
    n = max(1, proposal.get('source_node_count', 1))
    return len(proposal['unmapped_source_nodes']) / n <= CONFIG['max_unmapped_fraction']


def select_with_ec(strong_pool, ec_pools):
    """Strong-first selection.

    1. `retrieval.select` over complete (non-relaxed, non-partial) strong proposals.
    2. For each EC pool in order (expert-library EC first, unverified extensions last), select only among proposals
       that do not touch an already covered reaction, again with `retrieval.select`.
    EC fills gaps; it never displaces or overlaps a complete-equation or geometry selection.
    Returns (all selected, per-stage selected lists).
    """
    import copy
    import retrieval

    def is_incomplete(p):
        return p.get('match_level') == 'relaxed_equation' or bool(p.get('partial_match'))

    full = [p for p in strong_pool if not is_incomplete(p)]
    stages = [retrieval.select(copy.deepcopy(full if full else strong_pool))]
    covered = {r for p in stages[0] for r in p['reaction_ids']}
    for pool in ec_pools:
        free = [p for p in pool if usable(p) and not set(p['reaction_ids']) & covered]
        chosen = retrieval.select(copy.deepcopy(free))
        stages.append(chosen)
        covered |= {r for p in chosen for r in p['reaction_ids']}
    return [p for stage in stages for p in stage], stages
