"""Annotation-supported, equation-verified direct transfer for any source library.

Mappings are candidates with evidence, not destructive global entity merges.
Target annotations are legal inputs; no target expert coordinates or primary
node flags are used. Same engine and bounds apply to core and extension.
"""
import itertools
from functools import lru_cache
from core import *
import database

CONFIG=dict(max_assignments=2,module_combinations=16,proposal_budget=6000,seed=SEEDS[0])


def spelling(identifier):
    # Identifier escape spelling only: keep D/L, case, charge and all tokens.
    return re.sub(r'_{2,}', '_', identifier)



def _mam_bridge_links():
    """MAM → BiGG/ChEBI/MetaNetX from Human-GEM metabolites.tsv (registry licence).

    Built on demand from the pinned TSV; the JSON is gitignored. Used for licence
    checks and for callers that opt in — ``mapper_bridge`` does not merge these
    links into BiGG matching (keeps e_coli_core / iIT341 layout parity with main).
    """
    import canonical_ids as cid
    path = DATA / 'source_inputs/crosswalks/human_gem_metabolite_crosswalk.json'
    tsv = DATA / 'source_inputs/crosswalks/Human-GEM-metabolites.tsv'
    src = DATA / 'source_inputs/crosswalks/Human-GEM-metabolites.SOURCE.json'
    if not path.exists() and tsv.exists():
        cid.build_human_gem_metabolite_crosswalk(tsv_path=tsv, out_path=path)
    if not path.exists():
        return {}, dict(skipped=True, reason='missing_human_gem_metabolite_crosswalk')
    payload = json.loads(path.read_text())
    meta = dict(
        license=payload.get('license') or '',
        license_uri=payload.get('license_uri'),
        source_url=payload.get('source_url'),
        skipped=False,
    )
    if src.exists():
        src_meta = json.loads(src.read_text())
        for k in ('license', 'license_uri', 'url'):
            if src_meta.get(k):
                meta[k if k != 'url' else 'source_url'] = src_meta[k]
    links = defaultdict(lambda: defaultdict(set))
    for mam, row in (payload.get('metabolites') or {}).items():
        refs = row.get('references') or {}
        aliases = {spelling(chem(mam))}
        aliases.update(spelling(b) for b in (refs.get('bigg.metabolite') or []))
        for alias in aliases:
            for ns in ('chebi', 'metanetx.chemical', 'bigg.metabolite'):
                links[alias][ns].update(str(v) for v in (refs.get(ns) or []))
    return {k: {ns: sorted(v) for ns, v in refs.items()} for k, refs in links.items()}, meta


def _pick_assignment(assignments):
    """Deterministic choice among ≤2 complete assignments: keep compartments, then direction.

    Kept for tests and call-sites that resolve ambiguity explicitly. ``match_reaction``
    still drops ambiguous equations (main behaviour) so drawn coverage stays gate-safe.
    """
    if not assignments:
        return None
    if len(assignments) == 1:
        return assignments[0]
    ranked = sorted(assignments, key=lambda a: (
        0 if a.get('label_compartments_exact') else 1,
        0 if a.get('direction') == 'same' else 1,
        tuple(sorted((a.get('metabolites') or {}).items())),
    ))
    chosen = dict(ranked[0])
    chosen['assignment_resolution'] = 'deterministic_prefer_compartments_then_direction'
    chosen['alternate_assignments'] = len(assignments) - 1
    return chosen

def annotation_bridge(models):
    """Build a Mapper crosswalk from model annotations (legacy / enrichment path).

    Prefer ``mapper_bridge`` which seeds from ``data/identifier_bridge.json`` (the
    tracked identifier store) and only uses live models to add MetaNetX links.
    """
    links=defaultdict(lambda:defaultdict(set));evidence=defaultdict(list);inputs=[]
    for name,path,_ in models:
        model=load_model(path);inputs.append(dict(model=name,path=str(path.relative_to(ROOT)),sha256=sha(path.read_bytes())))
        for m in model['metabolites']:
            refs=xrefs(m.get('annotations',[]));aliases=set(refs.get('bigg.metabolite',[]))
            if name in ('e_coli_core','iML1515','iMM904','iYO844'):aliases.add(chem(m['id']))
            for original_alias in aliases:
                alias=spelling(original_alias)
                for ns in ('chebi','metanetx.chemical'):links[alias][ns].update(refs.get(ns,[]))
                if refs:evidence[alias].append(dict(model=name,metabolite=m['id'],references=refs))
    return dict(links={k:{ns:sorted(v) for ns,v in refs.items()} for k,refs in links.items()},
                evidence=dict(evidence),inputs=inputs,
                interpretation='Explicit annotations in fixed local target models plus underscore-escape spelling candidates; not a new MNXref release or identity certification')


def mapper_bridge(models=None):
    """Mapper crosswalk seeded from ``identifier_bridge`` (one identifier store).

    ChEBI links come from ``data/identifier_bridge.json``. When ``models`` are
    provided and readable, MetaNetX (and any extra ChEBI) annotations from those
    models are merged in — they are not a second checked-in store.
    """
    import identifier_bridge as ib
    links=defaultdict(lambda:defaultdict(set));evidence=defaultdict(list)
    for bigg,chebis in ib.metabolite_chebi().items():
        alias=spelling(bigg)
        links[alias]['chebi'].update(chebis)
    inputs=list(ib.sources())
    for name,path,_ in (models or ()):
        path=Path(path)
        if not path.exists():
            continue
        model=load_model(path)
        inputs.append(dict(model=name,path=str(path.relative_to(ROOT)) if ROOT in path.resolve().parents else str(path),
                           sha256=sha(path.read_bytes()),role='metanetx_enrichment'))
        for m in model['metabolites']:
            refs=xrefs(m.get('annotations',[]));aliases=set(refs.get('bigg.metabolite',[]))
            if name in ('e_coli_core','iML1515','iMM904','iYO844'):aliases.add(chem(m['id']))
            for original_alias in aliases:
                alias=spelling(original_alias)
                for ns in ('chebi','metanetx.chemical'):
                    links[alias][ns].update(refs.get(ns,[]))
                if refs.get('metanetx.chemical') or refs.get('chebi'):
                    evidence[alias].append(dict(model=name,metabolite=m['id'],references=refs))
    return dict(
        links={k:{ns:sorted(v) for ns,v in refs.items()} for k,refs in links.items()},
        evidence=dict(evidence),inputs=inputs,
        interpretation='ChEBI from data/identifier_bridge.json; MetaNetX (and extra ChEBI) enriched from available local models at load time — not a second identifier store')


def entity_parts(entity):
    if not isinstance(entity,str):return None,'unknown'
    if entity.lower().startswith('chebi:'):return 'chebi:'+entity.split('@')[0].split(':')[-1],comp(entity)
    return 'bigg:'+spelling(chem(entity)),comp(entity)


def unique_equation_mapping(coefficients, target, options, target_compartments):
    """Return at most2 distinct complete, stoichiometric assignments.

    Direction/proportional representation is recorded, not silently rewritten.
    Distinct source compartment roles cannot collapse to one target role.
    """
    if len(coefficients)!=len(target['metabolites']) or not coefficients:return []
    if any(not np.isfinite(v) or not v for v in coefficients.values()):return []
    entities=sorted(coefficients,key=lambda k:(len(options.get(k,[])),k))
    if any(not options.get(k) for k in entities):return []
    first=entities[0];factors=sorted({round(target['metabolites'][m]/coefficients[first],12) for m in options[first] if m in target['metabolites']})
    answers=[]
    for factor in factors:
        # Factor 0 would match any curated stoichiometry onto a target coefficient-0
        # species (A -> A); skip so a one-participant curated reaction cannot record
        # reversed_recording against that target.
        if not factor:continue
        choices={k:[m for m in options[k] if m in target['metabolites'] and abs(target['metabolites'][m]-coefficients[k]*factor)<1e-8] for k in entities}
        if any(not v for v in choices.values()):continue
        def visit(i,mapping,used,roles,reverse):
            if len(answers)>=CONFIG['max_assignments']:return
            if i==len(entities):
                answers.append(dict(metabolites=dict(mapping),factor=factor,
                    direction='same' if factor>0 else 'reversed_recording',
                    label_compartments_exact=all(comp(e) not in ('local','unknown') and comp(e)==target_compartments[m] for e,m in mapping.items()),compartment_roles=dict(roles)));return
            entity=entities[i];source_role=comp(entity)
            for mid in choices[entity]:
                if mid in used:continue
                target_role=target_compartments[mid]
                if source_role not in ('unknown','local'):
                    if source_role in roles and roles[source_role]!=target_role:continue
                    if target_role in reverse and reverse[target_role]!=source_role:continue
                newroles=dict(roles);newreverse=dict(reverse)
                if source_role not in ('unknown','local'):newroles[source_role]=target_role;newreverse[target_role]=source_role
                mapping[entity]=mid;visit(i+1,mapping,used|{mid},newroles,newreverse);mapping.pop(entity)
        visit(0,{},set(),{}, {})
        if len(answers)>=CONFIG['max_assignments']:break
    return answers


class Mapper:
    def __init__(self,model,bridge):
        self.model=model;self.bridge=bridge;self.reactions={r['id']:r for r in model['reactions']}
        self.met={m['id']:m for m in model['metabolites']};self.compartments={k:v.get('compartment','unknown') for k,v in self.met.items()}
        self.refs={mid:xrefs(m.get('annotations',[])) for mid,m in self.met.items()}
        self.inverted=defaultdict(set);self.uses=defaultdict(set)
        for mid,refs in self.refs.items():
            aliases=set(refs.get('bigg.metabolite',[]))
            if model['id'] in ('e_coli_core','iML1515','iMM904','iYO844'):aliases.add(chem(mid))
            for original_alias in aliases:
                alias=spelling(original_alias)
                self.inverted['bigg:'+alias].add(mid)
                for ns,values in bridge['links'].get(alias,{}).items():
                    for v in values:self.inverted[ns+':'+v].add(mid)
            for ns in ('chebi','metanetx.chemical'):
                for v in refs.get(ns,[]):self.inverted[ns+':'+v].add(mid)
        for r in model['reactions']:
            for mid in r['metabolites']:self.uses[mid].add(r['id'])
        self.cache={};self.entity_cache={};self.audit=Counter();self.mapping_rows={}

    def candidates(self,entity,label=None):
        cache_key=(entity,label)
        if cache_key in self.entity_cache:return self.entity_cache[cache_key]
        if entity in self.entity_cache and label is None:return self.entity_cache[entity]
        token,role=entity_parts(entity) if isinstance(entity,str) else (None,'unknown');out=set(self.inverted.get(token,set())) if token else set()
        if token and token.startswith('bigg:'):
            for ns,values in self.bridge['links'].get(token[5:],{}).items():
                for v in values:out.update(self.inverted.get(ns+':'+v,set()))
        # Name-only Atlas nodes (BRENDA ligands, some CellDesigner species): normalised name match.
        if not out and label:
            import identifier_bridge as ib
            want=ib.name_keys(label)
            if want:
                out={mid for mid,m in self.met.items() if ib.name_keys(m.get('name') or '') & want}
        # Keep all compartments until reaction-level role consistency is checked.
        answer=sorted(out);self.entity_cache[cache_key]=answer
        if label is None:self.entity_cache[entity]=answer
        return answer

    def match_reaction(self,record):
        coefficients=record.get('coefficients',{})
        signature=uid(sorted(coefficients.items()))
        if signature in self.cache:return self.cache[signature]
        options={entity:self.candidates(entity) for entity in coefficients}
        if not coefficients or any(not v for v in options.values()):
            self.audit['source_equations_with_unmapped_participants']+=1;self.mapping_rows[signature]=dict(coefficients=coefficients,status='unmapped_participants',unresolved_entities=[k for k,v in options.items() if not v],match_candidates=[]);self.cache[signature]=[];return []
        possible=None
        for entity,ms in sorted(options.items(),key=lambda x:len(x[1])):
            rs=set().union(*(self.uses[m] for m in ms));possible=rs if possible is None else possible&rs
            if not possible:break
        matches=[];ambiguous=[]
        for rid in sorted(possible or ()):
            target=self.reactions[rid]
            local_options={}
            for entity,values in options.items():
                token,_=entity_parts(entity);direct=set(self.inverted.get(token,set()))&set(target['metabolites'])
                local_options[entity]=sorted(direct) if direct else values
            assignments=unique_equation_mapping(coefficients,target,local_options,self.compartments)
            if len(assignments)==1:matches.append(dict(target_reaction=rid,**assignments[0]))
            elif len(assignments)>1:ambiguous.append(rid)
        self.audit['source_equations_checked']+=1;self.audit['ambiguous_reaction_pairs']+=len(ambiguous)
        self.audit['source_equations_matched' if matches else 'source_equations_no_unique_match']+=1
        self.mapping_rows[signature]=dict(coefficients=coefficients,match_candidates=matches,ambiguous_target_reactions=ambiguous)
        self.cache[signature]=matches;return matches

    def proposals(self,modules,source_label):
        g=model_graph(self.model);rlookup=self.reactions;out=[];skips=Counter();examined=0;matched_reactions=set();support=defaultdict(set)
        for m in modules:support[m['core_family']].add(m['lineage'])
        @lru_cache(maxsize=4096)
        def initial(rids):
            keys={'r:'+r for r in rids}
            for r in rids:keys.update(g['r:'+r])
            sub=nx.Graph();sub.add_nodes_from((k,dict(g.nodes[k])) for k in sorted(keys));sub.add_edges_from((a,b,dict(g.edges[a,b])) for a,b in sorted(tuple(sorted(e)) for e in g.subgraph(keys).edges()));return sub,nx.spring_layout(sub,seed=SEEDS[0],iterations=35,scale=1.)
        for m in sorted(modules,key=lambda m:(-m['size'],m['id'])):
            if m['size']<2 or not m.get('reactions') or len(m['reactions'])!=m['size']:continue
            examined+=1;choices=[self.match_reaction(r) for r in m['reactions']]
            parents=defaultdict(set)
            source_reaction_keys={r['key'] for r in m['reactions']}
            for e in m['edges']:
                if e['a'] in source_reaction_keys:parents[e['b']].add(e['a'])
                if e['b'] in source_reaction_keys:parents[e['a']].add(e['b'])
            if any(not v for v in choices):skips['unmatched_source_reaction']+=1;continue
            for combination in itertools.islice(itertools.product(*choices),CONFIG['module_combinations']):
                rids=tuple(sorted(c['target_reaction'] for c in combination))
                if len(set(rids))!=len(rids):skips['duplicate_target_reaction']+=1;continue
                comp_roles={};role_conflict=False
                for c in combination:
                    for sr,tr in c['compartment_roles'].items():
                        if sr in comp_roles and comp_roles[sr]!=tr:role_conflict=True
                        comp_roles[sr]=tr
                if role_conflict or len(set(comp_roles.values()))!=len(comp_roles):skips['module_compartment_role_conflict']+=1;continue
                by_key={r['key']:c for r,c in zip(m['reactions'],combination)}
                rmap={r['key']:c['target_reaction'] for r,c in zip(m['reactions'],combination)};node_map={k:'r:'+v for k,v in rmap.items()};valid=True
                for n in m['nodes']:
                    if n['role']=='reaction':continue
                    entity=n.get('entity');targets={by_key[rk]['metabolites'][entity] for rk in parents[n['key']] if entity in by_key[rk]['metabolites']}
                    if len(targets)!=1:valid=False;break
                    node_map[n['key']]='m:'+next(iter(targets))
                if not valid:skips['inconsistent_glyph_identity']+=1;continue
                if len(set(node_map.values()))!=len(node_map):skips['repeated_core_copy_requires_graphic_matcher']+=1;continue
                if any(not g.has_edge(node_map[e['a']],node_map[e['b']]) for e in m['edges']):skips['core_incidence_or_currency_conflict']+=1;continue
                tg,generic=initial(rids);source={n['key']:n for n in m['nodes']};ks=sorted(node_map)
                frame=fit_frame([source[k]['xy'] for k in ks],[generic[node_map[k]] for k in ks]);positions=transform({node_map[k]:source[k]['xy'] for k in ks},frame);source_xy={node_map[k]:list(source[k]['xy']) for k in ks}
                targetports={k:len(set(g[k])-{'r:'+rid for rid in rids}) for k in tg if g.nodes[k]['role']=='metabolite'}
                srcports=len(m.get('ports',[]));ports=sum(bool(v) for v in targetports.values());initial_fit=float(np.sqrt(np.mean([np.sum((positions[k]-generic[k])**2) for k in positions])))
                xy=np.array(list(positions.values()));extent=np.ptp(xy,axis=0);area=max(float(np.prod(extent)),.05);spread=float(np.std(xy[:,0])+np.std(xy[:,1]));vectors=[np.array(positions[node_map[e['a']]])-positions[node_map[e['b']]] for e in m['edges']];axis=float(np.mean([min(abs(v[0]),abs(v[1]))/max(np.linalg.norm(v),1e-8)<.1 for v in vectors])) if vectors else 0.;direction_cost=float(np.mean([c['factor']<0 for c in combination]))
                features=[len(rids),2,abs(ports-srcports),direction_cost,float(extent[0]/max(extent[1],.05)),len(tg)/area,axis,len(support[m['core_family']]),initial_fit,sum(targetports.values())*float(extent[0]/max(extent[1],.05)),ports/max(len(tg),1)*spread]
                exact_comp=all(c['label_compartments_exact'] for c in combination)
                checks=[dict(source_reaction=r['id'],source_key=r['key'],**c,complete_equation=True,evidence='annotation-supported candidates plus unique full participant/stoichiometry matching') for r,c in zip(m['reactions'],combination)]
                p=dict(id=uid(['direct-equation',m['id'],rids,sorted(node_map.items())]),template_id=m['id'],map_id=m['map_id'],lineage=m['lineage'],source=m['source'],species=m['species'],license=m['license'],source_citation=m.get('source_citation'),partition=m.get('partition','open_core'),
                    geometry_fingerprint=m['geometry_fingerprint'],reaction_ids=list(rids),node_mapping=node_map,reaction_mapping=rmap,source_glyph_mapping={k:v[2:] for k,v in node_map.items() if source[k]['role']=='metabolite'},
                    positions={k:np.array(v).tolist() for k,v in positions.items()},source_xy=source_xy,position_origins={node_map[k]:source[k]['position_origin'] for k in ks},
                    source_to_initial_frame=frame,match_level='complete_equation',complete_chemical_equation_verified=True,
                    complete_equation_verified=exact_comp,label_compartments_verified=exact_comp,
                    mapping_evidence=checks,features=features,ports=targetports,score=2*len(rids)+1.2-.1*features[2]-.2*direction_cost-.2*initial_fit,
                    expert_query_coordinates_used=False,template_quality=m['quality'],match_mode='direct_full_equation_partial_visible_core',unmapped_target_nodes=sorted(set(tg)-set(node_map.values())),
                    source_path_policy='Source decoded paths retained by template ID; target routes recomputed')
                out.append(p);matched_reactions.update(rids)
                if len(out)>=CONFIG['proposal_budget']:break
            if len(out)>=CONFIG['proposal_budget']:break
        return out,dict(library=source_label,examined_multi_modules=examined,candidates=len(out),proposal_budget=CONFIG['proposal_budget'],budget_reached=len(out)>=CONFIG['proposal_budget'],
            skips=dict(skips),candidate_reaction_ids=sorted(matched_reactions),candidate_chemical_reactions=len(matched_reactions),
            reaction_mapping_audit=dict(self.audit),config=CONFIG,
            interpretation='Conservative bounded search; no claim that unmatched reactions are absent from the entire source')
