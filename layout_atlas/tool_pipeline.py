"""Upload-a-GEM pipeline: match the model to curated structures, build a skeleton from them, then complete the rest.

Runs one job in its own process (started by tool_server.py, or by hand):

  PATH=$HOME/.local/node/bin:$PATH python3 tool_pipeline.py <job_folder>              # whole job
  PATH=$HOME/.local/node/bin:$PATH python3 tool_pipeline.py <job_folder> --view view-3  # draw one view of a finished job

A job folder holds `upload/model.xml|json` and `options.json`; the pipeline writes `job.json` (state polled by the
server), `skeleton.json/svg`, `map.*`, `modules.json`, `report.json`, `licenses.json` and `views/`.

Step 1 (skeleton) uses ONLY existing curated structures plus network bridges; step 2 completes everything else
algorithmically.  Reuses: semantic_transfer.Mapper (complete equations), retrieval.Library (id/geometry candidates),
ec_channel (EC candidates, strong-first selection), retrieval.bridges, drawing.pack_modules/complete/export and
micromap_experiment.draw_case (P0-compatible drawing with label avoidance and libavoid routing).
"""
import argparse
import copy
import gzip
import json
import os
import re
import sys
import time
import traceback

from core import *
import output_names

LIMITS = dict(max_bytes=25 * 1024 * 1024, max_reactions=20000, full_routed_max_reactions=150, whole_map_max_reactions=3000, view_size=50)
SOURCE_PATHS = False  # reuse the curated map's original curves (Escher Bezier chains) for the edges of matched structures
ORIENTATION = 'full'  # 'off' | 'sides' (currency copies on the substrate / product side) | 'full' (also reactions outside the curated structures between their substrates and products)
LAYOUT_STYLE = 'curated'  # 'legacy' | 'curated' (orthogonal polylines for algorithm edges; curated paths kept).

def normalize_layout_style(value):
    """Map user/API layout_style to the canonical value."""
    if value == 'legacy':
        return 'legacy'
    return 'curated'

COMPLETION_CONTEXT = 'auto'  # 'auto' (V6: uncovered reactions are placed beside the glyphs of their own metabolites) | 'off'
EDGE_ROUTING = 'libavoid'  # 'libavoid' (orthogonal, all edges) | 'hybrid' (routes only blocked algorithm-placed edges) | 'none' (straight lines)
# Routing behavior by layout style and model size:
#   - layout_style='curated': early edge_style.apply draws Bezier for layout stages; a final
#     edge_style.apply(..., mode='orthogonal') writes axis-parallel polylines (≤2 bends) for
#     algorithm edges. Curated Atlas paths / short cofactor stubs stay as stamped. No libavoid.
#   - layout_style='legacy' with <= ROUTING_THRESHOLD reactions: Uses libavoid orthogonal routing.
#   - layout_style='legacy' with > ROUTING_THRESHOLD reactions: Routing is deferred (edge_routing.measure only).
# For scalable=True (large models), straight connections are used initially; edge_style is applied separately.
ROUTING_THRESHOLD = 500  # reactions; above this, routing is deferred for legacy style
CONTEXT = 'auto'  # 'auto' (reactions without non-currency participants share nodes and sit next to related reactions) | 'off'
INSTANCES = 'auto'  # 'auto' (V3: decide how many visual instances each metabolite gets) | 'off'
LAYOUT = 'global'  # 'global' (one map for the whole model, <=3,000 reactions) | 'tiled' (50-reaction views tiled on a grid; also the automatic fallback)
GLOBAL_FAILURE = ''
COMPLETION = 'spaced'  # 'spaced' (free nodes keep a minimum distance) | 'classic' (drawing.complete only)
REACTION_STYLE = 'line'  # 'line' (reactions are lines between metabolites, like the curated maps) | 'marker' (earlier style: a dot per reaction)
PREVIOUS_ANCHORS = None  # optional pins from --previous map.json: {'r:id':[x,y], 'm:id':[x,y]}
PREVIOUS_SKIP_PACK = False  # True when ≥80% of reactions are pinned (skip module packing / composition)
PREVIOUS_REUSE_MAP = False  # True when every model reaction is already in the previous map
PREVIOUS_MAP_PATH = None  # absolute path to the --previous map.json


STEPS = [('parse', 'Parse the model'), ('map', 'Match library structures'),
         ('select', 'Select structures'), ('skeleton', 'Build the skeleton'),
         ('complete', 'Complete the map'), ('report', 'Write the report')]
SOURCE_COLORS = {'escher': '#128781', 'reactome': '#3d6fa3', 'wikipathways': '#8a5fb0', 'brenda': '#b04a4a', 'human_maps': '#2d6a8f', 'metabolismregulation': '#6b4f2a', 'panther': '#8a6b3d', 'ich360': '#3d7a5a'}
LEVEL_RANK = {'complete_equation': 0, 'reaction_reference': 1, 'partial_reference': 2, 'ec_topology': 3, 'geometry': 4}
LEVEL_NAMES = {'complete_equation': 'complete_equation', 'reaction_reference': 'reference_supported', 'partial_reference': 'reference_supported',
               'ec_topology': 'ec_topology', 'geometry': 'geometry_only'}


class Failure(Exception):
    def __init__(self, code, en):
        super().__init__(en)
        self.code, self.en = code, en


class Job:
    """State file the server polls; written atomically after every change."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.path = self.folder / 'job.json'
        self.t0 = time.time()
        self.state = read(self.path) if self.path.exists() else {}
        self.state.setdefault('id', self.folder.name)
        self.state.setdefault('status', 'queued')
        self.state.setdefault('step', 'parse')
        self.state.setdefault('steps', [dict(key=k, label_en=en, state='pending', seconds=None) for k, en in STEPS])
        self.state.setdefault('progress', 0.0)
        self.state.setdefault('log', [])
        self.state.setdefault('warnings', [])
        self.state.setdefault('error', None)
        self.state.setdefault('summary', None)
        self.state.setdefault('files', [])
        self.state.setdefault('views', [])
        self._step_started = {}

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))
        os.replace(tmp, self.path)

    def log(self, msg):
        self.state['log'].append(dict(t=round(time.time() - self.t0, 1), msg=msg))
        self.state['log'] = self.state['log'][-400:]
        self.save()

    def warn(self, code, en):
        self.state['warnings'].append(dict(code=code, message_en=en))
        self.log('WARNING: ' + en)

    def step(self, key):
        for s in self.state['steps']:
            if s['state'] == 'running':
                s['state'] = 'done'
                s['seconds'] = round(time.time() - self._step_started[s['key']], 1)
            if s['key'] == key:
                s['state'] = 'running'
                self._step_started[key] = time.time()
        self.state['step'] = key
        self.state['status'] = 'running'
        self.state['progress'] = round(([k for k, _ in STEPS].index(key)) / len(STEPS), 3)
        self.log(dict((k, en) for k, en in STEPS)[key])

    def finish(self):
        for s in self.state['steps']:
            if s['state'] == 'running':
                s['state'] = 'done'
                s['seconds'] = round(time.time() - self._step_started[s['key']], 1)
        self.state['status'], self.state['progress'] = 'done', 1.0
        self.save()

    def fail(self, code, en):
        for s in self.state['steps']:
            if s['state'] == 'running':
                s['state'] = 'failed'
        self.state['status'] = 'failed'
        self.state['error'] = dict(code=code, message_en=en)
        self.log('FAILED: ' + en)

    def add_file(self, rel, title_en, kind):
        p = self.folder / rel
        self.state['files'] = [f for f in self.state['files'] if f['name'] != rel] + [
            dict(name=rel, title_en=title_en, kind=kind, bytes=p.stat().st_size)]
        self.save()


# ------------------------------------------------------------------------------------------------ parsing
def reject_unsafe_xml(path):
    head = Path(path).read_bytes()
    if re.search(rb'<!DOCTYPE|<!ENTITY', head, re.I):
        raise Failure('unsafe_xml', 'The SBML file contains a DOCTYPE or ENTITY declaration, which is not accepted.')


def cobra_annotations(entry):
    """COBRA/BiGG JSON keeps annotations as a dict; the pipeline works on the SBML-style list of identifiers.org URIs."""
    out = list(entry.get('annotations') or [])
    ann = entry.get('annotation') or {}
    if isinstance(ann, dict):
        for ns, values in ann.items():
            for v in (values if isinstance(values, list) else [values]):
                v = str(v)
                if ns.lower() == 'rhea':
                    v = re.sub(r'^RHEA:', '', v)
                out.append(dict(qualifier='{http://biomodels.net/biology-qualifiers/}is', uri=f'http://identifiers.org/{ns}/{v}'))
    return out


def normalize_model(model, job):
    for key in ('metabolites', 'reactions'):
        if not isinstance(model.get(key), list) or not model[key]:
            raise Failure('bad_format', f'The model has no {key}.')
    for m in model['metabolites']:
        m['annotations'] = cobra_annotations(m)
        m.setdefault('compartment', 'unknown')
    for r in model['reactions']:
        r['annotations'] = cobra_annotations(r)
        r.setdefault('lower_bound', -1000.)
        r.setdefault('upper_bound', 1000.)
        # Gross participation (keys present), not net stoichiometry: coefficient 0 means
        # a species on both sides with cancelling amounts and still counts as a participant.
        if not isinstance(r.get('metabolites'), dict) or not r['metabolites']:
            raise Failure('bad_format', f"Reaction {r.get('id')} has no stoichiometry.")
    ids = {m['id'] for m in model['metabolites']}
    unknown = {k for r in model['reactions'] for k in r['metabolites']} - ids
    if unknown:
        raise Failure('bad_format', f'{len(unknown)} reaction participants are not declared as metabolites (e.g. {sorted(unknown)[0]}).')
    model.setdefault('id', 'uploaded_model')
    # BiGG-style identifiers (glu__L_c, PGI): add the same bigg.* identifiers the fixed BiGG models are given, transparently.
    pattern = re.compile(r'^[A-Za-z0-9_]+_[a-z][a-z0-9]?$')
    if np.mean([bool(pattern.match(m['id'])) for m in model['metabolites']]) >= .6:
        added = 0
        for m in model['metabolites']:
            if 'bigg.metabolite' not in xrefs(m['annotations']):
                m['annotations'].append(dict(qualifier='{http://biomodels.net/biology-qualifiers/}is', uri='http://identifiers.org/bigg.metabolite/' + chem(m['id'])))
                added += 1
        for r in model['reactions']:
            if 'bigg.reaction' not in xrefs(r['annotations']):
                r['annotations'].append(dict(qualifier='{http://biomodels.net/biology-qualifiers/}is', uri='http://identifiers.org/bigg.reaction/' + r['id']))
        if added:
            job.warn('bigg_like_ids', 'Identifiers look like BiGG IDs; they were used as BiGG identifiers for matching (inferred, not annotated in the file).')
    return model


def annotation_coverage(model):
    def frac(items, ns):
        return round(sum(bool(ns & set(xrefs(i['annotations']))) for i in items) / max(1, len(items)), 3)
    ec_cov = round(sum(bool(ec_values(r)) for r in model['reactions']) / max(1, len(model['reactions'])), 3)
    return dict(bigg=frac(model['reactions'], {'bigg.reaction'}), rhea=frac(model['reactions'], {'rhea'}), ec=ec_cov,
                chebi=frac(model['metabolites'], {'chebi'}), metabolite_bigg=frac(model['metabolites'], {'bigg.metabolite'}))


def ec_values(reaction):
    return {re.search(r'ec-code/([^/?#]+)', a.get('uri', ''))[1] for a in reaction.get('annotations', []) if 'ec-code/' in a.get('uri', '')}


# ------------------------------------------------------------------------------------------------ helpers
def fixed_views(model, g, max_size=50):
    """Fixed display scopes: reactions grouped by compartment set in breadth-first order, max_size reactions per view."""
    metas = {m['id']: m for m in model['metabolites']}
    by_comp = defaultdict(list)
    for r in model['reactions']:
        roles = tuple(sorted({metas[m].get('compartment', 'unknown') for m in r['metabolites']}))
        by_comp[roles].append(r['id'])
    out = []
    for roles, ids in sorted(by_comp.items()):
        allowed = {'r:' + rid for rid in ids}
        order, visited = [], set()
        for root in sorted(allowed):
            if root in visited:
                continue
            queue = [root]
            visited.add(root)
            for r in queue:
                order.append(r[2:])
                for mid in sorted(g[r]):
                    for nr in sorted(g[mid]):
                        if nr in allowed and nr not in visited:
                            visited.add(nr)
                            queue.append(nr)
        for i in range(0, len(order), max_size):
            out.append(dict(id='view-' + str(len(out)), compartments=list(roles), reaction_ids=order[i:i + max_size]))
    return out


def load_bridge():
    """Load the Mapper identifier crosswalk.

    Source of truth is ``data/identifier_bridge.json``. A gzipped local cache
    (``data/annotation_bridge.json.gz``) may be written for speed; the old
    uncompressed ``annotation_bridge.json`` is no longer tracked.
    """
    import semantic_transfer as st
    gz = DATA / 'annotation_bridge.json.gz'
    if gz.exists():
        with gzip.open(gz, 'rt') as f:
            return json.load(f)
    legacy = DATA / 'annotation_bridge.json'
    if legacy.exists():
        return read(legacy)
    available = [(n, p, s) for n, p, s in EXAMPLE_MODELS if Path(p).exists()]
    bridge = st.mapper_bridge(available)
    try:
        with gzip.open(gz, 'wt') as f:
            json.dump(bridge, f, ensure_ascii=False)
    except OSError:
        pass
    return bridge


def save_gz(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as f:
        json.dump(obj, f, ensure_ascii=False)


def render_skeleton_svg(model, blocks, containers, bridges, anchors, status, path):
    """Skeleton = curated modules (coloured by source library) plus network bridges; nothing else is drawn."""
    from xml.sax.saxutils import escape
    g = model_graph(model)
    xs, ys = [], []
    for b in blocks:
        for v in b['positions'].values():
            xs.append(v[0])
            ys.append(v[1])
    if not xs:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 120"><text x="20" y="60" font-family="sans-serif" font-size="16">No curated structure was matched.</text></svg>')
        return
    pad = 60
    x0, y0, x1, y1 = min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.0f} {y0:.0f} {x1 - x0:.0f} {y1 - y0:.0f}" font-family="Helvetica, Arial, sans-serif">',
           '<rect x="%.0f" y="%.0f" width="%.0f" height="%.0f" fill="#ffffff"/>' % (x0, y0, x1 - x0, y1 - y0)]
    for c in containers:
        out.append(f'<rect x="{c["x"]:.0f}" y="{c["y"]:.0f}" width="{c["width"]:.0f}" height="{c["height"]:.0f}" fill="#f4f7f8" stroke="#dbe4e8" rx="8"/>')
    for bridge in bridges:
        pts = [anchors[k] for k in bridge['nodes'] if k in anchors]
        if len(pts) >= 2:
            out.append('<polyline points="%s" fill="none" stroke="#9aa9b2" stroke-width="2" stroke-dasharray="6 5"><title>bridge: %s</title></polyline>'
                       % (' '.join(f'{p[0]:.1f},{p[1]:.1f}' for p in pts), escape(', '.join(bridge['reaction_ids']))))
    for b in blocks:
        color = SOURCE_COLORS.get(b['source'], '#128781')
        pos = b['positions']
        keys = set(pos)
        for a, c in sorted(tuple(sorted(e)) for e in g.subgraph(keys).edges()):
            out.append(f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" x2="{pos[c][0]:.1f}" y2="{pos[c][1]:.1f}" stroke="{color}" stroke-opacity=".55" stroke-width="2"/>')
        for k in sorted(keys):
            v = pos[k]
            if k.startswith('r:'):
                out.append(f'<rect x="{v[0] - 5:.1f}" y="{v[1] - 5:.1f}" width="10" height="10" fill="{color}"><title>{escape(k[2:])} ({escape(b["source"])}, {escape(b["match_level"])})</title></rect>')
            else:
                out.append(f'<circle cx="{v[0]:.1f}" cy="{v[1]:.1f}" r="4" fill="#fff" stroke="{color}" stroke-width="2"><title>{escape(k[2:])}</title></circle>')
    out.append('</svg>')
    path.write_text('\n'.join(out))



# ------------------------------------------------------------------------------------------------ step details (expandable steps in the page)
def thumbnail_svg(template, color, width=120, height=80):
    """Small inline preview of one library structure from its stored node positions (squares = reactions, dots = metabolites)."""
    nodes = {n['key']: n['xy'] for n in template.get('nodes', []) if n.get('xy') is not None}
    if not nodes:
        return ''
    xs, ys = [v[0] for v in nodes.values()], [v[1] for v in nodes.values()]
    x0, y0, w, h = min(xs), min(ys), (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0
    pad = 8
    scale = min((width - 2 * pad) / w, (height - 2 * pad) / h)
    ox, oy = (width - w * scale) / 2, (height - h * scale) / 2
    pt = lambda k: (round(ox + (nodes[k][0] - x0) * scale, 1), round(oy + (nodes[k][1] - y0) * scale, 1))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    for e in template.get('edges', []):
        if e['a'] in nodes and e['b'] in nodes:
            (ax, ay), (bx, by) = pt(e['a']), pt(e['b'])
            out.append(f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="#9aa9b2" stroke-width="1"/>')
    for n in template.get('nodes', []):
        if n['key'] not in nodes:
            continue
        x, y = pt(n['key'])
        out.append(f'<rect x="{x - 3}" y="{y - 3}" width="6" height="6" fill="{color}"/>' if n['role'] == 'reaction' else f'<circle cx="{x}" cy="{y}" r="2.4" fill="#5d7480"/>')
    out.append('</svg>')
    return ''.join(out)


def split_source_name(template):
    """Community maps are named '<source model>.<map>'; Reactome / WikiPathways entries are pathway titles."""
    name = template.get('name') or ''
    if template.get('source') == 'escher' and '.' in name:
        model, _, part = name.partition('.')
        return model, part
    return None, name


def structure_card(template, reaction_ids, model_reactions, extra=None, name_limit=6):
    source_model, source_map = split_source_name(template)
    ids = sorted(reaction_ids)
    card = dict(template_id=template.get('id'), source=template.get('source'), source_model=source_model, source_map=source_map, species=template.get('species'),
                license=template.get('license'), size=len(ids), reaction_count=len(ids),
                reactions=[dict(id=r, name=(model_reactions.get(r) or {}).get('name') or '') for r in ids[:name_limit]],
                svg=thumbnail_svg(template, SOURCE_COLORS.get(template.get('source'), '#128781')))
    card.update(extra or {})
    return card


def write_step_detail(job, key, payload):
    """Details of a pipeline step live in their own file (step_<key>.json); job.json only names the file."""
    name = f'step_{key}.json'
    dump(job.folder / name, payload)
    for st_ in job.state['steps']:
        if st_['key'] == key:
            st_['detail'] = name
    job.save()


def module_codes(blocks):
    """Stable display codes U0001.. in the order used by modules.json (largest first, then id)."""
    return {b['id']: 'U%04d' % (i + 1) for i, b in enumerate(sorted(blocks, key=lambda b: (-len(b['reaction_ids']), b['id'])))}


def grid_skeleton_svg(model, blocks, path):
    """Overview of all matched structures on a grid (used for large models, where the ELK macro layout is a tall stripe)."""
    from xml.sax.saxutils import escape
    g = model_graph(model)
    codes = module_codes(blocks)
    boxes = []
    for b in sorted(blocks, key=lambda b: (b['source'], -len(b['reaction_ids']), b['id'])):
        arr = np.array(list(b['positions'].values()))
        mn, mx = arr.min(0), arr.max(0)
        boxes.append((b, mn, mx - mn + 1))
    total = sum((w + 50) * (h + 70) for _, _, (w, h) in boxes)
    row_width = max(1200., float(np.sqrt(total)) * 1.5)
    x = y = row_h = 0.
    placed = []
    for b, mn, (w, h) in boxes:
        if x + w + 40 > row_width and x > 0:
            x, y, row_h = 0., y + row_h + 40, 0.
        placed.append((b, mn, x, y, w, h))
        x += w + 50
        row_h = max(row_h, h + 40)
    W, H = row_width + 80, y + row_h + 80
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-20 -20 {W:.0f} {H:.0f}" font-family="Helvetica, Arial, sans-serif"><rect x="-20" y="-20" width="{W:.0f}" height="{H:.0f}" fill="#fff"/>']
    for b, mn, ox, oy, w, h in placed:
        color = SOURCE_COLORS.get(b['source'], '#128781')
        out.append(f'<rect x="{ox - 12:.1f}" y="{oy - 24:.1f}" width="{w + 24:.1f}" height="{h + 36:.1f}" rx="6" fill="#f7fafb" stroke="#dbe4e8"/>')
        out.append(f'<text x="{ox - 6:.1f}" y="{oy - 9:.1f}" font-size="11" fill="#5b7480">{codes[b["id"]]} · {len(b["reaction_ids"])} rx</text>')
        pos = {k: (np.array(v) - mn + [ox, oy]) for k, v in b['positions'].items()}
        for a, c in sorted(tuple(sorted(e)) for e in g.subgraph(set(pos)).edges()):
            out.append(f'<line x1="{pos[a][0]:.1f}" y1="{pos[a][1]:.1f}" x2="{pos[c][0]:.1f}" y2="{pos[c][1]:.1f}" stroke="{color}" stroke-opacity=".55" stroke-width="2"/>')
        for k in sorted(pos):
            v = pos[k]
            if k.startswith('r:'):
                out.append(f'<rect x="{v[0] - 4:.1f}" y="{v[1] - 4:.1f}" width="8" height="8" fill="{color}"><title>{escape(k[2:])} ({escape(b["source"])}, {escape(b["match_level"])})</title></rect>')
            else:
                out.append(f'<circle cx="{v[0]:.1f}" cy="{v[1]:.1f}" r="3.5" fill="#fff" stroke="{color}" stroke-width="2"><title>{escape(k[2:])}</title></circle>')
    out.append('</svg>')
    path.write_text('\n'.join(out))



def draw_map_from_previous(model, selected, folder, full_model=None, log=None):
    """Rebuild from ``--previous`` when ≥80% of reactions are already drawn.

    Keeps the previous Escher geometry for every matched reaction (no global relayout).
    New / leftover reactions are grafted beside drawn participants when possible, or
    parked as a new component — never dropped.  A local relax of free nodes is only
    kept when it lowers crossings substantially (see ``keep_local_relayout``).
    """
    import previous_layout as prev
    import layout_quality
    from layout_quality import metric_details, assert_chemistry
    from jsonschema import Draft4Validator
    from xml.sax.saxutils import escape
    full_model = full_model or model
    original = uid(model)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data, report = prev.filter_previous_map(PREVIOUS_MAP_PATH, model)
    graft_report = None
    previous_relayout = dict(crossings_before=None, crossings_after=None, kept=False)
    if report.get('new_reactions'):
        data, graft_report = prev.graft_new_reactions(data, model, report['new_reactions'])
        report = dict(report, graft=graft_report)
        if log:
            log(f"  previous layout: grafted {graft_report['placed']} new reaction(s) "
                f"(embedded {graft_report['embedded']}, new components {graft_report['new_components']})")
        # Local relax of grafted midmarkers only; keep when crossings do not get worse
        # (and prefer a substantial drop when one is available).
        import copy as _copy_graft
        before_data = _copy_graft.deepcopy(data)
        data, relax_report = prev.relax_grafted_midmarkers(data, graft_report.get('midmarkers') or {})
        if not relax_report.get('kept', True):
            data = before_data
            relax_report = dict(relax_report, kept=False, reverted=True)
        previous_relayout = dict(
            crossings_before=relax_report.get('crossings_before'),
            crossings_after=relax_report.get('crossings_after'),
            kept=bool(relax_report.get('kept')),
            moved=relax_report.get('moved', 0))
        report = dict(report, graft_relax=relax_report)
        if log:
            log(f"  previous graft relax: crossings {relax_report.get('crossings_before')} -> "
                f"{relax_report.get('crossings_after')} "
                f"({'kept' if relax_report.get('kept') else 'reverted'}, "
                f"moved {relax_report.get('moved', 0)})")
    if log:
        extra = '' if not graft_report else f", +{graft_report['placed']} grafted"
        log(f"  previous layout: reused map ({report['kept_reactions']} reactions, "
            f"{report['removed_reactions']} removed, {report['nodes']} nodes{extra}); "
            f"skipped packing/redraw")
    # Hub split on the reused drawing (same gate as the full compose path).
    import split_cofactor_hubs as _sch
    from hub_drawing import pathway_from_model as _pathway_from_model
    curated_rids = {rid for b in (selected or []) for rid in b.get('reaction_ids') or []}
    pathway = _pathway_from_model(model)
    data, hub_report = _sch.apply_gated(
        data, model, pathway=pathway, curated=curated_rids, log=log)
    dump(folder / 'cofactor_hub_split.json', hub_report)
    dump(output_names.map_json(folder), data)
    try:
        Draft4Validator(read(ESCHER_SCHEMA)).validate(data)
    except Exception:
        data[0].setdefault('metadata', {}).pop('protected_connections', None)
        Draft4Validator(read(ESCHER_SCHEMA)).validate(data)
        dump(output_names.map_json(folder), data)
    # SVG: prefer line render when available; else a minimal placeholder from export-less path.
    if REACTION_STYLE == 'line':
        import line_render
        line_render.render_file(output_names.map_json(folder), output_names.map_svg(folder), {})
    else:
        # Lightweight SVG so the job always has a preview.
        (output_names.map_svg(folder)).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><text x="20" y="20">previous layout</text></svg>')
    quality_summary, quality_issues = layout_quality.evaluate(data)
    quality_summary['seconds'] = 0.0
    quality_summary['layout_style'] = LAYOUT_STYLE
    quality_summary['previous_layout'] = report
    dump(folder / 'layout_issues.json', quality_issues)
    metric = dict(
        method='previous_layout', previous_layout=report,
        layout_quality_final=quality_summary,
        completion=dict(method='previous_filter_graft', **report),
        previous_relayout=previous_relayout,
        cofactor_hub_split=hub_report,
        routing=dict(status='preserved_from_previous'),
        routing_attempts=[], orientation_mode=ORIENTATION,
        instance_optimization=dict(mode='off'), context_placement=dict(mode='off'),
        completion_context=dict(mode='off', previous_leftovers=graft_report),
        protected=dict(mode='off'),
        node_overlap_kinds=None, expert_geometry=None, curated_axes=None,
        constraints={}, peripheral=None,
    )
    metric.update(metric_details(output_names.map_json(folder)))
    assert_chemistry(model, output_names.map_json(folder))
    assert uid(model) == original
    sources = [{k: b.get(k) for k in ('id', 'template_id', 'map_id', 'source', 'lineage', 'license', 'partition', 'source_citation', 'source_sha256', 'original_instance')} for b in (selected or [])]
    dump(folder / 'source_licenses.json', dict(layout_sources=sources, policy='Original licenses preserved; research-partition content keeps its own terms'))
    data = read(output_names.map_json(folder))
    data[0].setdefault('metadata', {})['layout_sources'] = sources
    dump(output_names.map_json(folder), data)
    dump(folder / 'metrics.json', metric)
    # Minimal skeleton so downstream report steps find anchors.
    dump(folder / 'skeleton.json', dict(
        schema=4, model=model['id'], target_model_hash=original, modules=selected or [],
        bridges=[], containers=[], anchors={k: dict(xy=v) for k, v in (PREVIOUS_ANCHORS or {}).items()},
        display_positions={}, participants=[], reaction_status={},
        note='Map rebuilt from --previous; packing skipped; leftovers grafted.'))
    return metric


def draw_map(model, selected, folder, full_model=None, scalable=False, log=None, plan=None):
    """Same drawing pipeline as micromap_experiment.draw_case (P0-compatible), but robust: if routing fails or times out the map is
    re-exported with simpler routing and finally without routing instead of raising.  An empty `selected` draws the whole
    network from scratch.  micromap_experiment.py itself is not modified (its hash is recorded in saved results)."""
    import drawing
    import geometry
    import interfaces
    import retrieval
    import previous_layout as prev
    from layout_quality import metric_details, assert_chemistry
    from core import CanonicalGraph
    from jsonschema import Draft4Validator
    from xml.sax.saxutils import escape
    full_model = full_model or model
    original = uid(model)
    # High previous coverage: pin matched geometry and graft leftovers (no global pack/redraw).
    if PREVIOUS_SKIP_PACK and PREVIOUS_MAP_PATH and not scalable:
        return draw_map_from_previous(model, selected, folder, full_model=full_model, log=log)
    g = CanonicalGraph(model_graph(model))
    bridges = retrieval.bridges(model, selected)
    # Prefer an explicit global plan (connection-aware packing) whenever provided — including
    # small curated models where translation-only fit_frame needs adjacent neighbour placement.
    if PREVIOUS_SKIP_PACK and PREVIOUS_ANCHORS:
        # High previous coverage: keep module metadata but pin every known glyph; skip packing search.
        if plan is not None:
            plan = copy.deepcopy(plan)
            anchors, blocks, containers, routes = plan['anchors'], plan['blocks'], plan['containers'], plan['routes']
        else:
            anchors, blocks, containers, routes = drawing.pack_modules(model, copy.deepcopy(selected), bridges, physical_scale=True)
        anchors = {k: list(v) for k, v in PREVIOUS_ANCHORS.items()}
        blocks, n_prev = prev.apply_block_positions(blocks, PREVIOUS_ANCHORS)
        if log:
            log(f'  previous layout: pinned {len(PREVIOUS_ANCHORS)} anchors '
                f'({prev.reaction_anchor_share(PREVIOUS_ANCHORS, model):.0%} of reactions); skipped packing')
    elif plan is not None or scalable:
        import global_layout as gl_pack
        plan = copy.deepcopy(plan) if plan is not None else gl_pack.plan_global(model, selected, bridges, g, log=log)
        anchors, blocks, containers, routes = plan['anchors'], plan['blocks'], plan['containers'], plan['routes']
    else:
        anchors, blocks, containers, routes = drawing.pack_modules(model, copy.deepcopy(selected), bridges, physical_scale=True)
    proposal_by_id = {p['id']: p for p in selected}
    for b in blocks:
        p = proposal_by_id[b['id']]
        for key in ('license', 'partition', 'mapping_evidence', 'source_glyph_mapping', 'label_compartments_verified', 'source_to_initial_frame',
                    'source_xy', 'source_path_policy', 'source_citation', 'view_clipping', 'source_sha256', 'original_instance',
                    'same_map_rigid', 'source_xy_absolute'):
            if key in p:
                b[key] = p[key]
    # Small-model path: re-pack above drops skeleton rigid bake — restore same-source group here.
    # Skip when previous pins already own the canvas (≥80% coverage).
    if plan is None and not PREVIOUS_SKIP_PACK and not any(b.get('same_map_rigid') for b in blocks):
        import curated_axes
        blocks, same_map_bake = curated_axes.bake_same_map_positions(
            blocks, model_n_reactions=len(model.get('reactions') or []))
        if same_map_bake.get('blocks'):
            if log:
                log(f"Same-map rigid (draw): {same_map_bake['blocks']} block(s) in {same_map_bake['groups']} group(s) "
                    f"mode={same_map_bake.get('mode')}")
                pad = same_map_bake.get('padding_compression') or {}
                if pad.get('applied'):
                    log(f"  padding compress: {pad.get('segments_compressed', 0)} segment(s) "
                        f"(x={pad.get('gaps_x', 0)}/−{pad.get('removed_x', 0)}px, "
                        f"y={pad.get('gaps_y', 0)}/−{pad.get('removed_y', 0)}px) "
                        f"max_gap={pad.get('max_gap')}")
            anchors = {}
            met_acc = defaultdict(list)
            for b in blocks:
                for k, v in b['positions'].items():
                    if k.startswith('r:'):
                        anchors[k] = v
                    else:
                        met_acc[k].append(v)
            for k, vals in met_acc.items():
                anchors[k] = np.mean(vals, axis=0).tolist()
    if not PREVIOUS_SKIP_PACK:
        for bridge in bridges:
            a, c = anchors.get(bridge['nodes'][0]), anchors.get(bridge['nodes'][-1])
            if a is None or c is None:
                continue
            for i, rid in enumerate(bridge['reaction_ids']):
                key = 'r:' + rid
                anchors[key] = (np.array(a) + (np.array(c) - a) * (i + 1) / (len(bridge['reaction_ids']) + 1)).tolist()
    # Defer all --previous placement to the final Escher rigid translate so curated
    # structure relatives stay identical to the no-previous drawing.  Mid-pipeline
    # pins / leftovers change neighbour forces and deform structures.
    if PREVIOUS_ANCHORS and not PREVIOUS_SKIP_PACK and log:
        log(f'  previous layout: {len(PREVIOUS_ANCHORS)} anchors '
            f'({prev.reaction_anchor_share(PREVIOUS_ANCHORS, model):.0%} of reactions); '
            f'deferred to final rigid structure translate')
    timings = {}
    stamp = time.time()

    def lap(name):
        nonlocal stamp
        timings[name] = round(time.time() - stamp, 1)
        stamp = time.time()
        if log:
            log(f'  {name}: {timings[name]} s')

    def _pin_previous(scene, label=None):
        # Mid-pipeline previous pins are deferred; see final translate_structures_escher.
        return scene

    def _same_map_or_previous_restore(scene, label):
        """Same-map absolute restore (identical to the no-previous path)."""
        import curated_axes as _ca_restore
        scene, _rr = _ca_restore.restore_rigid_group(scene, blocks)
        if log and _rr.get('restored'):
            log(f'  same-map rigid restore ({label}): {_rr["restored"]} glyphs'
                + (f', n_rx={_rr["n_rx"]}' if _rr.get('n_rx') is not None else ''))
        return scene

    if scalable:
        import completion
        import global_layout as gl
        if plan['missing']:  # nodes no module placed (should not happen): fCoSE around everything that is placed
            core = gl.complete_global(g, anchors, iterations=350)
            lap('completion of unplaced nodes (fCoSE)')
        else:
            core = {k: list(v) for k, v in anchors.items() if k in g}
        metric_completion = dict(method='modules placed as rigid groups (curated blocks + automatic modules)', unplaced=len(plan['missing']), **{k: v for k, v in plan['stats'].items() if k in ('auto_modules', 'auto_modules_linked_to_expert', 'auto_modules_not_linked')})
    elif COMPLETION == 'spaced':
        import completion
        core = completion.complete_spaced(g, anchors, iterations=350)
        metric_completion = dict(method='spaced', **completion.complete_spaced.last)
    else:
        core = drawing.complete(g, anchors, iterations=350)
        metric_completion = dict(method='classic')
    context_report, context_assign = None, {}
    if CONTEXT == 'auto' and not PREVIOUS_SKIP_PACK:
        import context as context_module
        context_module.LAYOUT_STYLE_BACKBONE = (LAYOUT_STYLE == 'curated')
        covered = {rid for p in selected for rid in p['reaction_ids']} | {rid for b in bridges for rid in b['reaction_ids']}
        regions, context_report = context_module.find(model, currency(model), covered)
        core, context_assign, placed = context_module.place(model, core, currency(model), regions, covered)
        context_report.update(placed)
    scene = drawing.display_scene(model, core, blocks)
    if context_report is not None:
        context_report['isolated_reactions_before'] = len(context_module.isolated_reactions(model, scene))
        scene = context_module.apply(scene, context_assign, core)
        context_report['isolated_reactions_after'] = len(context_module.isolated_reactions(model, scene))
        if log:
            log(f"  reaction context: {context_report['reactions_without_core_participants']} reactions without non-currency participants, isolated {context_report['isolated_reactions_before']} -> {context_report['isolated_reactions_after']}")
    completion_report, completion_skip = None, set()
    if COMPLETION_CONTEXT == 'auto':
        import completion_context as completion_context_module
        if context_report is not None:
            completion_skip = {rid for region in regions if len(region['reactions']) > 1 for rid in region['reactions']}
        # With a previous map, embed a new reaction only when ≥2 participants are already drawn;
        # otherwise leave it as a new component (do not pull onto a single metabolite).
        # Same min_anchors as no-previous so curated geometry stays comparable;
        # --previous structure placement happens on the final map.
        scene, completion_report = completion_context_module.place(
            model, scene, blocks, currency(model), completion_skip, min_anchors=1)
        if log:
            log(f"  anchor-driven placement: {completion_report['placed']} of {completion_report['candidates']} reactions placed {completion_report['tiers']}, attachment {completion_report['attachment_rate_before']['rate']} -> {completion_report['attachment_rate_after']['rate']}")
        scene = _pin_previous(scene)
    peripheral_report = None
    instance_report = None
    if INSTANCES == 'auto' and not PREVIOUS_SKIP_PACK:
        import instances as instances_module
        scene, instance_report = instances_module.optimize(model, scene, blocks, currency(model))
        if log:
            log(f"  metabolite instances: +{instance_report['added']}, merged {instance_report['shared_nodes_merged']}, reattached {instance_report['reactions_reattached']}; edges >250: {instance_report['edges_over_250']}, >1000: {instance_report['edges_over_1000']}")
    regions_report = None
    if LAYOUT_STYLE == 'curated' and os.environ.get('OPM_REGIONS', 'off') == 'on':    # opt-in pathway-region organisation
        import regions as regions_module
        scene, regions_report = regions_module.organize(model, scene, blocks, currency(model))
        if log:
            log('  regions: ' + str(regions_report))
    composition_report = None
    if LAYOUT_STYLE == 'curated' and os.environ.get('OPM_COMPOSE', 'on') != 'off' and not PREVIOUS_SKIP_PACK:
        import composition
        scene, composition_report = composition.compose(model, scene, blocks, currency(model))
        if log:
            log('  composition: ' + str({k: v for k, v in composition_report.items() if k != 'moves'}))
        # Composition / landscape reflow must not break curated relatives.
        scene = _same_map_or_previous_restore(scene, 'after compose')
        scene = _pin_previous(scene)
    elif PREVIOUS_SKIP_PACK:
        scene = _pin_previous(scene)
    # Sync curated block positions from the composed scene (composition translates tiles
    # in the scene without rewriting blocks[].positions).
    def _sync_blocks_from_scene(blocks, scene):
        pos = scene.get('positions') or {}
        for b in blocks or []:
            bid = b.get('id')
            new_pos = {}
            for k, v in (b.get('positions') or {}).items():
                if k.startswith('r:') and k in pos:
                    new_pos[k] = list(pos[k])
                elif k.startswith('m:'):
                    ck = 'copy:%s:%s' % (bid, k[2:])
                    if ck in pos:
                        new_pos[k] = list(pos[ck])
                    elif k in pos:
                        new_pos[k] = list(pos[k])
                    else:
                        new_pos[k] = list(v)
                else:
                    new_pos[k] = list(v)
            if new_pos:
                b['positions'] = new_pos
        return blocks

    if composition_report is not None:
        blocks = _sync_blocks_from_scene(blocks, scene)
    # Re-place algorithm reactions against final curated AABB faces after composition.
    # Translation-only packing leaves algo midmarkers aimed at the old (rotated) frames;
    # a boundary-aware pass shortens curated×algorithm connectors and crossings.
    long_links_before = None
    if COMPLETION_CONTEXT == 'auto' and composition_report is not None:
        import completion_context as completion_context_module
        curated_rx = {rid for b in blocks for rid in b['reaction_ids']}
        long_links_before = completion_context_module.scene_longest_links(
            model, scene, currency(model), curated_rx, top=10)
        scene, completion_after = completion_context_module.place(
            model, scene, blocks, currency(model), completion_skip, boundary_aware=True,
            min_anchors=1)
        scene = _pin_previous(scene)
        if completion_report is not None:
            completion_report['after_compose'] = {
                k: completion_after.get(k)
                for k in ('placed', 'candidates', 'tiers', 'rounds', 'boundary_aware', 'prefer_ew')
            }
            # Summarise which attach sides were used (for algo×curated diagnosis).
            sides = Counter((d or {}).get('attach_side') for d in (completion_after.get('details') or {}).values()
                            if (d or {}).get('attach_side'))
            completion_report['after_compose']['attach_sides'] = dict(sides)
        else:
            completion_report = completion_after
        if log:
            log(f"  boundary-aware placement (after compose): {completion_after['placed']} of "
                f"{completion_after['candidates']} reactions {completion_after['tiers']}")
        # Compact stretched transport / exchange arms before orientation + export.
        import composition as composition_module
        scene, arm_report = composition_module.compact_long_arms_scene(
            model, scene, currency(model))
        if completion_report is not None:
            completion_report['compact_long_arms'] = arm_report
        if log and arm_report.get('shortened'):
            log(f"  compact long arms: shortened {arm_report['shortened']}, "
                f"pulled mids {arm_report['pulled_midmarkers']}")
        scene = _same_map_or_previous_restore(scene, 'after arms')
        # Private glyphs already move with each reaction in place(); a second full
        # instances.optimize here doubled runtime on iIT341 without cutting ac crossings.
    import orientation as orientation_module
    scene = orientation_module.apply(model, scene, blocks, ORIENTATION, 'before_relax')
    previous_relayout = None
    if scalable:
        scene, audit = gl.scalable_scene_relax(model, scene, blocks, extra_groups=plan['groups'])
        lap('scene relaxation (sparse)')
    else:
        scene, audit = geometry.complete_scene(model, scene, blocks)
    scene = orientation_module.apply(model, scene, blocks, ORIENTATION, 'after_relax')
    if INSTANCES == 'auto' and not PREVIOUS_SKIP_PACK:
        import instances as instances_module
        scene = instances_module.separate(model, scene, blocks)
        # After packing + separation, replace far cross-structure / far shared links with structure-local
        # inst: nodes.  Placement already enforces separate()'s metabolite gap / Escher primary diameter,
        # so a second separate pass is skipped (it shuffled neighbours and raised overlaps).
        scene, local_report = instances_module.structure_local(
            model, scene, blocks, currency(model),
            reasons=scene.get('instance_reasons'))
        if instance_report is not None:
            instance_report['structure_local'] = local_report
            instance_report['added'] = int(instance_report.get('added', 0)) + int(local_report.get('added', 0))
            instance_report['instances_separated'] = scene.get('instances_separated')
    import curated_axes as _ca3
    scene = _same_map_or_previous_restore(scene, 'after relax')
    scene = _pin_previous(scene)
    scene, _ov = _ca3.clear_rigid_node_overlaps(scene, blocks)
    if log and _ov.get('moved'):
        log(f"  clear rigid node overlaps: moved {_ov['moved']} non-rigid glyphs")
    scene = _pin_previous(scene)
    import layout_quality
    protected_report = None
    if completion_report is not None:
        import completion_context as completion_context_module
        broken = completion_context_module.verify_protected(scene)
        recs = scene.get('protected', [])
        protected_report = dict(groups=len(recs), strict=sum(bool(r.get('strict')) for r in recs), conditional=sum(not r.get('strict') for r in recs),
                                reactions=len({x for r in recs for x in r['attached_reaction_ids']}), violations=len(broken),
                                violated=[(r['canonical_metabolite_id'], r['attached_reaction_ids']) for r in broken][:20],
                                moved=(instance_report or {}).get('protected_groups'))
        if broken:
            raise Failure('protected_split', 'A protected group of connections was split (%d groups).' % len(broken))
    if INSTANCES == 'auto' and not PREVIOUS_SKIP_PACK:
        import instances as instances_module
        node_overlap_kinds = instances_module.overlap_breakdown(scene)
    else:
        node_overlap_kinds = None
    if completion_report is not None:
        completion_report['attachment_rate_final'] = completion_context_module.attachment_rate(model, scene, currency(model), {rid for b in blocks for rid in b['reaction_ids']}, completion_skip)
    expert_geometry = layout_quality.expert_geometry(blocks, scene)
    import curated_axes
    # Frame-only axes score: the fit_frame contract (no rot/reflect).  Scene residual Kabsch is
    # unstable on small / near-collinear modules after completion and is not the gate signal.
    axes = curated_axes.axes_report(blocks, scene=None)
    attempts = []
    # Routing decision: curated style uses Bezier curves (edge_style), legacy uses libavoid
    # For large models (> ROUTING_THRESHOLD reactions), defer routing to avoid long build times
    skip_libavoid = (scalable or 
                     EDGE_ROUTING in ('hybrid', 'none') or 
                     LAYOUT_STYLE == 'curated' or
                     len(model['reactions']) > ROUTING_THRESHOLD)
    for route, labels in (((False, False),) if skip_libavoid else ((True, True), (True, False), (False, False))):
        metric = drawing.export(model, core, folder, 'map', blocks=blocks, full_model=full_model, route=route, scene=scene, label_obstacles=labels)
        status = (metric.get('routing') or {}).get('status')
        attempts.append(dict(route=route, label_obstacles=labels, status=status if route else 'not_requested'))
        if not route or status == 'ok':
            break
    if scalable:
        lap('export (no routing)')
    sources = [{k: b.get(k) for k in ('id', 'template_id', 'map_id', 'source', 'lineage', 'license', 'partition', 'source_citation', 'source_sha256', 'original_instance')} for b in blocks]
    dump(folder / 'source_licenses.json', dict(layout_sources=sources, policy='Original licenses preserved; research-partition content keeps its own terms'))
    data = read(output_names.map_json(folder))
    if PREVIOUS_ANCHORS:
        data, n_pin = prev.pin_escher(data, PREVIOUS_ANCHORS)
        if n_pin:
            dump(output_names.map_json(folder), data)
            if log:
                log(f'  previous layout pin (escher): {n_pin} nodes')
    data[0]['metadata']['layout_sources'] = sources
    # copy: keys structure_local detached from — compact_long_arms_escher must not shorten them.
    data[0]['metadata']['structure_local_frozen_copies'] = list(
        scene.get('structure_local_frozen_copies') or ())
    if plan is not None:
        metric['global'] = dict(plan=(plan or {}).get('stats'), scene_nodes=len(scene['positions']),
                                routing_status=('none: straight connections between glyphs; structure edges use the structures\' own lines or curves'
                                                if scalable else 'curated/local routing'))
    if scalable:
        table = gl.instance_table(model, scene)
        data[0]['metadata']['metabolite_instances'] = table
        metric['instances'] = gl.instance_summary(table)
        metric['global'] = dict(metric.get('global') or {}, timings=timings, relax=audit)
    # edges whose metabolite glyph is a copy taken from an expert structure: the line renderer draws them straight, as in the structure
    if SOURCE_PATHS:
        import source_paths
        found, totals = [], Counter()
        for b in blocks:
            paths, info = source_paths.paths_for_block(b)
            totals.update({k: v for k, v in info.items() if isinstance(v, int) and not isinstance(v, bool)})
            found += [dict(reaction=r, metabolite=m, segs=v['segs'], deviation=v['deviation'], source=v['source']) for (r, m), v in paths.items()]
        # the final scene may move metabolite glyphs after the structure was placed (geometry.complete_scene): check the endpoints there;
        # a small move keeps the curve (preserved), a moderate one adjusts only the last segment (adjusted), a large one falls back (rerouted)
        kept, rerouted = [], Counter()
        for f in found:
            key = scene['participants'].get((f['reaction'], f['metabolite']))
            if key is None or ('r:' + f['reaction']) not in scene['positions']:
                rerouted['glyph_missing_in_final_scene'] += 1
                continue
            rel = np.array(scene['positions'][key]) - np.array(scene['positions']['r:' + f['reaction']])
            delta = rel - np.array(f['segs'][-1][-2:])
            shift = float(np.hypot(*delta))
            f['status'], f['shift_px'] = 'preserved', round(shift, 3)
            if shift > 3.0:
                segs, why = source_paths.adjust_chain(f['segs'], delta)
                if segs is None:
                    rerouted[why] += 1
                    continue
                f['segs'], f['status'] = segs, 'adjusted'
            f['deviation'] = round(float(np.hypot(*(rel - np.array(f['segs'][-1][-2:])))), 3)
            kept.append(f)
        # Record reuse for skip_edges / metrics now; stamp Beziers into map.json only after the
        # final-geometry evaluate so quality-gate cost stays comparable to maps without curves.
        data[0]['metadata']['source_paths'] = [dict(reaction=f['reaction'], metabolite=f['metabolite'], status=f['status'], shift_px=f['shift_px'], source=f['source'], segs=f['segs']) for f in kept]
        metric['source_paths'] = dict(structures=len(blocks), structures_with_paths=len({f['source']['map'] for f in found}), chains_eligible=totals.get('chains', 0),
                                      preserved=sum(f['status'] == 'preserved' for f in kept), adjusted=sum(f['status'] == 'adjusted' for f in kept),
                                      rerouted=sum(rerouted.values()), rerouted_by_reason=dict(rerouted), endpoint_mismatch=totals.get('endpoint_mismatch', 0),
                                      max_adjust_shift_px=max([f['shift_px'] for f in kept if f['status'] == 'adjusted'], default=0),
                                      transform_rejected=totals.get('transform_not_similarity', 0), not_in_module=totals.get('glyph_not_in_module', 0),
                                      max_endpoint_deviation_px=max([f['deviation'] for f in kept], default=0))
        metric['_source_path_chains'] = kept
    expert_reactions = {rid for b in blocks for rid in b['reaction_ids']}
    data[0]['metadata']['preserved_edges'] = sorted([rid, mid] for (rid, mid), key in scene['participants'].items() if key.startswith('copy:') and rid in expert_reactions)
    # Early expert_structures: whole atlas windows as corridor AABBs for edge_style.
    # Continuity later rewrites these to spatial pieces (see final evaluate below).
    codes_early = module_codes(blocks) if blocks else {}
    data[0]['metadata']['expert_structures'] = [
        dict(id=b.get('id') or b.get('template_id'), code=codes_early.get(b.get('id')),
             reactions=list(b['reaction_ids']))
        for b in blocks if len(b.get('reaction_ids') or []) >= 2]
    skip_edges = data[0]['metadata']['preserved_edges'] + [[p['reaction'], p['metabolite']] for p in data[0]['metadata'].get('source_paths', [])]
    if LAYOUT_STYLE == 'curated':
        import edge_style
        t_style = time.time()
        metric['edge_style'] = dict(edge_style.apply(data, skip_edges), seconds=round(time.time() - t_style, 1))
        if log:
                log(f"  curated-style connections: {metric['edge_style']}")
        if os.environ.get('OPM_CLEAR_THROUGH', 'on') != 'off':
            metric['clear_through_nodes'] = edge_style.clear_through_nodes(
                data, skip_edges, log=log)
        import label_placement
        t_labels = time.time()
        metric['label_placement'] = dict(label_placement.optimize(data), seconds=round(time.time() - t_labels, 1))
        if log:
            log(f"  label placement: {metric['label_placement']}")
        metric['edge_routing'] = dict(mode='curated (orthogonal algorithm connections)')
        if os.environ.get('OPM_PERIPHERAL', 'on') != 'off' and not PREVIOUS_SKIP_PACK:   # the switch exists for before / after comparisons
            import peripheral as peripheral_module
            import quality_gate
            _s, peripheral_report = peripheral_module.attach(model, scene, blocks, currency(model), apply=False)
            props = [dict(d, connection_px=d.get('connection_px', [0, 0])) for d in peripheral_report['details'] if d['state'] == 'proposed']
            # Mid/large GEMs: skip peripheral_pass. Its baseline evaluate_map alone is
            # O(minutes) on iIT341 and was exhausting the budget with 0 accepts.
            n_rx = len(model['reactions'])
            if n_rx >= 200:
                peripheral_report['gate'] = dict(proposed=len(props), accepted=0, rejected=0,
                                                 budget_exhausted=0, seconds=0.0, skipped='n_rx>=200')
                peripheral_report['gate_rows'] = []
                if log:
                    log('  peripheral modules: skipped gate (n_rx=%d); ' % n_rx
                        + str({k: v for k, v in peripheral_report.items() if k not in ('details', 'gate_rows')}))
            else:
                peri_max = 2
                data, gate = quality_gate.peripheral_pass(data, scene, props, skip_edges,
                                                         max_candidates=peri_max, log=log)
                for d in peripheral_report['details']:
                    row = next((r for r in gate['rows'] if r['exchange'] == d['exchange']), None)
                    if d['state'] == 'proposed' and row:
                        d['state'] = 'attached' if row['decision'] == 'accepted' else 'not_attached'
                        d['gate'] = row
                        if row['decision'] != 'accepted':
                            d.setdefault('x', None)
                peripheral_report['attached'] = sum(d['state'] == 'attached' for d in peripheral_report['details'])
                peripheral_report['not_attached'] = sum(d['state'] == 'not_attached' for d in peripheral_report['details'])
                peripheral_report['gate'] = {k: v for k, v in gate.items() if k != 'rows'}
                peripheral_report['gate_rows'] = gate['rows']
                if log:
                    log('  peripheral modules: ' + str({k: v for k, v in peripheral_report.items() if k not in ('details', 'gate_rows')}))
            dump(output_names.map_json(folder), data)
        if os.environ.get('OPM_BLOCK_RELOC', 'off') == 'on':     # demoted: stress packing should place blocks; enable with OPM_BLOCK_RELOC=on for A/B
            import block_relocation
            data, reloc = block_relocation.relocate(data, model, scene, blocks, currency(model), skip_edges, log=log)
            metric['block_relocation'] = reloc
            if log:
                log('  block relocation: ' + str({k: v for k, v in reloc.items() if k != 'rows'}))
            dump(output_names.map_json(folder), data)
        if os.environ.get('OPM_EXPERT_SUBMODULE', 'branch') in ('on', 'branch') and not PREVIOUS_SKIP_PACK:   # on by default: constrained relocation of exchange + transport modules inside big curated blocks; OPM_EXPERT_SUBMODULE=off disables it
            import block_relocation
            data, sub = block_relocation.relocate_sub(data, model, scene, blocks, currency(model), skip_edges,
                                                     log=log,
                                                     simple=os.environ.get('OPM_EXPERT_SUBMODULE', 'branch') != 'branch')
            metric['expert_submodule'] = sub
            data[0].setdefault('metadata', {})['regenerated_expert_edges'] = sub['regenerated_expert_edges']
            if log:
                log('  expert sub-module relocation: ' + str({k: v for k, v in sub.items() if k not in ('rows', 'regenerated_expert_edges')}))
            dump(output_names.map_json(folder), data)
        if os.environ.get('OPM_REGION_PLACE', 'off') == 'on' and not PREVIOUS_SKIP_PACK:     # opt-in pathway-aware region placement
            import region_placement
            data, rp = region_placement.place(data, model, scene, blocks, currency(model), skip_edges, log=log)
            metric['region_placement'] = rp
            data[0].setdefault('metadata', {}).setdefault('regenerated_expert_edges', [])
            data[0]['metadata']['regenerated_expert_edges'] += [e for row in rp.get('rows', []) for e in row.get('regenerated_expert_edges', [])]
            if log:
                log('  region placement: ' + str({k: v for k, v in rp.items() if k != 'rows'}))
            dump(output_names.map_json(folder), data)
        # Compact long transport/exchange arms after peripheral / submodule moves
        # (those passes can re-stretch arms that were shortened earlier on the scene).
        if not PREVIOUS_SKIP_PACK:
            import composition as composition_module
            arm_escher = composition_module.compact_long_arms_escher(data, currency(model))
            metric['compact_long_arms'] = arm_escher
            if composition_report is not None:
                composition_report['compact_long_arms_escher'] = arm_escher
            if log and arm_escher.get('shortened'):
                log(f"  compact long arms (escher): shortened {arm_escher['shortened']}")
        import curated_axes as _ca_es
        if not PREVIOUS_SKIP_PACK:
            data, _rr_es = _ca_es.restore_rigid_group_escher(data, blocks)
            if log and _rr_es.get('restored'):
                log(f"  same-map rigid restore (escher after arms): {_rr_es['restored']} nodes")
            data, _ov_es = _ca_es.clear_rigid_node_overlaps_escher(data, blocks)
            if log and _ov_es.get('moved'):
                log(f"  clear rigid node overlaps (escher): moved {_ov_es['moved']} nodes")
            dump(output_names.map_json(folder), data)
        # Final empty-band squeeze: collapse corridors wider than K × median reaction spacing
        # (nodes, labels and Bezier controls move together; monotone remap cannot add crossings).
        if not PREVIOUS_SKIP_PACK and os.environ.get('OPM_SQUEEZE', 'on') != 'off':
            import composition as composition_module
            spacing = composition_module.reaction_spacing_escher(data)
            k = float(os.environ.get('OPM_SQUEEZE_K', composition_module.SQUEEZE_K))
            squeeze = composition_module.squeeze_escher(data, max_gap=k * spacing)
            squeeze['spacing_px'] = round(spacing, 1)
            squeeze['k'] = k
            # Follow-up tighter gap so remaining BiGG alignment voids still shrink
            # (blank / empty-rect / long-p90), whether or not the first pass moved.
            squeeze2 = composition_module.squeeze_escher(
                data, max_gap=max(spacing * 1.25, 40.0))
            if squeeze2.get('applied'):
                squeeze2['spacing_px'] = round(spacing, 1)
                squeeze2['k'] = round(1.25, 2)
                squeeze2['fallback'] = True
                squeeze2['first'] = {kk: vv for kk, vv in squeeze.items()
                                     if kk not in ('first',)}
                squeeze = squeeze2
            metric['squeeze'] = squeeze
            if composition_report is not None:
                composition_report['squeeze'] = squeeze
            if log:
                log('  empty-band squeeze: ' + str(squeeze))
            # Monotone squeeze preserves relatives inside the rigid group — do not
            # restore here (that would re-expand BiGG-internal voids against a
            # squeezed surrounding canvas).
            dump(output_names.map_json(folder), data)
        # Shorten extreme empty axis-aligned corridors before hotspot so repair
        # can clean crossings introduced by midmarker slides.
        if not PREVIOUS_SKIP_PACK:
            import composition as composition_module
            import composition as _comp_fit
            spacing_now = composition_module.reaction_spacing_escher(data)
            corridor = _comp_fit.compact_empty_corridors_escher(data, spacing=spacing_now)
            # Whole-reaction corridor slides already keep private/EX partners attached;
            # a broad arm sweep after that retargets many spokes and costs continuity.
            # Collapse empty bands opened by the slide so neighbours can settle in.
            if corridor.get('shortened'):
                post_sq = composition_module.squeeze_escher(
                    data, max_gap=max(float(spacing_now or 150.0) * 1.25, 40.0))
                corridor['post_squeeze'] = {
                    k: post_sq.get(k) for k in ('applied', 'dx', 'dy', 'max_gap')}
            metric['empty_corridor_escher'] = corridor
            if composition_report is not None:
                composition_report['empty_corridor_escher'] = corridor
            if log and corridor.get('shortened'):
                log(f"  empty-corridor compact (escher): shortened {corridor['shortened']} "
                    f"max_gap={corridor.get('max_gap')} min_len={corridor.get('min_len')}")
            dump(output_names.map_json(folder), data)
        # Rebuild algorithm Beziers against post-restore geometry on small maps
        # (early edge_style ran before absolute snap; final orthogonal pass comes later).
        # Skip on mid/large — edge_style alone is costly on large GEMs.
        if len(model['reactions']) < 200:
            import edge_style as _es_post
            t_style2 = time.time()
            metric['edge_style_post_restore'] = dict(
                _es_post.apply(data, skip_edges), seconds=round(time.time() - t_style2, 1))
            if log:
                log(f"  curated-style connections (post-restore): {metric['edge_style_post_restore']}")
            if os.environ.get('OPM_CLEAR_THROUGH', 'on') != 'off':
                metric['clear_through_nodes_post_restore'] = _es_post.clear_through_nodes(
                    data, skip_edges, log=log)
            dump(output_names.map_json(folder), data)
        # Hotspot repair AFTER rigid restore + overlap clear + squeeze + corridor
        # so accepted moves are not undone by absolute snap / padding slides.
        if os.environ.get('OPM_HOTSPOT_REPAIR', 'on') != 'off' and not PREVIOUS_SKIP_PACK:
            import hotspot_repair
            codes_early = module_codes(blocks) if blocks else {}
            curated_structures_early = [
                dict(id=b.get('id') or b.get('template_id'), code=codes_early.get(b.get('id')),
                     reactions=list(b['reaction_ids']))
                for b in blocks if len(b.get('reaction_ids') or []) >= 2]
            data[0].setdefault('metadata', {})['expert_structures'] = curated_structures_early
            # After absolute rigid restore, more windows need search than the default
            # mass-scaled budget covers (e_coli formate cluster alone was 15 crossings).
            n_rx = len(model['reactions'])
            # Corridor slides can add a few local spoke crossings; give small maps
            # enough search to recover the pre-corridor counted-crossing floor.
            hs_budget = 35.0 if n_rx < 200 else 45.0
            hs_k = 10 if n_rx < 200 else 8
            # Mass-scaled try ceiling from budget_s (no wall-clock). Do not pass
            # PREVIOUS_ANCHORS as pinned_keys here: mid-pipeline --previous pins are
            # deferred to the final rigid structure translate so curated relatives
            # stay identical to the no-previous drawing.
            metric['hotspot_repair'] = hotspot_repair.repair(
                data, model=model, scene=scene, blocks=blocks, skip_edges=skip_edges, log=log,
                structures=curated_structures_early, k=hs_k, max_accepts=100,
                budget_s=hs_budget, max_rounds=8)
            if log:
                hr = metric['hotspot_repair']
                log('  hotspot repair: ' + str({k: hr.get(k) for k in ('accepted_count', 'tries', 'rounds', 'seconds', 'improved')}))
            if os.environ.get('OPM_CLEAR_THROUGH', 'on') != 'off':
                import edge_style as _es_clear
                metric['clear_through_nodes_post_hotspot'] = _es_clear.clear_through_nodes(
                    data, skip_edges, log=log)
            dump(output_names.map_json(folder), data)
        # Shrink canvas to content (same stage as main) before the final label pass.
        # Tip nudge may refit later only when it accepts a move.
        if not PREVIOUS_SKIP_PACK:
            import composition as _comp_fit
            fit = _comp_fit.fit_canvas_to_content(data)
            metric['canvas_fit'] = fit
            if log and fit.get('applied'):
                log(f"  canvas fit to content: {fit.get('content')} pad=40")
            dump(output_names.map_json(folder), data)
            # Final label pass after late geometry (rigid restore / overlap clear / squeeze /
            # hotspot). Root-cause stages already translate labels with nodes; this pass
            # re-anchors any residual detach and re-resolves text↔curve conflicts on the
            # finished glyph positions (early label_placement ran before those moves).
            import label_placement as _lp_final
            t_re = time.time()
            metric['label_placement_final'] = dict(
                _lp_final.optimize(data), seconds=round(time.time() - t_re, 1))
            if log:
                log(f"  label placement (final): {metric['label_placement_final']}")
            dump(output_names.map_json(folder), data)
    elif not scalable and len(model['reactions']) <= 400:
        import edge_routing
        er = dict(mode=EDGE_ROUTING, before=edge_routing.measure(data, skip_edges))
        if EDGE_ROUTING == 'hybrid':
            er.update(edge_routing.route(data, skip_edges))
            er['after'] = edge_routing.measure(data, skip_edges)
        metric['edge_routing'] = er
    else:
        metric['edge_routing'] = dict(mode=EDGE_ROUTING, not_applied='large model: routing of the algorithm-placed connections is a later step')
    # Stamp curated source curves before the final orthogonal pass so official
    # edge_orthogonality sees real waypoints (and so layout_quality describes the final map).
    pending = metric.pop('_source_path_chains', None)
    if pending:
        import source_paths
        written = source_paths.write_curves_to_escher(data, pending)
        sp = metric.get('source_paths') or {}
        sp['written'] = len(written)
        sp['not_written_to_map_json'] = len(pending) - len(written)
        metric['source_paths'] = sp
        data[0]['metadata']['source_paths'] = [
            dict(reaction=f['reaction'], metabolite=f['metabolite'], status=f['status'], shift_px=f['shift_px'], source=f['source'])
            for f in written
        ]
        dump(folder / 'map.json', data)
    # Final orthogonal polylines after curated source-path waypoints are present.
    gem_file = None
    for root in (folder, folder.parent):
        cand = root / 'upload' / 'model.xml'
        if cand.is_file():
            gem_file = cand
            break
        found = sorted((root / 'upload').glob('model.*')) if (root / 'upload').is_dir() else []
        if found:
            gem_file = found[0]
            break
    if LAYOUT_STYLE == 'curated':
        import edge_style as _es_final
        t_orth = time.time()
        # draw_map writes into <job>/full; the uploaded GEM lives in <job>/upload.
        metric['edge_style_orthogonal'] = dict(
            _es_final.apply(
                data, skip_edges, mode='orthogonal',
                gem_path=str(gem_file) if gem_file else None),
            seconds=round(time.time() - t_orth, 1))
        if log:
            log(f"  orthogonal algorithm routes (final): {metric['edge_style_orthogonal']}")
        # Orthogonal rewrite can reintroduce through-node hits; clear again on the
        # final polyline geometry (no canvas scale / map split).
        if os.environ.get('OPM_CLEAR_THROUGH', 'on') != 'off':
            metric['clear_through_nodes_post_orthogonal'] = _es_final.clear_through_nodes(
                data, skip_edges, log=log, mode='orthogonal',
                gem_path=str(gem_file) if gem_file else None)
        dump(folder / 'map.json', data)
    # Late per-reaction currency clones for Pathway-Scorer-style hubs (layout gate).
    # Runs after orthogonal multimarkers exist so the gate scores the same polylines
    # the official edge_orthogonality measure uses. Also runs on --previous pin paths
    # (including mid-size skip-packing), not only full redraws.
    import split_cofactor_hubs as _sch
    from hub_drawing import pathway_from_model as _pathway_from_model
    curated_rids = {rid for b in blocks for rid in b.get('reaction_ids') or []}
    pathway = _pathway_from_model(model)
    data, hub_report = _sch.apply_gated(
        data, model, pathway=pathway, curated=curated_rids, log=log)
    metric['cofactor_hub_split'] = hub_report
    dump(output_names.map_json(folder), data)
    dump(folder / 'cofactor_hub_split.json', hub_report)
    # Tip nudge after orthogonal routing and hub split; refit canvas only when a
    # tip is accepted so 0-accept maps stay byte-identical to main.
    if not PREVIOUS_SKIP_PACK:
        import composition as _comp_tip
        tip = _comp_tip.nudge_frontier_escher(
            # Keep tip identical to no-previous; structure previous is applied after.
            data, model, previous_anchors=None,
            skip_edges=skip_edges,
            gem_path=str(gem_file) if gem_file else None)
        metric['frontier_tip_nudge'] = tip
        if composition_report is not None:
            composition_report['frontier_tip_nudge'] = tip
        if log and tip.get('accepted'):
            log('  frontier tip nudge: ' + str(tip))
        if tip.get('accepted'):
            dump(output_names.map_json(folder), data)
            fit = _comp_tip.fit_canvas_to_content(data)
            metric['canvas_fit_after_tip'] = fit
            if log and fit.get('applied'):
                log(f"  canvas fit after tip: {fit.get('content')} pad=40")
            if (
                LAYOUT_STYLE == 'curated'
                and os.environ.get('OPM_CLEAR_THROUGH', 'on') != 'off'
            ):
                import edge_style as _es_tip
                metric['clear_through_nodes_post_tip'] = _es_tip.clear_through_nodes(
                    data, skip_edges, log=log, mode='orthogonal',
                    gem_path=str(gem_file) if gem_file else None)
            dump(output_names.map_json(folder), data)
    # Last geometry pass: tuck floating islands flush with the dense main hull so
    # empty AABB voids (e_coli TR/top floaters) do not survive into the PNG.
    if LAYOUT_STYLE == 'curated' and not PREVIOUS_SKIP_PACK:
        import composition as _comp_isl
        spacing_isl = _comp_isl.reaction_spacing_escher(data)
        island = _comp_isl.tuck_islands_escher(
            data, cur=currency(model), spacing=spacing_isl)
        metric['island_tuck_escher'] = island
        if composition_report is not None:
            composition_report['island_tuck_escher'] = island
        if log and island.get('moved'):
            log(f"  island silhouette tuck (escher, final): moved {island['moved']}/"
                f"{island.get('candidates', 0)} gap_thresh={island.get('gap_thresh')}")
        if island.get('moved'):
            fit_isl = _comp_isl.fit_canvas_to_content(data)
            metric['canvas_fit_after_island_tuck'] = fit_isl
            dump(output_names.map_json(folder), data)
            dump(folder / 'map.json', data)
            # Transport arms (FUMt2/MALt2) may still be long after rigid restore;
            # re-run empty-corridor with TCA-core freeze before through-clear.
            spacing_cor = _comp_isl.reaction_spacing_escher(data)
            cor2 = _comp_isl.compact_empty_corridors_escher(data, spacing=spacing_cor)
            metric['empty_corridor_escher_post_island'] = cor2
            if composition_report is not None:
                composition_report['empty_corridor_escher_post_island'] = cor2
            if log and cor2.get('shortened'):
                log(f"  empty-corridor compact (post-island): shortened {cor2['shortened']} "
                    f"max_gap={cor2.get('max_gap')} min_len={cor2.get('min_len')}")
                dump(output_names.map_json(folder), data)
                dump(folder / 'map.json', data)
            # Node translations can reintroduce official through-pairs; clear with
            # extra rounds — soft-corridor tuck can leave a thicker residual set.
            if os.environ.get('OPM_CLEAR_THROUGH', 'on') != 'off':
                import edge_style as _es_isl
                metric['clear_through_nodes_post_island_tuck'] = _es_isl.clear_through_nodes(
                    data, skip_edges, log=log, mode='orthogonal',
                    gem_path=str(gem_file) if gem_file else None,
                    max_rounds=6)
                # Second pass if a few official pairs remain after the first budget.
                post = metric['clear_through_nodes_post_island_tuck']
                if int(post.get('after_official') or 0) > 0:
                    metric['clear_through_nodes_post_island_tuck_2'] = _es_isl.clear_through_nodes(
                        data, skip_edges, log=log, mode='orthogonal',
                        gem_path=str(gem_file) if gem_file else None,
                        max_rounds=4)
                dump(output_names.map_json(folder), data)
                dump(folder / 'map.json', data)
    # Final --previous step: rigidly translate each curated structure toward previous
    # anchors.  Internal relatives stay identical to the no-previous drawing.
    # Do not clear overlaps afterwards — that pass can nudge duplicate rigid glyphs
    # and break within-structure relatives.
    if PREVIOUS_ANCHORS:
        data, n_final = prev.translate_structures_escher(
            data, PREVIOUS_ANCHORS, blocks=blocks, scene=scene)
        if log and n_final:
            log(f'  previous structure rigid translate (final map): {n_final} glyphs')
        import composition as _comp_prev_fit
        fit_prev = _comp_prev_fit.fit_canvas_to_content(data)
        if log and fit_prev.get('applied'):
            log(f"  canvas fit after previous translate: {fit_prev.get('content')} pad=40")
        dump(output_names.map_json(folder), data)
    t_quality = time.time()
    # Persist structure membership on the map so continuity can be scored from map.json alone.
    # Absolute same-map reuse keeps BiGG spacing; score spatial pieces, not whole atlas windows.
    codes = module_codes(blocks) if blocks else {}
    import curated_axes as _ca_final
    expert_structures = _ca_final.expert_structure_entries(blocks, codes)
    data[0].setdefault('metadata', {})['expert_structures'] = expert_structures
    quality_summary, quality_issues = layout_quality.evaluate(data, skip_edges, structures=expert_structures)
    quality_summary['seconds'] = round(time.time() - t_quality, 1)
    quality_summary['layout_style'] = LAYOUT_STYLE
    # Diagnosis reports are built inside evaluate (reuse crossing/edge pass). Attach the
    # pre-boundary longest-link snapshot when we have one.
    try:
        det = quality_summary.setdefault('details', {})
        ac_report = det.get('algorithm_curated_crossings') or {}
        long_pack = det.get('longest_links') or {}
        if isinstance(long_pack, dict):
            long_pack = dict(long_pack)
            if long_links_before is not None:
                long_pack['before'] = long_links_before
            det['longest_links'] = long_pack
        if log and ac_report.get('count'):
            top = ', '.join('%s×%d' % (r['algorithm_rid'], r['crossings'])
                            for r in (ac_report.get('top_algorithm_reactions') or [])[:8])
            log(f"  algorithm×curated crossings: {ac_report['count']} "
                f"(reasons {ac_report.get('by_reason')}); top algos: {top}")
        long_after = (long_pack or {}).get('after') or {}
        if log and long_after.get('top'):
            sample = '; '.join('%s–%s %.0fpx' % (r['reaction'], r['metabolite'], r['length_px'])
                               for r in long_after['top'][:5])
            log(f"  longest links p90={long_after.get('p90')} top: {sample}")
    except Exception as e:
        quality_summary.setdefault('details', {})['packing_diagnosis_error'] = str(e)[:200]
    if composition_report is not None:
        quality_summary['details']['composition_pass'] = {k: v for k, v in composition_report.items() if k != 'moves'}
        quality_issues['composition_moves'] = composition_report.get('moves', [])
    if peripheral_report is not None:
        quality_summary['details']['peripheral'] = dict({k: v for k, v in peripheral_report.items() if k not in ('details', 'gate_rows')},
                                                         unattached=[dict(exchange=d['exchange'], transports=d['transports'], anchors=d['anchors'], x=d.get('x'), y=d.get('y'))
                                                                     for d in peripheral_report['details'] if d['state'] == 'not_attached'][:12])
        quality_issues['peripheral'] = peripheral_report['details']
    metric['layout_quality_final'] = quality_summary
    metric['algorithm_curated_crossings'] = (quality_summary.get('details') or {}).get('algorithm_curated_crossings')
    metric['longest_links'] = (quality_summary.get('details') or {}).get('longest_links')
    metric['peripheral'] = None if peripheral_report is None else {k: v for k, v in peripheral_report.items() if k not in ('details', 'gate_rows')}
    dump(folder / 'layout_issues.json', quality_issues)
    metric['instance_optimization'] = instance_report if instance_report is not None else dict(mode='off')
    metric['regions'] = regions_report
    metric['context_placement'] = context_report if context_report is not None else dict(mode='off')
    metric['completion_context'] = completion_report if completion_report is not None else dict(mode='off')
    if previous_relayout is not None:
        metric['previous_relayout'] = previous_relayout
    metric['protected'] = protected_report if protected_report is not None else dict(mode='off')
    metric['node_overlap_kinds'] = node_overlap_kinds
    if protected_report is not None:
        data[0]['metadata']['protected_connections'] = [dict(r) for r in scene.get('protected', [])]
    # Node-level label_hidden is not in the Escher schema; keep ids under metadata (allowed).
    hidden_labels = [nid for nid, n in data[1]['nodes'].items() if n.pop('label_hidden', None)]
    for rx in data[1]['reactions'].values():
        rx.pop('label_hidden', None)
    if hidden_labels:
        data[0].setdefault('metadata', {})['hidden_metabolite_labels'] = hidden_labels
    try:
        Draft4Validator(read(ESCHER_SCHEMA)).validate(data)
    except Exception:
        # the extension keys are not accepted by the strict schema: keep the map interoperable and store the groups beside it
        dump(folder / 'protected_connections.json', data[0]['metadata'].pop('protected_connections', []))
        Draft4Validator(read(ESCHER_SCHEMA)).validate(data)
    dump(output_names.map_json(folder), data)
    svg = output_names.map_svg(folder)
    text = svg.read_text()
    offset = text.index('>') + 1
    svg.write_text(text[:offset] + '<metadata>' + escape(json.dumps(dict(layout_sources=sources), ensure_ascii=False)) + '</metadata>' + text[offset:])
    metric.update(metric_details(output_names.map_json(folder)))
    metric['constraints'] = audit
    metric['routing_attempts'] = attempts
    metric['expert_geometry'] = expert_geometry
    metric['curated_axes'] = axes
    metric['orientation_mode'] = ORIENTATION
    metric['completion'] = metric_completion
    assert_chemistry(model, output_names.map_json(folder))
    assert uid(model) == original
    used = {r for p in selected for r in p['reaction_ids']}
    bridge_ids = {r for b in bridges for r in b['reaction_ids']}
    allids = {r['id'] for r in model['reactions']}
    state = dict(schema=4, model=model['id'], target_model_hash=original, modules=blocks, bridges=bridges, containers=containers, anchors=scene['anchors'],
                 display_positions=scene['positions'], participants=[dict(reaction=r, metabolite=m, glyph=k) for (r, m), k in scene['participants'].items()],
                 reaction_status={rid: 'expert_module' if rid in used else 'algorithmic_bridge' if rid in bridge_ids else 'algorithm_completed' for rid in sorted(allids)},
                 pending_before_completion=sorted(allids - used - bridge_ids),
                 license_policy='Per-module source licenses are preserved', source_path_policy='Template centre-line controls indexed; final connections are algorithmic target-only paths')
    typed = copy.deepcopy(state['anchors'])
    state['anchors'] = {}
    state = interfaces.annotate(full_model, state)
    state['anchors'] = typed
    state['display_scope_reaction_ids'] = sorted(allids)
    state['full_target_model_hash'] = uid(full_model)
    dump(folder / 'skeleton.json', state)
    dump(folder / 'metrics.json', metric)
    return metric


def draw_one_view(model, selected, views, view_id, folder):
    """One fixed view with the P0-compatible pipeline; returns the metrics dict."""
    import drawing
    import retrieval
    sub = drawing.subset(model, views[view_id]['reaction_ids'])
    clipped = retrieval.clip_proposals(selected, sub)
    out = folder / 'views' / view_id
    metric = draw_map(sub, clipped, out, full_model=model)
    for ext in ('svg', 'json'):
        (folder / 'views' / (view_id + '.' + ext)).write_bytes((out / ('map.' + ext)).read_bytes())
    if (folder / 'reaction_provenance.json').exists():
        import ec_provenance as epv
        kinds = {r['reaction']: r['provenance'] for r in read(folder / 'reaction_provenance.json')}
        svg = folder / 'views' / (view_id + '.svg')
        if REACTION_STYLE == 'line':
            import line_render
            line_render.render_file(folder / 'views' / (view_id + '.json'), svg, kinds)
        else:
            svg.write_text(epv.recolor_svg(svg.read_text(), kinds))
    return metric


def merge_tiles(model, folder, view_ids, selected_count):
    """Tile the drawn views on a grid and merge them into one Escher JSON + one SVG (copies of a metabolite stay separate nodes)."""
    import re as _re
    maps, sizes = [], []
    for vid in view_ids:
        data = read(folder / 'views' / (vid + '.json'))
        maps.append(data)
        c = data[1]['canvas']
        sizes.append((c['width'], c['height']))
    cols = max(1, int(np.ceil(np.sqrt(len(maps)))))
    gap = 200.
    xs, ys, ox, oy, row_h, offsets, col = [], [], 0., 0., 0., [], 0
    for (w, h) in sizes:
        if col >= cols:
            ox, oy, row_h, col = 0., oy + row_h + gap, 0., 0
        offsets.append((ox, oy))
        ox += w + gap
        row_h = max(row_h, h)
        col += 1
    nodes, reactions, labels, counter, sources = {}, {}, {}, [0], []
    for data, (dx, dy) in zip(maps, offsets):
        body = data[1]
        idmap = {}
        for k, n in body['nodes'].items():
            counter[0] += 1
            idmap[k] = str(counter[0])
            n = dict(n)
            for a, b in (('x', dx), ('y', dy), ('label_x', dx), ('label_y', dy)):
                if a in n:
                    n[a] = n[a] + b
            nodes[idmap[k]] = n
        for k, r in body['reactions'].items():
            counter[0] += 1
            r = copy.deepcopy(r)
            r['label_x'] += dx
            r['label_y'] += dy
            for s in r['segments'].values():
                s['from_node_id'], s['to_node_id'] = idmap[s['from_node_id']], idmap[s['to_node_id']]
                for b in ('b1', 'b2'):
                    if s.get(b):
                        s[b] = dict(x=s[b]['x'] + dx, y=s[b]['y'] + dy)
            reactions[str(counter[0])] = r
        for k, tl in body.get('text_labels', {}).items():
            counter[0] += 1
            tl = dict(tl)
            tl['x'] += dx
            tl['y'] += dy
            labels[str(counter[0])] = tl
        sources += data[0].get('metadata', {}).get('layout_sources', [])
    width = max(o[0] + s[0] for o, s in zip(offsets, sizes))
    height = max(o[1] + s[1] for o, s in zip(offsets, sizes))
    header = dict(maps[0][0], map_name=model['id'] + ' (tiled views)', map_description='Fixed 50-reaction views tiled on a grid; metabolites shared between views appear as separate copies.')
    header['metadata'] = dict(header.get('metadata', {}), layout_sources=sources, tiled_views=len(maps))
    dump(output_names.map_json(folder), [header, dict(nodes=nodes, reactions=reactions, text_labels=labels, canvas=dict(x=0, y=0, width=width, height=height))])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" height="{height:.0f}"><rect width="100%" height="100%" fill="white"/>']
    for vid, (dx, dy), (w, h) in zip(view_ids, offsets, sizes):
        inner = (folder / 'views' / (vid + '.svg')).read_text()
        inner = _re.sub(r'^<svg[^>]*>', '', inner, count=1).rsplit('</svg>', 1)[0]
        parts.append(f'<svg x="{dx:.0f}" y="{dy:.0f}" width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}">{inner}</svg>')
        parts.append(f'<text x="{dx + 20:.0f}" y="{dy + 30:.0f}" font-family="Helvetica, Arial, sans-serif" font-size="22" fill="#5b7480">{vid}</text>')
    parts.append('</svg>')
    (output_names.map_svg(folder)).write_text('\n'.join(parts))


def _write_map_exports(job, folder, model, upload):
    """Write SBGN-ML, SBML Layout and CellDesigner SBML beside the Escher result (layout unchanged)."""
    try:
        import map_exports
        stem = output_names.stem_from_folder(folder)
        written = map_exports.write_all_exports(folder, model_path=upload, model=model, stem=stem)
        titles = {name: title for name, _kind, title in map_exports.export_file_entries(stem)}
        kinds = {name: kind for name, kind, _title in map_exports.export_file_entries(stem)}
        for name in written:
            job.add_file(name, titles.get(name, name), kinds.get(name, 'xml'))
        if written:
            job.log('Export formats: ' + ', '.join(written))
    except Exception as e:
        job.warn('export_formats_failed', f'Map exports could not be written: {type(e).__name__}: {str(e)[:180]}')


def check_integrity(model, original_hash, upload, map_path):
    """Three levels (the map may repeat reactions/metabolites as visual copies; it must never invent or alter biochemistry).

    model_integrity       the GEM is unchanged by the pipeline (hash of the working copy and of a fresh re-read of the upload)
    visualization_coverage every GEM reaction has at least one visual reaction in the map
    identity_integrity    every visual reaction is a real GEM reaction with the same stoichiometry and reversibility
    """
    out = {}
    fresh = normalize_model(load_model(upload), _Quiet())
    out['model_integrity'] = dict(passed=bool(uid(model) == original_hash and uid(fresh) == original_hash))
    gem = {r['id']: r for r in model['reactions']}
    if map_path is None or not Path(map_path).exists():
        out['visualization_coverage'] = dict(passed=None, reason='no map file was produced')
        out['identity_integrity'] = dict(passed=None, reason='no map file was produced')
        return out
    reactions = read(map_path)[1]['reactions']
    seen, wrong, normalised = Counter(), [], set()
    for r in reactions.values():
        rid = r['bigg_id']
        if rid not in gem:
            wrong.append(dict(reaction=rid, problem='not a GEM reaction'))
            continue
        coeffs = {m['bigg_id']: float(m['coefficient']) for m in r['metabolites']}
        g = gem[rid]
        # drawing.export draws reverse-only reactions (upper bound <= 0, lower bound < 0) in the direction the flux actually runs, i.e.
        # with the sign of every coefficient flipped; that is a documented direction normalisation, reported explicitly below.
        flipped = g['upper_bound'] <= 0 and g['lower_bound'] < 0
        expected = {k: (-1.0 if flipped else 1.0) * float(v) for k, v in g['metabolites'].items()}
        if flipped:
            normalised.add(rid)
        if coeffs != expected:
            wrong.append(dict(reaction=rid, problem='stoichiometry differs'))
        if bool(r.get('reversibility')) != bool(gem[rid]['lower_bound'] < 0 < gem[rid]['upper_bound']):
            wrong.append(dict(reaction=rid, problem='reversibility differs'))
        seen[rid] += 1
    missing = sorted(set(gem) - set(seen))
    out['visualization_coverage'] = dict(passed=not missing, gem_reactions=len(gem), drawn_reactions=len(seen), missing=missing[:50], missing_count=len(missing))
    out['identity_integrity'] = dict(passed=not wrong, visual_reactions=sum(seen.values()), visual_copies=sum(v - 1 for v in seen.values()), problems=wrong[:50], problem_count=len(wrong),
                                     direction_normalized_reverse_only=sorted(normalised))
    return out

def install_tool_environment():
    """Router binaries for this architecture (no-op on the arm64 machine the checked-in binaries came from)."""
    import routing_arch
    routing_arch.use_native_binaries()


# ------------------------------------------------------------------------------------------------ main job
def global_network_metrics(model, map_path):
    """Faithfulness and shape of the visual network (numbers only): edge lengths, connected components of the GEM and of the drawing."""
    import networkx as nx
    data = read(map_path)
    nodes = data[1]['nodes']
    lengths, drawn = [], nx.Graph()
    for rx in data[1]['reactions'].values():
        mid = nodes[next(iter(rx['segments'].values()))['from_node_id']]
        for seg in rx['segments'].values():
            end = nodes[seg['to_node_id']]
            if end['node_type'] == 'metabolite':
                lengths.append(math.hypot(end['x'] - mid['x'], end['y'] - mid['y']))
                drawn.add_edge('r:' + rx['bigg_id'], seg['to_node_id'])
    cur = set(currency(model))
    gem = nx.Graph()
    for r in model['reactions']:
        gem.add_node('r:' + r['id'])
        for m in r['metabolites']:
            if m not in cur:
                gem.add_edge('r:' + r['id'], 'm:' + m)
    arr = np.array(lengths) if lengths else np.zeros(1)
    canvas = data[1]['canvas']
    return dict(canvas=[round(canvas['width']), round(canvas['height'])], edges=len(lengths), edge_length_median=round(float(np.median(arr))), edge_length_p90=round(float(np.percentile(arr, 90))),
                edges_over_250=int((arr > 250).sum()), edges_over_1000=int((arr > 1000).sum()), gem_components=nx.number_connected_components(gem),
                drawn_components_by_instance=nx.number_connected_components(drawn),
                note='components: the GEM graph without currency metabolites vs the drawing, where every copy of a metabolite is its own node')


def draw_global_map(job, model, selected, folder, prov_kinds, n_reactions, plan=None):
    """One map for the whole model (scalable global steps, see global_layout).  Returns the map note, or None when the global layout failed
    (the caller then falls back to the tiled views; the failure is recorded and shown, never hidden)."""
    global GLOBAL_FAILURE
    job.log(f'Large model ({n_reactions} reactions): global layout of the whole network')
    t0 = time.time()
    try:
        # No wall-clock deadline: stages use deterministic iteration/eval budgets.
        metric = draw_map(model, selected, folder / 'full', full_model=model, scalable=True, log=job.log, plan=plan)
        if (folder / 'full' / 'layout_issues.json').is_file():
            (folder / 'layout_issues.json').write_bytes((folder / 'full' / 'layout_issues.json').read_bytes())
        output_names.map_json(folder).write_bytes((folder / 'full' / 'map.json').read_bytes())
        output_names.map_svg(folder).write_bytes((folder / 'full' / 'map.svg').read_bytes())
        if REACTION_STYLE == 'line':
            import line_render
            line_render.render_file(output_names.map_json(folder), output_names.map_svg(folder), prov_kinds)
        else:
            (output_names.map_svg(folder)).write_text(epv.recolor_svg((output_names.map_svg(folder)).read_text(), prov_kinds))
    except Exception as e:
        GLOBAL_FAILURE = f'{type(e).__name__}: {str(e)[:200]}'
        job.warn('global_failed', f'The global layout failed ({GLOBAL_FAILURE}); the map is built from tiled views instead.')
        output_names.map_json(folder).unlink(missing_ok=True)
        output_names.map_svg(folder).unlink(missing_ok=True)
        return None
    job.add_file(output_names.map_svg(folder).name, 'Completed map (global layout)', 'svg')
    job.add_file(output_names.map_json(folder).name, 'Completed map, Escher JSON', 'json')
    # Describe the final map JSON (after any orthogonal pass), not the pre-style export.
    ortho = metric.get('edge_style_orthogonal') or {}
    n_ortho = int(ortho.get('orthogonal') or 0)
    if n_ortho > 0:
        degraded = [f'routing: orthogonal polylines on {n_ortho} algorithm edge(s); stubs/curated paths unchanged']
        kind = 'global'
    else:
        degraded = ['routing: none (straight connections)']
        kind = 'global_degraded'
        job.warn('global_degraded', 'Global map without routing: connections are straight lines (structure edges keep their own lines or curves); long edges and crossings are not optimised yet.')
    network = global_network_metrics(model, output_names.map_json(folder))
    job.state['views'] = []  # a global map has no separate views; the modules tab lists the structures
    job.save()
    return dict(kind=kind, requested_layout='global', degraded=degraded, seconds=round(time.time() - t0, 1), network=network,
                metrics={k: metric.get(k) for k in ('node_overlap_pairs', 'label_overlap_pairs', 'straight_crossings')}, source_paths=metric.get('source_paths'),
                instances=metric.get('instances'), global_layout=metric.get('global'), completion=metric.get('completion'), expert_geometry=metric.get('expert_geometry'), instance_optimization=metric.get('instance_optimization'), context_placement=metric.get('context_placement'), completion_context=metric.get('completion_context'), layout_quality_final=metric.get('layout_quality_final'), edge_style=metric.get('edge_style'), edge_style_orthogonal=metric.get('edge_style_orthogonal'), peripheral=metric.get('peripheral'), block_relocation=metric.get('block_relocation'), region_placement=metric.get('region_placement'), regions=metric.get('regions'), expert_submodule=metric.get('expert_submodule'), label_placement=metric.get('label_placement'), hotspot_repair=metric.get('hotspot_repair'), clear_through_nodes=metric.get('clear_through_nodes'), clear_through_nodes_post_restore=metric.get('clear_through_nodes_post_restore'), clear_through_nodes_post_hotspot=metric.get('clear_through_nodes_post_hotspot'), clear_through_nodes_post_orthogonal=metric.get('clear_through_nodes_post_orthogonal'), clear_through_nodes_post_tip=metric.get('clear_through_nodes_post_tip'), protected=metric.get('protected'), node_overlap_kinds=metric.get('node_overlap_kinds'), edge_routing=metric.get('edge_routing'))


def run_job(folder):
    job = Job(folder)
    folder = job.folder
    options = read(folder / 'options.json') if (folder / 'options.json').exists() else {}
    libs = dict(core=True, ec=False)  # EC = extended mode, off by default
    libs.update(options.get('libraries', {}))
    libs.pop('micromap', None)  # MicroMap (NC-SA) removed; ignore stale option keys
    global REACTION_STYLE, COMPLETION, SOURCE_PATHS, LAYOUT, ORIENTATION, INSTANCES, CONTEXT, EDGE_ROUTING, COMPLETION_CONTEXT, LAYOUT_STYLE, PREVIOUS_ANCHORS, PREVIOUS_SKIP_PACK, PREVIOUS_REUSE_MAP, PREVIOUS_MAP_PATH
    LAYOUT_STYLE = normalize_layout_style(options.get('layout_style'))
    COMPLETION_CONTEXT = options.get('completion_context') if options.get('completion_context') in ('auto', 'off') else 'auto'
    EDGE_ROUTING = options.get('edge_routing') if options.get('edge_routing') in ('libavoid', 'hybrid', 'none') else 'libavoid'
    CONTEXT = options.get('context') if options.get('context') in ('auto', 'off') else 'auto'
    INSTANCES = options.get('instances') if options.get('instances') in ('auto', 'off') else 'auto'
    ORIENTATION = {True: 'full', False: 'off'}.get(options.get('orientation'), options.get('orientation')) if options.get('orientation') in (True, False, 'off', 'sides', 'full') else 'full'
    SOURCE_PATHS = bool(options.get('source_paths'))
    LAYOUT = options.get('layout') if options.get('layout') in ('global', 'tiled') else 'global'
    import global_layout as _gl
    _gl.BACKBONE = (LAYOUT_STYLE == 'curated')
    _gl.BLOCK_LAYOUT = options.get('block_layout') if options.get('block_layout') in ('auto', 'single') else 'auto'
    REACTION_STYLE = options.get('reaction_style') if options.get('reaction_style') in ('line', 'marker') else 'line'
    COMPLETION = options.get('completion') if options.get('completion') in ('spaced', 'classic') else 'spaced'
    PREVIOUS_ANCHORS = None
    PREVIOUS_SKIP_PACK = False
    PREVIOUS_REUSE_MAP = False
    PREVIOUS_MAP_PATH = None
    library_kind = options.get('library', 'windows')  # module-candidate source; 'windows' (default, Atlas v0.3) | 'units_sub' (structure units + derived sub-windows)
    if library_kind not in ('windows', 'units_sub'):
        library_kind = 'windows'
    try:
        job.step('parse')
        upload = next(iter(sorted((folder / 'upload').glob('model.*'))), None)
        if upload is None:
            raise Failure('bad_format', 'No uploaded model found.')
        if upload.stat().st_size > LIMITS['max_bytes']:
            raise Failure('too_large', 'The file is larger than 25 MB.')
        if upload.suffix.lower() == '.xml':
            reject_unsafe_xml(upload)
        try:
            model = load_model(upload)
        except Failure:
            raise
        except Exception as e:
            raise Failure('bad_format', f'The file could not be read as SBML or COBRA JSON: {str(e)[:200]}')
        model = normalize_model(model, job)
        if len(model['reactions']) > LIMITS['max_reactions']:
            raise Failure('too_large', f"The model has {len(model['reactions'])} reactions (limit {LIMITS['max_reactions']}).")
        coverage = annotation_coverage(model)
        job.log(f"Model {model['id']}: {len(model['reactions'])} reactions, {len(model['metabolites'])} metabolites; annotation coverage {coverage}")
        if coverage['bigg'] < .2 and coverage['rhea'] < .2 and coverage['ec'] < .2:
            job.warn('few_annotations', 'Almost no reaction carries BiGG, Rhea or EC annotations; few structures can be matched by identity.')
        original = uid(model)
        g = model_graph(model)
        allids = {r['id'] for r in model['reactions']}
        if options.get('previous'):
            import previous_layout as prev
            try:
                PREVIOUS_MAP_PATH = str(Path(options['previous']).resolve())
                PREVIOUS_ANCHORS = prev.load_previous_anchors(PREVIOUS_MAP_PATH, model)
            except Exception as e:
                raise Failure('bad_previous', f"Could not read --previous map: {e}") from e
            PREVIOUS_SKIP_PACK = prev.should_skip_packing(PREVIOUS_ANCHORS, model)
            missing_rx = prev.missing_model_reactions(PREVIOUS_MAP_PATH, model)
            PREVIOUS_REUSE_MAP = PREVIOUS_SKIP_PACK and not missing_rx
            job.log(f"Previous layout: {len(PREVIOUS_ANCHORS)} anchors from {PREVIOUS_MAP_PATH} "
                    f"({prev.reaction_anchor_share(PREVIOUS_ANCHORS, model):.0%} of reactions"
                    f"{'; reuse map' if PREVIOUS_REUSE_MAP else ('; skip packing' if PREVIOUS_SKIP_PACK else '')})")
            if PREVIOUS_SKIP_PACK:
                import global_layout as _gl_prev
                _gl_prev.BLOCK_LAYOUT = 'single'  # avoid candidate packing search

        job.step('map')
        install_tool_environment()
        import database
        import retrieval
        import semantic_transfer as st
        import ec_channel as ec
        import ec_provenance as epv
        ec_screening = {}
        if libs['core'] and library_kind != 'windows':
            import units_library
            core = units_library.modules(library_kind)
        else:
            core = database.modules() if libs['core'] else []
        bridge = load_bridge()
        job.log(f'Library ({library_kind}): {len(core)} open-core module instances loaded' if core else 'Open-core library switched off: no expert structures will be retrieved from it')
        pools_strong, pools_ec, templates = [], [], {m['id']: m for m in core}
        mapper = st.Mapper(model, bridge)
        if core:
            library = retrieval.Library(core)
            # Scale retrieval roots with model size: small models stay thorough; large ones cap VF2 work
            n_rx = len(model['reactions'])
            max_roots = 120 if n_rx <= 200 else (80 if n_rx <= 800 else 48)
            ps, audit = library.candidates(g, model=model, max_roots=max_roots)
            strong, sinfo = mapper.proposals(core, 'open_core')
        else:
            ps, audit, strong, sinfo = [], dict(levels={}), [], dict(candidate_chemical_reactions=0)
        job.log(f"Retrieval candidates: {len(ps)} (levels {audit['levels']})")
        job.log(f'Complete-equation candidates: {len(strong)} covering {sinfo["candidate_chemical_reactions"]} reactions')
        pools_strong = ps + strong
        if libs['ec']:
            ec.enrich(core)
            ecm = ec.ECMapper(model, bridge)
            core_ec, einfo = ecm.proposals(core, 'open_core EC')
            core_ec, ec_screening['open_core'] = epv.screen(core_ec, templates, ecm)
            pools_ec.append(core_ec)
            job.log(f'EC candidates (open core): {einfo["candidates"]} found, screened against the equations: {ec_screening["open_core"]}')
        if libs['ec']:
            job.warn('ec_extended', 'EC extended mode is on: structures matched only through a shared EC number are marked as functional layout suggestions, not as curated.')
        for p in pools_strong + [p for pool in pools_ec for p in pool]:
            t = templates.get(p['template_id'])
            if t:
                for key in ('license', 'source_sha256', 'original_instance', 'source_citation'):
                    if key in t:
                        p[key] = t[key]
                p.setdefault('partition', t.get('partition', 'open_core'))

        model_reactions = {r['id']: r for r in model['reactions']}
        try:
            all_candidates = pools_strong + [p for pool in pools_ec for p in pool]
            ranked, seen_templates = [], set()
            for p in sorted(all_candidates, key=lambda p: (-len(p['reaction_ids']), p['template_id'])):
                if p['template_id'] in seen_templates or p['template_id'] not in templates:
                    continue
                seen_templates.add(p['template_id'])
                ranked.append(structure_card(templates[p['template_id']], p['reaction_ids'], model_reactions,
                                             dict(match_level=p['match_level'], via='ec' if p['match_level'] == 'ec_topology' else 'id_equation_or_geometry')))
                if len(ranked) >= 24:
                    break
            write_step_detail(job, 'map', dict(
                library=library_kind, library_modules=len(core),
                candidates=dict(retrieval=len(ps), complete_equation=len(strong), ec=sum(len(x) for x in pools_ec)),
                candidates_by_source=dict(Counter(templates[p['template_id']]['source'] for p in all_candidates if p['template_id'] in templates)),
                distinct_structures=len({p['template_id'] for p in all_candidates}),
                shown=len(ranked), shown_note='largest candidate structures first (one card per library structure)', cards=ranked))
        except Exception as e:  # details are informational: never fail the job for them
            job.log(f'step details (map) not written: {e!r}'[:200])

        job.step('select')
        selected, stages = ec.select_with_ec(pools_strong, pools_ec)
        # a topology-only structure (no chemical identity) must not take over a reaction with dozens of participants (biomass): a 2-reaction pathway shape says
        # nothing about how 56 metabolites should be arranged.  Such a reaction is drawn by the algorithm (side-bundled, `instances` summary reaction).
        _cur = currency(model)
        _big = {r['id'] for r in model['reactions'] if sum(1 for m in r['metabolites'] if m not in _cur) >= 12}
        _drop = [p for p in selected if p.get('match_level') == 'geometry' and _big & set(p['reaction_ids'])]
        if _drop and os.environ.get('OPM_DROP_TOPO_BIG', 'off') == 'on':   # off by default: on iIT341 it raised crossings 470 -> 610-628 (the shared hub glyphs then spread the star)
            selected = [p for p in selected if p not in _drop]
            job.log(f'Dropped {len(_drop)} topology-only structure(s) that covered very large reactions: {sorted(_big & {r for p in _drop for r in p["reaction_ids"]})}')
        lowconf_report = None
        if LAYOUT_STYLE == 'curated' and os.environ.get('OPM_LOWCONF', 'off') == 'on':    # opt-in low-confidence structure decomposition
            import lowconf
            selected, lowconf_report = lowconf.split(model, currency(model), selected)
            job.log(f"Low-confidence decomposition: {len(lowconf_report['split'])} structure(s) split, {lowconf_report['kept']} kept, of {lowconf_report['checked']} topology-only: {lowconf_report['split']}")
        # Translation-only curated axes: one common scale per source map, keep relative offsets.
        import curated_axes
        selected, same_map_report = curated_axes.reapply_same_map_layout(selected)
        if same_map_report.get('modules'):
            job.log(f"Same-map axes: {same_map_report['modules']} module(s) across {same_map_report['maps']} map(s) reapplied (common scale, no rot/reflect)")
        used = {r for p in selected for r in p['reaction_ids']}
        job.log(f'Selected {len(selected)} structures covering {len(used)} of {len(allids)} reactions (stages {[len(s) for s in stages]})')
        save_gz(folder / 'selected.json.gz', selected)
        try:
            sel_cards = [structure_card(templates[p['template_id']], p['reaction_ids'], model_reactions,
                                        dict(match_level=p['match_level'], stage=next((i for i, st_ in enumerate(stages) if any(x is p or x.get('id') == p.get('id') for x in st_)), None)))
                         for p in sorted(selected, key=lambda p: (-len(p['reaction_ids']), p['id'])) if p['template_id'] in templates]
            write_step_detail(job, 'select', dict(library=library_kind, selected=len(selected), reactions_covered=len(used), reactions_total=len(allids),
                                                  stage_sizes=[len(x) for x in stages], cards=sel_cards))
        except Exception as e:
            job.log(f'step details (select) not written: {e!r}'[:200])

        job.step('skeleton')
        import drawing
        import interfaces
        bridges = retrieval.bridges(model, selected)
        global_plan = None
        # Large models: connection-aware global packing.  Small models keep ELK (baseline quality);
        # translation-only fit + landscape composition shelf still apply there.
        if LAYOUT == 'global' and LIMITS['full_routed_max_reactions'] < len(allids) <= LIMITS['whole_map_max_reactions']:
            import global_layout as gl_pack
            try:
                global_plan = gl_pack.plan_global(model, selected, bridges, g, log=job.log)  # the skeleton shows the same expert-block positions as the final global map
                anchors, blocks, containers, macro = [copy.deepcopy(global_plan[k]) for k in ('anchors', 'blocks', 'containers', 'routes')]
                expert_keys = {k for b in blocks for k in b['positions']}
                anchors = {k: v for k, v in anchors.items() if k in expert_keys}
            except Exception as e:
                global_plan = None
                job.log(f'global plan failed ({type(e).__name__}: {str(e)[:120]}); the skeleton uses the ELK layout')
        if global_plan is None:
            anchors, blocks, containers, macro = drawing.pack_modules(model, copy.deepcopy(selected), bridges, physical_scale=True)
        proposal_by_id = {p['id']: p for p in selected}
        for b in blocks:
            p = proposal_by_id[b['id']]
            for key in ('license', 'partition', 'mapping_evidence', 'source_citation', 'source_sha256', 'original_instance', 'layout_origin', 'counted_as_expert_design'):
                if key in p:
                    b[key] = p[key]
        # Small-model ELK path: same-source fragments → one rigid group (global_plan already bakes).
        if global_plan is None:
            import curated_axes
            blocks, same_map_bake = curated_axes.bake_same_map_positions(blocks, model_n_reactions=len(allids))
            if same_map_bake.get('blocks'):
                job.log(f"Same-map rigid: {same_map_bake['blocks']} block(s) in {same_map_bake['groups']} group(s) "
                        f"mode={same_map_bake.get('mode')}")
                pad = same_map_bake.get('padding_compression') or {}
                if pad.get('applied'):
                    job.log(f"  padding compress: {pad.get('segments_compressed', 0)} segment(s) "
                            f"(x={pad.get('gaps_x', 0)}/−{pad.get('removed_x', 0)}px, "
                            f"y={pad.get('gaps_y', 0)}/−{pad.get('removed_y', 0)}px) "
                            f"max_gap={pad.get('max_gap')}")
                # Rebuild anchors from rewritten block positions (bridges re-anchored below).
                anchors = {}
                met_acc = defaultdict(list)
                for b in blocks:
                    for k, v in b['positions'].items():
                        if k.startswith('r:'):
                            anchors[k] = v
                        else:
                            met_acc[k].append(v)
                for k, vals in met_acc.items():
                    anchors[k] = np.mean(vals, axis=0).tolist()
        for bridge in bridges:
            a, c = anchors.get(bridge['nodes'][0]), anchors.get(bridge['nodes'][-1])
            if a is None or c is None:
                continue
            for i, rid in enumerate(bridge['reaction_ids']):
                anchors['r:' + rid] = (np.array(a) + (np.array(c) - a) * (i + 1) / (len(bridge['reaction_ids']) + 1)).tolist()
        bridge_ids = {r for b in bridges for r in b['reaction_ids']}
        status = {rid: 'expert_module' if rid in used else 'algorithmic_bridge' if rid in bridge_ids else 'pending_algorithm' for rid in sorted(allids)}
        codes = module_codes(blocks)
        provenance = epv.reaction_provenance(model, selected, bridge_ids, codes, templates=templates, mapper=mapper)
        evidence_summary = epv.evidence_summary(provenance)
        job.log('Evidence: ' + ', '.join(f'{k} {v}' for k, v in evidence_summary['chemical_compatibility'].items()) + f"; identity revoked {evidence_summary['identity_revoked']}")
        dump(folder / 'reaction_provenance.json', list(provenance.values()))
        job.add_file('reaction_provenance.json', 'Reaction provenance', 'json')
        prov_kinds = {rid: v['provenance'] for rid, v in provenance.items()}
        prov_counts = Counter(prov_kinds.values())
        legend = [(k, prov_counts.get(k, 0)) for k in epv.KINDS]
        job.log('Provenance: ' + ', '.join(f'{k} {n}' for k, n in legend))
        skeleton = dict(schema=5, model=model['id'], target_model_hash=original, modules=blocks, bridges=bridges, containers=containers,
                        anchors={k: dict(xy=v) for k, v in anchors.items()}, reaction_status=status,
                        note='Skeleton = curated structures plus network bridges only; every other reaction is completed algorithmically in step 2.')
        skeleton = interfaces.annotate(model, skeleton)
        dump(folder / 'skeleton.json', skeleton)
        if len(allids) <= LIMITS['full_routed_max_reactions']:
            render_skeleton_svg(model, blocks, containers, bridges, anchors, status, folder / 'skeleton.svg')
        else:
            grid_skeleton_svg(model, blocks, folder / 'skeleton.svg')
        job.add_file('skeleton.svg', 'Skeleton (curated structures + bridges)', 'svg')
        job.add_file('skeleton.json', 'Skeleton data', 'json')
        try:
            skel_cards = []
            for b in sorted(blocks, key=lambda b: (-len(b['reaction_ids']), b['id'])):
                if b['template_id'] in templates:
                    skel_cards.append(structure_card(templates[b['template_id']], b['reaction_ids'], model_reactions,
                                                     dict(code=codes[b['id']], match_level=b['match_level'], provenance=dict(Counter(prov_kinds[r] for r in b['reaction_ids'])))))
            write_step_detail(job, 'skeleton', dict(
                structures=len(blocks), structure_reactions=len(used), bridges=len(bridges), bridge_reactions=len(bridge_ids),
                bridge_reaction_list=[dict(id=r, name=(model_reactions.get(r) or {}).get('name') or '') for r in sorted(bridge_ids)[:40]],
                cards=skel_cards))
        except Exception as e:
            job.log(f'step details (skeleton) not written: {e!r}'[:200])

        job.step('complete')
        views = fixed_views(model, g, LIMITS['view_size'])
        dump(folder / 'views.json', views)
        job.state['views'] = [dict(id=v['id'], reactions=len(v['reaction_ids']), state='none') for v in views]
        job.save()
        map_note = None
        if len(allids) <= LIMITS['full_routed_max_reactions']:
            # Pass job.log so clear_through / hotspot progress reaches the UI
            # (otherwise "Complete the map" looks frozen while CrossingIndex runs).
            metric = draw_map(model, selected, folder / 'full', full_model=model, log=job.log)
            if (folder / 'full' / 'layout_issues.json').is_file():
                (folder / 'layout_issues.json').write_bytes((folder / 'full' / 'layout_issues.json').read_bytes())
                job.add_file('layout_issues.json', 'Layout issues (where the problems are)', 'json')
            output_names.map_json(folder).write_bytes((folder / 'full' / 'map.json').read_bytes())
            output_names.map_svg(folder).write_bytes((folder / 'full' / 'map.svg').read_bytes())
            job.add_file(output_names.map_svg(folder).name, 'Completed map (routed)', 'svg')
            job.add_file(output_names.map_json(folder).name, 'Completed map, Escher JSON', 'json')
            if REACTION_STYLE == 'line':
                import line_render
                line_render.render_file(output_names.map_json(folder), output_names.map_svg(folder), prov_kinds)
            else:
                (output_names.map_svg(folder)).write_text(epv.recolor_svg((output_names.map_svg(folder)).read_text(), prov_kinds))
            map_note = dict(kind='routed_full', routing=(metric.get('routing') or {}).get('status'), routing_attempts=metric.get('routing_attempts'),
                            metrics={k: metric.get(k) for k in ('node_overlap_pairs', 'label_overlap_pairs', 'straight_crossings')}, source_paths=metric.get('source_paths'), expert_geometry=metric.get('expert_geometry'), instance_optimization=metric.get('instance_optimization'), context_placement=metric.get('context_placement'), completion_context=metric.get('completion_context'), layout_quality_final=metric.get('layout_quality_final'), edge_style=metric.get('edge_style'), peripheral=metric.get('peripheral'), block_relocation=metric.get('block_relocation'), region_placement=metric.get('region_placement'), regions=metric.get('regions'), expert_submodule=metric.get('expert_submodule'), label_placement=metric.get('label_placement'), hotspot_repair=metric.get('hotspot_repair'), clear_through_nodes=metric.get('clear_through_nodes'), clear_through_nodes_post_restore=metric.get('clear_through_nodes_post_restore'), clear_through_nodes_post_hotspot=metric.get('clear_through_nodes_post_hotspot'), clear_through_nodes_post_orthogonal=metric.get('clear_through_nodes_post_orthogonal'), clear_through_nodes_post_tip=metric.get('clear_through_nodes_post_tip'), protected=metric.get('protected'), node_overlap_kinds=metric.get('node_overlap_kinds'), edge_routing=metric.get('edge_routing'))
        elif len(allids) <= LIMITS['whole_map_max_reactions']:
            global_note = None
            if LAYOUT == 'global':
                global_note = draw_global_map(job, model, selected, folder, prov_kinds, len(allids), plan=global_plan)
            if global_note is None:
                job.log(f'Large model: drawing {len(views)} views, then tiling them into one map')
                done, failed = [], []
                vmetrics = {}
                for i, v in enumerate(views):
                    try:
                        vmetrics[v['id']] = draw_one_view(model, selected, {x['id']: x for x in views}, v['id'], folder)
                        done.append(v['id'])
                        job.add_file(f"views/{v['id']}.svg", f"View {v['id']}", 'svg')
                    except Exception as e:
                        failed.append(v['id'])
                        job.log(f"View {v['id']} failed: {str(e)[:120]}")
                    for sv in job.state['views']:
                        if sv['id'] == v['id']:
                            sv['state'] = 'done' if v['id'] in done else 'none'
                    job.state['progress'] = round(4 / 6 + (i + 1) / len(views) / 6, 3)
                    job.save()
                if failed:
                    job.warn('views_failed', f'{len(failed)} of {len(views)} views could not be drawn and are missing from the tiled map.')
                if done:
                    merge_tiles(model, folder, done, len(selected))
                    (output_names.map_svg(folder)).write_text(epv.recolor_svg((output_names.map_svg(folder)).read_text(), {}))
                    job.add_file(output_names.map_svg(folder).name, 'Completed map (tiled views, routed)', 'svg')
                    job.add_file(output_names.map_json(folder).name, 'Completed map, Escher JSON', 'json')
                map_note = dict(kind='tiled_fallback' if LAYOUT == 'global' else 'tiled_views', requested_layout=LAYOUT, fallback_reason=(GLOBAL_FAILURE or None), views_drawn=len(done), views_total=len(views), views_failed=failed,
                                metrics={k: sum((m.get(k) or 0) for m in vmetrics.values()) for k in ('node_overlap_pairs', 'label_overlap_pairs', 'straight_crossings')})
            else:
                map_note = global_note
        else:
            job.log('Very large model: no whole map; views can be drawn on demand')
            job.warn('no_whole_map', f'The model has more than {LIMITS["whole_map_max_reactions"]} reactions: only the skeleton is produced; draw individual views on demand.')
            map_note = dict(kind='skeleton_only', views_total=len(views))

        job.step('report')
        if (output_names.map_json(folder)).exists():
            data = read(output_names.map_json(folder))
            data[0].setdefault('metadata', {})['reaction_provenance'] = prov_kinds
            dump(output_names.map_json(folder), data)
            _write_map_exports(job, folder, model, upload)
        integrity = check_integrity(model, original, upload, output_names.map_json(folder) if (output_names.map_json(folder)).exists() else None)
        job.log('Integrity: ' + ', '.join(f"{k}={v['passed']}" for k, v in integrity.items()))
        best = {}
        for p in selected:
            for r in p['reaction_ids']:
                level = 'complete_equation' if p.get('complete_chemical_equation_verified') else p['match_level']
                if r not in best or LEVEL_RANK[level] < LEVEL_RANK[best[r]]:
                    best[r] = level
        counts = Counter(LEVEL_NAMES[v] for v in best.values())
        kind_of = {r['id']: reaction_kind(r) for r in model['reactions']}
        kind_totals = Counter(kind_of.values())
        kind_curated = Counter(kind_of[rid] for rid in used)
        covered = dict(expert_module_reactions=len(used), complete_equation=counts['complete_equation'], reference_supported=counts['reference_supported'],
                       ec_topology=counts['ec_topology'], geometry_only=counts['geometry_only'], bridge_reactions=len(bridge_ids),
                       algorithm_completed=len(allids) - len(used) - len(bridge_ids), total=len(allids),
                       by_kind=dict(
                           metabolic=dict(total=kind_totals['metabolic'], curated=kind_curated['metabolic'],
                                          coverage=round(kind_curated['metabolic'] / kind_totals['metabolic'], 4) if kind_totals['metabolic'] else None),
                           transport=dict(total=kind_totals['transport'], curated=kind_curated['transport'],
                                          coverage=round(kind_curated['transport'] / kind_totals['transport'], 4) if kind_totals['transport'] else None),
                           exchange=dict(total=kind_totals['exchange'], curated=kind_curated['exchange'],
                                         coverage=round(kind_curated['exchange'] / kind_totals['exchange'], 4) if kind_totals['exchange'] else None)))
        modules = []
        codes = module_codes(blocks)
        for b in sorted(blocks, key=lambda b: (-len(b['reaction_ids']), b['id'])):
            t = templates.get(b['template_id'], {})
            modules.append(dict(code=codes[b['id']], module_id=b['template_id'], source=b['source'], map_name=t.get('name', ''), species=b['species'],
                                license=b.get('license') or t.get('license'), size=len(b['reaction_ids']),
                                tier='micro' if len(b['reaction_ids']) == 1 else 'small' if len(b['reaction_ids']) <= 3 else 'structure',
                                reactions=b['reaction_ids'], match_level=b['match_level'], partition=b.get('partition', 'open_core'),
                                provenance=dict(Counter(prov_kinds[r] for r in b['reaction_ids']))))
        dump(folder / 'modules.json', modules)
        job.add_file('modules.json', 'Used structures', 'json')
        licenses = dict(policy='Per-module source licenses are preserved; research-partition content keeps its own terms.',
                        layout_sources=[{k: b.get(k) for k in ('id', 'template_id', 'map_id', 'source', 'lineage', 'license', 'partition', 'source_citation', 'source_sha256', 'original_instance')} for b in blocks])
        dump(folder / 'licenses.json', licenses)
        job.add_file('licenses.json', 'Source licenses', 'json')
        pending = sorted(r for r, s in status.items() if s == 'pending_algorithm')
        import orientation as orientation_report
        layout_quality_report = dict(orientation_mode=ORIENTATION, expert_geometry=(map_note or {}).get('expert_geometry'), final_geometry=(map_note or {}).get('layout_quality_final'))
        if (output_names.map_json(folder)).exists():
            layout_quality_report['orientation'] = orientation_report.summary(read(output_names.map_json(folder)), prov_kinds)
        run_settings = dict(
            structure_library=library_kind, library_label={'windows': 'pathway-fragment reuse', 'units_sub': 'structure-unit reuse'}.get(library_kind, library_kind),
            sources=dict(core_open_cc0_ccby=bool(libs['core']), brenda_pathway_maps=True),
            ec_extension=bool(libs['ec']), ec_note='layout hints only; the reactions and stoichiometry of the target GEM are never replaced',
            drawing_engine=dict(version='V7', layout_style=LAYOUT_STYLE, completion_context=COMPLETION_CONTEXT, instances=INSTANCES, orientation=ORIENTATION, expert_curves=bool(SOURCE_PATHS),
                                edge_routing=EDGE_ROUTING, layout=LAYOUT),
            licences=sorted({str(t.get('license')) for t in (templates.values() if 'templates' in dir() else []) if t.get('license')})[:12] if False else 'see licenses.json')
        report = dict(run_settings=run_settings, model=dict(id=model['id'], name=model.get('name', ''), reactions=len(allids), metabolites=len(model['metabolites']), hash=original),
                      options=dict(libraries=libs, library=library_kind, source_paths=SOURCE_PATHS, reaction_style=REACTION_STYLE, completion=COMPLETION), annotation_coverage=coverage, covered=covered, modules=len(blocks),
                      by_source=dict(Counter(b['source'] for b in blocks)), levels=dict(counts), map=map_note, candidates=dict(retrieval=len(ps), complete_equation=len(strong), ec=sum(len(x) for x in pools_ec)),
                      channels=dict(retrieval_levels=audit.get('levels', {}), complete_equation_candidates=len(strong), ec_candidates=sum(len(x) for x in pools_ec),
                                    selected_structures=len(selected), libraries=libs, library=library_kind),
                      integrity=integrity, provenance=dict(prov_counts), evidence=evidence_summary, layout_quality=layout_quality_report, ec_screening=ec_screening,
                      pending_reaction_ids=pending[:5000], code_sha256={p: sha((BASE / p).read_bytes()) for p in ('tool_pipeline.py', 'ec_channel.py', 'retrieval.py', 'semantic_transfer.py', 'drawing.py')},
                      seconds=round(time.time() - job.t0, 1),
                      limits='Matching depends on the model annotations; the skeleton covers only reactions with an expert structure; completion is algorithmic.')
        dump(folder / 'report.json', report)
        job.add_file('report.json', 'Report', 'json')
        result_stem = output_names.stem_from_folder(folder)
        job.state['summary'] = dict(model=report['model'] | dict(id=model['id']), annotation_coverage=coverage, covered=covered, provenance=dict(prov_counts), ec_screening=ec_screening, layout_quality=layout_quality_report,
                                    modules=dict(count=len(blocks), by_source=report['by_source']), map_kind=(map_note or {}).get('kind'), map_layout=dict(requested=(map_note or {}).get('requested_layout'), fallback_reason=(map_note or {}).get('fallback_reason'), degraded=(map_note or {}).get('degraded'), network=(map_note or {}).get('network')), seconds=report['seconds'],
                                    result_stem=result_stem, result_basename=output_names.result_basename(result_stem))
        failed = [k for k, v in integrity.items() if v['passed'] is False]
        if failed:
            job.fail('integrity', 'Integrity check failed: ' + ', '.join(failed) + '. The files were kept for inspection but must not be used.')
            return job.state['status']
        job.finish()
        job.log('Done')
    except Failure as f:
        job.fail(f.code, f.en)
    except Exception as e:  # keep the traceback in the log, show a short message
        job.log(traceback.format_exc()[-1500:])
        job.fail('internal_error', f'Internal error: {str(e)[:200]}')
    return job.state['status']


class _Quiet:
    def warn(self, *args, **kwargs):
        pass


def draw_view(folder, view_id):
    """Draw one fixed view of a finished job with the P0-compatible pipeline (label avoidance, libavoid routing)."""
    job = Job(folder)
    folder = job.folder
    install_tool_environment()
    import drawing
    options = read(folder / 'options.json') if (folder / 'options.json').exists() else {}
    global REACTION_STYLE, COMPLETION, SOURCE_PATHS
    SOURCE_PATHS = bool(options.get('source_paths'))
    REACTION_STYLE = options.get('reaction_style') if options.get('reaction_style') in ('line', 'marker') else 'line'
    COMPLETION = options.get('completion') if options.get('completion') in ('spaced', 'classic') else 'spaced'
    upload = next(iter(sorted((folder / 'upload').glob('model.*'))))
    model = normalize_model(load_model(upload), _Quiet())
    views = {v['id']: v for v in read(folder / 'views.json')}
    with gzip.open(folder / 'selected.json.gz', 'rt') as f:
        selected = json.load(f)
    for v in job.state['views']:
        if v['id'] == view_id:
            v['state'] = 'queued'
    job.save()
    try:
        (folder / 'views').mkdir(exist_ok=True)
        draw_one_view(model, selected, views, view_id, folder)
        for v in job.state['views']:
            if v['id'] == view_id:
                v['state'] = 'done'
        job.add_file(f'views/{view_id}.svg', f'View {view_id}', 'svg')
        job.add_file(f'views/{view_id}.json', f'View {view_id}, Escher JSON', 'json')
        job.log(f'View {view_id} drawn')
    except Exception as e:
        job.log(traceback.format_exc()[-1500:])
        for v in job.state['views']:
            if v['id'] == view_id:
                v['state'] = 'none'
        job.warn('view_failed', f'View {view_id} could not be drawn: {str(e)[:160]}')


if __name__ == '__main__':
    # Fixed hash seed: variant/ordering choices inside the matching code depend on set iteration order.
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--view')
    a = ap.parse_args()
    if a.view:
        draw_view(a.folder, a.view)
    else:
        print(run_job(a.folder))
