"""Export a composed Escher result JSON to SBGN-ML, SBML Layout, and CellDesigner SBML.

Exports read the final map geometry and the input GEM. They do not change layout
coordinates or the model stoichiometry. Shared helpers live here so CLI compose,
map edits and the Map Generator write the same files.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from core import DATA, comp, read
from model_io import load_model

SBGN_NS = 'http://sbgn.org/libsbgn/0.3'
SBGN_SCHEMA = DATA / 'sbgn_schema' / 'SBGN.xsd'
CD_NS = 'http://www.sbml.org/2001/ns/celldesigner'
SBML_L2_NS = 'http://www.sbml.org/sbml/level2/version4'

PROCESS_W = PROCESS_H = 20.0
CHEM_PRIMARY = 40.0
CHEM_SECONDARY = 28.0
MACRO_W, MACRO_H = 80.0, 40.0

# SBO terms that map to macromolecule glyphs / PROTEIN in CellDesigner.
SBO_MACRO = {245, 252, 253, 254, 297, 336}

EXPORT_KINDS = (
    ('sbgn', 'sbgn', 'Completed map, SBGN-ML'),
    ('sbml', 'sbml', 'Completed map, SBML with Layout'),
    ('celldesigner.xml', 'xml', 'Completed map, CellDesigner SBML'),
)


def export_file_entries(stem):
    """Return (filename, kind, title) tuples for the three map export formats."""
    import output_names
    return tuple(
        (output_names.result_filename(stem, ext), kind, title)
        for ext, kind, title in EXPORT_KINDS
    )

_NCNAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_\-.]*$')


def _body(data):
    return data[1] if isinstance(data, list) else data


def _xy(node):
    return float(node['x']), float(node['y'])


def _segment_controls(reaction, a, b):
    for seg in reaction['segments'].values():
        if seg['from_node_id'] == a and seg['to_node_id'] == b:
            return seg.get('b1'), seg.get('b2')
        if seg['from_node_id'] == b and seg['to_node_id'] == a:
            return seg.get('b2'), seg.get('b1')
    return None, None


def _curve_segments(reaction, path, nodes):
    """Yield (start, c1_or_None, c2_or_None, end) along an Escher node path."""
    for a, b in zip(path, path[1:]):
        start = _xy(nodes[a])
        end = _xy(nodes[b])
        b1, b2 = _segment_controls(reaction, a, b)
        if b1 and b2:
            yield start, (float(b1['x']), float(b1['y'])), (float(b2['x']), float(b2['y'])), end
        else:
            yield start, None, None, end


def _reverse_curve_segments(segments):
    """Reverse a sequence of cubic/straight segments (swap ends; swap control order)."""
    out = []
    for start, c1, c2, end in reversed(list(segments)):
        if c1 is not None and c2 is not None:
            out.append((end, c2, c1, start))
        else:
            out.append((end, None, None, start))
    return out

def _reaction_paths(reaction, nodes):
    """Node-id paths from midmarker to each metabolite, following stored segments."""
    out_edges = {}
    for seg in reaction['segments'].values():
        out_edges.setdefault(seg['from_node_id'], []).append(seg['to_node_id'])
        out_edges.setdefault(seg['to_node_id'], [])
    targets = {t for ts in out_edges.values() for t in ts}
    roots = [n for n in out_edges if n not in targets] or list(out_edges)[:1]
    lines = []

    def walk(node, path):
        nxt = out_edges.get(node, [])
        if not nxt:
            lines.append(path)
            return
        for n in nxt:
            if n in nodes:
                walk(n, path + [n])

    for root in roots:
        if root in nodes:
            walk(root, [root])
    return lines


def _midmarker(reaction, nodes):
    for seg in reaction['segments'].values():
        for nid in (seg['from_node_id'], seg['to_node_id']):
            if nodes.get(nid, {}).get('node_type') == 'midmarker':
                return nid
    return None


def _model_lookup(model):
    return {m['id']: m for m in model['metabolites']}, {r['id']: r for r in model['reactions']}


def _sbo_term(model_met, sbml_species=None):
    if sbml_species is not None and sbml_species.isSetSBOTerm():
        return sbml_species.getSBOTerm()
    for a in model_met.get('annotations') or []:
        uri = a.get('uri') or ''
        m = re.search(r'SBO[:_](\d+)', uri, re.I)
        if m:
            return int(m.group(1))
    return None


def _is_macromolecule(model_met, sbml_species=None):
    return _sbo_term(model_met, sbml_species) in SBO_MACRO


def _entity_class(model_met, sbml_species=None):
    return 'macromolecule' if _is_macromolecule(model_met, sbml_species) else 'simple chemical'


def _cd_species_class(model_met):
    return 'PROTEIN' if _is_macromolecule(model_met) else 'SIMPLE_MOLECULE'


def _glyph_size(entity_class, primary):
    if entity_class == 'macromolecule':
        return MACRO_W, MACRO_H
    side = CHEM_PRIMARY if primary else CHEM_SECONDARY
    return side, side


def _bbox(cx, cy, w, h):
    return dict(x=cx - w / 2.0, y=cy - h / 2.0, w=w, h=h)


def _safe_id(prefix, raw):
    """Build an XML NCName/ID. Prefix supplies the leading letter when raw starts with a digit."""
    text = re.sub(r'[^A-Za-z0-9_\-.]', '_', str(raw))
    full = f'{prefix}{text}'
    if not full or not re.match(r'^[A-Za-z_]', full):
        full = 'n' + full
    return full


def _scheme_glyph_id(decoded_id, sbml_id):
    """Glyph id for layout-benchmark: SBML id without M_/R_ when that form is a valid NCName."""
    if _NCNAME.match(decoded_id):
        return decoded_id
    return sbml_id


def _sbml_id_maps(doc):
    model = doc.getModel()
    species, reactions, compartments = {}, {}, {}
    for i in range(model.getNumSpecies()):
        sid = model.getSpecies(i).getId()
        species[sid] = sid
        decoded = sid[2:] if sid.startswith('M_') else sid
        species.setdefault(decoded, sid)
    for i in range(model.getNumReactions()):
        rid = model.getReaction(i).getId()
        reactions[rid] = rid
        decoded = rid[2:] if rid.startswith('R_') else rid
        reactions.setdefault(decoded, rid)
    for i in range(model.getNumCompartments()):
        cid = model.getCompartment(i).getId()
        compartments[cid] = cid
    return species, reactions, compartments


def model_to_sbml_document(model):
    """Build an SBML Level 3 Version 1 document from the unified model dict (JSON GEMs)."""
    import libsbml
    doc = libsbml.SBMLDocument(3, 1)
    sbml = doc.createModel()
    sbml.setId(_safe_id('model_', model.get('id') or 'map'))
    if model.get('name'):
        sbml.setName(model['name'])
    comps = sorted({m.get('compartment') or 'c' for m in model['metabolites']} or {'c'})
    for cid in comps:
        c = sbml.createCompartment()
        c.setId(cid)
        c.setConstant(True)
        c.setSpatialDimensions(3)
        c.setSize(1.0)
    for met in model['metabolites']:
        s = sbml.createSpecies()
        sid = 'M_' + met['id'] if not met['id'].startswith('M_') else met['id']
        # Keep decoded BiGG ids as the SBML id when already prefixed-free in GEM JSON.
        s.setId(sid if _NCNAME.match(sid) else _safe_id('M_', met['id']))
        s.setName(met.get('name') or met['id'])
        s.setCompartment(met.get('compartment') or comps[0])
        s.setConstant(False)
        s.setBoundaryCondition(False)
        s.setHasOnlySubstanceUnits(False)
        s.setInitialAmount(0.0)
        if _is_macromolecule(met):
            s.setSBOTerm(245)
        else:
            s.setSBOTerm(247)
    # Map decoded id → SBML species id
    sid_of = {}
    for i in range(sbml.getNumSpecies()):
        sp = sbml.getSpecies(i)
        sid = sp.getId()
        decoded = sid[2:] if sid.startswith('M_') else sid
        sid_of[decoded] = sid
        sid_of[sid] = sid
    for rx in model['reactions']:
        r = sbml.createReaction()
        rid = 'R_' + rx['id'] if not rx['id'].startswith('R_') else rx['id']
        r.setId(rid if _NCNAME.match(rid) else _safe_id('R_', rx['id']))
        r.setName(rx.get('name') or rx['id'])
        r.setReversible(rx['lower_bound'] < 0 < rx['upper_bound'])
        r.setFast(False)
        for mid, coef in rx['metabolites'].items():
            sid = sid_of.get(mid) or sid_of.get('M_' + mid)
            if sid is None:
                continue
            if coef < 0:
                sr = r.createReactant()
                sr.setSpecies(sid)
                sr.setStoichiometry(abs(float(coef)))
                sr.setConstant(True)
            elif coef > 0:
                sr = r.createProduct()
                sr.setSpecies(sid)
                sr.setStoichiometry(float(coef))
                sr.setConstant(True)
    return doc


def _ensure_sbml_path(model_path, model, folder=None):
    """Return a path to an SBML file for the model (write into the job folder for JSON GEMs)."""
    if model_path is not None:
        path = Path(model_path)
        if path.suffix.lower() in ('.xml', '.sbml'):
            return path, False
    if folder is None:
        raise ValueError('folder is required to materialise SBML for a JSON model')
    doc = model_to_sbml_document(model)
    import libsbml
    tmp_path = Path(folder) / 'export_model.xml'
    if not libsbml.writeSBMLToFile(doc, str(tmp_path)):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError('could not write temporary SBML for layout export')
    return tmp_path, True

def validate_sbgn_xsd(path):
    """Validate SBGN-ML against the vendored libSBGN 0.3 XSD. Returns (ok, errors)."""
    from lxml import etree
    if not SBGN_SCHEMA.is_file():
        return False, [f'missing schema: {SBGN_SCHEMA}']
    schema = etree.XMLSchema(etree.parse(str(SBGN_SCHEMA)))
    doc = etree.parse(str(path))
    ok = schema.validate(doc)
    errors = [str(e) for e in schema.error_log.filter_from_errors()]
    return ok, errors


def validate_sbml_libsbml(path):
    """Validate SBML with libSBML. Returns (ok, error_messages) — warnings are ignored."""
    import libsbml
    doc = libsbml.readSBMLFromFile(str(path))
    messages = []
    for i in range(doc.getNumErrors()):
        err = doc.getError(i)
        if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            messages.append(err.getMessage())
    doc.checkConsistency()
    for i in range(doc.getNumErrors()):
        err = doc.getError(i)
        if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            messages.append(err.getMessage())
    return not messages, messages


# ------------------------------------------------------------------------------------------------ SBGN-ML


def build_sbgn(data, model, sbml_path=None):
    """Build an SBGN-ML 0.3 Process Description document from Escher map + model."""
    ET.register_namespace('', SBGN_NS)
    body = _body(data)
    nodes = body['nodes']
    reactions = body['reactions']
    canvas = body.get('canvas') or {}
    mets, _ = _model_lookup(model)

    # SBML ids for layout-benchmark scheme matching
    if sbml_path and Path(sbml_path).is_file():
        import libsbml
        doc = libsbml.readSBMLFromFile(str(sbml_path))
        sp_map, rx_map, _ = _sbml_id_maps(doc)
    else:
        sp_map = {m['id']: ('M_' + m['id'] if _NCNAME.match('M_' + m['id']) else _safe_id('M_', m['id']))
                  for m in model['metabolites']}
        for mid, sid in list(sp_map.items()):
            sp_map[sid] = sid
        rx_map = {r['id']: ('R_' + r['id'] if _NCNAME.match('R_' + r['id']) else _safe_id('R_', r['id']))
                  for r in model['reactions']}
        for rid, sid in list(rx_map.items()):
            rx_map[sid] = sid

    clone_count = defaultdict(int)
    for n in nodes.values():
        if n.get('node_type') == 'metabolite' and n.get('bigg_id'):
            clone_count[n['bigg_id']] += 1

    root = ET.Element(f'{{{SBGN_NS}}}sbgn')
    map_el = ET.SubElement(root, f'{{{SBGN_NS}}}map', {
        'id': 'map1',
        'language': 'process description',
    })
    ET.SubElement(map_el, f'{{{SBGN_NS}}}bbox', {
        'x': f"{float(canvas.get('x', 0)):.4f}",
        'y': f"{float(canvas.get('y', 0)):.4f}",
        'w': f"{float(canvas.get('width', 1000)):.4f}",
        'h': f"{float(canvas.get('height', 1000)):.4f}",
    })

    by_comp = defaultdict(list)
    for n in nodes.values():
        if n.get('node_type') != 'metabolite':
            continue
        mid = n.get('bigg_id')
        c = (mets.get(mid) or {}).get('compartment') or comp(mid or '')
        by_comp[c].append(_xy(n))
    pad = 40.0
    for cid, pts in sorted(by_comp.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0 = min(xs) - pad, min(ys) - pad
        w, h = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
        gid = _safe_id('comp_', cid)
        g = ET.SubElement(map_el, f'{{{SBGN_NS}}}glyph', {
            'id': gid, 'class': 'compartment', 'compartmentOrder': '1',
        })
        ET.SubElement(g, f'{{{SBGN_NS}}}label', {'text': cid})
        ET.SubElement(g, f'{{{SBGN_NS}}}bbox', {
            'x': f'{x0:.4f}', 'y': f'{y0:.4f}', 'w': f'{max(w, 80):.4f}', 'h': f'{max(h, 80):.4f}',
        })

    # Prefer primary clone for the scheme glyph id (one position per pathway node).
    scheme_owner = {}
    for nid, n in nodes.items():
        if n.get('node_type') != 'metabolite' or not n.get('bigg_id'):
            continue
        mid = n['bigg_id']
        if mid not in scheme_owner or n.get('node_is_primary', True):
            scheme_owner[mid] = nid

    entity_glyph = {}
    used_glyph_ids = set()
    drawn_mids = set()
    for nid, n in nodes.items():
        if n.get('node_type') != 'metabolite':
            continue
        mid = n.get('bigg_id') or nid
        drawn_mids.add(mid)
        met = mets.get(mid) or dict(id=mid, name=n.get('name') or mid, compartment=comp(mid))
        cls = _entity_class(met)
        primary = bool(n.get('node_is_primary', True))
        w, h = _glyph_size(cls, primary)
        cx, cy = _xy(n)
        bb = _bbox(cx, cy, w, h)
        sbml_sid = sp_map.get(mid) or ('M_' + mid)
        if scheme_owner.get(mid) == nid:
            gid = _scheme_glyph_id(mid, sbml_sid)
        else:
            gid = _safe_id('ep_', f'{mid}_{nid}')
        if gid in used_glyph_ids:
            gid = _safe_id('ep_', f'{mid}_{nid}')
        used_glyph_ids.add(gid)
        entity_glyph[nid] = gid
        compartment = (met.get('compartment') or comp(mid) or '')
        attrs = {'id': gid, 'class': cls}
        if compartment and compartment in by_comp:
            attrs['compartmentRef'] = _safe_id('comp_', compartment)
        g = ET.SubElement(map_el, f'{{{SBGN_NS}}}glyph', attrs)
        ET.SubElement(g, f'{{{SBGN_NS}}}label', {'text': met.get('name') or mid})
        if clone_count.get(mid, 0) > 1:
            ET.SubElement(g, f'{{{SBGN_NS}}}clone')
        ET.SubElement(g, f'{{{SBGN_NS}}}bbox', {
            'x': f"{bb['x']:.4f}", 'y': f"{bb['y']:.4f}", 'w': f"{bb['w']:.4f}", 'h': f"{bb['h']:.4f}",
        })

    canvas_w = float(canvas.get('width', 1000))
    canvas_h = float(canvas.get('height', 1000))
    park_x, park_y = 40.0, canvas_h + 60.0
    parked = False
    for mid, met in mets.items():
        if mid in drawn_mids:
            continue
        parked = True
        cls = _entity_class(met)
        w, h = _glyph_size(cls, False)
        sbml_sid = sp_map.get(mid) or ('M_' + mid)
        gid = _scheme_glyph_id(mid, sbml_sid)
        if gid in used_glyph_ids:
            gid = _safe_id('ep_park_', mid)
        used_glyph_ids.add(gid)
        g = ET.SubElement(map_el, f'{{{SBGN_NS}}}glyph', {'id': gid, 'class': cls})
        ET.SubElement(g, f'{{{SBGN_NS}}}label', {'text': met.get('name') or mid})
        ET.SubElement(g, f'{{{SBGN_NS}}}bbox', {
            'x': f'{park_x:.4f}', 'y': f'{park_y:.4f}', 'w': f'{w:.4f}', 'h': f'{h:.4f}',
        })
        park_x += w + 10
        if park_x > canvas_w:
            park_x = 40.0
            park_y += h + 10
    if parked:
        map_el.find(f'{{{SBGN_NS}}}bbox').set('h', f'{park_y + CHEM_PRIMARY + 20:.4f}')

    by_bigg = defaultdict(list)
    for nid, n in nodes.items():
        if n.get('node_type') == 'metabolite' and n.get('bigg_id'):
            by_bigg[n['bigg_id']].append(nid)

    pending_arcs = []
    for rk, rx in reactions.items():
        rid = rx.get('bigg_id') or rk
        mid_id = _midmarker(rx, nodes)
        if mid_id is None:
            continue
        mx, my = _xy(nodes[mid_id])
        bb = _bbox(mx, my, PROCESS_W, PROCESS_H)
        sbml_rid = rx_map.get(rid) or ('R_' + rid)
        gid = _scheme_glyph_id(rid, sbml_rid)
        if gid in used_glyph_ids:
            gid = _safe_id('re_', rid)
        used_glyph_ids.add(gid)
        g = ET.SubElement(map_el, f'{{{SBGN_NS}}}glyph', {'id': gid, 'class': 'process'})
        ET.SubElement(g, f'{{{SBGN_NS}}}bbox', {
            'x': f"{bb['x']:.4f}", 'y': f"{bb['y']:.4f}", 'w': f"{bb['w']:.4f}", 'h': f"{bb['h']:.4f}",
        })
        port_in = f'{gid}.1'
        port_out = f'{gid}.2'
        ET.SubElement(g, f'{{{SBGN_NS}}}port', {'id': port_in, 'x': f"{bb['x']:.4f}", 'y': f"{my:.4f}"})
        ET.SubElement(g, f'{{{SBGN_NS}}}port', {
            'id': port_out, 'x': f"{bb['x'] + bb['w']:.4f}", 'y': f"{my:.4f}",
        })

        coeffs = {m['bigg_id']: float(m['coefficient']) for m in rx.get('metabolites') or []}
        used_nodes = set()
        for path in _reaction_paths(rx, nodes):
            end = nodes.get(path[-1])
            if not end or end.get('node_type') != 'metabolite':
                continue
            met_nid = path[-1]
            if met_nid in used_nodes:
                continue
            used_nodes.add(met_nid)
            mid = end.get('bigg_id')
            coef = coeffs.get(mid, 0.0)
            segs = list(_curve_segments(rx, path, nodes))
            if not segs:
                continue
            eg = entity_glyph.get(met_nid)
            if not eg:
                continue
            if coef < 0:
                pending_arcs.append(('consumption', eg, port_in, _reverse_curve_segments(segs)))
            else:
                pending_arcs.append(('production', port_out, eg, segs))

        attached = {nodes[n].get('bigg_id') for n in used_nodes}
        for mid, coef in coeffs.items():
            if mid in attached:
                continue
            candidates = [n for n in by_bigg.get(mid, []) if n in entity_glyph]
            if not candidates:
                continue
            met_nid = candidates[0]
            eg = entity_glyph[met_nid]
            ex, ey = _xy(nodes[met_nid])
            if coef < 0:
                pending_arcs.append(('consumption', eg, port_in, [((ex, ey), None, None, (bb['x'], my))]))
            else:
                pending_arcs.append(('production', port_out, eg, [((bb['x'] + bb['w'], my), None, None, (ex, ey))]))

    for arc_i, (cls, source, target, segs) in enumerate(pending_arcs):
        _append_sbgn_arc(map_el, f'arc{arc_i}', cls, source, target, segs)

    return root


def _append_sbgn_arc(map_el, arc_id, cls, source, target, segments):
    """Write one SBGN arc; cubic segments carry two <point> control children on next/end."""
    if not segments:
        return
    arc = ET.SubElement(map_el, f'{{{SBGN_NS}}}arc', {
        'id': arc_id, 'class': cls, 'source': source, 'target': target,
    })
    start = segments[0][0]
    ET.SubElement(arc, f'{{{SBGN_NS}}}start', {'x': f'{start[0]:.4f}', 'y': f'{start[1]:.4f}'})
    for i, (_s, c1, c2, end) in enumerate(segments):
        tag = 'end' if i == len(segments) - 1 else 'next'
        el = ET.SubElement(arc, f'{{{SBGN_NS}}}{tag}', {'x': f'{end[0]:.4f}', 'y': f'{end[1]:.4f}'})
        if c1 is not None and c2 is not None:
            ET.SubElement(el, f'{{{SBGN_NS}}}point', {'x': f'{c1[0]:.4f}', 'y': f'{c1[1]:.4f}'})
            ET.SubElement(el, f'{{{SBGN_NS}}}point', {'x': f'{c2[0]:.4f}', 'y': f'{c2[1]:.4f}'})


def write_sbgn(path, data, model, sbml_path=None):
    root = build_sbgn(data, model, sbml_path=sbml_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return path


# ------------------------------------------------------------------------------------------------ SBML Layout


def _set_bb(go, x, y, w, h, bb_id):
    bb = go.getBoundingBox()
    bb.setId(bb_id)
    bb.setX(float(x))
    bb.setY(float(y))
    bb.setWidth(float(w))
    bb.setHeight(float(h))


def _add_curve(srg, segments, reverse=False):
    """Attach Layout curve segments (CubicBezier when controls exist, else LineSegment)."""
    seq = list(reversed(segments)) if reverse else list(segments)
    for start, c1, c2, end in seq:
        if reverse:
            start, end = end, start
            if c1 is not None and c2 is not None:
                c1, c2 = c2, c1
        if c1 is not None and c2 is not None:
            cb = srg.createCubicBezier()
            cb.getStart().setX(start[0]); cb.getStart().setY(start[1])
            cb.getBasePoint1().setX(c1[0]); cb.getBasePoint1().setY(c1[1])
            cb.getBasePoint2().setX(c2[0]); cb.getBasePoint2().setY(c2[1])
            cb.getEnd().setX(end[0]); cb.getEnd().setY(end[1])
        else:
            ls = srg.createLineSegment()
            ls.getStart().setX(start[0]); ls.getStart().setY(start[1])
            ls.getEnd().setX(end[0]); ls.getEnd().setY(end[1])


def _add_basic_render(layout):
    """Attach minimal local render information when the render package is present."""
    rplugin = layout.getPlugin('render')
    if rplugin is None:
        return
    info = rplugin.createLocalRenderInformation()
    info.setId('OpenMetaMaps_render')
    info.setName('Open Meta Maps')
    info.setProgramName('Open Meta Maps')
    style = info.createStyle('speciesStyle')
    style.getGroup().setStroke('black')
    style.getGroup().setStrokeWidth(1.0)
    style.getGroup().setFill('#dcebdf')


def build_sbml_layout(model_path, data, model=None):
    """Return an SBML document: input model unchanged + Layout (+ Render) package."""
    import libsbml
    model_path = Path(model_path)
    doc = libsbml.readSBMLFromFile(str(model_path))
    if doc.getModel() is None:
        raise ValueError(f'cannot read SBML model: {model_path}')
    if model is None:
        model = load_model(model_path)

    doc.enablePackage(libsbml.LayoutExtension.getXmlnsL3V1V1(), 'layout', True)
    doc.setPackageRequired('layout', False)
    render_ok = doc.enablePackage(libsbml.RenderExtension.getXmlnsL3V1V1(), 'render', True) == libsbml.LIBSBML_OPERATION_SUCCESS
    if render_ok:
        doc.setPackageRequired('render', False)

    sbml_model = doc.getModel()
    plugin = sbml_model.getPlugin('layout')
    while plugin.getNumLayouts() > 0:
        plugin.removeLayout(0)

    body = _body(data)
    nodes = body['nodes']
    reactions = body['reactions']
    canvas = body.get('canvas') or {}
    mets, _ = _model_lookup(model)
    sp_map, rx_map, comp_map = _sbml_id_maps(doc)

    layout = plugin.createLayout()
    layout.setId('Layout_1')
    layout.setName('Open Meta Maps layout')
    layout.getDimensions().setWidth(float(canvas.get('width', 1000)))
    layout.getDimensions().setHeight(float(canvas.get('height', 1000)))

    by_comp = defaultdict(list)
    for n in nodes.values():
        if n.get('node_type') != 'metabolite':
            continue
        mid = n.get('bigg_id')
        c = (mets.get(mid) or {}).get('compartment') or comp(mid or '')
        by_comp[c].append(_xy(n))
    pad = 40.0
    for cid, pts in sorted(by_comp.items()):
        sbml_cid = comp_map.get(cid)
        if not sbml_cid:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0 = min(xs) - pad, min(ys) - pad
        w, h = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
        cg = layout.createCompartmentGlyph()
        cg.setId(_safe_id('cg_', cid))
        cg.setCompartmentId(sbml_cid)
        _set_bb(cg, x0, y0, max(w, 80), max(h, 80), _safe_id('bb_cg_', cid))

    species_glyph_for_node = {}
    glyph_index = defaultdict(int)
    for nid, n in nodes.items():
        if n.get('node_type') != 'metabolite':
            continue
        mid = n.get('bigg_id')
        sbml_sid = sp_map.get(mid)
        if not sbml_sid:
            continue
        met = mets.get(mid) or {}
        cls = _entity_class(met)
        primary = bool(n.get('node_is_primary', True))
        w, h = _glyph_size(cls, primary)
        cx, cy = _xy(n)
        idx = glyph_index[sbml_sid]
        glyph_index[sbml_sid] += 1
        sgid = _safe_id('sg_', f'{mid}_{idx}')
        sg = layout.createSpeciesGlyph()
        sg.setId(sgid)
        sg.setSpeciesId(sbml_sid)
        _set_bb(sg, cx - w / 2, cy - h / 2, w, h, _safe_id('bb_', sgid))
        species_glyph_for_node[nid] = sgid
        tg = layout.createTextGlyph()
        tg.setId(_safe_id('tg_', sgid))
        tg.setGraphicalObjectId(sgid)
        label = n.get('name') or met.get('name') or mid
        tg.setText(label)
        lx = float(n.get('label_x', cx))
        ly = float(n.get('label_y', cy - h / 2 - 8))
        _set_bb(tg, lx, ly - 6, max(len(label) * 6.0, 20), 12, _safe_id('bb_tg_', sgid))

    placed_species = {sp_map[n['bigg_id']] for n in nodes.values()
                      if n.get('node_type') == 'metabolite' and n.get('bigg_id') in sp_map}
    canvas_w = float(canvas.get('width', 1000))
    park_y = float(canvas.get('height', 1000)) + 40
    park_x = 40.0
    for mid, met in mets.items():
        sbml_sid = sp_map.get(mid)
        if not sbml_sid or sbml_sid in placed_species:
            continue
        sg = layout.createSpeciesGlyph()
        sgid = _safe_id('sg_park_', mid)
        sg.setId(sgid)
        sg.setSpeciesId(sbml_sid)
        _set_bb(sg, park_x, park_y, CHEM_SECONDARY, CHEM_SECONDARY, _safe_id('bb_', sgid))
        park_x += CHEM_SECONDARY + 10
        if park_x > canvas_w:
            park_x = 40.0
            park_y += CHEM_SECONDARY + 10

    by_bigg_nodes = defaultdict(list)
    for nid, n in nodes.items():
        if n.get('node_type') == 'metabolite' and n.get('bigg_id'):
            by_bigg_nodes[n['bigg_id']].append(nid)

    for rk, rx in reactions.items():
        rid = rx.get('bigg_id') or rk
        sbml_rid = rx_map.get(rid)
        if not sbml_rid:
            continue
        mid_id = _midmarker(rx, nodes)
        if mid_id is None:
            continue
        mx, my = _xy(nodes[mid_id])
        rg = layout.createReactionGlyph()
        rg.setId(_safe_id('rg_', rid))
        rg.setReactionId(sbml_rid)
        _set_bb(rg, mx - PROCESS_W / 2, my - PROCESS_H / 2, PROCESS_W, PROCESS_H, _safe_id('bb_rg_', rid))

        coeffs = {m['bigg_id']: float(m['coefficient']) for m in rx.get('metabolites') or []}
        used = set()
        srg_i = 0
        for path in _reaction_paths(rx, nodes):
            end = nodes.get(path[-1])
            if not end or end.get('node_type') != 'metabolite':
                continue
            met_nid = path[-1]
            if met_nid in used:
                continue
            used.add(met_nid)
            mid = end.get('bigg_id')
            sgid = species_glyph_for_node.get(met_nid)
            if not sgid:
                continue
            coef = coeffs.get(mid, 0.0)
            srg = rg.createSpeciesReferenceGlyph()
            srg.setId(_safe_id('srg_', f'{rid}_{srg_i}'))
            srg_i += 1
            srg.setSpeciesGlyphId(sgid)
            srg.setRole(libsbml.SPECIES_ROLE_SUBSTRATE if coef < 0 else libsbml.SPECIES_ROLE_PRODUCT)
            segs = list(_curve_segments(rx, path, nodes))
            _add_curve(srg, segs, reverse=(coef < 0))

        attached = {nodes[n].get('bigg_id') for n in used}
        for mid, coef in coeffs.items():
            if mid in attached:
                continue
            candidates = [n for n in by_bigg_nodes.get(mid, []) if n in species_glyph_for_node]
            if not candidates:
                continue
            met_nid = candidates[0]
            sgid = species_glyph_for_node[met_nid]
            ex, ey = _xy(nodes[met_nid])
            srg = rg.createSpeciesReferenceGlyph()
            srg.setId(_safe_id('srg_', f'{rid}_{srg_i}'))
            srg_i += 1
            srg.setSpeciesGlyphId(sgid)
            srg.setRole(libsbml.SPECIES_ROLE_SUBSTRATE if coef < 0 else libsbml.SPECIES_ROLE_PRODUCT)
            if coef < 0:
                _add_curve(srg, [((ex, ey), None, None, (mx, my))])
            else:
                _add_curve(srg, [((mx, my), None, None, (ex, ey))])

    drawn_rx = {rx_map[rx['bigg_id']] for rx in reactions.values() if rx.get('bigg_id') in rx_map}
    park_rx_x, park_rx_y = 40.0, park_y + 60
    for r in model['reactions']:
        sbml_rid = rx_map.get(r['id'])
        if not sbml_rid or sbml_rid in drawn_rx:
            continue
        rg = layout.createReactionGlyph()
        rg.setId(_safe_id('rg_park_', r['id']))
        rg.setReactionId(sbml_rid)
        _set_bb(rg, park_rx_x, park_rx_y, PROCESS_W, PROCESS_H, _safe_id('bb_rg_park_', r['id']))
        park_rx_x += PROCESS_W + 10
        if park_rx_x > canvas_w:
            park_rx_x = 40.0
            park_rx_y += PROCESS_H + 10

    if render_ok:
        _add_basic_render(layout)
    return doc


def write_sbml_layout(path, model_path, data, model=None):
    import libsbml
    doc = build_sbml_layout(model_path, data, model=model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not libsbml.writeSBMLToFile(doc, str(path)):
        raise RuntimeError(f'libsbml failed to write {path}')
    return path


# ------------------------------------------------------------------------------------------------ CellDesigner SBML


def _global_to_local(start, end, points):
    """Inverse of celldesigner.local_to_global for editPoints."""
    a = np.array(start, float)
    v = np.array(end, float) - a
    if float(np.linalg.norm(v)) < 1e-8:
        return []
    perp = np.array([-v[1], v[0]], float)
    mat = np.column_stack([v, perp])
    out = []
    for p in points:
        xy, _, _, _ = np.linalg.lstsq(mat, np.array(p, float) - a, rcond=None)
        out.append((float(xy[0]), float(xy[1])))
    return out


def _cd(tag):
    return f'{{{CD_NS}}}{tag}'


def _sbml2(tag):
    return f'{{{SBML_L2_NS}}}{tag}'


def _add_species_identity(parent, class_name, name):
    ident = ET.SubElement(parent, _cd('speciesIdentity'))
    ET.SubElement(ident, _cd('class')).text = class_name
    ET.SubElement(ident, _cd('name')).text = name


def _add_species_alias(list_sa, aid, species_id, x, y, w, h, paint='ffccff66'):
    sa = ET.SubElement(list_sa, _cd('speciesAlias'), {'id': aid, 'species': species_id})
    ET.SubElement(sa, _cd('activity')).text = 'inactive'
    ET.SubElement(sa, _cd('bounds'), {
        'x': f'{x:.2f}', 'y': f'{y:.2f}', 'w': f'{w:.2f}', 'h': f'{h:.2f}',
    })
    ET.SubElement(sa, _cd('font'), {'size': '12'})
    ET.SubElement(sa, _cd('view'), {'state': 'usual'})
    usual = ET.SubElement(sa, _cd('usualView'))
    ET.SubElement(usual, _cd('innerPosition'), {'x': '0.0', 'y': '0.0'})
    ET.SubElement(usual, _cd('boxSize'), {'width': f'{w:.2f}', 'height': f'{h:.2f}'})
    ET.SubElement(usual, _cd('singleLine'), {'width': '1.0'})
    ET.SubElement(usual, _cd('paint'), {'color': paint, 'scheme': 'Color'})
    return sa


def build_celldesigner(data, model):
    """Build CellDesigner 4.x SBML (L2V4) with speciesIdentity, aliases and editPoints."""
    ET.register_namespace('celldesigner', CD_NS)
    ET.register_namespace('', SBML_L2_NS)
    body = _body(data)
    nodes = body['nodes']
    reactions = body['reactions']
    canvas = body.get('canvas') or {}
    mets, _ = _model_lookup(model)

    root = ET.Element(_sbml2('sbml'), {'level': '2', 'version': '4'})
    model_el = ET.SubElement(root, _sbml2('model'), {
        'id': _safe_id('model_', model.get('id') or 'map'),
        'name': model.get('name') or model.get('id') or 'map',
    })

    ann = ET.SubElement(model_el, _sbml2('annotation'))
    ext = ET.SubElement(ann, _cd('extension'))
    ET.SubElement(ext, _cd('modelVersion')).text = '4.0'
    ET.SubElement(ext, _cd('modelDisplay'), {
        'sizeX': str(int(float(canvas.get('width', 1000)))),
        'sizeY': str(int(float(canvas.get('height', 1000)))),
    })

    comps = sorted({m.get('compartment') or 'c' for m in model['metabolites']})
    loc = ET.SubElement(model_el, _sbml2('listOfCompartments'))
    list_ca = ET.SubElement(ext, _cd('listOfCompartmentAliases'))
    for i, cid in enumerate(comps):
        ET.SubElement(loc, _sbml2('compartment'), {'id': cid, 'name': cid, 'size': '1'})
        pts = [_xy(n) for n in nodes.values()
               if n.get('node_type') == 'metabolite'
               and (mets.get(n.get('bigg_id')) or {}).get('compartment', comp(n.get('bigg_id') or '')) == cid]
        if not pts:
            pts = [(0.0, 0.0), (100.0, 100.0)]
        pad = 40.0
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0 = min(xs) - pad, min(ys) - pad
        w, h = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
        ca = ET.SubElement(list_ca, _cd('compartmentAlias'), {'id': f'ca{i}', 'compartment': cid})
        ET.SubElement(ca, _cd('bounds'), {
            'x': f'{x0:.2f}', 'y': f'{y0:.2f}', 'w': f'{max(w, 80):.2f}', 'h': f'{max(h, 80):.2f}',
        })

    ET.SubElement(ext, _cd('listOfComplexSpeciesAliases'))
    list_sa = ET.SubElement(ext, _cd('listOfSpeciesAliases'))
    ET.SubElement(ext, _cd('listOfGroups'))
    ET.SubElement(ext, _cd('listOfProteins'))
    ET.SubElement(ext, _cd('listOfGenes'))
    ET.SubElement(ext, _cd('listOfRNAs'))
    ET.SubElement(ext, _cd('listOfAntisenseRNAs'))
    ET.SubElement(ext, _cd('listOfLayers'))
    ET.SubElement(ext, _cd('listOfBlockDiagrams'))

    los = ET.SubElement(model_el, _sbml2('listOfSpecies'))
    species_el = {}
    for mid, met in mets.items():
        sid = _safe_id('s_', mid)
        species_el[mid] = sid
        sp = ET.SubElement(los, _sbml2('species'), {
            'id': sid,
            'name': met.get('name') or mid,
            'compartment': met.get('compartment') or comp(mid) or comps[0],
            'initialAmount': '0',
        })
        s_ann = ET.SubElement(sp, _sbml2('annotation'))
        s_ext = ET.SubElement(s_ann, _cd('extension'))
        ET.SubElement(s_ext, _cd('positionToCompartment')).text = 'inside'
        _add_species_identity(s_ext, _cd_species_class(met), met.get('name') or mid)

    alias_for_node = {}
    alias_by_mid = defaultdict(list)
    alias_xy = {}
    alias_i = 0
    for nid, n in nodes.items():
        if n.get('node_type') != 'metabolite':
            continue
        mid = n.get('bigg_id')
        if mid not in species_el:
            continue
        met = mets[mid]
        cls = _entity_class(met)
        primary = bool(n.get('node_is_primary', True))
        w, h = _glyph_size(cls, primary)
        cx, cy = _xy(n)
        aid = f'a{alias_i}'
        alias_i += 1
        alias_for_node[nid] = aid
        alias_by_mid[mid].append(aid)
        alias_xy[aid] = (cx, cy)
        _add_species_alias(
            list_sa, aid, species_el[mid], cx - w / 2, cy - h / 2, w, h,
            paint='ffccff66' if primary else 'ffffffff',
        )

    park_x, park_y = 40.0, float(canvas.get('height', 1000)) + 40
    for mid in mets:
        if alias_by_mid.get(mid):
            continue
        aid = f'a{alias_i}'
        alias_i += 1
        alias_by_mid[mid].append(aid)
        alias_xy[aid] = (park_x + CHEM_SECONDARY / 2, park_y + CHEM_SECONDARY / 2)
        _add_species_alias(list_sa, aid, species_el[mid], park_x, park_y, CHEM_SECONDARY, CHEM_SECONDARY)
        park_x += CHEM_SECONDARY + 10

    lor = ET.SubElement(model_el, _sbml2('listOfReactions'))
    drawn = {rx.get('bigg_id'): rx for rk, rx in reactions.items()}

    for r in model['reactions']:
        rid = r['id']
        rx_el = ET.SubElement(lor, _sbml2('reaction'), {
            'id': _safe_id('r_', rid),
            'name': r.get('name') or rid,
            'reversible': 'true' if r['lower_bound'] < 0 < r['upper_bound'] else 'false',
        })
        reactants = [(mid, -c) for mid, c in r['metabolites'].items() if c < 0]
        products = [(mid, c) for mid, c in r['metabolites'].items() if c > 0]

        escher = drawn.get(rid)
        node_aliases = {}
        mid_xy = None
        if escher:
            mid_nid = _midmarker(escher, nodes)
            if mid_nid:
                mid_xy = _xy(nodes[mid_nid])
            for path in _reaction_paths(escher, nodes):
                end = nodes.get(path[-1])
                if end and end.get('node_type') == 'metabolite' and path[-1] in alias_for_node:
                    node_aliases[end['bigg_id']] = alias_for_node[path[-1]]

        def pick_alias(mid):
            if mid in node_aliases:
                return node_aliases[mid]
            return (alias_by_mid.get(mid) or [None])[0]

        # Empty side (exchange / sink / demand): DEGRADED species + alias as base participant.
        degraded_sid = degraded_aid = None
        if not reactants or not products:
            degraded_sid = _safe_id('s_degraded_', rid)
            degraded_aid = f'a{alias_i}'
            alias_i += 1
            deg_x = (mid_xy[0] if mid_xy else park_x) + 30
            deg_y = (mid_xy[1] if mid_xy else park_y)
            alias_xy[degraded_aid] = (deg_x, deg_y)
            sp = ET.SubElement(los, _sbml2('species'), {
                'id': degraded_sid,
                'name': f'degraded_{rid}',
                'compartment': comps[0],
                'initialAmount': '0',
            })
            s_ann = ET.SubElement(sp, _sbml2('annotation'))
            s_ext = ET.SubElement(s_ann, _cd('extension'))
            ET.SubElement(s_ext, _cd('positionToCompartment')).text = 'inside'
            _add_species_identity(s_ext, 'DEGRADED', f'degraded_{rid}')
            _add_species_alias(list_sa, degraded_aid, degraded_sid, deg_x - 10, deg_y - 10, 20, 20, paint='ffcccccc')

        base_r = pick_alias(reactants[0][0]) if reactants else degraded_aid
        base_p = pick_alias(products[0][0]) if products else degraded_aid
        base_r_species = species_el[reactants[0][0]] if reactants else degraded_sid
        base_p_species = species_el[products[0][0]] if products else degraded_sid

        r_ann = ET.SubElement(rx_el, _sbml2('annotation'))
        r_ext = ET.SubElement(r_ann, _cd('extension'))
        ET.SubElement(r_ext, _cd('reactionType')).text = 'STATE_TRANSITION'
        brs = ET.SubElement(r_ext, _cd('baseReactants'))
        bps = ET.SubElement(r_ext, _cd('baseProducts'))
        if base_r and base_r_species:
            ET.SubElement(brs, _cd('baseReactant'), {'species': base_r_species, 'alias': base_r})
        if base_p and base_p_species:
            ET.SubElement(bps, _cd('baseProduct'), {'species': base_p_species, 'alias': base_p})

        if len(reactants) > 1:
            lrl = ET.SubElement(r_ext, _cd('listOfReactantLinks'))
            for mid, _sto in reactants[1:]:
                aid = pick_alias(mid)
                if not aid:
                    continue
                link = ET.SubElement(lrl, _cd('reactantLink'), {
                    'reactant': species_el[mid], 'alias': aid, 'targetLineIndex': '-1,0',
                })
                ET.SubElement(link, _cd('line'), {'width': '1.0', 'color': 'ff000000', 'type': 'Straight'})
        if len(products) > 1:
            lpl = ET.SubElement(r_ext, _cd('listOfProductLinks'))
            for mid, _sto in products[1:]:
                aid = pick_alias(mid)
                if not aid:
                    continue
                link = ET.SubElement(lpl, _cd('productLink'), {
                    'product': species_el[mid], 'alias': aid, 'targetLineIndex': '-1,1',
                })
                ET.SubElement(link, _cd('line'), {'width': '1.0', 'color': 'ff000000', 'type': 'Straight'})

        edit_text = ''
        rect_index = '0'
        if base_r and base_p and base_r in alias_xy and base_p in alias_xy:
            start = alias_xy[base_r]
            end = alias_xy[base_p]
            interiors = [mid_xy] if mid_xy is not None else []
            locals_ = _global_to_local(start, end, interiors)
            if locals_:
                edit_text = ' '.join(f'{x:.8f},{y:.8f}' for x, y in locals_)
        scheme = ET.SubElement(r_ext, _cd('connectScheme'), {
            'connectPolicy': 'direct',
            'rectangleIndex': rect_index,
        })
        lod = ET.SubElement(scheme, _cd('listOfLineDirection'))
        ET.SubElement(lod, _cd('lineDirection'), {'index': '0', 'value': 'unknown'})
        if edit_text:
            ET.SubElement(lod, _cd('lineDirection'), {'index': '1', 'value': 'unknown'})
            ET.SubElement(r_ext, _cd('editPoints')).text = edit_text
        ET.SubElement(r_ext, _cd('line'), {'width': '1.0', 'color': 'ff000000'})

        # Omit empty listOfReactants / listOfProducts (libSBML 21103).
        if reactants:
            lorct = ET.SubElement(rx_el, _sbml2('listOfReactants'))
            for mid, sto in reactants:
                sr = ET.SubElement(lorct, _sbml2('speciesReference'), {
                    'species': species_el[mid],
                    'stoichiometry': str(sto),
                })
                aid = pick_alias(mid)
                if aid:
                    s_ann = ET.SubElement(sr, _sbml2('annotation'))
                    s_ext = ET.SubElement(s_ann, _cd('extension'))
                    ET.SubElement(s_ext, _cd('alias')).text = aid
        if products:
            lop = ET.SubElement(rx_el, _sbml2('listOfProducts'))
            for mid, sto in products:
                sr = ET.SubElement(lop, _sbml2('speciesReference'), {
                    'species': species_el[mid],
                    'stoichiometry': str(sto),
                })
                aid = pick_alias(mid)
                if aid:
                    s_ann = ET.SubElement(sr, _sbml2('annotation'))
                    s_ext = ET.SubElement(s_ann, _cd('extension'))
                    ET.SubElement(s_ext, _cd('alias')).text = aid

    return root


def write_celldesigner(path, data, model):
    root = build_celldesigner(data, model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return path


# ------------------------------------------------------------------------------------------------ Orchestration


def write_all_exports(folder, model_path=None, map_path=None, model=None, stem=None):
    """Write SBGN-ML, SBML Layout and CellDesigner SBML next to the Escher result JSON.

    Returns a list of written filenames. Missing map JSON yields an empty list.
    """
    import output_names
    folder = Path(folder)
    stem = stem or output_names.stem_from_folder(folder)
    map_path = Path(map_path) if map_path else output_names.map_json(folder, stem)
    if not map_path.is_file():
        return []
    data = read(map_path)
    if model is None:
        if model_path is None:
            upload = next(iter(sorted((folder / 'upload').glob('model.*'))), None)
            if upload is None:
                raise FileNotFoundError(f'no model for export in {folder}')
            model_path = upload
        model = load_model(model_path)
    model_path = Path(model_path) if model_path else None

    sbgn_path = output_names.map_export(folder, 'sbgn', stem)
    sbml_out = output_names.map_export(folder, 'sbml', stem)
    cd_path = output_names.map_export(folder, 'celldesigner.xml', stem)
    sbml_path, temporary = _ensure_sbml_path(model_path, model, folder=folder)
    try:
        write_sbgn(sbgn_path, data, model, sbml_path=sbml_path)
        write_sbml_layout(sbml_out, sbml_path, data, model=model)
        write_celldesigner(cd_path, data, model)
    finally:
        if temporary:
            Path(sbml_path).unlink(missing_ok=True)
    return [sbgn_path.name, sbml_out.name, cd_path.name]
