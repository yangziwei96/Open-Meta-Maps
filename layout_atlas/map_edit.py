"""Apply interactive map edits to a finished tool job (positions only).

Keeps the first original once as map.original.json (intermediate), writes the
edited OpenMetaMaps_<stem>_result.json, regenerates the matching result SVG with
line_render, refreshes layout_quality in report.json, and rewrites the
SBGN-ML / SBML Layout / CellDesigner result files.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import jsonschema
import output_names

# Patch body accepted by POST /api/jobs/<id>/map — only geometry fields.
EDIT_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'nodes': {
            'type': 'object',
            'additionalProperties': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'x': {'type': 'number'},
                    'y': {'type': 'number'},
                    'label_x': {'type': 'number'},
                    'label_y': {'type': 'number'},
                },
                'minProperties': 1,
            },
        },
        'reactions': {
            'type': 'object',
            'additionalProperties': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'label_x': {'type': 'number'},
                    'label_y': {'type': 'number'},
                    'segments': {
                        'type': 'object',
                        'additionalProperties': {
                            'type': 'object',
                            'additionalProperties': False,
                            'properties': {
                                'b1': {
                                    'anyOf': [
                                        {'type': 'null'},
                                        {
                                            'type': 'object',
                                            'additionalProperties': False,
                                            'required': ['x', 'y'],
                                            'properties': {
                                                'x': {'type': 'number'},
                                                'y': {'type': 'number'},
                                            },
                                        },
                                    ],
                                },
                                'b2': {
                                    'anyOf': [
                                        {'type': 'null'},
                                        {
                                            'type': 'object',
                                            'additionalProperties': False,
                                            'required': ['x', 'y'],
                                            'properties': {
                                                'x': {'type': 'number'},
                                                'y': {'type': 'number'},
                                            },
                                        },
                                    ],
                                },
                            },
                            'minProperties': 1,
                        },
                    },
                },
                'minProperties': 1,
            },
        },
    },
    'minProperties': 1,
}


def translate_node(n, dx, dy, digits=2):
    """Move a glyph by ``(dx, dy)`` and keep ``label_x``/``label_y`` attached.

    Stages that relocate metabolites or midmarkers must use this (or
    ``set_node_xy``) so text never drifts from its node.  Midmarkers without a
    label are still translated; reaction labels are updated separately via
    ``translate_reaction_label``.
    """
    dx, dy = float(dx), float(dy)
    n['x'] = round(float(n['x']) + dx, digits)
    n['y'] = round(float(n['y']) + dy, digits)
    if 'label_x' in n:
        n['label_x'] = round(float(n['label_x']) + dx, digits)
        n['label_y'] = round(float(n['label_y']) + dy, digits)
    return dx, dy


def set_node_xy(n, x, y, digits=2):
    """Set absolute glyph coordinates while preserving the label offset."""
    return translate_node(n, float(x) - float(n['x']), float(y) - float(n['y']), digits=digits)


def translate_reaction_label(rx, dx, dy, digits=2):
    """Translate a reaction's label with the same delta as its midmarker."""
    if 'label_x' not in rx:
        return
    rx['label_x'] = round(float(rx['label_x']) + float(dx), digits)
    rx['label_y'] = round(float(rx['label_y']) + float(dy), digits)


def validate_edit(edit):
    """Raise jsonschema.ValidationError when the edit patch is not valid."""
    if not isinstance(edit, dict):
        raise jsonschema.ValidationError('edit must be an object')
    jsonschema.validate(edit, EDIT_SCHEMA)


def apply_edit(data, edit):
    """Return a deep-copied Escher map with geometry from `edit` applied."""
    validate_edit(edit)
    out = copy.deepcopy(data)
    body = out[1]
    nodes, reactions = body['nodes'], body['reactions']
    for nid, patch in (edit.get('nodes') or {}).items():
        if nid not in nodes:
            raise ValueError('unknown node id: %s' % nid)
        node = nodes[nid]
        for key in ('x', 'y', 'label_x', 'label_y'):
            if key in patch:
                if key.startswith('label_') and node.get('node_type') != 'metabolite' and key not in node:
                    # mid/multimarkers have no metabolite labels; ignore silently only if absent
                    if key not in node:
                        raise ValueError('node %s has no %s' % (nid, key))
                node[key] = float(patch[key])
    for rk, patch in (edit.get('reactions') or {}).items():
        if rk not in reactions:
            raise ValueError('unknown reaction id: %s' % rk)
        rx = reactions[rk]
        for key in ('label_x', 'label_y'):
            if key in patch:
                rx[key] = float(patch[key])
        for sid, seg_patch in (patch.get('segments') or {}).items():
            if sid not in rx.get('segments', {}):
                raise ValueError('unknown segment %s on reaction %s' % (sid, rk))
            seg = rx['segments'][sid]
            for bk in ('b1', 'b2'):
                if bk not in seg_patch:
                    continue
                val = seg_patch[bk]
                if val is None:
                    seg[bk] = None
                else:
                    seg[bk] = {'x': float(val['x']), 'y': float(val['y'])}
    return out


def _prov_kinds(folder: Path):
    """Reaction id → provenance kind for SVG colouring (legend lives in the tool UI)."""
    kinds = {}
    prov_path = folder / 'reaction_provenance.json'
    if prov_path.is_file():
        rows = json.loads(prov_path.read_text())
        kinds = {r['reaction']: r['provenance'] for r in rows if 'reaction' in r and 'provenance' in r}
    return kinds


def _refresh_quality(folder: Path, data):
    """Recompute layout_quality.final_geometry and write it into report.json / job summary."""
    import layout_quality
    import orientation as orientation_module

    structures = (data[0].get('metadata') or {}).get('expert_structures')
    skip = (data[0].get('metadata') or {}).get('preserved_edges')
    summary, _issues = layout_quality.evaluate(data, skip_edges=skip, structures=structures)
    report_path = folder / 'report.json'
    layout_quality_report = None
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        lq = report.setdefault('layout_quality', {})
        lq['final_geometry'] = summary
        # orientation summary uses the map; keep mode if present
        mode = lq.get('orientation_mode')
        if mode and mode != 'off':
            kinds = _prov_kinds(folder)
            try:
                lq['orientation'] = orientation_module.summary(data, kinds)
            except Exception:
                pass
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        layout_quality_report = report.get('layout_quality')
    job_path = folder / 'job.json'
    if job_path.is_file() and layout_quality_report is not None:
        state = json.loads(job_path.read_text())
        if isinstance(state.get('summary'), dict):
            state['summary']['layout_quality'] = layout_quality_report
            tmp = folder / 'job.tmp'
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            os.replace(tmp, job_path)
    return summary


def _update_file_bytes(folder: Path, name: str):
    job_path = folder / 'job.json'
    if not job_path.is_file():
        return
    state = json.loads(job_path.read_text())
    path = folder / name
    if not path.is_file():
        return
    size = path.stat().st_size
    for f in state.get('files') or []:
        if f.get('name') == name:
            f['bytes'] = size
    tmp = folder / 'job.tmp'
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    os.replace(tmp, job_path)


def _resolve_map_filename(folder: Path, filename=None) -> str:
    """Return the result Escher JSON basename for this job folder."""
    folder = Path(folder)
    if filename:
        name = Path(filename).name
        if not output_names.is_result_map_json(name):
            raise ValueError('only OpenMetaMaps_<name>_result.json can be edited')
        if not (folder / name).is_file():
            raise FileNotFoundError('%s missing' % name)
        return name
    path = output_names.find_map_json(folder)
    if path is None:
        raise FileNotFoundError('result map JSON missing')
    return path.name


def _result_export_names(folder: Path):
    return [
        output_names.map_export(folder, 'sbgn').name,
        output_names.map_export(folder, 'sbml').name,
        output_names.map_export(folder, 'celldesigner.xml').name,
    ]


def _refresh_exports(folder: Path):
    """Rewrite SBGN-ML / SBML Layout / CellDesigner beside the current result map."""
    upload = next(iter(sorted((folder / 'upload').glob('model.*'))), None)
    if upload is None:
        return
    import map_exports
    map_exports.write_all_exports(folder, model_path=upload)


def save_map_edit(folder: Path, edit, *, filename=None):
    """Validate and persist an edit into the job folder.

    Returns dict(ok=True, layout_quality=..., original_kept=bool).
    Raises ValueError / jsonschema.ValidationError on bad input.
    """
    folder = Path(folder)
    map_name = _resolve_map_filename(folder, filename)
    map_path = folder / map_name
    svg_path = output_names.map_svg(folder)
    data = json.loads(map_path.read_text())
    if not (isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict)):
        raise ValueError('result map JSON is not a valid Escher map array')
    edited = apply_edit(data, edit)
    original = folder / 'map.original.json'
    kept = original.is_file()
    if not kept:
        # first edit: freeze the pre-edit map once
        original.write_text(json.dumps(data, ensure_ascii=False))
        kept = True
        first_backup = True
    else:
        first_backup = False
    tmp = folder / (map_name + '.tmp')
    tmp.write_text(json.dumps(edited, ensure_ascii=False))
    os.replace(tmp, map_path)
    kinds = _prov_kinds(folder)
    import line_render
    line_render.render_file(map_path, svg_path, kinds)
    quality = _refresh_quality(folder, edited)
    _refresh_exports(folder)
    for name in (map_name, svg_path.name, 'report.json', *_result_export_names(folder)):
        _update_file_bytes(folder, name)
    return dict(ok=True, layout_quality=quality, original_kept=True, first_backup=first_backup)


def reset_map(folder: Path, *, filename=None):
    """Restore the result map JSON from map.original.json and regenerate SVG + quality.

    Raises FileNotFoundError when no original backup exists.
    """
    folder = Path(folder)
    map_name = _resolve_map_filename(folder, filename)
    map_path = folder / map_name
    svg_path = output_names.map_svg(folder)
    original = folder / 'map.original.json'
    if not original.is_file():
        raise FileNotFoundError('map.original.json missing')
    data = json.loads(original.read_text())
    if not (isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict)):
        raise ValueError('map.original.json is not a valid Escher map array')
    tmp = folder / (map_name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, map_path)
    kinds = _prov_kinds(folder)
    import line_render
    line_render.render_file(map_path, svg_path, kinds)
    quality = _refresh_quality(folder, data)
    _refresh_exports(folder)
    for name in (map_name, svg_path.name, 'report.json', *_result_export_names(folder)):
        _update_file_bytes(folder, name)
    return dict(ok=True, layout_quality=quality, original_kept=True)
