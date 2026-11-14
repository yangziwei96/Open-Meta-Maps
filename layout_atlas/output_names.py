"""Final map download filenames: OpenMetaMaps_<stem>_result.<ext>.

The stem is the uploaded (or CLI) model file name without extension, sanitised
to letters, digits, '-', '_' and '.'. Intermediate and debug artefacts
(report.json, skeleton.*, map.original.json, views/view-*/map.json, …) keep
their own names.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

RESULT_PREFIX = 'OpenMetaMaps_'
RESULT_SUFFIX = '_result'
# Compound extensions first so stem stripping is unambiguous.
RESULT_EXTS = (
    'celldesigner.xml',
    'json',
    'sbgn',
    'sbml',
    'svg',
    'png',
)
_SAFE = re.compile(r'[^A-Za-z0-9._-]+')
_RESULT_JSON = re.compile(r'^OpenMetaMaps_[A-Za-z0-9._-]+_result\.json$')
# Working map files inside scratch folders (not final downloads).
_VIEW_SCRATCH = re.compile(r'^view-\d+$')


def sanitize_stem(filename_or_stem: str) -> str:
    """Return a safe model stem from a file name or stem."""
    raw = Path(filename_or_stem or '').name or 'model'
    lower = raw.lower()
    for suf in ('.celldesigner.xml', '.sbml', '.sbgn', '.xml', '.json', '.svg', '.png'):
        if lower.endswith(suf):
            raw = raw[: len(raw) - len(suf)]
            break
    else:
        raw = Path(raw).stem
    cleaned = _SAFE.sub('_', raw).strip('._')
    return cleaned or 'model'


def result_basename(stem: str) -> str:
    return f'{RESULT_PREFIX}{sanitize_stem(stem)}{RESULT_SUFFIX}'


def result_filename(stem: str, ext: str) -> str:
    """Build OpenMetaMaps_<stem>_result.<ext> (ext may be compound, e.g. celldesigner.xml)."""
    ext = (ext or '').lstrip('.')
    if ext not in RESULT_EXTS:
        raise ValueError(f'unsupported result extension: {ext!r}')
    return f'{result_basename(stem)}.{ext}'


def stem_from_folder(folder) -> str:
    """Resolve the result stem from meta.json original_name."""
    folder = Path(folder)
    meta = folder / 'meta.json'
    if meta.is_file():
        try:
            name = json.loads(meta.read_text()).get('original_name')
            if name:
                return sanitize_stem(name)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return 'model'


def is_result_map_json(name: str) -> bool:
    return bool(_RESULT_JSON.match(Path(name).name))


def is_working_map_dir(folder) -> bool:
    """True for intermediate dirs that still use map.json / map.svg."""
    folder = Path(folder)
    if folder.name == 'full':
        return True
    return folder.parent.name == 'views' and bool(_VIEW_SCRATCH.match(folder.name))


def map_json(folder, stem=None) -> Path:
    """Path of the final Escher JSON in a job folder (or map.json in a working dir)."""
    folder = Path(folder)
    if is_working_map_dir(folder):
        return folder / 'map.json'
    return folder / result_filename(stem or stem_from_folder(folder), 'json')


def map_svg(folder, stem=None) -> Path:
    folder = Path(folder)
    if is_working_map_dir(folder):
        return folder / 'map.svg'
    return folder / result_filename(stem or stem_from_folder(folder), 'svg')


def map_export(folder, ext: str, stem=None) -> Path:
    """Final export path for sbgn / sbml / celldesigner.xml / png."""
    folder = Path(folder)
    return folder / result_filename(stem or stem_from_folder(folder), ext)


def find_map_json(folder) -> Path | None:
    """Locate the final Escher JSON in a finished job folder."""
    folder = Path(folder)
    path = map_json(folder)
    return path if path.is_file() else None


def clear_stale_results(folder) -> None:
    """Remove previous result downloads and map.original.json from a job folder.

    Deletes every OpenMetaMaps_*_result.* file in the folder root, plus any
    map.json / map.svg / map.sbgn / map.sbml / map.celldesigner.xml / map.png
    and map.original.json sitting beside them, so the next compose or tool run
    starts from an empty download set.
    """
    folder = Path(folder)
    for path in folder.glob(f'{RESULT_PREFIX}*{RESULT_SUFFIX}.*'):
        if path.is_file():
            path.unlink(missing_ok=True)
    for name in (
        'map.json', 'map.svg', 'map.sbgn', 'map.sbml', 'map.celldesigner.xml', 'map.png',
        'map.original.json',
    ):
        (folder / name).unlink(missing_ok=True)


def write_meta(folder, original_name: str, size: int | None = None) -> Path:
    """Persist the original model file name used to build result stems."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(
        original_name=Path(original_name).name,
        bytes=int(size if size is not None else 0),
        created=time.time(),
    )
    path = folder / 'meta.json'
    path.write_text(json.dumps(payload))
    return path
