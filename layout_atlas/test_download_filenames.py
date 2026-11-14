"""UI / server download filenames use OpenMetaMaps_<stem>_result.<ext>."""
from __future__ import annotations

from pathlib import Path

import output_names
import tool_server

TOOL_HTML = Path(__file__).resolve().parent / 'results' / 'tool' / 'index.html'


def test_tool_page_has_result_download_helpers():
    html = TOOL_HTML.read_text()
    assert 'function resultFile(ext)' in html
    assert 'function resultBase()' in html
    assert 'download="\'+esc(x[0])+\'"' in html
    assert "resultFile('json')" in html
    assert "resultFile('sbgn')" in html
    assert "resultFile('celldesigner.xml')" in html
    assert 'saveBlob(svgName' in html
    # no hard-coded legacy download basenames in the download row
    assert "['map.svg','Download SVG']" not in html
    assert "['map.json','Download Escher JSON']" not in html
    assert 'OpenMetaMaps_model_result' not in html


def test_content_disposition_uses_result_filename():
    """Server download headers use the on-disk result basename."""
    stem = 'e_coli_core'
    name = output_names.result_filename(stem, 'json')
    assert tool_server.SAFE_FILE.match(name)
    assert name == 'OpenMetaMaps_e_coli_core_result.json'
    # Same construction as Handler.do_GET for ?download=1
    header = 'attachment; filename="%s"' % name
    assert header == 'attachment; filename="OpenMetaMaps_e_coli_core_result.json"'


def test_dlmain_resolves_names_from_job_summary():
    """Mirror the tool page resultFile() resolution from job.summary."""
    stem = 'iIT341'
    job = {
        'summary': {
            'result_stem': stem,
            'result_basename': output_names.result_basename(stem),
        },
        'files': [
            {'name': output_names.result_filename(stem, ext)}
            for ext in ('json', 'svg', 'sbgn', 'sbml', 'celldesigner.xml')
        ],
    }
    base = job['summary']['result_basename']
    assert base == 'OpenMetaMaps_iIT341_result'
    for ext in ('json', 'svg', 'sbgn', 'sbml', 'celldesigner.xml'):
        name = f'{base}.{ext}'
        assert any(f['name'] == name for f in job['files'])
