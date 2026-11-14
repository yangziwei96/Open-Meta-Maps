"""Unit tests for OpenMetaMaps_<stem>_result.<ext> naming."""
from __future__ import annotations

from pathlib import Path

import output_names


def test_sanitize_stem_strips_extension_and_unsafe_chars():
    assert output_names.sanitize_stem('e_coli_core.xml') == 'e_coli_core'
    assert output_names.sanitize_stem('iIT341.SBML') == 'iIT341'
    assert output_names.sanitize_stem('my model (v2).json') == 'my_model_v2'
    assert output_names.sanitize_stem('...bad...') == 'bad'
    assert output_names.sanitize_stem('') == 'model'


def test_result_filename_patterns():
    assert output_names.result_filename('e_coli_core', 'json') == 'OpenMetaMaps_e_coli_core_result.json'
    assert output_names.result_filename('iIT341', 'sbgn') == 'OpenMetaMaps_iIT341_result.sbgn'
    assert output_names.result_filename('iIT341', 'celldesigner.xml') == (
        'OpenMetaMaps_iIT341_result.celldesigner.xml'
    )


def test_stem_from_folder_uses_meta(tmp_path: Path):
    output_names.write_meta(tmp_path, 'e_coli_core.xml', size=12)
    assert output_names.stem_from_folder(tmp_path) == 'e_coli_core'
    assert output_names.map_json(tmp_path).name == 'OpenMetaMaps_e_coli_core_result.json'


def test_working_dirs_keep_map_json(tmp_path: Path):
    full = tmp_path / 'full'
    full.mkdir()
    assert output_names.map_json(full) == full / 'map.json'
    view = tmp_path / 'views' / 'view-3'
    view.mkdir(parents=True)
    assert output_names.map_svg(view) == view / 'map.svg'


def test_is_result_map_json():
    assert output_names.is_result_map_json('OpenMetaMaps_e_coli_core_result.json')
    assert not output_names.is_result_map_json('map.json')
    assert not output_names.is_result_map_json('OpenMetaMaps_iIT341_result.sbgn')
    assert not output_names.is_result_map_json('report.json')
