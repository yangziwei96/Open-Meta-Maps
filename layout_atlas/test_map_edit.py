"""Unit tests for interactive map-edit save (python3 test_map_edit.py)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import jsonschema

import map_edit
import output_names
import line_render


def _toy_map():
    nodes = {
        '1': {'node_type': 'midmarker', 'x': 100.0, 'y': 100.0},
        '2': {'node_type': 'metabolite', 'x': 200.0, 'y': 100.0, 'bigg_id': 'A_c', 'name': 'A',
              'node_is_primary': True, 'label_x': 210.0, 'label_y': 90.0},
        '3': {'node_type': 'metabolite', 'x': 100.0, 'y': 200.0, 'bigg_id': 'B_c', 'name': 'B',
              'node_is_primary': True, 'label_x': 110.0, 'label_y': 210.0},
    }
    reactions = {
        '10': {
            'bigg_id': 'R1', 'name': 'R1', 'reversibility': False,
            'label_x': 120.0, 'label_y': 80.0, 'gene_reaction_rule': '', 'genes': [],
            'metabolites': [{'bigg_id': 'A_c', 'coefficient': -1}, {'bigg_id': 'B_c', 'coefficient': 1}],
            'segments': {
                '0': {'from_node_id': '1', 'to_node_id': '2',
                      'b1': {'x': 130.0, 'y': 100.0}, 'b2': {'x': 170.0, 'y': 100.0}},
                '1': {'from_node_id': '1', 'to_node_id': '3', 'b1': None, 'b2': None},
            },
        }
    }
    return [
        {'schema': 'https://escher.github.io/escher/jsonschema/1-0-0#',
         'homepage': 'https://escher.github.io', 'map_id': 'toy', 'map_name': 'toy',
         'map_description': 'unit test', 'metadata': {}},
        {'nodes': nodes, 'reactions': reactions, 'text_labels': {},
         'canvas': {'x': 0, 'y': 0, 'width': 400, 'height': 400}},
    ]


class TestMapEditValidate(unittest.TestCase):
    def test_rejects_non_object(self):
        with self.assertRaises(jsonschema.ValidationError):
            map_edit.validate_edit([])

    def test_rejects_unknown_field(self):
        with self.assertRaises(jsonschema.ValidationError):
            map_edit.validate_edit({'nodes': {'1': {'x': 1}}, 'extra': 1})

    def test_rejects_bad_point(self):
        with self.assertRaises(jsonschema.ValidationError):
            map_edit.validate_edit({'nodes': {'1': {'x': 'nope'}}})

    def test_accepts_minimal_node_move(self):
        map_edit.validate_edit({'nodes': {'1': {'x': 1.5, 'y': 2.5}}})


class TestMapEditApply(unittest.TestCase):
    def test_roundtrip_positions(self):
        data = _toy_map()
        edit = {
            'nodes': {
                '2': {'x': 250.0, 'y': 110.0, 'label_x': 260.0, 'label_y': 100.0},
                '1': {'x': 105.0, 'y': 105.0},
            },
            'reactions': {
                '10': {
                    'label_x': 130.0, 'label_y': 70.0,
                    'segments': {
                        '0': {'b1': {'x': 140.0, 'y': 105.0}, 'b2': {'x': 200.0, 'y': 110.0}},
                    },
                }
            },
        }
        out = map_edit.apply_edit(data, edit)
        self.assertEqual(out[1]['nodes']['2']['x'], 250.0)
        self.assertEqual(out[1]['nodes']['2']['label_y'], 100.0)
        self.assertEqual(out[1]['nodes']['1']['y'], 105.0)
        self.assertEqual(out[1]['reactions']['10']['label_x'], 130.0)
        self.assertEqual(out[1]['reactions']['10']['segments']['0']['b1']['x'], 140.0)
        # original untouched
        self.assertEqual(data[1]['nodes']['2']['x'], 200.0)

    def test_unknown_node_rejected(self):
        with self.assertRaises(ValueError):
            map_edit.apply_edit(_toy_map(), {'nodes': {'999': {'x': 1, 'y': 2}}})


class TestMapEditSave(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='omm_map_edit_'))
        self.map = _toy_map()
        self.stem = 'toy'
        self.map_name = output_names.result_filename(self.stem, 'json')
        self.svg_name = output_names.result_filename(self.stem, 'svg')
        output_names.write_meta(self.tmp, 'toy.xml', size=10)
        (self.tmp / self.map_name).write_text(json.dumps(self.map))
        (self.tmp / 'job.json').write_text(json.dumps({
            'id': 'j_testdummy1', 'status': 'done',
            'summary': {'layout_quality': {}, 'result_stem': self.stem,
                        'result_basename': output_names.result_basename(self.stem)},
            'files': [
                {'name': self.map_name, 'bytes': 10, 'kind': 'json'},
                {'name': self.svg_name, 'bytes': 10, 'kind': 'svg'},
                {'name': 'report.json', 'bytes': 10, 'kind': 'json'},
            ],
        }))
        (self.tmp / 'report.json').write_text(json.dumps({
            'layout_quality': {'orientation_mode': 'off', 'final_geometry': {}},
        }))
        line_render.render_file(self.tmp / self.map_name, self.tmp / self.svg_name)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_keeps_original_once_and_updates_files(self):
        edit = {'nodes': {'2': {'x': 260.0, 'y': 120.0}}}
        r1 = map_edit.save_map_edit(self.tmp, edit)
        self.assertTrue(r1['ok'])
        self.assertTrue(r1['first_backup'])
        self.assertTrue((self.tmp / 'map.original.json').is_file())
        original = json.loads((self.tmp / 'map.original.json').read_text())
        self.assertEqual(original[1]['nodes']['2']['x'], 200.0)
        saved = json.loads((self.tmp / self.map_name).read_text())
        self.assertEqual(saved[1]['nodes']['2']['x'], 260.0)
        svg = (self.tmp / self.svg_name).read_text()
        self.assertIn('data-n="2"', svg)
        self.assertIn('cx="260.0"', svg)
        # second save must not overwrite the original
        r2 = map_edit.save_map_edit(self.tmp, {'nodes': {'2': {'x': 300.0, 'y': 120.0}}})
        self.assertFalse(r2['first_backup'])
        original2 = json.loads((self.tmp / 'map.original.json').read_text())
        self.assertEqual(original2[1]['nodes']['2']['x'], 200.0)
        self.assertEqual(json.loads((self.tmp / self.map_name).read_text())[1]['nodes']['2']['x'], 300.0)
        self.assertIsNotNone(r2.get('layout_quality'))
        report = json.loads((self.tmp / 'report.json').read_text())
        self.assertIn('core', report['layout_quality']['final_geometry'])

    def test_save_rejects_bad_filename(self):
        with self.assertRaises(ValueError):
            map_edit.save_map_edit(self.tmp, {'nodes': {'2': {'x': 1, 'y': 2}}}, filename='evil.json')

    def test_reset_restores_original_and_svg(self):
        map_edit.save_map_edit(self.tmp, {'nodes': {'2': {'x': 260.0, 'y': 120.0, 'label_x': 270.0, 'label_y': 110.0}}})
        self.assertEqual(json.loads((self.tmp / self.map_name).read_text())[1]['nodes']['2']['x'], 260.0)
        r = map_edit.reset_map(self.tmp)
        self.assertTrue(r['ok'])
        restored = json.loads((self.tmp / self.map_name).read_text())
        self.assertEqual(restored[1]['nodes']['2']['x'], 200.0)
        self.assertEqual(restored[1]['nodes']['2']['label_x'], 210.0)
        svg = (self.tmp / self.svg_name).read_text()
        self.assertIn('cx="200.0"', svg)
        # original backup stays
        self.assertTrue((self.tmp / 'map.original.json').is_file())
        self.assertEqual(json.loads((self.tmp / 'map.original.json').read_text())[1]['nodes']['2']['x'], 200.0)

    def test_reset_without_original_raises(self):
        with self.assertRaises(FileNotFoundError):
            map_edit.reset_map(self.tmp)

    def test_endpoint_helpers_match_server_guards(self):
        # mirror the server's job-id / safe-filename checks used before save_map_edit
        import re
        JOB_ID = re.compile(r'^j_[0-9a-f]{10}$')
        SAFE_FILE = re.compile(r'^(?:views/view-\d+\.(?:svg|json)|[A-Za-z0-9_.-]+)$')
        self.assertTrue(JOB_ID.match('j_0123456789'))
        self.assertFalse(JOB_ID.match('j_short'))
        self.assertTrue(SAFE_FILE.match(self.map_name))
        self.assertFalse(SAFE_FILE.match('../' + self.map_name))
        with self.assertRaises(jsonschema.ValidationError):
            map_edit.validate_edit({'nodes': {'2': {'x': 1, 'y': 2, 'z': 3}}})


if __name__ == '__main__':
    unittest.main()
