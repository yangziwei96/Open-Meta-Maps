"""Stability when a few reactions are removed (e_coli_core only).

Compares reduced-model layouts to the full-model map (Procrustes, median displacement)
and records crossings / opposite-side share / longest-edge change.  Also checks
``--previous`` rebuild of the k=3 model.

Requires compose build assets (``layout_atlas/results/routing_build.json`` and the
libavoid binary).  Skips cleanly with a clear reason when they are missing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cobra
import pytest

from core import ROOT, BASE
from evaluation.escher_bridge import escher_positions, load_escher
from evaluation.pathway import read_sbml
from evaluation.stability import position_displacement
from layout_quality import evaluate as lq_evaluate
import output_names

RM3 = ('ME1', 'PPCK', 'FRD7')
RM5 = ('MALt2_2', 'FBP', 'PFK', 'ADK1', 'ALCD2x')  # seed-7 list from the gap analysis


def _compose_assets_ready():
    """Return None when compose can run; else a short skip reason."""
    build_json = BASE / 'results' / 'routing_build.json'
    if not build_json.is_file():
        return f'missing compose asset {build_json.relative_to(ROOT)}'
    try:
        meta = json.loads(build_json.read_text())
    except Exception as e:
        return f'unreadable routing_build.json: {e}'
    if meta.get('status') not in (None, 'built', 'ok'):
        # Older files may omit status; only fail on an explicit failure.
        if meta.get('status') == 'failed':
            return 'routing_build.json reports status=failed'
    binary = BASE / 'build' / 'avoid_router'
    # Curated style often skips libavoid, but the tool env still expects the build record.
    # If the binary is absent and build status failed, skip; if binary absent but JSON ok,
    # still allow (line/Bezier path does not need the router).
    if not binary.is_file() and meta.get('status') == 'failed':
        return f'missing libavoid binary {binary} (routing build failed)'
    gem = ROOT / 'examples' / 'e_coli_core.xml'
    if not gem.is_file():
        return f'missing example model {gem.relative_to(ROOT)}'
    return None


def _write_reduced(gem: Path, remove: tuple[str, ...], out: Path) -> Path:
    model = cobra.io.read_sbml_model(str(gem))
    for rid in remove:
        rx = model.reactions.get_by_id(rid)
        model.remove_reactions([rx])
    out.parent.mkdir(parents=True, exist_ok=True)
    cobra.io.write_sbml_model(model, str(out))
    return out


def _compose(gem: Path, out: Path, previous: Path | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'project.py'), 'compose',
           '--gem', str(gem), '--out', str(out), '--source-paths']
    if previous is not None:
        cmd += ['--previous', str(previous)]
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join([
        str(Path.home() / '.local' / 'bin'),
        env.get('PATH', ''),
    ])
    node_bin = Path.home() / '.local' / 'node' / 'bin'
    if node_bin.is_dir():
        env['PATH'] = str(node_bin) + os.pathsep + env['PATH']
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f'compose failed ({proc.returncode}):\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}')
    path = output_names.map_json(out)
    assert path.is_file()
    return path


def _quality(map_path: Path) -> dict:
    data = json.loads(map_path.read_text())
    summary, _ = lq_evaluate(data)
    core = summary['core']
    return dict(
        crossings=int(core['crossings']['counted']),
        opposite_side=float(core['participant_placement']['inout_opposite_share'] or 0.0),
        max_edge_rel=float(core['long_lines']['edge_len_max_rel'] or 0.0),
    )


def _compare(gem: Path, full_map: Path, other_map: Path) -> dict:
    pathway = read_sbml(gem)
    a = escher_positions(pathway, load_escher(full_map))
    b = escher_positions(pathway, load_escher(other_map))
    return position_displacement(a, b)


def test_e_coli_core_stability_k3_k5_and_previous(tmp_path):
    reason = _compose_assets_ready()
    if reason:
        pytest.skip(f'stability compose skipped: {reason}')

    gem = ROOT / 'examples' / 'e_coli_core.xml'
    assert gem.is_file()
    base = tmp_path / 'stab'
    t0 = time.time()
    full_dir = base / 'full'
    full_map = _compose(gem, full_dir)
    q_full = _quality(full_map)

    rm3_gem = _write_reduced(gem, RM3, base / 'rm3.xml')
    rm5_gem = _write_reduced(gem, RM5, base / 'rm5.xml')

    rm3_map = _compose(rm3_gem, base / 'rm3')
    rm5_map = _compose(rm5_gem, base / 'rm5')
    rm3_prev_map = _compose(rm3_gem, base / 'rm3_prev', previous=full_map)

    d3 = _compare(rm3_gem, full_map, rm3_map)
    d5 = _compare(rm5_gem, full_map, rm5_map)
    d3p = _compare(rm3_gem, full_map, rm3_prev_map)
    q3 = _quality(rm3_map)
    q5 = _quality(rm5_map)
    q3p = _quality(rm3_prev_map)

    report = dict(
        seconds=round(time.time() - t0, 1),
        full=q_full,
        k3=dict(stability=d3, quality=q3),
        k5=dict(stability=d5, quality=q5),
        k3_previous=dict(stability=d3p, quality=q3p),
    )
    (base / 'stability_report.json').write_text(json.dumps(report, indent=2))
    print('STABILITY', json.dumps(report, indent=2))

    # --previous must keep the map almost identical to the full layout.
    assert d3p['procrustes_disparity'] is not None
    assert d3p['procrustes_disparity'] < 0.05, d3p
    assert q3p['crossings'] <= q_full['crossings'] + 2, (q3p, q_full)
    assert q3p['opposite_side'] + 1e-6 >= q_full['opposite_side'] - 0.05, (q3p, q_full)
    assert q3p['max_edge_rel'] <= q_full['max_edge_rel'] * 1.15 + 1e-6, (q3p, q_full)
