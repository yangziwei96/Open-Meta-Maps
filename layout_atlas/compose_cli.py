"""Command-line entry of the upload tool: GEM in, map + report out (same pipeline as the web page).

  python3 project.py compose --gem examples/e_coli_core.xml --out results/local_mvp/ [--ec] [--no-library] [--library windows|units_sub]

Writes the whole job folder (OpenMetaMaps_<stem>_result.{json,svg,sbgn,...}, report.json,
skeleton.*, modules.json, licenses.json, views/ ...).
Exit code 0 only when the pipeline finished and the integrity checks passed.
"""
import shutil
import subprocess
import sys
import time

from core import *
import output_names
import tool_server

RESULT_FILES = ['job.json', 'report.json', 'skeleton.json', 'skeleton.svg', 'modules.json', 'licenses.json', 'selected.json.gz', 'views.json', 'layout_issues.json', 'reaction_provenance.json', 'meta.json', 'options.json']


def run(gem, out, ec=False, no_library=False, library='windows', reaction_style='line', completion='spaced', source_paths=False, layout='global', orientation='full', block_layout='auto', instances='auto', context='auto', edge_routing='libavoid', completion_context='auto', layout_style='curated', previous=None):
    gem, out = Path(gem).resolve(), Path(out).resolve()  # the pipeline runs in layout_atlas/, so paths must be absolute
    if not gem.is_file():
        raise SystemExit(f'GEM file not found: {gem}')
    if gem.suffix.lower() not in ('.xml', '.sbml', '.json'):
        raise SystemExit('The GEM must be an SBML .xml/.sbml or a COBRA/BiGG .json file.')
    previous_path = None
    if previous:
        previous_path = Path(previous).resolve()
        if not previous_path.is_file():
            raise SystemExit(f'Previous map not found: {previous_path}')
    problems = tool_server.check_environment()
    if problems:
        raise SystemExit('The tool cannot run:\n  - ' + '\n  - '.join(problems))
    out.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:  # drop stale results of an earlier run in the same folder
        (out / name).unlink(missing_ok=True)
    output_names.clear_stale_results(out)
    shutil.rmtree(out / 'views', ignore_errors=True)
    shutil.rmtree(out / 'full', ignore_errors=True)
    shutil.rmtree(out / 'upload', ignore_errors=True)
    (out / 'upload').mkdir()
    shutil.copy(gem, out / 'upload' / ('model' + gem.suffix.lower()))
    output_names.write_meta(out, gem.name, size=gem.stat().st_size)
    libraries = dict(core=not no_library, ec=bool(ec))
    options = dict(libraries=libraries, library=library, reaction_style=reaction_style, completion=completion, source_paths=bool(source_paths), layout=layout, orientation=orientation, block_layout=block_layout, instances=instances, context=context, edge_routing=edge_routing, completion_context=completion_context, layout_style=layout_style)
    if previous_path is not None:
        options['previous'] = str(previous_path)
    dump(out / 'options.json', options)
    import tool_pipeline as tp
    tp.Job(out).save()
    proc = subprocess.Popen([sys.executable, str(BASE / 'tool_pipeline.py'), str(out)], cwd=str(BASE), env=tool_server.tool_environment(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    shown = 0
    while proc.poll() is None:
        time.sleep(1.5)
        state = tool_server.read_state(out) or {}
        for line in state.get('log', [])[shown:]:
            print(f"[{line['t']:6.1f}s] {line['msg']}", flush=True)
        shown = len(state.get('log', []))
    tail = proc.communicate()[0]
    state = tool_server.read_state(out) or {}
    for line in state.get('log', [])[shown:]:
        print(f"[{line['t']:6.1f}s] {line['msg']}", flush=True)
    if state.get('status') != 'done':
        err = state.get('error') or dict(message_en='the pipeline ended without a result')
        print('FAILED:', err.get('message_en'), (tail or '')[-400:], file=sys.stderr)
        raise SystemExit(1)
    s = state['summary']
    c = s['covered']
    print(f"\nDone: {s['model']['id']} ({s['model']['reactions']} reactions) -> {out}")
    pv = s.get('provenance', {})
    print(f"  structures used: {s['modules']['count']}; provenance of the {c['total']} reactions: curated {pv.get('expert_derived', 0)}, "
          f"topology-only suggestion {pv.get('topology_layout_suggestion', 0)}, functional (EC) suggestion {pv.get('functional_layout_suggestion', 0)}, "
          f"bridge {pv.get('bridge', 0)}, generated {pv.get('generated', 0)}")
    print('  files: ' + ', '.join(f['name'] for f in state['files']))
    return 0
