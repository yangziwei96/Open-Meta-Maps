"""Local server for the upload-a-GEM tool (stdlib only, 127.0.0.1).

  python3 project.py tool [--port 8765]        (or: python3 tool_server.py)

* GET /                      the tool page results/tool/index.html
* GET /docs/*.md, /README.md, /THIRD_PARTY_NOTICES.md, /LICENSE, /data/README.md
                             allowlisted repo docs rendered as HTML (same chrome/nav); GET /docs/images/* for embedded images
* GET /<path>                read-only static files from results/ (catalog, atlas_browser, experiment pages, ...); tool_runs is NOT served this way
* POST /api/jobs             multipart/form-data: `model` (.xml SBML or .json COBRA/BiGG) + `options` (JSON string)
* GET  /api/jobs/<id>        job state (see tool_pipeline.Job)
* GET  /api/jobs/<id>/files/<name>   result files (skeleton.svg, OpenMetaMaps_<stem>_result.*, modules.json, report.json, licenses.json, views/<view>.svg ...)
* POST /api/jobs/<id>/map            save interactive position edits into the result Escher JSON (keeps map.original.json once; regenerates result SVG + layout quality)
* POST /api/jobs/<id>/views/<view_id>  draw one view (queued behind running jobs)
* DELETE /api/jobs/<id>      cancel a running job and delete its folder
* GET  /api/atlas/meta       Atlas Browser filter vocabulary + counts (builds index on first use)
* GET  /api/atlas/search     paginated module search (`q`, `ec`, `reaction`, `metabolite`, `map`, `source`, `license`, `page`, `per_page`)
* GET  /api/atlas/modules/<id>                module detail + provenance
* GET  /api/atlas/modules/<id>/thumbnail.svg  lazy thumbnail (no rotate/mirror)
* GET  /api/atlas/modules/<id>/escher.json    single-module Escher download
* POST /api/atlas/download   JSON `{ids:[...]}` → zip of Escher JSON + provenance manifest (CSV+JSON)

Lifetime: each open tab sends /api/bye on pagehide (primary signal) and also pings /api/ping while visible. When the last tab is
gone (bye, or a multi-minute stale fallback that tolerates Chrome background-timer throttling) and a short grace (~8 s) has
passed so reloads survive, and no job is running or queued, the server stops itself (disable with --keep-running). If a job
is still running when the last tab closes, the server exits after that job finishes unless a tab reconnects. If the port is
taken by an older instance of this same tool, that instance is stopped and replaced; any other program on the port is left
alone and the next free port is used.

Jobs run one at a time in a separate process (tool_pipeline.py).  The server only listens on 127.0.0.1, checks the Host
header (DNS-rebinding guard), limits uploads to 25 MB, never executes uploaded content and never serves job inputs back.
"""
import email.parser
import errno
import signal
import email.policy
import http.server
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'
RUNS = RESULTS / 'tool_runs'
MAX_BYTES = 25 * 1024 * 1024
EXAMPLES = {  # example models selectable in the page; fixed table, no client-supplied paths
    'e_coli_core': ('E. coli core', BASE.parent / 'examples' / 'e_coli_core.xml'),  # ordered by size: 95, 554, 1,577, 2,712 reactions
    'iIT341': ('H. pylori iIT341', BASE.parent / 'examples' / 'iIT341.xml'),
    'iMM904': ('S. cerevisiae iMM904', BASE.parent / 'examples' / 'iMM904.xml'),
    'iML1515': ('E. coli iML1515', BASE.parent / 'examples' / 'iML1515.xml'),
}
_example_info = {}


def example_list():
    out = []
    for key, (name, path) in EXAMPLES.items():
        if not path.is_file():
            continue
        if key not in _example_info:
            _example_info[key] = len(re.findall(rb'<reaction ', path.read_bytes()))  # SBML reaction elements
        out.append(dict(id=key, name=name, file=path.name, reactions=_example_info[key], bytes=path.stat().st_size))
    return out

MAX_QUEUE = 4
JOB_TIMEOUT = 45 * 60
JOB_ID = re.compile(r'^j_[0-9a-f]{10}$')
VIEW_ID = re.compile(r'^view-\d+$')
MODULE_ID = re.compile(r'^[A-Za-z0-9._:-]{8,128}$')
SAFE_FILE = re.compile(r'^(?:views/view-\d+\.(?:svg|json)|[A-Za-z0-9_.-]+)$')
BLOCKED_TOP = {'upload', 'options.json', 'selected.json.gz', 'views.json', 'full', 'whole', 'job.json'}
MAX_ATLAS_DOWNLOAD = 200

tasks = queue.Queue()
processes = {}
lock = threading.Lock()


def error(code, en):
    return dict(error=dict(code=code, message_en=en))


def tool_environment():
    env = dict(os.environ)
    node_dir = Path.home() / '.local/node/bin'
    if node_dir.exists():
        env['PATH'] = str(node_dir) + os.pathsep + env.get('PATH', '')
    env['PYTHONHASHSEED'] = '0'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def check_environment():
    """Fail early with a readable message when the drawing stack is missing."""
    problems = []
    probe = subprocess.run([sys.executable, '-c', 'import PIL, jsonschema, scipy, networkx, numpy, markdown'], capture_output=True, text=True)
    if probe.returncode:
        problems.append('Python packages missing (need pillow, jsonschema, scipy, networkx, numpy, markdown): ' + probe.stderr.strip().splitlines()[-1])
    node = subprocess.run(['node', '--version'], capture_output=True, text=True, env=tool_environment()) if shutil.which('node', path=tool_environment()['PATH']) else None
    if node is None or node.returncode:
        problems.append('node not found (needed for the ELK/fCoSE layout backend); put it on PATH or in ~/.local/node/bin')
    return problems


def read_state(job_dir):
    p = job_dir / 'job.json'
    for _ in range(5):  # the pipeline replaces the file atomically; retry on a rare race
        try:
            return json.loads(p.read_text())
        except Exception:
            time.sleep(.05)
    return None


def write_state(job_dir, state):
    tmp = job_dir / 'job.tmp'
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    os.replace(tmp, job_dir / 'job.json')


def queued_state(job_id):
    import tool_pipeline as tp
    steps = [dict(key=k, label_en=en, state='pending', seconds=None) for k, en in tp.STEPS]
    return dict(id=job_id, status='queued', step='parse', steps=steps, progress=0.0, log=[], warnings=[], error=None, summary=None, files=[], views=[])


def worker():
    while True:
        job_dir, view = tasks.get()
        state = read_state(job_dir)
        if state is None or state.get('status') == 'cancelled':
            continue
        cmd = [sys.executable, str(BASE / 'tool_pipeline.py'), str(job_dir)] + (['--view', view] if view else [])
        try:
            with lock:
                proc = subprocess.Popen(cmd, cwd=str(BASE), env=tool_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                processes[job_dir.name] = proc
            try:
                out, _ = proc.communicate(timeout=JOB_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                state = read_state(job_dir) or state
                state.update(status='failed', error=dict(code='timeout', message_en='The job exceeded the time limit and was stopped.'))
                write_state(job_dir, state)
                continue
            if proc.returncode not in (0, None) and not view:
                state = read_state(job_dir) or state
                if state.get('status') not in ('done', 'failed', 'cancelled'):
                    state.update(status='failed', error=dict(code='internal_error', message_en='The pipeline process ended unexpectedly.'))
                    state['log'] = state.get('log', []) + [dict(t=0, msg=(out or '')[-1500:])]
                    write_state(job_dir, state)
        finally:
            with lock:
                processes.pop(job_dir.name, None)
            WAKE.set()


def parse_multipart(content_type, body):
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(b'Content-Type: ' + content_type.encode() + b'\r\n\r\n' + body)
    fields = {}
    if not msg.is_multipart():
        return fields
    for part in msg.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if name:
            fields[name] = (part.get_filename(), part.get_payload(decode=True) or b'')
    return fields


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'OpenPathwayMapsTool/1'

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    # ---- helpers
    def send_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, status, body, ctype, extra=None):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def host_ok(self):
        host = (self.headers.get('Host') or '').split(':')[0].lower()
        return host in ('127.0.0.1', 'localhost', '[::1]')

    def job_dir(self, job_id):
        return RUNS / job_id if JOB_ID.match(job_id or '') and (RUNS / job_id).is_dir() else None

    # ---- routing
    def do_GET(self):
        if not self.host_ok():
            return self.send_json(403, error('bad_host', 'Forbidden host.'))
        path = unquote(urlparse(self.path).path)
        m = re.match(r'^/api/jobs/([^/]+)(?:/files/(.+))?$', path)
        if m:
            jd = self.job_dir(m[1])
            if jd is None:
                return self.send_json(404, error('not_found', 'Unknown job.'))
            if m[2] is None:
                state = read_state(jd)
                return self.send_json(200, state) if state else self.send_json(500, error('state_unreadable', 'Job state unreadable.'))
            name = m[2]
            if not SAFE_FILE.match(name) or name.split('/')[0] in BLOCKED_TOP or '..' in name:
                return self.send_json(403, error('forbidden', 'This file is not available.'))
            f = jd / name
            if not f.is_file():
                return self.send_json(404, error('not_found', 'File not found.'))
            ctype = mimetypes.guess_type(f.name)[0] or 'application/octet-stream'
            extra = {'Content-Disposition': 'attachment; filename="%s"' % f.name} if self.path.endswith('?download=1') else {}
            return self.send_bytes(200, f.read_bytes(), ctype + ('; charset=utf-8' if ctype.startswith(('text/', 'application/json', 'image/svg')) else ''), extra)
        if path == '/api/health':
            return self.send_json(200, dict(tool='open-pathway-maps', pid=os.getpid(), examples=True, atlas=True, auto_exit=not LIFE.get('keep', False)))
        if path == '/api/examples':
            return self.send_json(200, dict(examples=example_list()))
        if path.startswith('/api/atlas'):
            return self.atlas_get(path)
        if path.startswith('/api/'):
            return self.send_json(404, error('not_found', 'Unknown endpoint.'))
        return self.static(path)

    def atlas_get(self, path):
        import atlas_browser
        try:
            atlas_browser.ensure_index()
        except Exception:
            return self.send_json(500, error('atlas_index', 'Could not build or load the Atlas Browser index.'))
        if path == '/api/atlas/meta':
            return self.send_json(200, atlas_browser.meta())
        if path == '/api/atlas/search':
            qs = parse_qs(urlparse(self.path).query)
            filters = {k: (qs.get(k) or [''])[0] for k in ('q', 'ec', 'reaction', 'metabolite', 'map', 'source', 'license', 'licence', 'database', 'ec_number', 'reaction_id', 'metabolite_id', 'map_id', 'model', 'variants', 'show_variants')}
            page = (qs.get('page') or ['1'])[0]
            per = (qs.get('per_page') or [str(atlas_browser.PAGE_DEFAULT)])[0]
            try:
                return self.send_json(200, atlas_browser.search(filters, page=page, per_page=per))
            except Exception:
                return self.send_json(500, error('atlas_search', 'Search failed.'))
        m = re.match(r'^/api/atlas/modules/([^/]+)(?:/(thumbnail\.svg|escher\.json))?$', path)
        if not m or not MODULE_ID.match(m[1]):
            return self.send_json(404, error('not_found', 'Unknown Atlas endpoint.'))
        mid = m[1]
        kind = m[2]
        if kind == 'thumbnail.svg':
            mod = atlas_browser.load_module(mid)
            if mod is None or not atlas_browser.license_ok(mod.get('license') or mod.get('licence') or ''):
                return self.send_json(404, error('not_found', 'Unknown module.'))
            svg = atlas_browser.thumbnail_svg(mod)
            return self.send_bytes(200, svg.encode(), 'image/svg+xml; charset=utf-8', {'Cache-Control': 'public, max-age=3600'})
        if kind == 'escher.json':
            mod = atlas_browser.load_module(mid)
            if mod is None or not atlas_browser.license_ok(mod.get('license') or mod.get('licence') or ''):
                return self.send_json(404, error('not_found', 'Unknown module.'))
            try:
                data = atlas_browser.module_to_escher(mod)
            except Exception:
                return self.send_json(400, error('no_geometry', 'This module has no drawable geometry.'))
            body = json.dumps(data, ensure_ascii=False, indent=2).encode()
            return self.send_bytes(200, body, 'application/json; charset=utf-8',
                                   {'Content-Disposition': 'attachment; filename="%s.escher.json"' % mid[:40]})
        detail = atlas_browser.module_detail(mid)
        if detail is None:
            return self.send_json(404, error('not_found', 'Unknown module.'))
        return self.send_json(200, detail)

    HEARTBEAT = (b"<script>(function(){if(window.__opmHB)return;window.__opmHB=1;"
                 b"var id=Math.random().toString(36).slice(2,10);"
                 b"function ping(){fetch('/api/ping?c='+id,{method:'POST',keepalive:true}).catch(function(){})}"
                 b"function bye(){try{navigator.sendBeacon('/api/bye?c='+encodeURIComponent(id),'')}catch(e){}}"
                 b"ping();setInterval(ping,4000);"
                 b"document.addEventListener('visibilitychange',function(){if(document.visibilityState==='visible')ping()});"
                 b"window.addEventListener('focus',ping);"
                 b"window.addEventListener('pagehide',bye);window.addEventListener('unload',bye)})();</script>")

    def html_bytes(self, data):
        """Every HTML page this server hands out keeps the server alive while it is open (the catalog and home pages have no script of their own; without
        this the server stopped shortly after the tool page was left for another page)."""
        if b'__opmHB' in data or b"/api/ping?c=" in data:
            return data
        i = data.rfind(b'</body>')
        return data[:i] + self.HEARTBEAT + data[i:] if i >= 0 else data + self.HEARTBEAT

    def static(self, path):
        import doc_pages
        doc = doc_pages.handle(path)
        if doc is not None:
            status, body, ctype = doc
            if status != 200 or body is None:
                return self.send_json(404, error('not_found', 'Not found.'))
            if ctype.startswith('text/html'):
                body = self.html_bytes(body)
            return self.send_bytes(200, body, ctype)
        rel = 'tool/index.html' if path in ('/', '') else path.lstrip('/')
        if rel.startswith('layout_atlas/results/'):
            rel = rel[len('layout_atlas/results/'):]  # the project home page links to results by their path from the project root
        if rel == 'index.html':
            target = (BASE.parent / 'index.html').resolve()  # the project home page (Home link of every page)
            if not target.is_file():
                return self.send_json(404, error('not_found', 'Not found.'))
            return self.send_bytes(200, self.html_bytes(target.read_bytes()), 'text/html; charset=utf-8')
        target = (RESULTS / rel).resolve()
        # files under results/ are served; a developer may keep big generated folders outside the repository and link them in as FIRST-LEVEL symlinks of results/
        # (`results/catalog -> ../../../Experiments/...`): those link targets are allowed too, nothing else outside results/ is
        roots = [RESULTS.resolve()] + [c.resolve() for c in RESULTS.iterdir() if c.is_symlink()]
        if not any(r == target or r in target.parents for r in roots) or RUNS.resolve() in (target, *target.parents) or not target.is_file():
            return self.send_json(404, error('not_found', 'Not found.'))
        ctype = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        if target.suffix == '.js':
            ctype = 'text/javascript'
        body = target.read_bytes()
        if target.suffix in ('.html', '.htm'):
            body = self.html_bytes(body)
        self.send_bytes(200, body, ctype + ('; charset=utf-8' if ctype.startswith(('text/', 'application/json', 'image/svg')) else ''))

    def do_POST(self):
        if not self.host_ok():
            return self.send_json(403, error('bad_host', 'Forbidden host.'))
        path = urlparse(self.path).path
        if path in ('/api/ping', '/api/bye'):
            length = int(self.headers.get('Content-Length') or 0)
            if length > 0:
                self.rfile.read(min(length, 4096))  # drain beacon/keepalive body
            cid = (parse_qs(urlparse(self.path).query).get('c') or ['page'])[0][:40]
            (client_touch if path == '/api/ping' else client_bye)(cid)
            return self.send_json(200, dict(ok=True))
        if path == '/api/jobs':
            return self.create_job()
        if path == '/api/atlas/download':
            return self.atlas_download()
        m = re.match(r'^/api/jobs/([^/]+)/map/reset$', path)
        if m:
            return self.reset_map(m[1])
        m = re.match(r'^/api/jobs/([^/]+)/map$', path)
        if m:
            return self.save_map(m[1])
        m = re.match(r'^/api/jobs/([^/]+)/views/([^/]+)$', path)
        if m:
            jd = self.job_dir(m[1])
            if jd is None or not VIEW_ID.match(m[2]):
                return self.send_json(404, error('not_found', 'Unknown job or view.'))
            state = read_state(jd)
            if not state or state.get('status') != 'done' or not any(v['id'] == m[2] for v in state.get('views', [])):
                return self.send_json(409, error('not_ready', 'The job is not finished or the view does not exist.'))
            for v in state['views']:
                if v['id'] == m[2]:
                    v['state'] = 'queued'
            write_state(jd, state)
            tasks.put((jd, m[2]))
            return self.send_json(202, dict(id=m[1], view=m[2], status='queued'))
        return self.send_json(404, error('not_found', 'Unknown endpoint.'))

    def save_map(self, job_id):
        jd = self.job_dir(job_id)
        if jd is None:
            return self.send_json(404, error('not_found', 'Unknown job.'))
        state = read_state(jd)
        if not state or state.get('status') != 'done':
            return self.send_json(409, error('not_ready', 'The job is not finished.'))
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > MAX_BYTES:
            return self.send_json(413 if length > MAX_BYTES else 400,
                                  error('too_large' if length > MAX_BYTES else 'bad_format', 'Edit payload missing or larger than 25 MB.'))
        ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip().lower()
        if ctype not in ('application/json', 'text/json'):
            return self.send_json(400, error('bad_format', 'Expected application/json.'))
        try:
            edit = json.loads(self.rfile.read(length).decode())
        except Exception:
            return self.send_json(400, error('bad_format', 'The body is not valid JSON.'))
        try:
            import map_edit
            result = map_edit.save_map_edit(jd, edit)
        except Exception as e:
            import jsonschema
            if isinstance(e, jsonschema.ValidationError):
                return self.send_json(400, error('bad_schema', 'Edit JSON failed schema validation: ' + e.message))
            if isinstance(e, (ValueError, FileNotFoundError, KeyError, TypeError)):
                return self.send_json(400, error('bad_edit', str(e)))
            return self.send_json(500, error('internal_error', 'Could not save the map edit.'))
        state = read_state(jd) or state
        return self.send_json(200, dict(id=job_id, status='saved', summary=state.get('summary'),
                                        layout_quality=result.get('layout_quality'),
                                        original_kept=result.get('original_kept', True)))

    def reset_map(self, job_id):
        jd = self.job_dir(job_id)
        if jd is None:
            return self.send_json(404, error('not_found', 'Unknown job.'))
        state = read_state(jd)
        if not state or state.get('status') != 'done':
            return self.send_json(409, error('not_ready', 'The job is not finished.'))
        length = int(self.headers.get('Content-Length') or 0)
        if length > 0:
            self.rfile.read(min(length, MAX_BYTES))
        try:
            import map_edit
            result = map_edit.reset_map(jd)
        except FileNotFoundError:
            return self.send_json(404, error('no_original', 'No original map backup to restore.'))
        except Exception as e:
            if isinstance(e, (ValueError, KeyError, TypeError)):
                return self.send_json(400, error('bad_edit', str(e)))
            return self.send_json(500, error('internal_error', 'Could not reset the map.'))
        state = read_state(jd) or state
        return self.send_json(200, dict(id=job_id, status='reset', summary=state.get('summary'),
                                        layout_quality=result.get('layout_quality'),
                                        original_kept=True))

    def atlas_download(self):
        import atlas_browser
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > 2_000_000:
            return self.send_json(413 if length > 2_000_000 else 400,
                                  error('too_large' if length > 2_000_000 else 'bad_format', 'Download request missing or too large.'))
        ctype = (self.headers.get('Content-Type') or '').split(';')[0].strip().lower()
        if ctype not in ('application/json', 'text/json'):
            return self.send_json(400, error('bad_format', 'Expected application/json.'))
        try:
            body = json.loads(self.rfile.read(length).decode())
            ids = body.get('ids') or body.get('modules') or []
            assert isinstance(ids, list)
        except Exception:
            return self.send_json(400, error('bad_format', 'Expected JSON with an ids array.'))
        ids = [str(x) for x in ids if MODULE_ID.match(str(x or ''))]
        if not ids:
            return self.send_json(400, error('bad_format', 'No valid module ids.'))
        if len(ids) > MAX_ATLAS_DOWNLOAD:
            return self.send_json(400, error('too_many', f'At most {MAX_ATLAS_DOWNLOAD} modules per download.'))
        try:
            atlas_browser.ensure_index()
            blob, rows, missing = atlas_browser.build_download_zip(ids)
        except Exception:
            return self.send_json(500, error('atlas_download', 'Could not build the download archive.'))
        if not rows:
            return self.send_json(404, error('not_found', 'None of the requested modules were found.'))
        return self.send_bytes(200, blob, 'application/zip',
                               {'Content-Disposition': 'attachment; filename="atlas_modules.zip"',
                                'X-Atlas-Modules': str(len(rows)), 'X-Atlas-Missing': str(len(missing))})

    def create_job(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > MAX_BYTES + 200_000:
            return self.send_json(413, error('too_large', 'The upload is larger than 25 MB.'))
        ctype = self.headers.get('Content-Type') or ''
        if not ctype.startswith('multipart/form-data'):
            return self.send_json(400, error('bad_format', 'Expected multipart/form-data.'))
        fields = parse_multipart(ctype, self.rfile.read(length))
        example = fields.get('example', (None, b''))[1].decode(errors='ignore').strip()
        if example:
            if example not in EXAMPLES or not EXAMPLES[example][1].is_file():
                return self.send_json(400, error('bad_format', 'Unknown example model.'))
            fields['model'] = (EXAMPLES[example][1].name, EXAMPLES[example][1].read_bytes())
        if 'model' not in fields or not fields['model'][1]:
            return self.send_json(400, error('bad_format', 'No model file in the upload.'))
        filename, data = fields['model']
        ext = Path(filename or '').suffix.lower()
        if ext not in ('.xml', '.json') or len(data) > MAX_BYTES:
            return self.send_json(400 if len(data) <= MAX_BYTES else 413, error('too_large' if len(data) > MAX_BYTES else 'bad_format', 'Only .xml (SBML) or .json (COBRA/BiGG) files up to 25 MB are accepted.'))
        try:
            options = json.loads(fields.get('options', (None, b'{}'))[1].decode() or '{}')
            assert isinstance(options, dict)
        except Exception:
            return self.send_json(400, error('bad_format', 'The options field is not valid JSON.'))
        options = dict(libraries={k: bool(v) for k, v in (options.get('libraries') or {}).items() if k in ('core', 'ec')}, library=options.get('library') if options.get('library') in ('windows', 'units_sub') else 'windows', source_paths=bool(options.get('source_paths')), layout=options.get('layout') if options.get('layout') in ('global', 'tiled') else 'global', layout_style=('legacy' if options.get('layout_style')=='legacy' else 'curated'), completion_context=options.get('completion_context') if options.get('completion_context') in ('auto', 'off') else 'auto', edge_routing=options.get('edge_routing') if options.get('edge_routing') in ('libavoid', 'hybrid', 'none') else 'libavoid', context=options.get('context') if options.get('context') in ('auto', 'off') else 'auto', instances=options.get('instances') if options.get('instances') in ('auto', 'off') else 'auto', block_layout=options.get('block_layout') if options.get('block_layout') in ('auto', 'single') else 'single', orientation=options.get('orientation') if options.get('orientation') in ('off', 'sides', 'full') else ('off' if options.get('orientation') is False else 'full'))
        if tasks.qsize() + len(processes) >= MAX_QUEUE:
            return self.send_json(429, error('busy', 'The tool is busy; try again when the running job has finished.'))
        job_id = 'j_' + uuid.uuid4().hex[:10]
        jd = RUNS / job_id
        (jd / 'upload').mkdir(parents=True)
        (jd / 'upload' / ('model' + ext)).write_bytes(data)
        (jd / 'options.json').write_text(json.dumps(options))
        import output_names
        output_names.write_meta(jd, filename or 'model', size=len(data))
        write_state(jd, queued_state(job_id))
        tasks.put((jd, None))
        self.send_json(201, dict(id=job_id, status='queued'))

    def do_DELETE(self):
        if not self.host_ok():
            return self.send_json(403, error('bad_host', 'Forbidden host.'))
        m = re.match(r'^/api/jobs/([^/]+)$', urlparse(self.path).path)
        jd = self.job_dir(m[1]) if m else None
        if jd is None:
            return self.send_json(404, error('not_found', 'Unknown job.'))
        with lock:
            proc = processes.get(jd.name)
            if proc:
                state = read_state(jd) or {}
                state.update(status='cancelled')
                write_state(jd, state)
                proc.kill()
        shutil.rmtree(jd, ignore_errors=True)
        self.send_json(200, dict(id=jd.name, status='removed'))


CLIENTS = {}  # open page id -> last time it pinged
LIFE = dict(ever=False, empty_since=None, keep=False, pending_exit=False)
GRACE = 8.0     # seconds with no open tab before stop (covers reload/navigation gaps)
STALE = 300.0   # fallback only: hidden Chrome tabs may ping only ~once/min; several minutes of silence before counting closed
WAKE = threading.Event()


def client_touch(cid):
    with lock:
        CLIENTS[cid] = time.time()
        LIFE['ever'], LIFE['empty_since'], LIFE['pending_exit'] = True, None, False
    WAKE.set()


def client_bye(cid):
    with lock:
        CLIENTS.pop(cid, None)
        if LIFE['ever'] and not CLIENTS:
            LIFE['pending_exit'] = True
    WAKE.set()


def _busy():
    return bool(processes) or not tasks.empty()


def idle_watch(httpd):
    """Stop after explicit bye (or multi-minute stale fallback) and GRACE, once nothing is running."""
    while True:
        WAKE.wait(timeout=1.0)
        WAKE.clear()
        with lock:
            if LIFE.get('keep'):
                continue
            now = time.time()
            for cid in [c for c, t in CLIENTS.items() if now - t > STALE]:
                CLIENTS.pop(cid, None)
            busy = _busy()
            if LIFE['ever'] and not CLIENTS and not busy:
                LIFE['empty_since'] = LIFE['empty_since'] or now
                done = now - LIFE['empty_since'] >= GRACE
            else:
                # Job still running after the last tab closed: wait, then take the grace once it finishes.
                if LIFE['ever'] and not CLIENTS and busy:
                    LIFE['pending_exit'] = True
                LIFE['empty_since'], done = None, False
        if done:
            print('All pages are closed and no job is running: stopping the tool server.', flush=True)
            httpd.shutdown()
            return


def listeners(port):
    out = subprocess.run(['lsof', '-tiTCP:%d' % port, '-sTCP:LISTEN'], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.isdigit()]


def stop_older_instance(port):
    """Stop a previous instance of THIS tool (command line contains this folder) that still holds the port; leave anything else alone."""
    stopped = False
    for pid in listeners(port):
        if pid == os.getpid():
            continue
        cmd = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
        if str(BASE) in cmd or 'tool_server' in cmd:
            print(f'Port {port} is held by an older instance of this tool (pid {pid}); stopping it.', flush=True)
            os.kill(pid, signal.SIGTERM)
            stopped = True
    if stopped:
        for _ in range(30):
            if not listeners(port):
                break
            time.sleep(0.2)
    return stopped


def bind(port):
    try:
        return http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler), port
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
    if stop_older_instance(port):
        try:
            return http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler), port
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
    for alt in range(port + 1, port + 30):
        try:
            print(f'Port {port} is in use by another program; using port {alt} instead.', flush=True)
            return http.server.ThreadingHTTPServer(('127.0.0.1', alt), Handler), alt
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
    raise SystemExit(f'No free port between {port} and {port + 29}.')


def main(port=8765, keep_running=False):
    RUNS.mkdir(parents=True, exist_ok=True)
    problems = check_environment()
    if problems:
        print('The tool cannot start:\n  - ' + '\n  - '.join(problems), file=sys.stderr)
        raise SystemExit(1)
    LIFE['keep'] = bool(keep_running)
    threading.Thread(target=worker, daemon=True).start()
    httpd, port = bind(port)
    if not keep_running:
        threading.Thread(target=idle_watch, args=(httpd,), daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *a: threading.Thread(target=httpd.shutdown, daemon=True).start())
    print(f'Open Meta Maps tool: http://127.0.0.1:{port}/tool/index.html   ' +
          ('(stops by itself when the page is closed; ' if not keep_running else '(') + f'Ctrl-C to stop; jobs are stored in {RUNS})', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8765, keep_running='--keep-running' in sys.argv)
