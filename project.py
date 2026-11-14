#!/usr/bin/env python3
"""Root entry point for Open Meta Maps (tool, compose, evaluate, catalog, setup)."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parent

def main():
 args=sys.argv[1:]
 if not args or args in (['-h'],['--help']):
  print('Open Meta Maps\n\nRead: README.md; docs/\n\nCommands:\n  python3 project.py setup        # one-time: build the edge routers (needs g++ or clang++, and network)\n  python3 project.py tool         # local web tool: upload a GEM, build the map, preview and download (127.0.0.1:8765)\n  python3 project.py compose --gem examples/e_coli_core.xml --out results/e_coli_core/ --source-paths   # GEM in -> map + report out (Bézier/curated layout by default; --layout-style legacy for orthogonal routing; add --ec for extended EC mode, --no-library for a from-scratch layout)\n  python3 project.py evaluate --gem MODEL.xml --map DIR/OpenMetaMaps_<stem>_result.json [--out DIR/evaluation.json]   # layout metrics (layout-benchmark metrics; optional: pip install -r layout_atlas/requirements-eval.txt)\n  python3 project.py data export  # whole curated Atlas as one archive (Escher JSON per module + manifest + notices); --source, --licence, --all-windows\n  python3 project.py data list    # optional data packages (full Atlas); `data fetch atlas-full` installs the full local Atlas snapshot\n  python3 project.py catalog      # rebuild the module catalog pages\n\nStage options: python3 layout_atlas/cli.py --help')
  return
 if args and args[0]=='data':
  sys.path.insert(0,str(ROOT/'layout_atlas'));import data_packages;data_packages.main(args[1:]);return
 if args==['setup']:
  # one-time build of the libavoid edge routers (needs g++ or clang++, and network access for the pinned source)
  sys.path.insert(0,str(ROOT/'layout_atlas'));import routing;routing.build();routing.build_labels();print('routers built');return
 result=subprocess.run([sys.executable,str(ROOT/'layout_atlas/cli.py'),*args],cwd=ROOT);raise SystemExit(result.returncode)
if __name__=='__main__':main()
