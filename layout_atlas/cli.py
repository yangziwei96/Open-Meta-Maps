"""Command-line stages of Open Meta Maps: compose (GEM in, map out), tool (local web tool), routing (build the edge routers), catalog (module catalog pages), evaluate (layout metrics)."""
import argparse,json
from core import *

def main():
 p=argparse.ArgumentParser(description='Open Meta Maps stages: compose, tool, routing, catalog, evaluate')
 p.add_argument('stage',choices=['routing','catalog','tool','compose','evaluate'])
 p.add_argument('--port',type=int,default=8765,help='tool server port (default 8765)')
 p.add_argument('--gem',help='SBML or COBRA/BiGG JSON model')
 p.add_argument('--out',help='output directory')
 p.add_argument('--ec',action='store_true',help='enable EC similarity extension (off by default)')
 p.add_argument('--no-library',action='store_true',help='skip Atlas retrieval; algorithmic layout only')
 p.add_argument('--library',choices=['windows','units_sub'],default='windows',help='pathway-fragment reuse (default) or structure-unit reuse')
 p.add_argument('--reaction-style',choices=['line','marker'],default='line')
 p.add_argument('--completion',choices=['spaced','classic'],default='spaced')
 p.add_argument('--source-paths',action='store_true',help='reuse original Escher curves on matched edges')
 p.add_argument('--keep-running',action='store_true',help='keep the tool server running after the last tab closes')
 p.add_argument('--layout',choices=['global','tiled'],default='global')
 p.add_argument('--orientation',choices=['off','sides','full'],default='full')
 p.add_argument('--block-layout',choices=['auto','single'],default='auto')
 p.add_argument('--instances',choices=['auto','off'],default='auto')
 p.add_argument('--context',choices=['auto','off'],default='auto')
 p.add_argument('--layout-style',choices=['legacy','curated','expert'],default='curated',help='curated = Bézier (default); expert = deprecated alias of curated; legacy = orthogonal libavoid')
 p.add_argument('--completion-context',choices=['auto','off'],default='auto')
 p.add_argument('--edge-routing',choices=['libavoid','hybrid','none'],default='libavoid')
 p.add_argument('--previous',help='previous Escher OpenMetaMaps_<stem>_result.json whose reaction/metabolite positions are pinned as anchors')
 p.add_argument('--map',help='Escher OpenMetaMaps_<stem>_result.json for evaluate')
 p.add_argument('--reference',help='optional reference map for evaluate')
 p.add_argument('--compare-map',help='optional second result map JSON for reproducibility metrics')
 p.add_argument('--batch',help='evaluate batch manifest or folder')
 p.add_argument('--no-geometry',action='store_true',help='skip layout_quality final-geometry block in evaluate')
 args=p.parse_args()
 if args.stage=='compose':
  if not args.gem:p.error('compose needs --gem MODEL.xml (SBML or COBRA JSON)')
  import compose_cli;return compose_cli.run(args.gem,args.out or 'results/local_mvp',ec=args.ec,no_library=args.no_library,library=args.library,reaction_style=args.reaction_style,completion=args.completion,source_paths=args.source_paths,layout=args.layout,orientation=args.orientation,block_layout=args.block_layout,instances=args.instances,context=args.context,edge_routing=args.edge_routing,completion_context=args.completion_context,layout_style=args.layout_style,previous=args.previous)
 if args.stage=='evaluate':
  import evaluation.cli as evaluate_cli
  argv=[]
  if args.batch:argv+=['--batch',args.batch]
  if args.gem:argv+=['--gem',args.gem]
  if args.map:argv+=['--map',args.map]
  if args.reference:argv+=['--reference',args.reference]
  if args.compare_map:argv+=['--compare-map',args.compare_map]
  if args.out:argv+=['--out',args.out]
  if args.no_geometry:argv+=['--no-geometry']
  return evaluate_cli.main(argv)
 if args.stage=='tool':
  import tool_server;return tool_server.main(args.port,keep_running=args.keep_running)
 if args.stage=='routing':
  import routing;return routing.build()
 if args.stage=='catalog':
  import gallery;return gallery.make()
if __name__=='__main__':main()
