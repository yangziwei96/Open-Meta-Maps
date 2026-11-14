"""v0.3 shared contracts and independently stored source-observation inputs."""
from pathlib import Path
import hashlib,json,sys,re,os,sqlite3,math,gzip
from collections import defaultdict,Counter
import importlib.util
import numpy as np
import networkx as nx
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
EXAMPLES=ROOT/'examples';HUMAN_GEM=Path(os.environ.get('OPM_HUMAN_GEM',ROOT/'external/Human-GEM.xml'))  # reference models; Human-GEM is only needed to rebuild the identifier tables
DATA=ROOT/'data';OUT=BASE/'results';INPUTS=DATA/'source_inputs'
ESCHER_SCHEMA=DATA/'escher_schema/map-schema-1-0-0.json'  # the official Escher map schema (MIT, licence next to it) used to validate every exported map.json
if not (DATA/'atlas.sqlite').exists() and (DATA/'open_core.sqlite').exists():__import__('shutil').copyfile(DATA/'open_core.sqlite',DATA/'atlas.sqlite')  # fresh clone: default to the redistributable open core
sys.dont_write_bytecode=True
from model_io import load_model
SEEDS=[20261105,20261106,20261107]
CURRENCY={e['id'] for e in json.loads((DATA/'currency_metabolites.json').read_text())['currency']}  # BiGG ids of currency / cofactor metabolites (see the file for provenance)
# Shared example GEM list for bridge rebuilds / experiments (not model-specific layout rules)
EXAMPLE_MODELS=[('e_coli_core',EXAMPLES/'e_coli_core.xml','Escherichia coli'),('iML1515',EXAMPLES/'iML1515.xml','Escherichia coli'),
                ('iMM904',EXAMPLES/'iMM904.xml','Saccharomyces cerevisiae'),('Human-GEM',HUMAN_GEM,'Homo sapiens'),
                ('iYO844',DATA/'models/iYO844.xml','Bacillus subtilis'),('AraCore_v2_1',DATA/'models/AraCore_v2_1.xml','Arabidopsis thaliana')]
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def uid(v):return sha(json.dumps(v,sort_keys=True,ensure_ascii=False).encode())[:24]
def local(e):return e.tag.rsplit('}',1)[-1]
def freeze():
 dest=DATA/'input_manifest.json'
 if dest.exists():return read(dest)
 result=dict(version='0.3-source-inputs',files={str(p.relative_to(INPUTS)):sha(p.read_bytes()) for p in sorted(INPUTS.rglob('*')) if p.is_file()})
 dump(dest,result);return result

def source_module_records():
 with gzip.open(INPUTS/'module_observations.json.gz','rt') as f:return json.load(f)
def source_modules(quality=None):return [r['observation'] for r in source_module_records() if quality is None or r['observation']['quality']==quality]
def source_maps():
 with gzip.open(INPUTS/'map_observations.json.gz','rt') as f:return json.load(f)
def chem(mid):return mid.split('@')[0] if '@' in mid else re.sub(r'_[a-z][0-9]?$','',mid)
def comp(mid):return mid.split('@')[-1] if '@' in mid else (re.search(r'_([a-z][0-9]?)$',mid)[1] if re.search(r'_([a-z][0-9]?)$',mid) else 'unknown')

def reaction_kind(reaction):
 """Classify a GEM reaction as exchange, transport or metabolic."""
 mets=list((reaction.get('metabolites') or {}))
 if len(mets)==1:return 'exchange'
 if len({comp(m) for m in mets})>1:return 'transport'
 return 'metabolic'

def xrefs(annotations):
 out=defaultdict(set)
 for a in annotations:
  if not a.get('qualifier','').endswith('}is'):continue
  u=a.get('uri','');m=re.search(r'/(chebi|bigg.metabolite|rhea|reactome|bigg.reaction|metanetx.chemical|metanetx.reaction)[/:](?:CHEBI[:_])?([^/?#]+)',u,re.I)
  if m:out[m[1].lower()].add(m[2])
 return {k:sorted(v) for k,v in out.items()}
def currency(model):
 out=set();names={'water','h2o','proton','h+','atp','adp','amp','nad+','nadh','nadp+','nadph','coenzyme a','coa','phosphate','diphosphate'}
 for m in model['metabolites']:
  if chem(m['id']) in CURRENCY or m.get('name','').lower() in names or set(xrefs(m.get('annotations',[])).get('bigg.metabolite',[]))&CURRENCY:out.add(m['id'])
 return out

def participation_roles(coefficient):
 """Signed edge roles for one net stoichiometric coefficient.

 A species on both sides with cancelling amounts is stored as 0 and draws as
 both substrate and product (gross participation, not net stoichiometry).
 """
 c=float(coefficient)
 if c:return (c,)
 return (-1.0,1.0)

def model_graph(model):
 g=nx.Graph();cur=currency(model);met={m['id']:m for m in model['metabolites']}
 for r in model['reactions']:
  rk='r:'+r['id'];g.add_node(rk,role='reaction',reversible=r['lower_bound']<0<r['upper_bound'],compartment='|'.join(sorted({met[m].get('compartment','unknown') for m in r['metabolites']})))
  for mid,c in r['metabolites'].items():
   if mid in cur:continue
   mk='m:'+mid;g.add_node(mk,role='metabolite',compartment=met[mid].get('compartment','unknown'))
   # Undirected graph: one edge; coefficient 0 means dual participation (both roles).
   role='substrate' if c<0 else ('product' if c>0 else 'both')
   g.add_edge(rk,mk,coefficient=float(c),role=role)
 return g

class CanonicalGraph(nx.Graph):
    """Keep seeded fallback layouts independent of Python set iteration order."""
    def subgraph(self, nodes):
        keys = set(nodes) & set(self)
        out = CanonicalGraph()
        out.add_nodes_from((k, dict(self.nodes[k])) for k in sorted(keys))
        edges = sorted(tuple(sorted((a, b))) for a, b in self.edges() if a in keys and b in keys)
        out.add_edges_from((a, b, dict(self.edges[a, b])) for a, b in edges)
        return out


def normalize(xy):
 arr=np.array(list(xy.values()),float);center=arr.mean(0);sc=max(float(np.sqrt(np.mean(np.sum((arr-center)**2,1)))),1e-8);return {k:((np.array(v)-center)/sc).tolist() for k,v in xy.items()}
def fit_frame(source, target, rotate=False):
 """Fit source points onto target centroids.

 Default (rotate=False): uniform scale + translation only — curated axes are preserved
 (no rotation, no reflection).  rotate=True restores the old Kabsch rotation for callers
 that score geometric similarity (e.g. database matching); reflection is still forbidden.
 """
 a = np.asarray(source, float)
 b = np.asarray(target, float)
 ac, bc = a - a.mean(0), b - b.mean(0)
 var = float(np.sum(ac * ac))
 if rotate:
  u, _, vt = np.linalg.svd(ac.T @ bc)
  rot = u @ vt
  # Forbid reflection: force det(rot) = +1
  if np.linalg.det(rot) < 0:
   u = u.copy()
   u[:, -1] *= -1
   rot = u @ vt
  scale = max(float(np.sum((ac @ rot) * bc) / max(var, 1e-12)), 1e-8)
 else:
  rot = np.eye(2)
  # Isotropic LS scale with axes fixed (no rotation).
  scale = max(float(np.sqrt(np.sum(bc * bc) / max(var, 1e-12))), 1e-8)
 return dict(rotation=rot.tolist(), scale=scale, source_center=a.mean(0).tolist(),
             target_center=b.mean(0).tolist(), rotate=bool(rotate), reflect=False)
def transform(xy,frame):return {k:((np.array(v)-frame['source_center'])@np.array(frame['rotation'])*frame['scale']+frame['target_center']).tolist() for k,v in xy.items()}


def unify_map_scale(frames):
 """One common scale for a list of fit frames (median of positive scales)."""
 scales = [float(f['scale']) for f in frames if f and float(f.get('scale') or 0) > 0]
 if not scales:
  return 1.0
 return float(np.median(scales))
