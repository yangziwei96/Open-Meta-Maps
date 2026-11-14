"""Compound ELK placement, explicit chemical copies, truthful target-only exports."""
import subprocess,html
from itertools import combinations
from PIL import ImageFont
from jsonschema import Draft4Validator
from scipy.spatial import cKDTree
from core import *

def backend(payload,timeout=90):
 r=subprocess.run(['node',str(BASE/'runtime/layout_backend.cjs')],input=json.dumps(payload),text=True,capture_output=True,timeout=timeout,check=True);return json.loads(r.stdout)

def initial(g,anchors,seed=SEEDS[0]):
 rng=np.random.default_rng(seed);xy={k:list(v) for k,v in anchors.items() if k in g};offset=0
 for component in sorted(nx.connected_components(g),key=lambda c:sorted(c)):
  fixed=set(component)&set(xy)
  if fixed:
   queue=sorted(fixed);visited=set(fixed)
   for k in queue:
    for i,n in enumerate(sorted(g[k])):
     if n not in visited:xy[n]=(np.array(xy[k])+[70*math.cos(i*2.4),70*math.sin(i*2.4)]+rng.normal(0,2,2)).tolist();visited.add(n);queue.append(n)
  else:
   pos=nx.spring_layout(g.subgraph(component),seed=seed,iterations=50,scale=max(100,30*math.sqrt(len(component))))
   xy.update({k:(v+[offset,0]).tolist() for k,v in pos.items()});offset+=max(250,80*math.sqrt(len(component)))
 return xy

def complete(g,anchors,seed=SEEDS[0],iterations=250,method='fcose'):
 if any(isinstance(v,dict) for v in anchors.values()):
  import geometry;return geometry.complete_core(g,anchors,seed,iterations)
 xy=initial(g,anchors,seed)
 if method=='initial':return xy
 els=[dict(data=dict(id=k),position=dict(x=xy[k][0],y=xy[k][1])) for k in sorted(g)]
 els.extend(dict(data=dict(id='edge'+str(i),source=a,target=b)) for i,(a,b) in enumerate(sorted(g.edges())))
 result=backend(dict(backend='fcose',elements=els,anchors=[dict(nodeId=k,position=dict(x=v[0],y=v[1])) for k,v in anchors.items() if k in g],seed=seed,iterations=iterations))
 out={k:[v['x'],v['y']] for k,v in result.items()}
 assert all(np.linalg.norm(np.array(out[k])-v)<1e-5 for k,v in anchors.items() if k in g)
 return out

def pack_modules(model,chosen,bridge_list,physical_scale=False):
 metadata={m['id']:m for m in model['metabolites']};rlookup={r['id']:r for r in model['reactions']};groups={};owners=defaultdict(list);children=[]
 # Translation-only fit can leave tiny modules with huge module_scale (density/spacing blow-up).
 # Clamp each scale to [0.5, 2]× the median so curated continuity stays near the source spacing.
 raw_scales=[]
 if physical_scale:
  import geometry
  raw_scales=[geometry.module_scale(model,p) for p in chosen]
 med=float(np.median(raw_scales)) if raw_scales else 100.0
 for i,p in enumerate(chosen):
  scale=100
  if physical_scale:
   # Tight clamp: translation-only fit otherwise inflates tiny modules and breaks continuity.
   scale=float(np.clip(raw_scales[i], 0.75*med, 1.35*med))
  p['physical_scale']=scale
  xy={k:np.array(v)*scale for k,v in p['positions'].items()};arr=np.array(list(xy.values()));mn=arr.min(0);extent=np.ptp(arr,axis=0)+80;localpos={k:(v-mn+40).tolist() for k,v in xy.items()};roles=sorted({metadata[mid].get('compartment','unknown') for rid in p['reaction_ids'] for mid in rlookup[rid]['metabolites']});group='comp:'+('|'.join(roles));groupid='comp'+uid(roles)
  entry=groups.setdefault(groupid,dict(id=groupid,roles=roles,children=[]));ports=[];bid='block'+str(i)
  for k,v in localpos.items():
   if not k.startswith('m:'):continue
   x,y=v;side='WEST' if x<extent[0]/2 else 'EAST';pid=bid+':p'+str(len(ports));ports.append(dict(id=pid,width=2,height=2,x=0 if side=='WEST' else float(extent[0]),y=y,layoutOptions={'elk.port.side':side}));owners[k].append((bid,pid))
  entry['children'].append(dict(id=bid,width=float(max(extent[0],100)),height=float(max(extent[1],80)),ports=ports,layoutOptions={'elk.portConstraints':'FIXED_POS'}));p['local_positions']=localpos;p['block_id']=bid;p['container']=groupid
 # Junctions are explicit display connectors. Copies inside modules remain untouched.
 edges=[]
 for mid,os in sorted(owners.items()):
  if len(os)<2:continue
  roles=[metadata[mid[2:]].get('compartment','unknown')];gid='comp'+uid(roles);entry=groups.setdefault(gid,dict(id=gid,roles=roles,children=[]));j='junction'+uid(mid);entry['children'].append(dict(id=j,width=12,height=12))
  for bid,pid in os:edges.append(dict(id=uid([mid,pid]),sources=[pid],targets=[j]))
 root=dict(id='root',layoutOptions={'elk.algorithm':'layered','elk.direction':'RIGHT','elk.hierarchyHandling':'INCLUDE_CHILDREN','elk.edgeRouting':'ORTHOGONAL','elk.randomSeed':str(SEEDS[0]),'elk.spacing.nodeNode':'60','elk.layered.spacing.nodeNodeBetweenLayers':'100'},children=[dict(id=e['id'],children=e['children'],layoutOptions={'elk.algorithm':'layered','elk.padding':'[top=45,left=30,bottom=30,right=30]'}) for e in groups.values()],edges=edges)
 if not chosen:return {},[],[],[]
 result=backend(dict(backend='elk',graph=root));absolute={};containers=[]
 def visit(node,offset):
  xy=offset+np.array([node.get('x',0),node.get('y',0)]);absolute[node['id']]=xy
  for ch in node.get('children',[]):visit(ch,xy)
 for child in result['children']:
  visit(child,np.zeros(2));containers.append(dict(id=child['id'],roles=groups[child['id']]['roles'],x=child['x'],y=child['y'],width=child['width'],height=child['height']))
 reaction_anchors={};met_suggestions=defaultdict(list);blocks=[]
 for p in chosen:
  origin=absolute[p['block_id']];positions={k:(np.array(v)+origin).tolist() for k,v in p['local_positions'].items()}
  for k,v in positions.items():
   if k.startswith('r:'):reaction_anchors[k]=v
   else:met_suggestions[k].append(v)
  blk=dict(id=p['id'],template_id=p['template_id'],map_id=p['map_id'],lineage=p['lineage'],source=p['source'],species=p['species'],reaction_ids=p['reaction_ids'],positions=positions,node_mapping=p['node_mapping'],reaction_mapping=p['reaction_mapping'],match_level=p['match_level'],template_quality=p['template_quality'],match_mode=p['match_mode'],complete_equation_verified=p['complete_equation_verified'],complete_chemical_equation_verified=p.get('complete_chemical_equation_verified',p['complete_equation_verified']),container=p['container'],physical_scale=p['physical_scale'],anchor_confidence=.9 if p['complete_equation_verified'] else .7 if p['match_level']=='reaction_reference' else .35,movable=True)
  for key in ('source_xy','source_to_initial_frame','source_sha256','original_instance'):
   if key in p:blk[key]=p[key]
  blocks.append(blk)
 for mid,vals in met_suggestions.items():
  junction='junction'+uid(mid)
  # Shared identity is a separate derived junction, not the first source copy.
  reaction_anchors[mid]=(absolute[junction]+6).tolist() if junction in absolute else vals[0]
 return reaction_anchors,blocks,containers,result.get('edges',[])

FONT_PATH='/System/Library/Fonts/Menlo.ttc'
FONT=ImageFont.truetype(FONT_PATH,12) if Path(FONT_PATH).exists() else ImageFont.load_default(size=12)

def label_layout(positions,labels,avoid_nodes=False):
 boxes=[];offsets={};overlaps=0;grid=defaultdict(set)
 def cells(box):
  return [(x,y) for x in range(math.floor(box[0]/100),math.floor(box[2]/100)+1) for y in range(math.floor(box[1]/100),math.floor(box[3]/100)+1)]
 for k in sorted(positions):
  x,y=positions[k];b=FONT.getbbox(labels[k]);w,h=b[2]-b[0],b[3]-b[1];candidates=[]
  for dx,dy in [(9,-15),(9,20),(-w-9,-15),(-w-9,20),(9,-35),(9,40),(-w/2,-25),(-w/2,35)]:
   box=(x+dx,y+dy-h,x+dx+w,y+dy);near=set()
   for cell in cells(box):near.update(grid[cell])
   cost=sum(box[0]<boxes[j][2] and boxes[j][0]<box[2] and box[1]<boxes[j][3] and boxes[j][1]<box[3] for j in near)
   node_cost=sum(box[0]<v[0]+6 and box[2]>v[0]-6 and box[1]<v[1]+6 and box[3]>v[1]-6 for v in positions.values()) if avoid_nodes else 0
   candidates.append((cost+node_cost,abs(dx)+abs(dy),dx,dy,box))
  cost,_,dx,dy,box=min(candidates);index=len(boxes);boxes.append(box)
  for cell in cells(box):grid[cell].add(index)
  offsets[k]=[dx,dy];overlaps+=sum(box[0]<b[2] and b[0]<box[2] and box[1]<b[3] and b[1]<box[3] for b in boxes[:-1])
 if avoid_nodes:
  keys=sorted(positions);arr=np.array(boxes)
  for order in (list(reversed(range(len(keys)))),list(range(len(keys))),list(reversed(range(len(keys))))):
   for i in order:
    k=keys[i];x,y=positions[k];b=FONT.getbbox(labels[k]);w,h=b[2]-b[0],b[3]-b[1];options=[]
    for dx,dy in [(9,-15),(9,20),(-w-9,-15),(-w-9,20),(9,-35),(9,40),(-w/2,-25),(-w/2,35)]:
     box=np.array([x+dx,y+dy-h,x+dx+w,y+dy]);mask=(box[0]<arr[:,2])&(arr[:,0]<box[2])&(box[1]<arr[:,3])&(arr[:,1]<box[3]);mask[i]=False
     node_cost=sum(box[0]<v[0]+6 and box[2]>v[0]-6 and box[1]<v[1]+6 and box[3]>v[1]-6 for v in positions.values())
     options.append((int(mask.sum())+node_cost,abs(dx)+abs(dy),dx,dy,box))
    _,_,dx,dy,box=min(options,key=lambda c:c[:4]);offsets[k]=[dx,dy];arr[i]=box
  overlaps=sum(a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3] for i,a in enumerate(arr) for b in arr[i+1:])
 return offsets,int(overlaps)

_FULL_USES={}
def full_uses(model):
 key=id(model)
 if key not in _FULL_USES or _FULL_USES[key][0] is not model:
  uses=defaultdict(list)
  for r in model['reactions']:
   for mid in r['metabolites']:uses[mid].append(r['id'])
  _FULL_USES[key]=(model,uses)
 return _FULL_USES[key][1]

def crossings(xy,segments):
 keys=sorted(xy);ids={k:i for i,k in enumerate(keys)};arr=np.array([xy[k] for k in keys]);indices=np.array([[ids[a],ids[b]] for a,b in segments]);starts=arr[indices[:,0]];ends=arr[indices[:,1]];total=0
 def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
 for i in range(0,len(segments),128):
  aa=starts[i:i+128,None,:];bb=ends[i:i+128,None,:];cc=starts[None,:,:];dd=ends[None,:,:]
  mask=(cross(bb-aa,cc-aa)*cross(bb-aa,dd-aa)<0)&(cross(dd-cc,aa-cc)*cross(dd-cc,bb-cc)<0)
  ia=indices[i:i+128]
  shared=(ia[:,0,None]==indices[None,:,0])|(ia[:,0,None]==indices[None,:,1])|(ia[:,1,None]==indices[None,:,0])|(ia[:,1,None]==indices[None,:,1]);mask&=~shared;mask&=np.arange(len(segments))[None,:]>np.arange(i,i+len(ia))[:,None];total+=int(mask.sum())
 return total

def display_scene(model,corexy,blocks=()):
 cur=currency(model);positions={};participants={};labels={};copies={};byreaction={rid:b for b in blocks for rid in b['reaction_ids']}
 import orientation as orientation_module
 # Build keys first, then seed currency stubs with the adaptive spoke scale (median over reactions) so
 # display_scene / sides / separate share one rule while the pre-relax prior stays spatially coherent.
 for r in model['reactions']:
  rk='r:'+r['id'];positions[rk]=corexy[rk];labels[rk]=r['id'];b=byreaction.get(r['id'])
  for mid,coeff in r['metabolites'].items():
   mk='m:'+mid
   if mid in cur:
    key='side:'+r['id']+':'+mid;xy=list(corexy[rk])
   elif b and mk in b['positions']:
    key='copy:'+b['id']+':'+mid;xy=b['positions'][mk]
   else:key=mk;xy=corexy.get(mk,(np.array(corexy[rk])+[0,80]).tolist())
   positions[key]=xy;labels[key]=mid;participants[r['id'],mid]=key;copies[key]=mid
 params=orientation_module.reaction_spoke_params(model,dict(positions=positions,participants=participants),cur,blocks=blocks)
 for r in model['reactions']:
  # Cap initial stubs at SIDE_MAX so a later median-raised ceiling (applied in
  # orientation.sides) cannot inflate pre-compose bboxes.
  scale=min(params[r['id']]['distance'],orientation_module.SIDE_MAX)/orientation_module.SIDE_MAX;i=0
  here=np.array(corexy['r:'+r['id']],float)
  for mid in r['metabolites']:
   key=participants[r['id'],mid]
   if key.startswith('side:'):
    positions[key]=(here+[35*scale*(i%3-1),(45+28*(i//3))*scale]).tolist();i+=1
 return dict(positions=positions,labels=labels,copies=copies,participants=participants)

def export(model,corexy,folder,name,blocks=(),full_model=None,route=False,scene=None,label_obstacles=False):
 folder.mkdir(parents=True,exist_ok=True);cur=currency(model);rlookup={r['id']:r for r in model['reactions']};metlookup={m['id']:m for m in model['metabolites']}
 scene=scene if scene is not None else display_scene(model,corexy,blocks)
 positions={k:list(v) for k,v in scene['positions'].items()};labels=scene['labels'];copies=scene['copies'];participants=scene['participants']
 assert all(('r:'+r['id']) in positions and (r['id'],mid) in participants and participants[r['id'],mid] in positions for r in model['reactions'] for mid in r['metabolites'])
 mins=np.min(list(positions.values()),0);positions={k:(np.array(v)-mins+50).tolist() for k,v in positions.items()};offsets,overlap=label_layout(positions,labels,avoid_nodes=label_obstacles)
 label_min=np.min([[v[0]+offsets[k][0],v[1]+offsets[k][1]-14] for k,v in positions.items()],0);shift=np.maximum(20-label_min,0);positions={k:(np.array(v)+shift).tolist() for k,v in positions.items()}
 ids={k:str(i) for i,k in enumerate(sorted(positions))};nodes={}
 for k,v in positions.items():
  x,y=v
  if k.startswith('r:'):nodes[ids[k]]=dict(node_type='midmarker',x=x,y=y)
  else:nodes[ids[k]]=dict(node_type='metabolite',x=x,y=y,bigg_id=copies[k],name=metlookup.get(copies[k],{}).get('name',copies[k]),node_is_primary=copies[k] not in cur,label_x=x+offsets[k][0],label_y=y+offsets[k][1])
 reactions={};segments=[];original={};sid=0
 for i,r in enumerate(model['reactions']):
  rk='r:'+r['id'];x,y=positions[rk];sign=-1 if r['upper_bound']<=0 and r['lower_bound']<0 else 1;segs={}
  for mid,c in r['metabolites'].items():
   key=participants[r['id'],mid]
   for role_c in participation_roles(c):
    segs[str(sid)]=dict(from_node_id=ids[rk],to_node_id=ids[key],b1=None,b2=None);segments.append((key,rk) if role_c*sign<0 else (rk,key));sid+=1
  reactions[str(i)]=dict(bigg_id=r['id'],name=r.get('name') or r['id'],reversibility=r['lower_bound']<0<r['upper_bound'],label_x=x+offsets[rk][0],label_y=y+offsets[rk][1],gene_reaction_rule=r.get('gene_reaction_rule',''),genes=[],metabolites=[dict(bigg_id=mid,coefficient=c*sign) for mid,c in r['metabolites'].items()],segments=segs)
  original[r['id']]=dict(stoichiometry=r['metabolites'],lower_bound=r['lower_bound'],upper_bound=r['upper_bound'])
 values=np.array(list(positions.values()));label_max=np.max([[v[0]+offsets[k][0]+FONT.getlength(labels[k]),v[1]+offsets[k][1]+5] for k,v in positions.items()],0);width,height=(np.maximum(values.max(0),label_max)+[40,40]).tolist();data=[dict(schema='https://escher.github.io/escher/jsonschema/1-0-0#',homepage='https://escher.github.io',map_id=uid([model['id'],name,corexy]),map_name=model['id']+' '+name,map_description='v0.3; observed expert layouts, derived arrangements and algorithmic completion distinguished in skeleton manifest',metadata=dict(source_reactions=original,source_metabolites={k:{field:v for field,v in m.items() if field in ('id','name','compartment','chemicalFormula','charge')} for k,m in metlookup.items()},model_hash=uid(model),glyph_identity=copies,external_reactions={mid:[rid for rid in full_uses(full_model)[mid] if rid not in rlookup] for mid in metlookup} if full_model else {})),dict(nodes=nodes,reactions=reactions,text_labels={},canvas=dict(x=0,y=0,width=width,height=height))]
 Draft4Validator(read(ESCHER_SCHEMA)).validate(data);dump(folder/(name+'.json'),data)
 paths={};routing=dict(status='straight',method='straight')
 if route:
  try:
   import routing as routing_module
   obstacles=None
   if label_obstacles:
    obstacles=[]
    for k,(x,y) in positions.items():
     dx,dy=offsets[k];bbox=FONT.getbbox(labels[k]);obstacles.append([x+dx-2,y+dy-(bbox[3]-bbox[1])-2,x+dx+FONT.getlength(labels[k])+2,y+dy+2])
   paths=routing_module.avoid(positions,segments,obstacles=obstacles);routing=dict(status='ok',method='libavoid',label_obstacles=bool(label_obstacles),edges=len(paths),metrics=route_metrics(positions,segments,paths))
  except Exception as e:routing=dict(status='failed',method='straight_fallback',error=str(e))
 if paths:
  bend_count=0;edge_lookup=defaultdict(list)
  for i,(a,b) in enumerate(segments):edge_lookup[frozenset((a,b))].append(i)
  for r in model['reactions']:
   rk='r:'+r['id'];rx=next(v for v in reactions.values() if v['bigg_id']==r['id']);newsegments={};used=Counter()
   for mid,c in r['metabolites'].items():
    mk=participants[r['id'],mid];pair=frozenset((rk,mk))
    for _role_c in participation_roles(c):
     ei=edge_lookup[pair][used[pair]];used[pair]+=1;points=paths[ei]
     if segments[ei][0]!=rk:points=list(reversed(points))
     chain=[ids[rk]]
     for point in points[1:-1]:
      key='bend'+str(bend_count);bend_count+=1;nodes[key]=dict(node_type='multimarker',x=point[0],y=point[1]);chain.append(key)
     chain.append(ids[mk])
     for a,b in zip(chain,chain[1:]):newsegments['route'+str(len(newsegments))]=dict(from_node_id=a,to_node_id=b,b1=None,b2=None)
   rx['segments']=newsegments
  Draft4Validator(read(ESCHER_SCHEMA)).validate(data);dump(folder/(name+'.json'),data)
 svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0L8,4L0,8Z" fill="#70838c"/></marker></defs>']
 for i,(a,b) in enumerate(segments):
  points=paths.get(i,[positions[a],positions[b]]);d='M'+' L'.join(f'{p[0]},{p[1]}' for p in points);r=rlookup[(a if a.startswith('r:') else b)[2:]];marker='' if r['lower_bound']==r['upper_bound']==0 else ' marker-end="url(#a)"'
  if r['lower_bound']<0<r['upper_bound']:marker+=' marker-start="url(#a)"'
  svg.append(f'<path d="{d}" stroke="#a9b9be" fill="none"{marker}/>')
 for k,(x,y) in positions.items():
  color='#d77d38' if k.startswith('r:') else '#197d87';dx,dy=offsets[k];svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/><text x="{x+dx}" y="{y+dy}" font-family="Menlo,monospace" font-size="12">{html.escape(labels[k])}</text>')
 (folder/(name+'.svg')).write_text(''.join(svg)+'</svg>');dump(folder/(name+'.routing.json'),routing)
 crossing_count=crossings(positions,segments) if len(segments)<1200 else None
 return dict(reactions=len(reactions),participants=sid,escher_schema=True,label_overlap_pairs=overlap,node_overlap_pairs=len(cKDTree(values).query_pairs(10)),straight_crossings=crossing_count,routing=routing,display_glyphs=len(nodes),boundary_copy_count=sum(k.startswith('copy:') for k in positions),file=name+'.json')

def subset(model,ids):
 selected=set(ids);rx=[r for r in model['reactions'] if r['id'] in selected];mids={k for r in rx for k in r['metabolites']};return dict(model,reactions=rx,metabolites=[m for m in model['metabolites'] if m['id'] in mids])

def route_metrics(positions,edges,paths):
 pieces=[];owners=[];sourceids={k:i for i,k in enumerate(sorted(positions))};ends=[]
 for i,(a,b) in enumerate(edges):
  pts=paths.get(i,[positions[a],positions[b]])
  for p,q in zip(pts,pts[1:]):pieces.append((p,q));owners.append(i);ends.append((sourceids[a],sourceids[b]))
 if not pieces:return dict(proper_crossings=0,collinear_piece_overlap_pairs=0,bends=0)
 p=np.array(pieces);a=p[:,0];b=p[:,1];owners=np.array(owners);ends=np.array(ends);proper=0;overlap=0
 def cross(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]
 for i in range(0,len(p),128):
  aa=a[i:i+128,None];bb=b[i:i+128,None];cc=a[None];dd=b[None];mask=np.arange(len(p))[None,:]>np.arange(i,min(i+128,len(p)))[:,None];mask&=owners[i:i+128,None]!=owners[None,:]
  e=ends[i:i+128];mask&=~((e[:,0,None]==ends[None,:,0])|(e[:,0,None]==ends[None,:,1])|(e[:,1,None]==ends[None,:,0])|(e[:,1,None]==ends[None,:,1]))
  proper+=int(((cross(bb-aa,cc-aa)*cross(bb-aa,dd-aa)<0)&(cross(dd-cc,aa-cc)*cross(dd-cc,bb-cc)<0)&mask).sum())
  for axis in (0,1):
   other=1-axis;aligned=(np.abs(aa[...,axis]-bb[...,axis])<1e-6)&(np.abs(cc[...,axis]-dd[...,axis])<1e-6)&(np.abs(aa[...,axis]-cc[...,axis])<1e-6);length=np.minimum(np.maximum(aa[...,other],bb[...,other]),np.maximum(cc[...,other],dd[...,other]))-np.maximum(np.minimum(aa[...,other],bb[...,other]),np.minimum(cc[...,other],dd[...,other]));overlap+=int((aligned&(length>1e-6)&mask).sum())
 return dict(proper_crossings=proper,collinear_piece_overlap_pairs=overlap,bends=sum(max(0,len(points)-2) for points in paths.values()))


class CanonicalGraph(nx.Graph):
    """Keep seeded fallback layouts independent of Python set iteration."""
    def subgraph(self, nodes):
        keys=set(nodes)&set(self);out=CanonicalGraph()
        out.add_nodes_from((k,dict(self.nodes[k])) for k in sorted(keys))
        edges=sorted(tuple(sorted((a,b))) for a,b in self.edges() if a in keys and b in keys)
        out.add_edges_from((a,b,dict(self.edges[a,b])) for a,b in edges)
        return out


def clip_proposals(selected, sub):
    """Keep only the parts of selected proposals that fall inside a view subgraph."""
    import copy
    ids={r['id'] for r in sub['reactions']};g=model_graph(sub);out=[]
    for p in selected:
        rids=sorted(set(p['reaction_ids'])&ids)
        # Single leftover reactions are drawing units, not multi-reaction skeletons.
        if len(rids)<2:continue
        keys={'r:'+r for r in rids}
        for r in rids:keys.update(g['r:'+r])
        q=copy.deepcopy(p);q['reaction_ids']=rids;q['positions']={k:v for k,v in p['positions'].items() if k in keys}
        q['node_mapping']={k:v for k,v in p['node_mapping'].items() if v in q['positions']}
        q['reaction_mapping']={k:v for k,v in p['reaction_mapping'].items() if v in rids}
        q['mapping_evidence']=[e for e in p.get('mapping_evidence',[]) if e.get('target_reaction') in rids or e.get('target_reaction') is None]
        q['source_glyph_mapping']={k:v for k,v in p.get('source_glyph_mapping',{}).items() if 'm:'+v in q['positions']}
        q['view_clipping']=dict(original_proposal=p['id'],full_reaction_ids=p['reaction_ids'],visible_reaction_ids=rids)
        q['id']=uid([p['id'],'fixed-view',rids]);out.append(q)
    return out
