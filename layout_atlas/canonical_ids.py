"""Map extension reaction references onto canonical chemical namespaces.

Human-maps store Human-GEM MAR ids under ``bigg.reaction``; this module moves the
native MAR id to ``human.gem`` and attaches BiGG / MetaNetX / Rhea when the
Human-GEM reactions.tsv crosswalk (CC BY 4.0) provides them.

PANTHER and Metabolism Regulation keep their native local ids and gain ``rhea``
(and ``ec-code`` when present) when participant ChEBI sets uniquely match a
Rhea equation in ``data/rhea/``.
"""
from __future__ import annotations

import csv
import datetime
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from core import DATA, INPUTS, dump, sha
from semantics import rhea_index

CROSSWALK_DIR = INPUTS / 'crosswalks'
HUMAN_GEM_TSV = CROSSWALK_DIR / 'Human-GEM-reactions.tsv'
HUMAN_GEM_JSON = CROSSWALK_DIR / 'human_gem_reaction_crosswalk.json'
HUMAN_GEM_SOURCE = CROSSWALK_DIR / 'Human-GEM-reactions.SOURCE.json'
HUMAN_GEM_MET_TSV = CROSSWALK_DIR / 'Human-GEM-metabolites.tsv'
HUMAN_GEM_MET_JSON = CROSSWALK_DIR / 'human_gem_metabolite_crosswalk.json'
HUMAN_GEM_MET_SOURCE = CROSSWALK_DIR / 'Human-GEM-metabolites.SOURCE.json'

CANONICAL_NS = ('bigg.reaction', 'metanetx.reaction', 'rhea')
NATIVE_NS = {
    'human_maps': 'human.gem',
    'panther': 'celldesigner.reaction',
    'metabolismregulation': 'sbgn.process',
}

CURRENCY_CHEBI = {
    '15377', '15378', '15379', '16236', '18009', '15422', '16761', '18367',
    '16474', '16908', '30616', '456216', '57287', '57288', '57945', '57925',
    '16027', '17877', '16311', '456215', '57540',
}


def _chebi_id(entity):
    if not entity:
        return None
    m = re.search(r'chebi[:_](\d+)', str(entity), re.I)
    return m.group(1) if m else None


def build_human_gem_crosswalk(tsv_path=None, out_path=None):
    """Compile Human-GEM reactions.tsv into a MAR → canonical refs JSON."""
    tsv_path = Path(tsv_path or HUMAN_GEM_TSV)
    out_path = Path(out_path or HUMAN_GEM_JSON)
    CROSSWALK_DIR.mkdir(parents=True, exist_ok=True)
    table = {}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            mar = (row.get('rxns') or '').strip()
            if not mar:
                continue
            refs = {}
            bigg = [x.strip() for x in re.split(r'[;,]', row.get('rxnBiGGID') or '') if x.strip()]
            mnx = [x.strip() for x in re.split(r'[;,]', row.get('rxnMetaNetXID') or '') if x.strip()]
            rhea = []
            for col in ('rxnRheaMasterID', 'rxnRheaID'):
                for part in re.split(r'[;,]', row.get(col) or ''):
                    part = part.strip()
                    if not part:
                        continue
                    rhea.append(part.split(':')[-1])
            if bigg:
                refs['bigg.reaction'] = sorted(set(bigg))
            if mnx:
                refs['metanetx.reaction'] = sorted(set(mnx))
            if rhea:
                refs['rhea'] = sorted(set(rhea))
            table[mar] = refs
    payload = dict(
        format_version=1,
        source='Human-GEM model/reactions.tsv',
        source_url='https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/v2.1.1/model/reactions.tsv',
        license='CC-BY-4.0',
        license_uri='https://creativecommons.org/licenses/by/4.0/',
        tsv_sha256=sha(tsv_path.read_bytes()),
        reactions=table,
    )
    dump(out_path, payload)
    dump(HUMAN_GEM_SOURCE, dict(
        source='Human-GEM',
        file='model/reactions.tsv',
        url=payload['source_url'],
        license=payload['license'],
        license_uri=payload['license_uri'],
        sha256=payload['tsv_sha256'],
        path=str(tsv_path.relative_to(DATA.parent)),
        access_date=datetime.date.today().isoformat(),
        note='Reaction identifier crosswalk used to attach BiGG/MetaNetX/Rhea to Human-maps MAR ids',
    ))
    return payload


def build_human_gem_metabolite_crosswalk(tsv_path=None, out_path=None):
    """Compile Human-GEM metabolites.tsv into a MAM → BiGG/ChEBI/MetaNetX JSON."""
    from core import chem, comp

    tsv_path = Path(tsv_path or HUMAN_GEM_MET_TSV)
    out_path = Path(out_path or HUMAN_GEM_MET_JSON)
    CROSSWALK_DIR.mkdir(parents=True, exist_ok=True)
    table = {}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            mam = (row.get('mets') or '').strip()
            if not mam:
                continue
            refs = {}
            bigg = [x.strip() for x in re.split(r'[;,]', row.get('metBiGGID') or '') if x.strip()]
            chebi = []
            for part in re.split(r'[;,]', row.get('metChEBIID') or ''):
                part = part.strip()
                if not part:
                    continue
                chebi.append(part.split(':')[-1])
            mnx = [x.strip() for x in re.split(r'[;,]', row.get('metMetaNetXID') or '') if x.strip()]
            seed = [x.strip() for x in re.split(r'[;,]', row.get('metSeedID') or '') if x.strip()]
            if bigg:
                refs['bigg.metabolite'] = sorted(set(bigg))
            if chebi:
                refs['chebi'] = sorted(set(chebi))
            if mnx:
                refs['metanetx.chemical'] = sorted(set(mnx))
            if seed:
                refs['seed.metabolite'] = sorted(set(seed))
            base = (row.get('metsNoComp') or '').strip() or chem(mam)
            table[mam] = dict(base=base, compartment=comp(mam), references=refs)
    payload = dict(
        format_version=1,
        source='Human-GEM model/metabolites.tsv',
        source_url='https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/v2.1.1/model/metabolites.tsv',
        license='CC-BY-4.0',
        license_uri='https://creativecommons.org/licenses/by/4.0/',
        tsv_sha256=sha(tsv_path.read_bytes()),
        metabolites=table,
    )
    dump(out_path, payload)
    dump(HUMAN_GEM_MET_SOURCE, dict(
        source='Human-GEM',
        file='model/metabolites.tsv',
        url=payload['source_url'],
        license=payload['license'],
        license_uri=payload['license_uri'],
        sha256=payload['tsv_sha256'],
        path=str(tsv_path.relative_to(DATA.parent)),
        access_date=datetime.date.today().isoformat(),
        note='Metabolite identifier crosswalk (MAM → BiGG/ChEBI/MetaNetX); Human-GEM is CC BY 4.0',
    ))
    return payload


def load_human_gem_crosswalk():
    if not HUMAN_GEM_JSON.exists():
        if not HUMAN_GEM_TSV.exists():
            raise FileNotFoundError(HUMAN_GEM_TSV)
        return build_human_gem_crosswalk()['reactions']
    return read_json(HUMAN_GEM_JSON)['reactions']


def read_json(path):
    return json.loads(Path(path).read_text())


def _rhea_chebi_index():
    rhea = rhea_index()
    by_exact = defaultdict(list)
    for rid, rx in rhea['reactions'].items():
        if not rx.get('complete'):
            continue
        chebs = []
        for mid, coeff in rx['coefficients'].items():
            c = _chebi_id(mid)
            if not c or c in CURRENCY_CHEBI:
                continue
            chebs.append((c, 1 if coeff > 0 else -1))
        if len({c for c, _ in chebs}) < 2:
            continue
        by_exact[frozenset(chebs)].append(str(rid))
        by_exact[frozenset(c for c, _ in chebs)].append(str(rid))
    return by_exact


def _reaction_chebi_signature(reaction, nodes_by_key=None):
    signed = []
    coeffs = reaction.get('coefficients') or {}
    if coeffs:
        for mid, coeff in coeffs.items():
            c = _chebi_id(mid)
            if not c or c in CURRENCY_CHEBI:
                continue
            signed.append((c, 1 if float(coeff) > 0 else -1))
    elif nodes_by_key:
        for e in reaction.get('edges') or []:
            pass
    return frozenset(signed), frozenset(c for c, _ in signed)


def match_rhea(reaction, index=None):
    index = index or _rhea_chebi_index()
    signed, unsigned = _reaction_chebi_signature(reaction)
    if len(unsigned) < 2:
        return []
    hits = index.get(signed) or []
    if not hits:
        hits = index.get(unsigned) or []
    # unique only
    uniq = sorted(set(hits))
    return uniq[:1] if len(uniq) == 1 else []


def _is_mar(value):
    return bool(re.match(r'^MAR\d+$', str(value or '')))


def _has_canonical(refs):
    return any(refs.get(ns) for ns in CANONICAL_NS if ns != 'bigg.reaction' or any(not _is_mar(v) for v in refs.get(ns) or []))


def measure_canonical_share(modules, *, source=None):
    """Share of reaction occurrences with a chemically usable canonical id."""
    n = 0
    hit = 0
    by_ns = Counter()
    for m in modules:
        if source and m.get('source') != source:
            continue
        for r in m.get('reactions') or []:
            n += 1
            refs = r.get('references') or {}
            ok = False
            for ns in CANONICAL_NS:
                vals = refs.get(ns) or []
                if ns == 'bigg.reaction':
                    vals = [v for v in vals if not _is_mar(v)]
                if vals:
                    by_ns[ns] += 1
                    ok = True
            if ok:
                hit += 1
    return dict(
        reactions=n,
        with_canonical=hit,
        share=(hit / n) if n else 0.0,
        by_namespace=dict(by_ns),
    )


def remap_human_maps_module(module, crosswalk):
    """Rewrite one Human-maps module reaction references in place."""
    changed = 0
    for r in module.get('reactions') or []:
        refs = dict(r.get('references') or {})
        native = []
        for v in refs.get('bigg.reaction') or []:
            if _is_mar(v):
                native.append(v)
        if _is_mar(r.get('id')):
            native.append(r['id'])
        native = sorted(set(native))
        if not native:
            continue
        # Move MAR out of bigg.reaction
        bigg_true = [v for v in (refs.get('bigg.reaction') or []) if not _is_mar(v)]
        refs['human.gem'] = native
        extras = {}
        for mar in native:
            for ns, vals in (crosswalk.get(mar) or {}).items():
                extras.setdefault(ns, [])
                extras[ns].extend(vals)
        for ns, vals in extras.items():
            merged = sorted(set(vals))
            if ns == 'bigg.reaction':
                bigg_true = sorted(set(bigg_true) | set(merged))
            else:
                refs[ns] = sorted(set(refs.get(ns) or []) | set(merged))
        if bigg_true:
            refs['bigg.reaction'] = bigg_true
        elif 'bigg.reaction' in refs:
            del refs['bigg.reaction']
        r['references'] = refs
        changed += 1
    return changed


def remap_local_module(module, *, rhea_index_map=None):
    """Attach unique Rhea ids to PANTHER / SBGN modules when equations match."""
    rhea_index_map = rhea_index_map or _rhea_chebi_index()
    changed = 0
    for r in module.get('reactions') or []:
        refs = dict(r.get('references') or {})
        # Preserve native namespace
        src = module.get('source')
        native_ns = NATIVE_NS.get(src)
        if native_ns and native_ns not in refs:
            native_val = r.get('id') or (refs.get(native_ns) or [None])[0]
            if native_val:
                refs[native_ns] = [native_val]
        hits = match_rhea(r, rhea_index_map)
        if hits:
            refs['rhea'] = sorted(set(refs.get('rhea') or []) | set(hits))
            changed += 1
        r['references'] = refs
    return changed


def patch_extension_modules(path, remap_fn, *, source_label):
    path = Path(path)
    with gzip.open(path, 'rt') as f:
        mods = json.load(f)
    before = measure_canonical_share(mods, source=None)
    n_changed = 0
    for m in mods:
        n_changed += remap_fn(m)
        # refresh reaction index fields used elsewhere
        m.setdefault('source', source_label)
    after = measure_canonical_share(mods, source=None)
    with gzip.open(path, 'wt') as f:
        json.dump(mods, f, ensure_ascii=False, separators=(',', ':'))
    return dict(path=str(path), modules=len(mods), reactions_touched=n_changed, before=before, after=after)


def run(human_maps=True, panther=True, metabolismregulation=True):
    report = dict(canonical_namespaces=list(CANONICAL_NS))
    if human_maps:
        if not HUMAN_GEM_JSON.exists():
            build_human_gem_crosswalk()
        crosswalk = load_human_gem_crosswalk()
        report['human_maps'] = patch_extension_modules(
            DATA / 'extensions/human_maps/modules.json.gz',
            lambda m: remap_human_maps_module(m, crosswalk),
            source_label='human_maps',
        )
    rhea_map = _rhea_chebi_index() if (panther or metabolismregulation) else None
    if panther:
        report['panther'] = patch_extension_modules(
            DATA / 'extensions/panther/modules.json.gz',
            lambda m: remap_local_module(m, rhea_index_map=rhea_map),
            source_label='panther',
        )
    if metabolismregulation:
        report['metabolismregulation'] = patch_extension_modules(
            DATA / 'extensions/metabolismregulation/modules.json.gz',
            lambda m: remap_local_module(m, rhea_index_map=rhea_map),
            source_label='metabolismregulation',
        )
    return report


if __name__ == '__main__':
    if not HUMAN_GEM_TSV.exists():
        raise SystemExit(f'missing {HUMAN_GEM_TSV}; download Human-GEM reactions.tsv first')
    build_human_gem_crosswalk()
    print(json.dumps(run(), indent=2))
