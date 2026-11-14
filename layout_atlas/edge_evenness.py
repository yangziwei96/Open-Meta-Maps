"""Core mid→metabolite chord lengths for edge-evenness targets.

The official layout-benchmark ``edge_length_deviation`` scores polyline lengths
between markers (Bézier handles ignored). Callers derive currency stub ceilings
from these lengths so algorithm-placed stubs track the same scale as core chords.
"""
from __future__ import annotations

import numpy as np

# Enough independent core chords for a stable median beyond one local neighbourhood.
MIN_CORE_SAMPLES = 8


def core_edge_lengths(model, scene, cur=None):
    """Euclidean midmarker→metabolite lengths for non-currency participants."""
    cur = cur or set()
    pos = scene.get('positions') or {}
    parts = scene.get('participants') or {}
    out = []
    for r in model['reactions']:
        rk = 'r:' + r['id']
        if rk not in pos:
            continue
        here = np.asarray(pos[rk], float)
        for mid in r['metabolites']:
            if mid in cur:
                continue
            key = parts.get((r['id'], mid))
            if not key or key not in pos or key.startswith('side:'):
                continue
            xy = np.asarray(pos[key], float)
            d = float(np.hypot(*(xy - here)))
            if d > 1e-6:
                out.append(d)
    return out


def median_core_edge(model, scene, cur=None, *, default):
    """Median core chord length, or ``default`` when fewer than MIN_CORE_SAMPLES."""
    lengths = core_edge_lengths(model, scene, cur=cur)
    if len(lengths) < MIN_CORE_SAMPLES:
        return float(default)
    return float(np.median(lengths))
