#!/usr/bin/env python3
"""Crop an Open Meta Maps Escher result JSON to a preview PNG.

White background, content-tight crop, every reaction and metabolite label
of the result SVG drawn. The default width keeps the smallest label at least
MIN_LABEL_PX tall; strokes are widened when they would drop below
TARGET_STROKE px.

  PYTHONPATH=layout_atlas python3 tools/render_map_preview.py OpenMetaMaps_<stem>_result.json out.png
"""
from __future__ import annotations

import argparse
import io
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'layout_atlas'))

from core import read  # noqa: E402
import line_render  # noqa: E402

# Sizes in output pixels. SMALLEST_FONT is the smallest label font of
# line_render in map units.
TARGET_STROKE = 1.2
MIN_LABEL_PX = 8.0
SMALLEST_FONT = 10.5
CROP_PAD_FRAC = 0.03
MIN_WIDTH = 2800
MAX_WIDTH = 8000
MAX_BYTES = 5_000_000


def _content_bbox(data):
    body = data[1]
    nodes = body['nodes']
    xs, ys = [], []
    for n in nodes.values():
        xs.append(float(n['x']))
        ys.append(float(n['y']))
        if n.get('label_x') is not None:
            xs.append(float(n['label_x']))
        if n.get('label_y') is not None:
            ys.append(float(n['label_y']))
    for rx in body.get('reactions', {}).values():
        if rx.get('label_x') is not None:
            xs.append(float(rx['label_x']))
        if rx.get('label_y') is not None:
            ys.append(float(rx['label_y']))
        for seg in (rx.get('segments') or {}).values():
            for key in ('b1', 'b2'):
                pt = seg.get(key)
                if pt and 'x' in pt and 'y' in pt:
                    xs.append(float(pt['x']))
                    ys.append(float(pt['y']))
    if not xs:
        canvas = body.get('canvas') or {}
        return 0.0, 0.0, float(canvas.get('width') or 1000), float(canvas.get('height') or 1000)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = CROP_PAD_FRAC * max(xmax - xmin, ymax - ymin, 1.0)
    return xmin - pad, ymin - pad, xmax + pad, ymax + pad


def _scale_svg(svg: str, mul: float) -> str:
    """Multiply stroke, radius and marker sizes so strokes hit the target pixel size."""

    def num(attr, x):
        return f'{attr}="{float(x.group(1)) * mul:.3f}"'

    svg = re.sub(r'stroke-width="([0-9.]+)"', lambda m: num('stroke-width', m), svg)
    svg = re.sub(r'\br="([0-9.]+)"', lambda m: num('r', m), svg)
    svg = re.sub(r'markerWidth="([0-9.]+)"', lambda m: num('markerWidth', m), svg)
    svg = re.sub(r'markerHeight="([0-9.]+)"', lambda m: num('markerHeight', m), svg)
    return svg


def _split_label_halos(svg: str) -> str:
    """Draw each label's halo and its glyphs as two texts; cairosvg ignores paint-order."""

    def split(m):
        attrs, body = m.group(1), m.group(2)
        glyphs = re.sub(r'\s(stroke|stroke-width|paint-order)="[^"]*"', '', attrs)
        return f'<text{attrs}>{body}</text><text{glyphs}>{body}</text>'

    return re.sub(r'<text\b([^>]*paint-order="stroke"[^>]*)>(.*?)</text>', split, svg, flags=re.DOTALL)


def _set_viewbox(svg: str, bbox, width: int, height: int) -> str:
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    new = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} {bw:.2f} {bh:.2f}" width="{width}" height="{height}">'
    return re.sub(r'<svg\b[^>]*>', new, svg, count=1)


def default_width(bbox) -> int:
    """Width at which the smallest label is MIN_LABEL_PX tall, within MIN_WIDTH..MAX_WIDTH."""
    bw = max(bbox[2] - bbox[0], 1.0)
    return min(MAX_WIDTH, max(MIN_WIDTH, math.ceil(bw * MIN_LABEL_PX / SMALLEST_FONT)))


def render_preview(map_path: Path, out_path: Path, width: int | None = None, max_bytes: int = MAX_BYTES) -> Path:
    from PIL import Image
    import cairosvg

    data = read(map_path)
    if not isinstance(data, list):
        import json
        data = json.loads(Path(map_path).read_text())
    svg = line_render.render(data)
    x0, y0, x1, y1 = _content_bbox(data)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    if width is None:
        width = default_width((x0, y0, x1, y1))
    height = max(1, int(round(width * bh / bw)))
    scale = width / bw  # map units -> output pixels
    # Keep map-unit sizes unless a stroke would drop below TARGET_STROKE px.
    mul = max(1.0, TARGET_STROKE / (2.4 * scale))
    svg = _scale_svg(svg, mul)
    svg = _split_label_halos(svg)
    svg = _set_viewbox(svg, (x0, y0, x1, y1), width, height)
    # Force the background rect to the viewBox.
    svg = re.sub(
        r'<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bw:.2f}" height="{bh:.2f}" fill="white"/>',
        svg,
        count=1,
    )
    png = cairosvg.svg2png(bytestring=svg.encode('utf-8'), output_width=width, output_height=height)
    im = Image.open(io.BytesIO(png)).convert('RGB')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format='PNG', optimize=True)
    q_w = width
    while out_path.stat().st_size > max_bytes and q_w > MIN_WIDTH:
        q_w = max(MIN_WIDTH, int(q_w * 0.9))
        q_h = max(1, int(round(q_w * bh / bw)))
        im.resize((q_w, q_h), Image.Resampling.LANCZOS).save(out_path, format='PNG', optimize=True)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('map', type=Path, help='Escher OpenMetaMaps_<stem>_result.json')
    p.add_argument('out', type=Path, help='output PNG')
    p.add_argument('--width', type=int, default=None, help='output width in px (default: smallest label MIN_LABEL_PX tall)')
    p.add_argument('--max-bytes', type=int, default=MAX_BYTES)
    args = p.parse_args()
    path = render_preview(args.map, args.out, width=args.width, max_bytes=args.max_bytes)
    from PIL import Image
    im = Image.open(path)
    print(f'{path} {im.size[0]}x{im.size[1]} {path.stat().st_size} bytes')


if __name__ == '__main__':
    main()
