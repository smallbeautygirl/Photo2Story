#!/usr/bin/env python3
"""Figure 3-1: the four-stage Photo2Story pipeline, drawn in a plain academic style.

One layout definition, two renderers: SVG (vector, for the thesis document) and
PNG via Pillow (preview / fallback). Grayscale, thin strokes, no fills but white.
Run from the repo root:  .venv/bin/python docs/thesis/figures/make_fig3_1_pipeline.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_FAMILY = "'Noto Sans CJK TC', 'Noto Sans CJK SC', 'PingFang TC', 'Microsoft JhengHei', sans-serif"

W, H = 1140, 400
PNG_SCALE = 3
INK = "#000000"
MUTED = "#555555"
RULE = 1.2


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    title: str
    subtitle: str
    lines: tuple[str, ...]
    note: str = ""
    dashed: bool = False

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


BOXES = [
    Box(20, 110, 120, 120, "輸入", "", ("N 張照片", "主題脈絡（可選）", "繪本風格", "目標語言"), dashed=True),
    Box(180, 95, 170, 150, "Stage 0", "照片選取", ("CLIP 分群／LLM 選片", "VLM 英文描述", "依拍攝時間排序"), "實驗變因：選片模式、脈絡來源"),
    Box(540, 40, 180, 130, "Stage 1", "故事文字生成", ("先推論故事大綱（可關閉）", "逐頁生成 k 頁文字"), "實驗變因：因果推論開關"),
    Box(540, 230, 180, 130, "Stage 2", "插畫生成", ("FLUX.1-dev 文生圖", "風格 LoRA", "角色延續（文字／影像）"), "實驗變因：前頁插畫參考（規劃中）"),
    Box(800, 95, 170, 150, "Stage 3", "跨頁組裝", ("文字安全區偵測", "半透明墊底", "注音標註、PDF 輸出"), ""),
    Box(1010, 125, 110, 90, "輸出", "", ("繪本 PDF", "（k 個跨頁）"), dashed=True),
]
INPUT, S0, S1, S2, S3, OUTPUT = BOXES

# (from-box, to-box, label, dy_from, dy_to): edges leave the right side of `from` and enter
# the left side of `to`; the dy offsets keep forking / merging edges on distinct heights.
FORK = 28
EDGES = [
    (INPUT, S0, "", 0, 0),
    (S0, S1, "描述、主題", -FORK, 0),
    (S0, S2, "描述", FORK, 0),
    (S1, S3, "k 頁文字", 0, -FORK),
    (S2, S3, "k 張插畫", 0, FORK),
    (S3, OUTPUT, "", 0, 0),
]


def polyline(a: Box, b: Box, dy_from: float, dy_to: float) -> list[tuple[float, float]]:
    """Orthogonal route: out of a's right edge, one elbow at the midpoint, into b's left edge."""
    (x1, y1), (x2, y2) = (a.x + a.w, a.cy + dy_from), (b.x, b.cy + dy_to)
    if abs(y1 - y2) < 1:
        return [(x1, y1), (x2, y2)]
    xm = (x1 + x2) / 2
    return [(x1, y1), (xm, y1), (xm, y2), (x2, y2)]


def label_anchor(pts: list[tuple[float, float]]) -> tuple[float, float, str]:
    """Just above the final horizontal segment: centred when the segment is long enough,
    otherwise right-aligned at the arrowhead so the text runs back over the elbow."""
    (px, py), (qx, qy) = pts[-2], pts[-1]
    if qx - px >= 80:
        return (px + qx) / 2, qy - 6, "middle"
    return qx - 4, qy - 6, "end"


# ---------------------------------------------------------------- SVG
def svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT_FAMILY}">',
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='8' markerHeight='8' orient='auto-start-reverse'>"
        f"<path d='M 0 0 L 10 5 L 0 10 z' fill='{INK}'/></marker></defs>",
        f'<rect width="{W}" height="{H}" fill="white"/>',
    ]
    for a, b, label, dy_from, dy_to in EDGES:
        pts = polyline(a, b, dy_from, dy_to)
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f"<polyline points='{d}' fill='none' stroke='{INK}' stroke-width='{RULE}' marker-end='url(#arrow)'/>")
        if label:
            lx, ly, anchor = label_anchor(pts)
            parts.append(f"<text x='{lx:.1f}' y='{ly:.1f}' font-size='11' fill='{MUTED}' text-anchor='{anchor}'>{label}</text>")
    for box in BOXES:
        dash = " stroke-dasharray='4 3'" if box.dashed else ""
        parts.append(f"<rect x='{box.x}' y='{box.y}' width='{box.w}' height='{box.h}' fill='white' stroke='{INK}' stroke-width='{RULE}'{dash}/>")
        y = box.y + 22
        parts.append(f"<text x='{box.cx:.1f}' y='{y}' font-size='13' font-weight='bold' text-anchor='middle' fill='{INK}'>{box.title}</text>")
        if box.subtitle:
            y += 18
            parts.append(f"<text x='{box.cx:.1f}' y='{y}' font-size='13' text-anchor='middle' fill='{INK}'>{box.subtitle}</text>")
            y += 6
            parts.append(f"<line x1='{box.x + 14}' y1='{y}' x2='{box.x + box.w - 14}' y2='{y}' stroke='{INK}' stroke-width='0.8'/>")
        for line in box.lines:
            y += 19
            parts.append(f"<text x='{box.cx:.1f}' y='{y}' font-size='11.5' text-anchor='middle' fill='{INK}'>{line}</text>")
        if box.note:
            parts.append(f"<text x='{box.cx:.1f}' y='{box.y + box.h + 16}' font-size='10.5' text-anchor='middle' fill='{MUTED}'>{box.note}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- PNG (Pillow)
def _font(size: float) -> ImageFont.FreeTypeFont:
    px = int(round(size * PNG_SCALE))
    for index in (3, 2, 0):  # prefer the TC face; fall back to any face in the collection
        try:
            return ImageFont.truetype(FONT_PATH, px, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def _s(v: float) -> int:
    return int(round(v * PNG_SCALE))


def _arrowhead(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float) -> None:
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    size = 8
    p1 = (x2 - size * math.cos(ang - 0.4), y2 - size * math.sin(ang - 0.4))
    p2 = (x2 - size * math.cos(ang + 0.4), y2 - size * math.sin(ang + 0.4))
    draw.polygon([(_s(x2), _s(y2)), (_s(p1[0]), _s(p1[1])), (_s(p2[0]), _s(p2[1]))], fill=INK)


def _text(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, size: float, *, anchor: str, fill: str, bold: bool = False) -> None:
    font = _font(size)
    draw.text((_s(x), _s(y)), text, font=font, fill=fill, anchor=anchor)
    if bold:  # fake bold: overdraw with a 1px offset
        draw.text((_s(x) + 1, _s(y)), text, font=font, fill=fill, anchor=anchor)


def png() -> Image.Image:
    img = Image.new("RGB", (_s(W), _s(H)), "white")
    draw = ImageDraw.Draw(img)
    for a, b, label, dy_from, dy_to in EDGES:
        pts = polyline(a, b, dy_from, dy_to)
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            draw.line([(_s(x1), _s(y1)), (_s(x2), _s(y2))], fill=INK, width=_s(RULE))
        (px, py), (qx, qy) = pts[-2], pts[-1]
        _arrowhead(draw, px, py, qx, qy)
        if label:
            lx, ly, anchor = label_anchor(pts)
            _text(draw, lx, ly, label, 11, anchor={"middle": "ms", "end": "rs"}[anchor], fill=MUTED)
    for box in BOXES:
        rect = [_s(box.x), _s(box.y), _s(box.x + box.w), _s(box.y + box.h)]
        if box.dashed:
            _dashed_rect(draw, rect)
        else:
            draw.rectangle(rect, fill="white", outline=INK, width=_s(RULE))
        y = box.y + 22
        _text(draw, box.cx, y, box.title, 13, anchor="ms", fill=INK, bold=True)
        if box.subtitle:
            y += 18
            _text(draw, box.cx, y, box.subtitle, 13, anchor="ms", fill=INK)
            y += 6
            draw.line([(_s(box.x + 14), _s(y)), (_s(box.x + box.w - 14), _s(y))], fill=INK, width=_s(0.8))
        for line in box.lines:
            y += 19
            _text(draw, box.cx, y, line, 11.5, anchor="ms", fill=INK)
        if box.note:
            _text(draw, box.cx, box.y + box.h + 16, box.note, 10.5, anchor="ms", fill=MUTED)
    return img


def _dashed_rect(draw: ImageDraw.ImageDraw, rect: list[int]) -> None:
    x1, y1, x2, y2 = rect
    draw.rectangle(rect, fill="white")
    dash, gap = _s(4), _s(3)
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x1, y2, x2, y2), (x1, y1, x1, y2)):
        length = max(bx - ax, by - ay)
        pos = 0
        while pos < length:
            end = min(pos + dash, length)
            seg = ((ax + pos, ay), (ax + end, ay)) if ay == by else ((ax, ay + pos), (ax, ay + end))
            draw.line(list(seg), fill=INK, width=_s(RULE))
            pos += dash + gap


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fig3-1-pipeline.svg").write_text(svg(), encoding="utf-8")
    png().save(OUT_DIR / "fig3-1-pipeline.png", dpi=(96 * PNG_SCALE, 96 * PNG_SCALE))
    print("wrote", OUT_DIR / "fig3-1-pipeline.svg", "and .png")


if __name__ == "__main__":
    main()
