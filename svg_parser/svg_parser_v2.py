#!/usr/bin/env python3
from __future__ import annotations

"""
Supported primitives:
- LINE
- ARC
- POLYLINE
- TEXT
- ANNOTATION

Start:
    python svg_parser_v1.py input.svg -o output.json

"""

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]

def safe_float(x: Optional[str], default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x) if x is not None else default
    except Exception:
        return default

# Parse "rgb(r,g,b)" or "#rrggbb" into [r, g, b] list
def parse_rgb(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    value = value.strip()
    m = re.match(r"rgb\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", value)
    if m:
        return [int(float(m.group(1))), int(float(m.group(2))), int(float(m.group(3)))]
    if value.startswith("#") and len(value) == 7:
        return [int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)]
    return None

# compute bounding box [xmin, ymin, xmax, ymax] from a list of (x, y) points
def bbox_from_points(points: List[Tuple[float, float]]) -> List[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]

# extract viewBox as [vx, vy, vw, vh]
def extract_svg_viewbox(root: ET.Element) -> List[float]:
    vb = root.attrib.get("viewBox")
    if vb:
        nums = [float(x) for x in re.split(r"[ ,]+", vb.strip()) if x]
        if len(nums) == 4:
            return nums
    w = safe_float(root.attrib.get("width"), 100.0)
    h = safe_float(root.attrib.get("height"), 100.0)
    return [0.0, 0.0, w or 100.0, h or 100.0]

# tokenize SVG path data string into commands and numbers
def tokenize_path(d: str) -> List[str]:
    token_re = re.compile(r"[MLAZmlaz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    return token_re.findall(d)

# parse SVG path data string into a list of command dicts (MLAZ)
def parse_path_commands(d: str) -> List[Dict[str, Any]]:
    tokens = tokenize_path(d)
    i = 0
    current = None
    commands: List[Dict[str, Any]] = []

    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[MLAZmlaz]", tok): # capital letter and small letter of svg command
            current = tok
            i += 1
            if current in "Zz":
                commands.append({"cmd": "Z"})
                continue
        else:
            if current is None:
                raise ValueError("Path data missing initial command")

        cmd = current
        if cmd in "Mm":
            x = float(tokens[i])
            y = float(tokens[i + 1])
            i += 2
            commands.append({"cmd": "M", "x": x, "y": y})
            current = "L" if cmd == "M" else "l"
        elif cmd in "Ll":
            x = float(tokens[i])
            y = float(tokens[i + 1])
            i += 2
            commands.append({"cmd": "L", "x": x, "y": y})
        elif cmd in "Aa":
            rx = float(tokens[i])
            ry = float(tokens[i + 1])
            xrot = float(tokens[i + 2])
            laf = int(float(tokens[i + 3]))
            sf = int(float(tokens[i + 4]))
            x = float(tokens[i + 5])
            y = float(tokens[i + 6])
            i += 7
            commands.append(
                {
                    "cmd": "A",
                    "rx": rx,
                    "ry": ry,
                    "x_axis_rotation": xrot,
                    "large_arc": laf,
                    "sweep": sf,
                    "x": x,
                    "y": y,
                }
            )
        else:
            raise ValueError(f"unsupported path command: {cmd}")

    return commands

# compute signed angle from vector u to v
def signed_angle(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    dot = u[0] * v[0] + u[1] * v[1]
    lu = math.hypot(u[0], u[1])
    lv = math.hypot(v[0], v[1])
    if lu == 0 or lv == 0:
        return 0.0
    c = max(-1.0, min(1.0, dot / (lu * lv)))
    sign = 1.0 if (u[0] * v[1] - u[1] * v[0]) >= 0 else -1.0
    return sign * math.acos(c)

# convert svg elliptical arc endpoint representation to center representation
def svg_arc_to_center(
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    phi_deg: float,
    large_arc: int,
    sweep: int,
    x2: float,
    y2: float,
) -> Optional[Dict[str, Any]]:

    phi = math.radians(phi_deg % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dx2 = (x1 - x2) / 2.0
    dy2 = (y1 - y2) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    rx = abs(rx)
    ry = abs(ry)
    if rx == 0 or ry == 0:
        return None

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale

    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = 0.0 if den == 0 else math.sqrt(max(0.0, num / den))
    if large_arc == sweep:
        coef = -coef

    cxp = coef * (rx * y1p / ry)
    cyp = coef * (-ry * x1p / rx)

    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    v1 = ((x1p - cxp) / rx, (y1p - cyp) / ry)
    v2 = ((-x1p - cxp) / rx, (-y1p - cyp) / ry)

    theta1 = signed_angle((1.0, 0.0), v1)
    delta = signed_angle(v1, v2)

    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    theta2 = theta1 + delta

    return {
        "cx": cx,
        "cy": cy,
        "rx": rx,
        "ry": ry,
        "start_angle_rad": theta1,
        "end_angle_rad": theta2,
        "start_angle_deg": math.degrees(theta1),
        "end_angle_deg": math.degrees(theta2),
        "clockwise": bool(sweep),
        "x_axis_rotation_deg": phi_deg,
        "arc_length_approx": abs(delta) * (rx + ry) / 2.0,
    }

# estimate arc bounding box by sampling points along the arc
def estimate_arc_bbox(arc: Dict[str, Any], num_samples: int = 50) -> Tuple[List[float], List[Tuple[float, float]]]:
    pts: List[Tuple[float, float]] = []
    t1 = arc["start_angle_rad"]
    t2 = arc["end_angle_rad"]
    if t2 == t1:
        t2 = t1 + 2 * math.pi

    phi = math.radians(arc["x_axis_rotation_deg"])
    cphi = math.cos(phi)
    sphi = math.sin(phi)

    for i in range(num_samples + 1):
        t = t1 + (t2 - t1) * i / num_samples
        x = arc["cx"] + arc["rx"] * math.cos(t) * cphi - arc["ry"] * math.sin(t) * sphi
        y = arc["cy"] + arc["rx"] * math.cos(t) * sphi + arc["ry"] * math.sin(t) * cphi
        pts.append((x, y))

    return bbox_from_points(pts), pts

# convert integer number to English words
def number_to_words(n: int) -> str:
    small = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def under_1000(x: int) -> str:
        parts: List[str] = []
        if x >= 100:
            parts += [small[x // 100], "hundred"]
            x %= 100
        if x >= 20:
            parts.append(tens[x // 10])
            x %= 10
            if x:
                parts.append(small[x])
        elif x > 0 or not parts:
            parts.append(small[x])
        return " ".join(parts)

    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + number_to_words(-n)

    groups = [(10**9, "billion"), (10**6, "million"), (1000, "thousand")]
    parts: List[str] = []

    for value, name in groups:
        if n >= value:
            parts.append(under_1000(n // value))
            parts.append(name)
            n %= value

    if n:
        parts.append(under_1000(n))
    return " ".join(parts)

# normalize text
def normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = " ".join(s.split()).strip()
    return s if s else None


ANNOTATION_LAYER_HINTS = ("ANNOT", "DIM", "NOTE", "LABEL", "TEXT")


def infer_text_primitive_type(layer: Optional[str]) -> str:
    if not layer:
        return "TEXT"
    name = layer.upper()
    if any(token in name for token in ANNOTATION_LAYER_HINTS):
        return "ANNOTATION"
    return "TEXT"

# Chinese to English mapping, but just samples, not exhaustive
CN_TO_EN = {
    "门": "door",
    "窗": "window",
    "墙": "wall",
    "柱": "column",
    "梁": "beam",
    "出口": "exit",
    "楼梯": "stair",
    "备勤值班": "duty room",
    "烟道": "flue",
    "卫": "toilet",
    "水池": "sink",
    "消毒柜": "disinfection cabinet",
    "桌子": "table",
    "男": "male",
    "女": "female",
    "圆心": "center"
}

# convert raw text to English version
def english_text_version(raw: Optional[str]) -> Optional[str]:
    raw = normalize_text(raw)
    if raw is None:
        return None

    if re.fullmatch(r"[-+]?\d+", raw):
        return number_to_words(int(raw))

    if re.fullmatch(r"[-+]?(?:\d+\.\d+|\d+)", raw):
        if "." in raw:
            left, right = raw.split(".", 1)
            right_words = " ".join(number_to_words(int(ch)) for ch in right if ch.isdigit())
            return f"{number_to_words(int(left))} point {right_words}"
        return number_to_words(int(raw))

    if all(ord(c) < 128 for c in raw):
        return raw.lower()

    out = raw
    for k, v in CN_TO_EN.items():
        out = out.replace(k, v)
    return out

# parse style attribute string
def parse_style_attr(style_str: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not style_str:
        return out
    for item in style_str.split(";"):
        if ":" in item:
            k, v = item.split(":", 1)
            out[k.strip()] = v.strip()
    return out

# merge parent style with own style
def merge_styles(parent: Dict[str, Optional[str]], own: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    out = dict(parent)
    out.update({k: v for k, v in own.items() if v is not None})
    return out

# final style
def node_style(attrib: Dict[str, str], inherited: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    inline = parse_style_attr(attrib.get("style"))
    own = {
        "stroke": attrib.get("stroke", inline.get("stroke")),
        "stroke-width": attrib.get("stroke-width", inline.get("stroke-width")),
        "fill": attrib.get("fill", inline.get("fill")),
        "opacity": attrib.get("opacity", inline.get("opacity")),
        "font-family": attrib.get("font-family", inline.get("font-family")),
        "font-size": attrib.get("font-size", inline.get("font-size")),
        "transform": attrib.get("transform", inline.get("transform")),
    }
    return merge_styles(inherited, own)

# parse rotation angle in degrees from transform string
def parse_transform_rotation(transform_str: Optional[str]) -> Optional[float]:
    if not transform_str:
        return None

    m = re.search(r"rotate\(\s*([-+]?\d*\.?\d+)", transform_str)
    if m:
        return float(m.group(1))

    m = re.search(r"matrix\(([^)]+)\)", transform_str)
    if m:
        nums = [float(x) for x in re.split(r"[ ,]+", m.group(1).strip()) if x]
        if len(nums) >= 4:
            a, b, c, d = nums[:4]
            return math.degrees(math.atan2(b, a))

    return None

# main svg parser class
class SvgParser:
    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size
        self._pid = 0

    def next_id(self) -> str:
        self._pid += 1
        return f"p_{self._pid:06d}"

    def parse(self, svg_path: str | Path) -> Dict[str, Any]:
        svg_path = Path(svg_path)
        tree = ET.parse(svg_path)
        root = tree.getroot()
        viewbox = extract_svg_viewbox(root)

        primitives: List[Dict[str, Any]] = []
        self._walk(
            root,
            primitives,
            ctx={"style": {}, "group_id": None},
            viewbox=viewbox,
        )

        return {
            "source_file": str(svg_path),
            "viewBox": viewbox,
            "num_primitives": len(primitives),
            "primitives": primitives,
        }
    # extract primitives
    def _walk(
        self,
        elem: ET.Element,
        out: List[Dict[str, Any]],
        ctx: Dict[str, Any],
        viewbox: List[float],
    ) -> None:
        tag = strip_ns(elem.tag)
        style = node_style(elem.attrib, ctx["style"])
        group_id = ctx["group_id"]

        if tag == "g":
            group_id = elem.attrib.get("id") or group_id

        new_ctx = {
            "style": style,
            "group_id": group_id,
        }

        if tag == "path":
            prim = self._parse_path(elem, new_ctx, viewbox)
            if prim is not None:
                out.append(prim)

        elif tag == "text":
            prim = self._parse_text(elem, new_ctx, viewbox)
            if prim is not None:
                out.append(prim)

        for child in list(elem):
            self._walk(child, out, new_ctx, viewbox)
    # base primitive record
    def _base_record(
        self,
        typ: str,
        subtype: Optional[str],
        style: Dict[str, Any],
        position: Dict[str, Any],
        text: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "primitive_id": self.next_id(),
            "type": typ,
            "subtype": subtype,
            "geometry": {},
            "style": style,
            "position": position,
            "text": text,
            "meta": meta,
        }
    # style info
    def _style_record(self, style: Dict[str, Optional[str]], layer: Optional[str]) -> Dict[str, Any]:
        sw = safe_float(style.get("stroke-width"), None)
        return {
            "layer": layer,
            "stroke": style.get("stroke"),
            "stroke_width": sw,
            "fill": style.get("fill"),
            "opacity": safe_float(style.get("opacity"), 1.0),
        }
    # position info
    def _position_record(
        self,
        bbox: List[float],
        viewbox: List[float],
    ) -> Dict[str, Any]:
        xmin, ymin, xmax, ymax = bbox

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        return {
            "cx": cx,
            "cy": cy,
            "w": xmax - xmin,
            "h": ymax - ymin,
        }
    # empty text record
    def _empty_text(self) -> Dict[str, Any]:
        return {
            "normalized_text": None,
            "english_text": None,
            "font_family": None,
            "font_size": None,
        }
    # parse <path> element
    def _parse_path(
        self,
        elem: ET.Element,
        ctx: Dict[str, Any],
        viewbox: List[float],
    ) -> Optional[Dict[str, Any]]:
        d = elem.attrib.get("d")
        if not d:
            return None

        try:
            commands = parse_path_commands(d)
        except Exception:
            return None

        layer = ctx["group_id"]
        style_record = self._style_record(ctx["style"], layer)

        meta = {
            "parent_group_id": ctx["group_id"],
            "instance_id": elem.attrib.get("instance-id"),
            "semantic_id": elem.attrib.get("semantic-id"),
        }

        # line: M x,y  L x,y
        if len(commands) == 2 and commands[0]["cmd"] == "M" and commands[1]["cmd"] == "L":
            x1, y1 = commands[0]["x"], commands[0]["y"]
            x2, y2 = commands[1]["x"], commands[1]["y"]

            length = math.hypot(x2 - x1, y2 - y1)
            ang = math.atan2(y2 - y1, x2 - x1)
            bbox = bbox_from_points([(x1, y1), (x2, y2)])
            pos = self._position_record(bbox, viewbox)

            rec = self._base_record("LINE", None, style_record, pos, self._empty_text(), meta)
            rec["geometry"] = {
                "coords": [x1, y1, x2, y2, length, math.sin(ang), math.cos(ang), 0.0],
                "geom_mask": [1, 1, 1, 1, 1, 1, 1, 0],
            }
            return rec

        # polyline: M x,y  L x,y  L x,y ...
        if len(commands) >= 3 and commands[0]["cmd"] == "M" and all(c["cmd"] == "L" for c in commands[1:]):
            pts = [(commands[0]["x"], commands[0]["y"])] + [(c["x"], c["y"]) for c in commands[1:]]
            total = 0.0
            for a, b in zip(pts[:-1], pts[1:]):
                total += math.hypot(b[0] - a[0], b[1] - a[1])

            closed = pts[0] == pts[-1]
            bbox = bbox_from_points(pts)
            pos = self._position_record(bbox, viewbox)

            rec = self._base_record("POLYLINE", None, style_record, pos, self._empty_text(), meta)
            rec["geometry"] = {
                "coords": [
                    pts[0][0],
                    pts[0][1],
                    pts[-1][0],
                    pts[-1][1],
                    total,
                    float(len(pts)),
                    1.0 if closed else 0.0,
                    0.0,
                ],
                "geom_mask": [1, 1, 1, 1, 1, 1, 1, 0],
                "closed": closed,
            }
            return rec

        # arc: M x,y  A rx,ry rot large,sweep x,y 
        if len(commands) == 2 and commands[0]["cmd"] == "M" and commands[1]["cmd"] == "A":
            x1, y1 = commands[0]["x"], commands[0]["y"]
            a = commands[1]

            arc = svg_arc_to_center(
                x1,
                y1,
                a["rx"],
                a["ry"],
                a["x_axis_rotation"],
                a["large_arc"],
                a["sweep"],
                a["x"],
                a["y"],
            )
            if not arc:
                return None

            bbox, _ = estimate_arc_bbox(arc)
            pos = self._position_record(bbox, viewbox)

            rec = self._base_record("ARC", None, style_record, pos, self._empty_text(), meta)
            rec["geometry"] = {
                "coords": [
                    arc["cx"],
                    arc["cy"],
                    arc["rx"],
                    arc["ry"],
                    math.sin(arc["start_angle_rad"]),
                    math.cos(arc["start_angle_rad"]),
                    math.sin(arc["end_angle_rad"]),
                    math.cos(arc["end_angle_rad"]),
                ],
                "geom_mask": [1, 1, 1, 1, 1, 1, 1, 1],
                "arc_kind": "CIRCLE" if abs(arc["rx"] - arc["ry"]) < 1e-6 else "ELLIPSE",
                "clockwise": arc["clockwise"],
                "x_axis_rotation_deg": arc["x_axis_rotation_deg"],
            }
            return rec

        return None
    
    # <text> element
    def _parse_text(
        self,
        elem: ET.Element,
        ctx: Dict[str, Any],
        viewbox: List[float],
    ) -> Optional[Dict[str, Any]]:
        raw = "".join(elem.itertext())
        norm = normalize_text(raw)
        if norm is None:
            return None

        x = safe_float(elem.attrib.get("x"), 0.0)
        y = safe_float(elem.attrib.get("y"), 0.0)
        font_size = safe_float(elem.attrib.get("font-size", ctx["style"].get("font-size")), None)
        font_family = elem.attrib.get("font-family", ctx["style"].get("font-family"))
        rotation_deg = parse_transform_rotation(elem.attrib.get("transform") or ctx["style"].get("transform"))

        # bbox estimate for text
        width_est = max(len(norm), 1) * (font_size or 1.0) * 0.6
        height_est = font_size or 1.0
        bbox = [x, y - height_est, x + width_est, y]

        style_record = self._style_record(ctx["style"], ctx["group_id"])
        pos = self._position_record(bbox, viewbox)

        text_rec = {
            "normalized_text": norm,
            "english_text": english_text_version(norm),
            "font_family": font_family,
            "font_size": font_size,
        }

        meta = {
            "parent_group_id": ctx["group_id"],
            "instance_id": elem.attrib.get("instance-id"),
            "semantic_id": elem.attrib.get("semantic-id"),
        }

        rec = self._base_record(infer_text_primitive_type(ctx["group_id"]), None, style_record, pos, text_rec, meta)
        rr = math.radians(rotation_deg or 0.0)
        rec["geometry"] = {
            "coords": [x, y, width_est, height_est, math.sin(rr), math.cos(rr), font_size or 0.0, 0.0],
            "geom_mask": [1, 1, 1, 1, 1, 1, 1, 0],
        }
        return rec

# main function
def main() -> None:
    parser = argparse.ArgumentParser(description="Parse CAD-derived SVG into a canonical primitive schema.")
    parser.add_argument("svg_file", type=str, help="Input SVG file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--grid-size", type=int, default=16, help="Grid size for coarse position encoding")
    args = parser.parse_args()

    svg_parser = SvgParser(grid_size=args.grid_size)
    result = svg_parser.parse(args.svg_file)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.svg_file).with_suffix(".parsed.json")

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Parsed {result['num_primitives']} primitives")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
