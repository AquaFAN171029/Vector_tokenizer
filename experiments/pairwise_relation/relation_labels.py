from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


PRIMITIVE_TYPES = ("LINE", "ARC", "POLYLINE")
TYPE_ORDER = {name: idx for idx, name in enumerate(PRIMITIVE_TYPES)}

TYPE_PAIR_RELATION_LABELS = {
    "LINE-LINE": ("connected", "intersect", "collinear_overlap", "perpendicular", "parallel", "near"),
    "LINE-ARC": ("connected", "intersect", "tangent", "near"),
    "LINE-POLYLINE": (
        "connected",
        "intersect",
        "overlap_segment",
        "perpendicular_to_segment",
        "parallel_to_segment",
        "near",
    ),
    "ARC-ARC": ("connected", "intersect", "overlap", "concentric", "tangent", "near"),
    "ARC-POLYLINE": ("connected", "intersect", "tangent_to_segment", "near"),
    "POLYLINE-POLYLINE": (
        "connected",
        "intersect",
        "overlap_segment",
        "perpendicular_segment",
        "parallel_segment",
        "containment",
        "near",
    ),
}

TYPE_PAIR_RELATION_TO_ID = {
    type_pair: {label: idx for idx, label in enumerate(labels)}
    for type_pair, labels in TYPE_PAIR_RELATION_LABELS.items()
}

# Backward-compatible names used by earlier LINE-LINE experiments.
RELATION_LABELS = ("parallel", "perpendicular", "joined", "none")
RELATION_TO_ID = {label: idx for idx, label in enumerate(RELATION_LABELS)}

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Segment = Tuple[Point, Point]


@dataclass(frozen=True)
class RelationThresholds:
    parallel_dot: float = 0.95
    perpendicular_dot: float = 0.15
    joined_diag_ratio: float = 0.01
    near_diag_ratio: float = 0.03
    overlap_diag_ratio: float = 0.003
    tangent_diag_ratio: float = 0.01
    concentric_diag_ratio: float = 0.01


def viewbox_diagonal(viewbox: List[float]) -> float:
    if not isinstance(viewbox, list) or len(viewbox) != 4:
        return 1.0
    try:
        return math.hypot(float(viewbox[2]), float(viewbox[3])) or 1.0
    except (TypeError, ValueError):
        return 1.0


def canonical_type_pair(type_a: str, type_b: str) -> Optional[str]:
    if type_a not in TYPE_ORDER or type_b not in TYPE_ORDER:
        return None
    if TYPE_ORDER[type_a] <= TYPE_ORDER[type_b]:
        return f"{type_a}-{type_b}"
    return f"{type_b}-{type_a}"


def add_label(labels: List[str], type_pair: str, label: str) -> None:
    if label in TYPE_PAIR_RELATION_TO_ID[type_pair] and label not in labels:
        labels.append(label)


def label_vector(type_pair: str, labels: Iterable[str]) -> List[int]:
    vector = [0] * len(TYPE_PAIR_RELATION_LABELS[type_pair])
    relation_to_id = TYPE_PAIR_RELATION_TO_ID[type_pair]
    for label in labels:
        vector[relation_to_id[label]] = 1
    return vector


def coords(primitive: Dict[str, Any]) -> List[float]:
    values = (primitive.get("geometry") or {}).get("coords") or []
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            return []
    return out


def primitive_bbox(primitive: Dict[str, Any]) -> Optional[BBox]:
    position = primitive.get("position") or {}
    try:
        cx = float(position.get("cx"))
        cy = float(position.get("cy"))
        w = abs(float(position.get("w") or 0.0))
        h = abs(float(position.get("h") or 0.0))
    except (TypeError, ValueError):
        return None
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def bbox_distance(a: BBox, b: BBox) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def bbox_overlaps(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def bbox_contains(a: BBox, b: BBox) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def line_endpoints(primitive: Dict[str, Any]) -> Optional[Segment]:
    if primitive.get("type") != "LINE":
        return None
    values = coords(primitive)
    if len(values) < 4:
        return None
    return (values[0], values[1]), (values[2], values[3])


def polyline_points(primitive: Dict[str, Any]) -> List[Point]:
    if primitive.get("type") != "POLYLINE":
        return []
    values = coords(primitive)
    points = []
    for idx in range(0, len(values) - 1, 2):
        points.append((values[idx], values[idx + 1]))
    return points


def segments_for(primitive: Dict[str, Any]) -> List[Segment]:
    endpoints = line_endpoints(primitive)
    if endpoints is not None:
        return [endpoints]
    points = polyline_points(primitive)
    return [(points[idx], points[idx + 1]) for idx in range(len(points) - 1)]


def segment_direction(segment: Segment) -> Optional[Point]:
    (x1, y1), (x2, y2) = segment
    norm = math.hypot(x2 - x1, y2 - y1)
    if norm == 0:
        return None
    return (x2 - x1) / norm, (y2 - y1) / norm


def direction_abs_dot_from_segments(a: Segment, b: Segment) -> Optional[float]:
    da = segment_direction(a)
    db = segment_direction(b)
    if da is None or db is None:
        return None
    return abs(max(-1.0, min(1.0, da[0] * db[0] + da[1] * db[1])))


def point_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def endpoint_distance(segments_a: Iterable[Segment], segments_b: Iterable[Segment]) -> Optional[float]:
    distances = [
        point_distance(pa, pb)
        for seg_a in segments_a
        for seg_b in segments_b
        for pa in seg_a
        for pb in seg_b
    ]
    return min(distances) if distances else None


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_on_segment(point: Point, segment: Segment, tolerance: float) -> bool:
    (x, y) = point
    (x1, y1), (x2, y2) = segment
    if min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance:
        return abs(orientation((x1, y1), (x2, y2), point)) <= tolerance
    return False


def segments_intersect(a: Segment, b: Segment, tolerance: float) -> bool:
    p1, p2 = a
    p3, p4 = b
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        point_on_segment(p3, a, tolerance)
        or point_on_segment(p4, a, tolerance)
        or point_on_segment(p1, b, tolerance)
        or point_on_segment(p2, b, tolerance)
    )


def point_segment_distance(point: Point, segment: Segment) -> float:
    (x, y) = point
    (x1, y1), (x2, y2) = segment
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return point_distance(point, (x1, y1))
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / denom))
    return point_distance(point, (x1 + t * dx, y1 + t * dy))


def segment_distance(a: Segment, b: Segment, tolerance: float) -> float:
    if segments_intersect(a, b, tolerance):
        return 0.0
    return min(
        point_segment_distance(a[0], b),
        point_segment_distance(a[1], b),
        point_segment_distance(b[0], a),
        point_segment_distance(b[1], a),
    )


def collinear_overlap(a: Segment, b: Segment, tolerance: float) -> bool:
    if abs(orientation(a[0], a[1], b[0])) > tolerance or abs(orientation(a[0], a[1], b[1])) > tolerance:
        return False
    return segment_distance(a, b, tolerance) <= tolerance


def arc_center_radius(primitive: Dict[str, Any]) -> Optional[Tuple[Point, float]]:
    if primitive.get("type") != "ARC":
        return None
    values = coords(primitive)
    if len(values) < 4:
        return None
    return (values[0], values[1]), (abs(values[2]) + abs(values[3])) / 2.0


def add_segment_relations(
    labels: List[str],
    type_pair: str,
    segments_a: List[Segment],
    segments_b: List[Segment],
    diag: float,
    thresholds: RelationThresholds,
) -> None:
    join_threshold = thresholds.joined_diag_ratio * diag
    near_threshold = thresholds.near_diag_ratio * diag
    overlap_threshold = thresholds.overlap_diag_ratio * diag

    end_dist = endpoint_distance(segments_a, segments_b)
    if end_dist is not None and end_dist <= join_threshold:
        add_label(labels, type_pair, "connected")

    dots = []
    min_segment_distance = None
    for seg_a in segments_a:
        for seg_b in segments_b:
            dist = segment_distance(seg_a, seg_b, overlap_threshold)
            min_segment_distance = dist if min_segment_distance is None else min(min_segment_distance, dist)
            if dist <= overlap_threshold:
                add_label(labels, type_pair, "intersect")
            if collinear_overlap(seg_a, seg_b, overlap_threshold):
                add_label(labels, type_pair, "collinear_overlap")
                add_label(labels, type_pair, "overlap_segment")
            dot = direction_abs_dot_from_segments(seg_a, seg_b)
            if dot is not None:
                dots.append(dot)

    if dots:
        if min(dots) < thresholds.perpendicular_dot:
            add_label(labels, type_pair, "perpendicular")
            add_label(labels, type_pair, "perpendicular_to_segment")
            add_label(labels, type_pair, "perpendicular_segment")
        if max(dots) > thresholds.parallel_dot:
            add_label(labels, type_pair, "parallel")
            add_label(labels, type_pair, "parallel_to_segment")
            add_label(labels, type_pair, "parallel_segment")

    if min_segment_distance is not None and min_segment_distance <= near_threshold:
        add_label(labels, type_pair, "near")


def add_arc_segment_relations(
    labels: List[str],
    type_pair: str,
    arc: Dict[str, Any],
    segmented: Dict[str, Any],
    diag: float,
    thresholds: RelationThresholds,
) -> None:
    circle = arc_center_radius(arc)
    segments = segments_for(segmented)
    if circle is None or not segments:
        return

    center, radius = circle
    near_threshold = thresholds.near_diag_ratio * diag
    tangent_threshold = thresholds.tangent_diag_ratio * diag
    min_delta = None
    min_distance = None
    for segment in segments:
        distance = point_segment_distance(center, segment)
        delta = abs(distance - radius)
        min_delta = delta if min_delta is None else min(min_delta, delta)
        min_distance = distance if min_distance is None else min(min_distance, distance)
        if delta <= tangent_threshold:
            add_label(labels, type_pair, "tangent")
            add_label(labels, type_pair, "tangent_to_segment")
        if distance < radius - tangent_threshold:
            add_label(labels, type_pair, "intersect")

    arc_box = primitive_bbox(arc)
    seg_box = primitive_bbox(segmented)
    if arc_box and seg_box and bbox_distance(arc_box, seg_box) <= near_threshold:
        add_label(labels, type_pair, "near")
    if min_delta is not None and min_delta <= near_threshold:
        add_label(labels, type_pair, "near")
    if min_distance is not None and min_distance <= radius + near_threshold:
        add_label(labels, type_pair, "near")


def add_arc_arc_relations(
    labels: List[str],
    type_pair: str,
    a: Dict[str, Any],
    b: Dict[str, Any],
    diag: float,
    thresholds: RelationThresholds,
) -> None:
    circle_a = arc_center_radius(a)
    circle_b = arc_center_radius(b)
    if circle_a is None or circle_b is None:
        return

    center_dist = point_distance(circle_a[0], circle_b[0])
    radius_a = circle_a[1]
    radius_b = circle_b[1]
    radius_delta = abs(radius_a - radius_b)
    tangent_threshold = thresholds.tangent_diag_ratio * diag

    if center_dist <= thresholds.concentric_diag_ratio * diag:
        add_label(labels, type_pair, "concentric")
        if radius_delta <= tangent_threshold:
            add_label(labels, type_pair, "overlap")

    if abs(center_dist - (radius_a + radius_b)) <= tangent_threshold or abs(center_dist - radius_delta) <= tangent_threshold:
        add_label(labels, type_pair, "tangent")

    if radius_delta - tangent_threshold < center_dist < radius_a + radius_b + tangent_threshold:
        add_label(labels, type_pair, "intersect")

    box_a = primitive_bbox(a)
    box_b = primitive_bbox(b)
    if box_a and box_b and bbox_distance(box_a, box_b) <= thresholds.near_diag_ratio * diag:
        add_label(labels, type_pair, "near")


def classify_line_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
    viewbox: List[float],
    thresholds: RelationThresholds = RelationThresholds(),
) -> Optional[str]:
    if a.get("type") != "LINE" or b.get("type") != "LINE":
        return None
    result = classify_pair(a, b, viewbox, thresholds)
    if result is None:
        return None
    _, labels = result
    if "connected" in labels:
        return "joined"
    if "perpendicular" in labels:
        return "perpendicular"
    if "parallel" in labels:
        return "parallel"
    return "none"


def classify_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
    viewbox: List[float],
    thresholds: RelationThresholds = RelationThresholds(),
) -> Optional[Tuple[str, Tuple[str, ...]]]:
    type_a = a.get("type")
    type_b = b.get("type")
    type_pair = canonical_type_pair(type_a, type_b)
    if type_pair is None:
        return None

    diag = viewbox_diagonal(viewbox)
    labels: List[str] = []

    segments_a = segments_for(a)
    segments_b = segments_for(b)
    if segments_a and segments_b:
        add_segment_relations(labels, type_pair, segments_a, segments_b, diag, thresholds)

    if type_pair in {"LINE-ARC", "ARC-POLYLINE"}:
        arc = a if type_a == "ARC" else b
        segmented = b if type_a == "ARC" else a
        add_arc_segment_relations(labels, type_pair, arc, segmented, diag, thresholds)

    if type_pair == "ARC-ARC":
        add_arc_arc_relations(labels, type_pair, a, b, diag, thresholds)

    box_a = primitive_bbox(a)
    box_b = primitive_bbox(b)
    if box_a and box_b:
        if bbox_overlaps(box_a, box_b):
            add_label(labels, type_pair, "overlap")
        if bbox_contains(box_a, box_b) or bbox_contains(box_b, box_a):
            add_label(labels, type_pair, "containment")
        if bbox_distance(box_a, box_b) <= thresholds.near_diag_ratio * diag:
            add_label(labels, type_pair, "near")

    return type_pair, tuple(labels)

