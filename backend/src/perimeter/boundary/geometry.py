"""Boundary geometry primitives (PLAN.md section 6.1).

Coordinates are stored **normalised to 0-1** so zones survive a resolution change or a
swap to the camera's low-resolution substream. Conversion to pixels happens at the last
possible moment.

Everything tests the **foot point** - the bottom-centre of a detection box - not the
centroid. Under perspective a tall person standing outside a floor zone has a centroid
inside it, which is the classic false positive in tripwire systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from shapely.geometry import LineString, Point, Polygon

# A tripwire shorter than this fraction of the frame diagonal is almost always a
# mis-click rather than an intended line.
MIN_TRIPWIRE_LENGTH = 0.05

# A polygon covering nearly the whole frame is legal but almost never intended, so the
# editor warns rather than rejecting.
SUSPICIOUS_AREA_FRACTION = 0.80


class Side(Enum):
    """Which side of a tripwire a point lies on.

    A sign flip between consecutive evaluations is a crossing, and the direction of the
    flip distinguishes ENTRY from EXIT.
    """

    A = "a"
    B = "b"
    ON = "on"  # exactly collinear - rare, treated as "no change" by the state machine


@dataclass(frozen=True)
class ValidationError:
    field: str
    message: str


def foot_point(box: np.ndarray) -> tuple[float, float]:
    """Bottom-centre of an xyxy box - the point every zone test uses."""
    x1, _, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y2))


def foot_points(boxes: np.ndarray) -> np.ndarray:
    """Vectorised foot points for an (N, 4) array of xyxy boxes."""
    if boxes.size == 0:
        return np.zeros((0, 2), np.float32)
    xs = (boxes[:, 0] + boxes[:, 2]) / 2.0
    return np.stack([xs, boxes[:, 3]], axis=1).astype(np.float32)


# A box whose bottom edge sits within this fraction of the frame's bottom is treated as
# cut off by the frame: the person's feet are below the image (a close-up webcam, or
# someone walking right under a ceiling camera), so the box's bottom edge is where the
# frame ends, not where they stand.
FRAME_EDGE_TOLERANCE = 0.02
# Where the foot point of such a cut-off box is placed instead, as a fraction of frame
# height up from the bottom. A zone drawn by hand "to the bottom of the frame" rarely
# reaches the exact edge - hand-placed corners land anywhere from 3-8% short - so testing
# the raw edge point would read as outside a zone the operator clearly meant to cover.
TRUNCATED_FOOT_INSET = 0.10


def clamp_truncated_feet(points: np.ndarray, frame_height: int) -> np.ndarray:
    """Pull the foot points of bottom-truncated boxes into the frame's bottom band.

    `points` are (N, 2) pixel foot points from `foot_points`. Points whose y lies within
    `FRAME_EDGE_TOLERANCE` of the bottom are moved up to `1 - TRUNCATED_FOOT_INSET` of
    the frame height; every other point is returned unchanged.
    """
    if points.size == 0 or frame_height <= 0:
        return points
    out = points.copy()
    truncated = out[:, 1] >= frame_height * (1.0 - FRAME_EDGE_TOLERANCE)
    out[truncated, 1] = frame_height * (1.0 - TRUNCATED_FOOT_INSET)
    return out


def to_pixels(points: list[list[float]], width: int, height: int) -> np.ndarray:
    """Normalised 0-1 points -> pixel coordinates."""
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 2), np.float64)
    return array * np.array([width, height], dtype=np.float64)


def to_normalised(points: np.ndarray, width: int, height: int) -> list[list[float]]:
    """Pixel coordinates -> normalised 0-1 points."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    array = np.asarray(points, dtype=np.float64) / np.array([width, height], dtype=np.float64)
    return [[float(x), float(y)] for x, y in array]


def build_polygon(points: list[list[float]], width: int, height: int) -> Polygon:
    return Polygon(to_pixels(points, width, height))


def point_in_polygon(point: tuple[float, float], polygon: Polygon) -> bool:
    """Containment test.

    `covers` rather than `contains`: a foot point landing exactly on the boundary counts
    as inside. With hysteresis in front of it the difference rarely matters, but a
    consistent choice avoids a track flickering when someone stands on the line.
    """
    return bool(polygon.covers(Point(point)))


def side_of_line(
    point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> Side:
    """Which side of the directed segment a->b the point lies on.

    Sign of the 2-D cross product. Side.A is the left-hand side when travelling a->b.
    """
    cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    if cross > 0:
        return Side.A
    if cross < 0:
        return Side.B
    return Side.ON


def side_of_polyline(point: tuple[float, float], vertices: np.ndarray) -> Side:
    """Side relative to a multi-segment tripwire.

    The nearest segment decides. A polyline drawn along an irregular fenceline has no
    single global "side", so the local one is the only meaningful answer.
    """
    if len(vertices) < 2:
        raise ValueError("a tripwire needs at least two vertices")
    if len(vertices) == 2:
        return side_of_line(point, tuple(vertices[0]), tuple(vertices[1]))

    target = Point(point)
    best_index, best_distance = 0, float("inf")
    for index in range(len(vertices) - 1):
        segment = LineString([vertices[index], vertices[index + 1]])
        distance = segment.distance(target)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return side_of_line(point, tuple(vertices[best_index]), tuple(vertices[best_index + 1]))


# -- validation -----------------------------------------------------------


def validate_points(points: list[list[float]]) -> list[ValidationError]:
    """Structural checks applied to every zone type."""
    errors: list[ValidationError] = []
    if not isinstance(points, list) or not points:
        return [ValidationError("points", "points must be a non-empty list")]

    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            errors.append(ValidationError("points", f"point {index} must be [x, y]"))
            continue
        x, y = point
        if not all(isinstance(v, (int, float)) for v in (x, y)):
            errors.append(ValidationError("points", f"point {index} must be numeric"))
        elif not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            errors.append(
                ValidationError(
                    "points",
                    f"point {index} = ({x}, {y}) is outside 0-1; coordinates are normalised",
                )
            )
    return errors


def validate_polygon(points: list[list[float]]) -> list[ValidationError]:
    """A polygon must be closeable and simple.

    Self-intersection is the check that matters: shapely accepts a bow-tie without
    complaint and then returns nonsense from containment tests. Catching it in the
    editor is far cheaper than debugging phantom events later.
    """
    errors = validate_points(points)
    if errors:
        return errors

    if len(points) < 3:
        return [ValidationError("points", "a polygon needs at least 3 points")]

    polygon = Polygon(points)
    if not polygon.is_valid:
        errors.append(
            ValidationError(
                "points",
                "polygon is self-intersecting; containment tests would be meaningless",
            )
        )
    elif polygon.area <= 0:
        errors.append(ValidationError("points", "polygon has zero area"))
    return errors


def validate_tripwire(points: list[list[float]]) -> list[ValidationError]:
    errors = validate_points(points)
    if errors:
        return errors

    if len(points) < 2:
        return [ValidationError("points", "a tripwire needs at least 2 points")]

    length = LineString(points).length  # in normalised units
    if length < MIN_TRIPWIRE_LENGTH:
        errors.append(
            ValidationError(
                "points",
                f"tripwire is only {length:.3f} of the frame; likely a mis-click "
                f"(minimum {MIN_TRIPWIRE_LENGTH})",
            )
        )
    return errors


def polygon_warnings(points: list[list[float]]) -> list[str]:
    """Non-fatal advisories surfaced by the editor."""
    warnings: list[str] = []
    if len(points) >= 3:
        polygon = Polygon(points)
        if polygon.is_valid and polygon.area > SUSPICIOUS_AREA_FRACTION:
            warnings.append(
                f"zone covers {polygon.area:.0%} of the frame - is that intended?"
            )
    return warnings
