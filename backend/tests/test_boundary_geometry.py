"""Tests for boundary geometry primitives (PLAN.md section 6.1)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from perimeter.boundary.geometry import (
    TRUNCATED_FOOT_INSET,
    Side,
    clamp_truncated_feet,
    foot_point,
    foot_points,
    point_in_polygon,
    side_of_line,
    side_of_polyline,
    to_normalised,
    to_pixels,
    validate_polygon,
    validate_tripwire,
)

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


# -- foot point -----------------------------------------------------------


def test_foot_point_is_bottom_centre():
    assert foot_point(np.array([100.0, 50.0, 200.0, 450.0])) == (150.0, 450.0)


def test_foot_point_is_not_the_centroid():
    """Under perspective a tall person outside a floor zone has a centroid inside it."""
    box = np.array([100.0, 50.0, 200.0, 450.0])
    centroid_y = (box[1] + box[3]) / 2
    assert foot_point(box)[1] != centroid_y


def test_foot_points_vectorised_matches_scalar():
    boxes = np.array([[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 15.0, 35.0]])
    assert foot_points(boxes).tolist() == [[5.0, 20.0], [10.0, 35.0]]


def test_foot_points_empty():
    assert foot_points(np.zeros((0, 4))).shape == (0, 2)


def test_truncated_foot_is_pulled_into_the_bottom_band():
    # A close-up webcam subject: box runs to the bottom of a 720px frame.
    points = foot_points(np.array([[354.0, 228.0, 1043.0, 715.0]]))
    clamped = clamp_truncated_feet(points, 720)
    assert clamped[0, 0] == points[0, 0]
    assert clamped[0, 1] == pytest.approx(720 * (1 - TRUNCATED_FOOT_INSET))


def test_truncated_foot_lands_inside_a_zone_drawn_near_the_bottom_edge():
    # The zone stops a few percent short of the bottom edge, as hand-drawn zones do.
    zone = Polygon([(58, 176), (1279, 150), (1257, 698), (123, 699)])
    points = foot_points(np.array([[354.0, 228.0, 1043.0, 715.0]]))
    assert not point_in_polygon(tuple(points[0]), zone)
    assert point_in_polygon(tuple(clamp_truncated_feet(points, 720)[0]), zone)


def test_feet_visible_in_frame_are_not_moved():
    points = foot_points(np.array([[100.0, 100.0, 200.0, 600.0]]))
    assert clamp_truncated_feet(points, 720).tolist() == points.tolist()


def test_clamp_truncated_feet_empty():
    assert clamp_truncated_feet(np.zeros((0, 2), np.float32), 720).shape == (0, 2)


# -- normalisation --------------------------------------------------------


def test_to_pixels_scales_by_frame_size():
    assert to_pixels([[0.5, 0.5]], 640, 480).tolist() == [[320.0, 240.0]]


def test_normalisation_round_trip_survives_resolution_change():
    """The reason coordinates are normalised: zones must survive a substream swap."""
    pixels_1080 = to_pixels(SQUARE, 1920, 1080)
    back = to_normalised(pixels_1080, 1920, 1080)
    assert np.allclose(back, SQUARE)

    pixels_360 = to_pixels(back, 640, 360)
    assert pixels_360.tolist() == to_pixels(SQUARE, 640, 360).tolist()


def test_to_normalised_rejects_zero_dimensions():
    with pytest.raises(ValueError):
        to_normalised(np.array([[1.0, 1.0]]), 0, 100)


# -- containment ----------------------------------------------------------


def test_point_in_polygon_inside_and_outside():
    polygon = Polygon(to_pixels(SQUARE, 100, 100))
    assert point_in_polygon((50.0, 50.0), polygon)
    assert not point_in_polygon((5.0, 5.0), polygon)


def test_point_exactly_on_boundary_counts_as_inside():
    """Consistency matters more than the choice: it stops a track flickering on the line."""
    polygon = Polygon(to_pixels(SQUARE, 100, 100))
    assert point_in_polygon((20.0, 50.0), polygon)


# -- tripwire sides -------------------------------------------------------


def test_side_of_line_opposite_sides():
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert side_of_line((5.0, 5.0), a, b) is Side.A
    assert side_of_line((5.0, -5.0), a, b) is Side.B


def test_side_of_line_collinear_is_on():
    assert side_of_line((5.0, 0.0), (0.0, 0.0), (10.0, 0.0)) is Side.ON


def test_side_flips_when_direction_reverses():
    """The a->b ordering defines which side is which, so reversing swaps them."""
    point = (5.0, 5.0)
    assert side_of_line(point, (0.0, 0.0), (10.0, 0.0)) is Side.A
    assert side_of_line(point, (10.0, 0.0), (0.0, 0.0)) is Side.B


def test_side_of_polyline_uses_nearest_segment():
    """An L-shaped wire has no global side; the local segment must decide."""
    vertices = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    assert side_of_polyline((2.0, 3.0), vertices) is Side.A
    assert side_of_polyline((12.0, 8.0), vertices) is Side.B


def test_side_of_polyline_needs_two_points():
    with pytest.raises(ValueError):
        side_of_polyline((0.0, 0.0), np.array([[1.0, 1.0]]))


# -- validation -----------------------------------------------------------


def test_valid_polygon_passes():
    assert validate_polygon(SQUARE) == []


def test_self_intersecting_polygon_is_rejected():
    """shapely accepts a bow-tie and then returns nonsense from containment tests."""
    bowtie = [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]]
    errors = validate_polygon(bowtie)
    assert errors and "self-intersecting" in errors[0].message


def test_polygon_needs_three_points():
    errors = validate_polygon([[0.1, 0.1], [0.5, 0.5]])
    assert errors and "at least 3" in errors[0].message


def test_points_outside_unit_range_are_rejected():
    """Catches pixel coordinates pasted in where normalised ones belong."""
    errors = validate_polygon([[0.1, 0.1], [640, 480], [0.5, 0.9]])
    assert errors and "normalised" in errors[0].message


def test_valid_tripwire_passes():
    assert validate_tripwire([[0.2, 0.7], [0.85, 0.66]]) == []


def test_tiny_tripwire_is_rejected_as_a_misclick():
    errors = validate_tripwire([[0.5, 0.5], [0.51, 0.5]])
    assert errors and "mis-click" in errors[0].message


def test_tripwire_needs_two_points():
    errors = validate_tripwire([[0.5, 0.5]])
    assert errors and "at least 2" in errors[0].message


def test_polyline_tripwire_is_allowed():
    assert validate_tripwire([[0.1, 0.5], [0.5, 0.5], [0.9, 0.8]]) == []
