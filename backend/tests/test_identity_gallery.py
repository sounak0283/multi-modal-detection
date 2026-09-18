"""Tests for FaceGallery's nearest-neighbour matching, backed by Chroma (Expansion Plan
Phase F redesign; PLATFORM_EXPANSION_PLAN.md section 7).

Every test constructs `FaceGallery()` with no `persist_dir`, which uses an in-memory
Chroma client - fully isolated per instance, no state leaking between tests and no
files written to disk.
"""

from __future__ import annotations

import numpy as np

from perimeter.identity.gallery import FaceGallery, PersonRecord


def test_match_returns_none_when_the_gallery_is_empty():
    gallery = FaceGallery()
    assert gallery.match(np.ones(128, dtype=np.float32)) is None


def test_match_finds_the_closest_enrolled_person_above_threshold():
    gallery = FaceGallery(threshold=0.5)
    alex = np.zeros(128, dtype=np.float32)
    alex[0] = 1.0
    sam = np.zeros(128, dtype=np.float32)
    sam[1] = 1.0
    gallery.rebuild(
        [
            PersonRecord(
                id="p1", name="Alex", external_id="", active=True, embeddings=[alex.tolist()]
            ),
            PersonRecord(
                id="p2", name="Sam", external_id="", active=True, embeddings=[sam.tolist()]
            ),
        ]
    )

    assert gallery.match(alex) == ("p1", "Alex")


def test_match_returns_none_below_threshold():
    gallery = FaceGallery(threshold=0.9)
    alex = np.zeros(128, dtype=np.float32)
    alex[0] = 1.0
    off_axis = np.zeros(128, dtype=np.float32)
    off_axis[0] = 0.7
    off_axis[1] = 0.7  # cosine similarity to `alex` is ~0.7, below a 0.9 threshold
    gallery.rebuild([
        PersonRecord(id="p1", name="Alex", external_id="", active=True, embeddings=[alex.tolist()])
    ])

    assert gallery.match(off_axis) is None


def test_match_returns_none_for_a_zero_query_vector():
    gallery = FaceGallery(threshold=0.1)
    gallery.rebuild(
        [
            PersonRecord(
                id="p1", name="Alex", external_id="", active=True,
                embeddings=[np.ones(128, dtype=np.float32).tolist()],
            )
        ]
    )

    assert gallery.match(np.zeros(128, dtype=np.float32)) is None


def test_inactive_persons_are_excluded_from_matching():
    gallery = FaceGallery(threshold=0.5)
    alex = np.zeros(128, dtype=np.float32)
    alex[0] = 1.0
    gallery.rebuild([
        PersonRecord(id="p1", name="Alex", external_id="", active=False, embeddings=[alex.tolist()])
    ])

    assert gallery.match(alex) is None


def test_persons_without_embeddings_are_excluded():
    gallery = FaceGallery()
    gallery.rebuild([
        PersonRecord(id="p1", name="Alex", external_id="", active=True, embeddings=[])
    ])

    assert gallery.match(np.ones(128, dtype=np.float32)) is None


def test_matches_against_any_of_a_persons_several_poses():
    """The whole point of storing 5 poses: a live face that looks nothing like the
    'front' enrollment photo should still match if it's close to one of the other
    stored poses (Expansion Plan Phase F redesign)."""
    gallery = FaceGallery(threshold=0.8)
    front = np.zeros(128, dtype=np.float32)
    front[0] = 1.0
    left = np.zeros(128, dtype=np.float32)
    left[0] = 0.1
    left[5] = 0.995  # far from `front`, but this IS the same person's left-pose vector
    gallery.rebuild(
        [
            PersonRecord(
                id="p1", name="Alex", external_id="", active=True,
                embeddings=[front.tolist(), left.tolist()],
            )
        ]
    )

    live_capture = np.zeros(128, dtype=np.float32)
    live_capture[0] = 0.1
    live_capture[5] = 1.0  # close to the stored `left` pose, nowhere near `front`

    assert gallery.match(live_capture) == ("p1", "Alex")


def test_rebuild_replaces_the_previous_contents():
    """A second rebuild() must not leave the first call's vectors queryable - this is
    the wholesale-replace guarantee `ZoneStore.reload` also makes."""
    gallery = FaceGallery(threshold=0.5)
    alex = np.zeros(128, dtype=np.float32)
    alex[0] = 1.0
    gallery.rebuild([
        PersonRecord(id="p1", name="Alex", external_id="", active=True, embeddings=[alex.tolist()])
    ])
    assert gallery.match(alex) == ("p1", "Alex")

    gallery.rebuild([])

    assert gallery.match(alex) is None
