"""Tests for IdentityResolver's gating logic (Expansion Plan Phase F; PLAN.md section 7),
using fake YuNet/SFace doubles - no real cv2 model load here, mirroring
EvidenceWriter's injectable-mux testing pattern."""

from __future__ import annotations

import numpy as np

from perimeter.identity.gallery import FaceGallery, PersonRecord
from perimeter.identity.resolver import IdentityResolver
from perimeter.identity.yunet import FaceBox


class FakeDetector:
    def __init__(self, face=None):
        self._face = face
        self.calls = 0

    def largest(self, image_bgr):
        self.calls += 1
        return self._face


class FakeEmbedder:
    def __init__(self, embedding):
        self._embedding = embedding

    def embed(self, image_bgr, face):
        return self._embedding


def make_face() -> FaceBox:
    raw = np.zeros(15, dtype=np.float32)
    raw[2] = raw[3] = 40
    return FaceBox(raw=raw, score=0.9)


def make_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_resolver(face, embedding, persons=(), min_person_box_px=120, threshold=0.5):
    gallery = FaceGallery(threshold=threshold)
    gallery.rebuild(list(persons))
    detector = FakeDetector(face)
    return IdentityResolver(detector, FakeEmbedder(embedding), gallery, min_person_box_px), detector


def test_no_face_when_the_person_box_is_too_small():
    resolver, detector = make_resolver(face=make_face(), embedding=np.ones(128, dtype=np.float32))

    result = resolver.resolve(make_frame(), (0, 0, 50, 60))  # height 60 < 120

    assert result.status == "no_face"
    assert detector.calls == 0, "a too-small box must skip inference entirely"


def test_no_face_when_yunet_finds_nothing():
    resolver, _detector = make_resolver(face=None, embedding=np.ones(128, dtype=np.float32))

    result = resolver.resolve(make_frame(), (100, 100, 200, 300))

    assert result.status == "no_face"


def test_unknown_face_when_no_gallery_match():
    resolver, _detector = make_resolver(face=make_face(), embedding=np.ones(128, dtype=np.float32))

    result = resolver.resolve(make_frame(), (100, 100, 200, 300))

    assert result.status == "unknown_face"


def test_known_when_the_gallery_matches():
    embedding = np.zeros(128, dtype=np.float32)
    embedding[0] = 1.0
    person = PersonRecord(
        id="p1", name="Alex", external_id="", active=True, embeddings=[embedding.tolist()]
    )
    resolver, _detector = make_resolver(
        face=make_face(), embedding=embedding, persons=[person], threshold=0.5
    )

    result = resolver.resolve(make_frame(), (100, 100, 200, 300))

    assert result.status == "known"
    assert result.person_id == "p1"
    assert result.name == "Alex"


def test_crop_at_the_frame_edge_does_not_raise():
    resolver, _detector = make_resolver(face=make_face(), embedding=np.ones(128, dtype=np.float32))

    result = resolver.resolve(make_frame(), (600, 400, 700, 550))  # x2 beyond the 640px width

    assert result.status in ("known", "unknown_face")


class TopCropBlindDetector(FakeDetector):
    """Finds a face only in crops taller than `min_height` - stands in for a close-up
    subject whose face sits below the top 35% of their box."""

    def __init__(self, face, min_height):
        super().__init__(face)
        self.min_height = min_height

    def largest(self, image_bgr):
        self.calls += 1
        return self._face if image_bgr.shape[0] >= self.min_height else None


def test_falls_back_to_the_full_box_when_the_top_crop_has_no_face():
    gallery = FaceGallery(threshold=0.5)
    gallery.rebuild([])
    detector = TopCropBlindDetector(make_face(), min_height=200)
    resolver = IdentityResolver(detector, FakeEmbedder(np.ones(128, dtype=np.float32)), gallery)

    result = resolver.resolve(make_frame(), (100, 100, 400, 400))  # top 35% = 105px

    assert result.status == "unknown_face"
    assert detector.calls == 2


class CutOffInTopCropDetector(FakeDetector):
    """Returns a face reaching the bottom of whatever crop it is given - in the top-35%
    crop that means a face sliced in half; in the full box it sits comfortably inside."""

    def largest(self, image_bgr):
        self.calls += 1
        raw = np.zeros(15, dtype=np.float32)
        raw[1], raw[2], raw[3] = 60, 80, 80  # y=60, h=80 -> ends at 140
        return FaceBox(raw=raw, score=0.9)


def test_falls_back_to_the_full_box_when_the_top_crop_cuts_the_face_off():
    gallery = FaceGallery(threshold=0.5)
    gallery.rebuild([])
    detector = CutOffInTopCropDetector()
    resolver = IdentityResolver(detector, FakeEmbedder(np.ones(128, dtype=np.float32)), gallery)

    # Box height 400 -> top crop is 140px tall, exactly where the face ends.
    result = resolver.resolve(make_frame(), (100, 50, 400, 450))

    assert result.status == "unknown_face"
    assert detector.calls == 2, "a face cut off by the top crop must trigger the full-box retry"
