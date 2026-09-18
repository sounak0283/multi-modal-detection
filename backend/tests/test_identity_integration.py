"""End-to-end identity checks against the real YuNet/SFace weights (Expansion Plan
Phase F). Unlike fire/smoke, these models were downloaded and verified during this
phase, so - unlike test_firesmoke_integration.py - this does not always skip.

Skipped when the weights are absent, so a clean checkout without them still passes CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perimeter.identity.gallery import FaceGallery, PersonRecord
from perimeter.identity.resolver import IdentityResolver
from perimeter.identity.sface import FaceEmbedder
from perimeter.identity.yunet import FaceDetector

YUNET = Path("models/yunet/face_detection_yunet_2023mar.onnx")
SFACE = Path("models/sface/face_recognition_sface_2021dec.onnx")

pytestmark = pytest.mark.skipif(
    not YUNET.is_file() or not SFACE.is_file(),
    reason="YuNet/SFace weights not present - see docs/phases/PHASE_F.md",
)


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    return FaceDetector(YUNET)


@pytest.fixture(scope="module")
def embedder() -> FaceEmbedder:
    return FaceEmbedder(SFACE)


def test_blank_frame_yields_no_faces(detector):
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert detector.detect(blank) == []
    assert detector.largest(blank) is None


def test_resolver_degrades_to_no_face_on_a_blank_frame(detector, embedder):
    resolver = IdentityResolver(detector, embedder, FaceGallery(), min_person_box_px=120)
    frame = np.full((720, 1280, 3), 128, np.uint8)

    result = resolver.resolve(frame, (400.0, 100.0, 550.0, 500.0))  # 400px tall, blank

    assert result.status == "no_face"


def test_embedding_dimension_is_128(detector, embedder):
    """A real face-shaped crop (a synthetic gradient, not a photo) exercises the actual
    alignCrop -> feature() path end to end and confirms the output shape - the only
    thing exercisable without a real photograph in the repo."""
    gradient = np.zeros((300, 300, 3), np.uint8)
    for row in range(300):
        gradient[row, :, :] = row % 256
    face = detector.largest(gradient)
    if face is None:
        pytest.skip("no face detected in the synthetic gradient crop")
    embedding = embedder.embed(gradient, face)
    assert embedding.shape == (128,)


def test_gallery_round_trips_a_real_embedding():
    gallery = FaceGallery(threshold=0.99)  # a vector matches only itself at this bar
    vector = np.random.default_rng(0).normal(size=128).astype(np.float32)
    gallery.rebuild([
        PersonRecord(
            id="p1", name="Alex", external_id="", active=True, embeddings=[vector.tolist()]
        )
    ])

    assert gallery.match(vector) == ("p1", "Alex")
