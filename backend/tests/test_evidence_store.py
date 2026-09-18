"""Tests for LocalEvidenceStore (Expansion Plan Phase C)."""

from __future__ import annotations

from perimeter.evidence.store import LocalEvidenceStore


def make_store(tmp_path) -> LocalEvidenceStore:
    return LocalEvidenceStore(
        root=tmp_path / "evidence", snapshot_dir="snapshots", clip_dir="clips"
    )


def test_save_snapshot_writes_the_file_and_returns_a_dated_key(tmp_path):
    store = make_store(tmp_path)

    key = store.save_snapshot("cam_01", "event123", b"\xff\xd8fake-jpeg-bytes")

    assert key.startswith("snapshots/cam_01/")
    assert key.endswith("/event123.jpg")
    assert (tmp_path / "evidence" / key).read_bytes() == b"\xff\xd8fake-jpeg-bytes"


def test_save_clip_moves_the_temp_file_into_place(tmp_path):
    store = make_store(tmp_path)
    tmp_file = tmp_path / "scratch.mp4"
    tmp_file.write_bytes(b"fake-mp4-bytes")

    key = store.save_clip("cam_01", "event456", tmp_file)

    assert key.startswith("clips/cam_01/")
    assert key.endswith("/event456.mp4")
    assert (tmp_path / "evidence" / key).read_bytes() == b"fake-mp4-bytes"
    assert not tmp_file.exists()  # moved, not copied


def test_resolve_roundtrips_a_saved_key(tmp_path):
    store = make_store(tmp_path)
    key = store.save_snapshot("cam_01", "event789", b"jpeg-bytes")

    resolved = store.resolve(key)

    assert resolved is not None
    assert resolved.is_file()
    assert resolved.read_bytes() == b"jpeg-bytes"


def test_resolve_returns_none_for_a_missing_key(tmp_path):
    store = make_store(tmp_path)
    assert store.resolve("snapshots/cam_01/2020-01-01/does-not-exist.jpg") is None


def test_resolve_refuses_to_escape_the_store_root(tmp_path):
    store = make_store(tmp_path)
    # A key that would climb out of `root` via traversal must never resolve, even if
    # something outside the store happens to exist at that path.
    secret = tmp_path / "secret.txt"
    secret.write_text("not evidence")

    assert store.resolve("../secret.txt") is None


def test_different_cameras_and_events_do_not_collide(tmp_path):
    store = make_store(tmp_path)
    key_a = store.save_snapshot("cam_01", "evt", b"aaa")
    key_b = store.save_snapshot("cam_02", "evt", b"bbb")

    assert key_a != key_b
    assert store.resolve(key_a).read_bytes() == b"aaa"
    assert store.resolve(key_b).read_bytes() == b"bbb"
