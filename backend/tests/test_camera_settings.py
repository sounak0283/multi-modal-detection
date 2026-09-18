"""Tests for camera settings, the use_cctv switch, and .env persistence.

Camera config is a per-deployment fact that lives in .env, never in the committed
app.yaml. One of its fields is a password, so the write path and the redaction path
both matter more than usual.
"""

from __future__ import annotations

import os

import pytest

from perimeter.settings import (
    CameraSettings,
    describe_source,
    load_settings,
    redact,
    save_camera_settings,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Camera settings come from the environment, so each test starts from a clean one."""
    for key in list(os.environ):
        if key.startswith("PERIMETER_"):
            monkeypatch.delenv(key, raising=False)


# -- the use_cctv switch --------------------------------------------------


def test_webcam_index_resolves_to_an_int():
    camera = CameraSettings(use_cctv=False, source="0")
    assert camera.resolved_source == 0


def test_video_file_path_resolves_to_a_string():
    """A file path in the webcam field is how development against clips works."""
    camera = CameraSettings(use_cctv=False, source="D:/footage/clip.mp4")
    assert camera.resolved_source == "D:/footage/clip.mp4"


def test_cctv_true_uses_the_rtsp_url():
    camera = CameraSettings(
        use_cctv=True, source="0", cctv_rtsp_url="rtsp://admin:pw@10.0.0.5:554/s1"
    )
    assert camera.resolved_source == "rtsp://admin:pw@10.0.0.5:554/s1"


def test_cctv_true_without_a_url_raises_an_actionable_error():
    camera = CameraSettings(use_cctv=True, cctv_rtsp_url="")
    with pytest.raises(ValueError, match="PERIMETER_CCTV_RTSP_URL"):
        _ = camera.resolved_source


def test_switching_to_webcam_ignores_a_stored_rtsp_url():
    """Turning the switch off must not keep using CCTV just because a URL is stored."""
    camera = CameraSettings(use_cctv=False, source="1", cctv_rtsp_url="rtsp://x@y/z")
    assert camera.resolved_source == 1


def test_substream_is_preferred_for_inference():
    camera = CameraSettings(
        use_cctv=True,
        cctv_rtsp_url="rtsp://admin:pw@10.0.0.5:554/main",
        substream="rtsp://admin:pw@10.0.0.5:554/sub",
    )
    assert camera.inference_source.endswith("/sub")
    assert camera.resolved_source.endswith("/main")


# -- boolean parsing ------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "on"])
def test_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("PERIMETER_USE_CCTV", value)
    monkeypatch.setenv("PERIMETER_CCTV_RTSP_URL", "rtsp://x@y/z")
    assert load_settings().camera.use_cctv is True


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", ""])
def test_falsy_spellings(monkeypatch, value):
    monkeypatch.setenv("PERIMETER_USE_CCTV", value)
    assert load_settings().camera.use_cctv is False


def test_a_typo_falls_back_to_the_default_not_silently_false(monkeypatch, caplog):
    """`USE_CCTV=Ture` quietly disabling the CCTV feed would be a confusing site visit."""
    monkeypatch.setenv("PERIMETER_USE_CCTV", "Ture")
    with caplog.at_level("WARNING"):
        settings = load_settings()
    assert settings.camera.use_cctv is False
    assert "not a boolean" in caplog.text


# -- redaction ------------------------------------------------------------


def test_redact_strips_the_password():
    assert redact("rtsp://admin:hunter2@10.0.0.5:554/s1") == "rtsp://admin:***@10.0.0.5:554/s1"


def test_redact_keeps_a_username_without_a_password():
    assert redact("rtsp://admin@10.0.0.5:554/s1") == "rtsp://admin@10.0.0.5:554/s1"


def test_redact_passes_through_plain_sources():
    assert redact(0) == "device:0"
    assert redact("clip.mp4") == "clip.mp4"
    assert redact(None) == "<unset>"


def test_str_of_camera_settings_never_leaks_the_password():
    camera = CameraSettings(use_cctv=True, cctv_rtsp_url="rtsp://admin:hunter2@10.0.0.5/s1")
    assert "hunter2" not in str(camera)


def test_describe_source_does_not_raise_on_a_half_configured_camera():
    """Fresh installs have CCTV selected and no URL yet; logging must survive it."""
    camera = CameraSettings(use_cctv=True, cctv_rtsp_url="")
    assert "no URL" in describe_source(camera)


# -- .env persistence -----------------------------------------------------


def test_env_written_with_a_utf8_bom_is_still_read(tmp_path):
    """Notepad and PowerShell's Set-Content write UTF-8 WITH a BOM on Windows.

    Decoded as plain utf-8 the first key becomes "\\ufeffPERIMETER_..." and that setting
    silently does not exist - the app reports "not set" while the file plainly shows it.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PERIMETER_MONGO_URL=mongodb://u:p@localhost:27017\nPERIMETER_CAMERA_ID=bom_cam\n",
        encoding="utf-8-sig",
    )

    settings = load_settings(env_file=env_file)

    assert settings.storage.mongo_url.endswith("27017"), "first line lost to the BOM"
    assert settings.camera.id == "bom_cam"


def test_env_without_a_bom_still_works(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("PERIMETER_CAMERA_ID=plain_cam\n", encoding="utf-8")
    assert load_settings(env_file=env_file).camera.id == "plain_cam"


def test_save_creates_env_and_round_trips(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    camera = CameraSettings(
        id="bay_cam",
        use_cctv=True,
        source="0",
        cctv_rtsp_url="rtsp://admin:pw@10.0.0.5:554/s1",
        width=1920,
        height=1080,
        fps=25,
        decode_fps=8.0,
        autostart=False,
    )

    save_camera_settings(camera, env_file=env_file)

    assert env_file.is_file()
    reloaded = load_settings(env_file=env_file).camera
    assert reloaded.id == "bay_cam"
    assert reloaded.use_cctv is True
    assert reloaded.cctv_rtsp_url == "rtsp://admin:pw@10.0.0.5:554/s1"
    assert (reloaded.width, reloaded.height, reloaded.fps) == (1920, 1080, 25)
    assert reloaded.decode_fps == 8.0
    assert reloaded.autostart is False


def test_save_updates_os_environ_for_the_running_process(tmp_path):
    """dotenv does not override already-set vars, so the save must do it itself -
    otherwise a reload in the same process silently returns the old camera."""
    save_camera_settings(
        CameraSettings(id="first", width=640, height=360), env_file=tmp_path / ".env"
    )
    assert os.environ["PERIMETER_CAMERA_ID"] == "first"

    save_camera_settings(
        CameraSettings(id="second", width=800, height=600), env_file=tmp_path / ".env"
    )
    assert os.environ["PERIMETER_CAMERA_ID"] == "second"
    assert os.environ["PERIMETER_CAMERA_WIDTH"] == "800"


def test_saving_twice_does_not_duplicate_keys(tmp_path):
    env_file = tmp_path / ".env"
    save_camera_settings(CameraSettings(id="a"), env_file=env_file)
    save_camera_settings(CameraSettings(id="b"), env_file=env_file)

    lines = [ln for ln in env_file.read_text(encoding="utf-8").splitlines()
             if ln.startswith("PERIMETER_CAMERA_ID")]
    assert len(lines) == 1


def test_rtsp_url_with_special_characters_survives_a_round_trip(tmp_path):
    """Camera passwords routinely contain punctuation that would break naive quoting."""
    url = "rtsp://admin:p@ss w0rd!#$@10.0.0.5:554/Streaming/Channels/101"
    env_file = tmp_path / ".env"

    save_camera_settings(CameraSettings(use_cctv=True, cctv_rtsp_url=url), env_file=env_file)

    assert load_settings(env_file=env_file).camera.cctv_rtsp_url == url


# -- yaml no longer owns the camera ---------------------------------------


def test_camera_is_not_read_from_app_yaml(tmp_path, monkeypatch):
    """Camera settings are per-deployment and must come from .env only - a stale camera
    block in a committed app.yaml must not override the environment."""
    config = tmp_path / "app.yaml"
    config.write_text(
        "camera:\n  id: from_yaml\n  decode_fps: 99\ninference:\n  person_every_n: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PERIMETER_CAMERA_ID", "from_env")

    settings = load_settings(config_path=config, env_file=tmp_path / "absent.env")

    assert settings.camera.id == "from_env"
    assert settings.camera.decode_fps == 12.0, "yaml must not supply decode_fps"


def test_inference_tuning_still_comes_from_app_yaml(tmp_path):
    config = tmp_path / "app.yaml"
    config.write_text("inference:\n  person_every_n: 5\n", encoding="utf-8")

    settings = load_settings(config_path=config, env_file=tmp_path / "absent.env")

    assert settings.inference.person_every_n == 5


def test_person_fps_derives_from_decode_fps_and_cadence(tmp_path):
    settings = load_settings(env_file=tmp_path / "absent.env")
    assert settings.inference.person_fps(12.0) == 6.0
