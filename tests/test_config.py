from pathlib import Path

import pytest
from pydantic import ValidationError

from handbinder.config import Config, ConfigError, load
from handbinder.gestures import Gesture
from handbinder.trigger import Timing

ROOT: Path = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, body: str) -> Path:
    path: Path = tmp_path / "gestures.toml"
    path.write_text(body)
    return path


def executable(tmp_path: Path) -> Path:
    path: Path = tmp_path / "s.sh"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_the_shipped_example_loads() -> None:
    config: Config = load(ROOT / "gestures.example.toml")
    assert config.gestures[Gesture.THUMB_UP].script == ROOT / "examples/notify.sh"


def test_binding_overrides_only_its_own_timing(tmp_path: Path) -> None:
    executable(tmp_path)
    config: Config = load(
        write(tmp_path, 'cooldown_ms = 3000\n[gestures.thumb_up]\nscript = "s.sh"\nhold_ms = 700\n')
    )
    assert config.timings() == {Gesture.THUMB_UP: Timing(0.7, 3)}


def test_missing_script_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load(write(tmp_path, '[gestures.thumb_up]\nscript = "nope.sh"\n'))


def test_script_must_be_executable(tmp_path: Path) -> None:
    executable(tmp_path).chmod(0o644)
    with pytest.raises(ConfigError, match="not executable"):
        load(write(tmp_path, '[gestures.thumb_up]\nscript = "s.sh"\n'))


def test_unknown_gesture_is_rejected(tmp_path: Path) -> None:
    executable(tmp_path)
    with pytest.raises(ValidationError):
        load(write(tmp_path, '[gestures.wave]\nscript = "s.sh"\n'))


def test_at_least_one_binding_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load(write(tmp_path, "gestures = {}\n"))


def test_missing_file_points_to_the_example(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"gestures\.example\.toml"):
        load(tmp_path / "gestures.toml")
