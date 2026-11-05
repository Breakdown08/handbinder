import time
from pathlib import Path

from handbinder.gestures import Gesture
from handbinder.runner import ScriptRunner


def script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def wait_for(path: Path) -> bool:
    deadline: float = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return False


def test_runs_the_script(tmp_path: Path) -> None:
    mark: Path = tmp_path / "ran"
    assert ScriptRunner().run(Gesture.THUMB_UP, script(tmp_path / "s.sh", f"touch {mark}"))
    assert wait_for(mark)


def test_running_script_is_not_started_twice(tmp_path: Path) -> None:
    mark: Path = tmp_path / "stop"
    path: Path = script(tmp_path / "s.sh", f"while [ ! -e {mark} ]; do sleep 0.01; done")
    runner: ScriptRunner = ScriptRunner()
    assert runner.run(Gesture.THUMB_UP, path)
    assert not runner.run(Gesture.THUMB_UP, path)
    mark.touch()
    runner._running[Gesture.THUMB_UP].wait(timeout=5)
    assert runner.run(Gesture.THUMB_UP, script(tmp_path / "t.sh", "true"))
