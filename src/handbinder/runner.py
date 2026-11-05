"""Starts the bound scripts. A script still running is not started a second time."""

import logging
import subprocess
from pathlib import Path

from handbinder.gestures import Gesture

log: logging.Logger = logging.getLogger(__name__)


class ScriptRunner:
    def __init__(self) -> None:
        self._running: dict[Gesture, subprocess.Popen[bytes]] = {}

    def run(self, gesture: Gesture, script: Path) -> bool:
        previous: subprocess.Popen[bytes] | None = self._running.get(gesture)
        if previous is not None and previous.poll() is None:
            log.info("%s: %s is still running", gesture, script)
            return False
        log.info("%s: starting %s", gesture, script)
        self._running[gesture] = subprocess.Popen(
            [str(script)], stdin=subprocess.DEVNULL, start_new_session=True
        )
        return True
