"""gestures.toml: the page of bindings, validated before the camera opens."""

import os
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from handbinder.gestures import Gesture
from handbinder.trigger import Timing

MS: int = 1000


class ConfigError(ValueError):
    pass


class Binding(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    script: Path
    hold_ms: int | None = Field(default=None, ge=0)
    cooldown_ms: int | None = Field(default=None, ge=0)


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    device: int = 0
    threads: int = Field(default=1, ge=1)
    idle_fps: float = Field(default=5, gt=0)
    active_fps: float = Field(default=15, gt=0)
    hold_ms: int = Field(default=500, ge=0)
    cooldown_ms: int = Field(default=2000, ge=0)
    gestures: dict[Gesture, Binding] = Field(min_length=1)

    def timings(self) -> dict[Gesture, Timing]:
        out: dict[Gesture, Timing] = {}
        gesture: Gesture
        binding: Binding
        for gesture, binding in self.gestures.items():
            hold: int = self.hold_ms if binding.hold_ms is None else binding.hold_ms
            cooldown: int = self.cooldown_ms if binding.cooldown_ms is None else binding.cooldown_ms
            out[gesture] = Timing(hold / MS, cooldown / MS)
        return out


def load(path: Path) -> Config:
    if not path.is_file():
        raise ConfigError(f"{path} not found; copy gestures.example.toml to start")
    raw: dict[str, Any] = tomllib.loads(path.read_text())
    config: Config = Config.model_validate(raw)
    gesture: Gesture
    binding: Binding
    for gesture, binding in config.gestures.items():
        script: Path = (path.resolve().parent / binding.script.expanduser()).resolve()
        if not script.is_file():
            raise ConfigError(f"{gesture}: {script} not found")
        if not os.access(script, os.X_OK):
            raise ConfigError(f"{gesture}: {script} is not executable")
        binding.script = script
    return config
