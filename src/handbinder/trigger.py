"""When a recognised gesture becomes an action: this is what keeps a scratched cheek from firing a script."""

from dataclasses import dataclass

from handbinder.gestures import Gesture

GRACE_S: float = 0.2  # a recogniser miss this short does not interrupt a hold


@dataclass(frozen=True)
class Timing:
    hold_s: float
    cooldown_s: float


class Trigger:
    """Fires once per hold. To fire again the hand has to leave the gesture and the cooldown has to pass."""

    def __init__(self, timings: dict[Gesture, Timing]) -> None:
        self._timings: dict[Gesture, Timing] = timings
        self._gesture: Gesture | None = None
        self._since: float = 0
        self._seen: float = 0
        self._fired: bool = False
        self._ready_at: dict[Gesture, float] = {}

    def update(self, gesture: Gesture | None, now: float) -> Gesture | None:
        if gesture is None or gesture not in self._timings:
            if now - self._seen > GRACE_S:
                self._gesture = None
                self._fired = False
            return None
        if gesture != self._gesture or now - self._seen > GRACE_S:
            self._gesture = gesture
            self._since = now
            self._fired = False
        self._seen = now
        timing: Timing = self._timings[gesture]
        if self._fired or now - self._since < timing.hold_s or now < self._ready_at.get(gesture, 0):
            return None
        self._fired = True
        self._ready_at[gesture] = now + timing.cooldown_s
        return gesture
