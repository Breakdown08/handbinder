from handbinder.gestures import Gesture
from handbinder.trigger import Timing, Trigger

THUMB: Gesture = Gesture.THUMB_UP


def trigger() -> Trigger:
    return Trigger({THUMB: Timing(hold_s=0.5, cooldown_s=2)})


def play(t: Trigger, events: list[tuple[Gesture | None, float]]) -> list[float]:
    fired: list[float] = []
    gesture: Gesture | None
    now: float
    for gesture, now in events:
        if t.update(gesture, now):
            fired.append(now)
    return fired


def held(start: float, end: float, step: float = 0.1) -> list[tuple[Gesture | None, float]]:
    return [(THUMB, start + i * step) for i in range(round((end - start) / step) + 1)]


def test_fires_after_the_hold() -> None:
    assert play(trigger(), held(0, 0.7)) == [0.5]


def test_short_flash_does_not_fire() -> None:
    assert play(trigger(), [*held(0, 0.3), (None, 0.4), (None, 0.5), (None, 0.6), *held(0.7, 1.0)]) == []


def test_one_missed_frame_does_not_interrupt_the_hold() -> None:
    assert play(trigger(), [*held(0, 0.2), (None, 0.3), *held(0.4, 0.5)]) == [0.5]


def test_long_hold_fires_once() -> None:
    assert len(play(trigger(), held(0, 10))) == 1


def test_cooldown_blocks_a_quick_repeat_and_then_lets_it_through() -> None:
    t: Trigger = trigger()
    assert play(t, [*held(0, 0.5), (None, 0.9), *held(1.0, 1.6)]) == [0.5]
    assert play(t, [(None, 2.0), *held(2.5, 3.0)]) == [3.0]


def test_held_through_the_cooldown_fires_when_it_ends() -> None:
    t: Trigger = trigger()
    assert play(t, [*held(0, 0.5), (None, 0.9), *held(1.0, 2.6)]) == [0.5, 2.5]


def test_unbound_gesture_does_nothing() -> None:
    assert play(Trigger({}), held(0, 2)) == []
