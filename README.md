# handbinder

Bind hand gestures to your own scripts. Think of it as a keymap page, with gestures instead of keys.

Everything runs locally. Frames stay in memory: nothing is written to disk or sent anywhere.

## How it works

```
camera → palm detector → upright hand crop → 21 hand points → gesture → hold check → your script
```

The vision path runs on OpenCV alone: `cv.dnn` runs the two ONNX models from
[opencv_zoo](https://github.com/opencv/opencv_zoo), and `warpAffine` turns the hand upright between them.
Gestures are plain geometry of the 21 points.

- **Idle:** the palm detector looks at about 5 frames per second.
- **Active:** once a hand is found, its points seed the next frame, so the detector is skipped.
- **Firing:** a gesture has to be held. It fires once per hold, then waits for a release and a cooldown.
  A script that is still running is not started again.

Idle load is about 10% of one core on an Intel Core Ultra 7 155H at 640×480.

## Usage

```bash
uv sync
cp gestures.example.toml gestures.toml
uv run handbinder run --preview    # debug window, q quits
uv run handbinder run              # no window, Ctrl+C quits
```

The weights (about 8 MB) are downloaded on the first run, checked by SHA-256 and kept in
`~/.cache/handbinder`.

On NixOS, run the commands inside `nix develop`: the wheels of `opencv-python` need system libraries.

## Bindings

`gestures.toml` maps a gesture to an executable file. handbinder knows only the path; what the script
does is up to you. Relative paths are resolved against the config file.

```toml
hold_ms = 500
cooldown_ms = 2000

[gestures.thumb_up]
script = "~/bin/open-trainer.sh"
hold_ms = 700
```

| Gesture | Name |
|---|---|
| 👍 | `thumb_up` |

## Development

```bash
uv run ruff check && uv run ruff format --check
uv run mypy && uv run ty check
uv run pytest
```

Every variable is typed; `tests/test_annotations.py` enforces it.

## License

Apache 2.0, see [LICENSE](LICENSE). The code and models adapted from opencv_zoo are listed in [NOTICE](NOTICE).
