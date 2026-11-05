"""Command line."""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from handbinder import app
from handbinder.config import Config, ConfigError, load
from handbinder.models import WeightsError


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="handbinder", description=__doc__)
    commands: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(
        dest="command", required=True
    )
    run: argparse.ArgumentParser = commands.add_parser("run", help="watch the camera and run bound scripts")
    run.add_argument("--config", type=Path, default=Path("gestures.toml"))
    run.add_argument("--preview", action="store_true", help="show the debug window; q quits")
    args: argparse.Namespace = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        config: Config = load(args.config)
        app.run(config, args.preview)
    except KeyboardInterrupt:
        return 0
    except (ConfigError, ValidationError, WeightsError, app.CameraError) as error:
        print(f"handbinder: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
