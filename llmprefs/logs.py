import logging
from pathlib import Path


def configure_logging(
    level: int,
    filepath: Path | None,
) -> None:
    """Configure log level format and handlers.

    This must be called before any loggers are created.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if filepath is not None:
        handlers.append(logging.FileHandler(filepath, mode="a"))
    logging.basicConfig(
        handlers=handlers,
        style="{",
        format="{asctime:s} {levelname:7s} {name:s}:{lineno:d} {message:s}",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=level,
    )
