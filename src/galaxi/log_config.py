"""Logging setup for GALAXI's command-line entry points.

GALAXI configures its own "galaxi" logger rather than calling
`logging.basicConfig`. `basicConfig` is a no-op once the root logger already has
a handler, and several scientific dependencies attach one at import time, so
root-level configuration is not reliable for a package's own output. Giving the
"galaxi" logger its own handler and setting `propagate = False` keeps GALAXI's
messages independent of whatever a dependency does to the root logger.

Library code must not call this. Only the console-script `main()` functions do,
so importing `galaxi` as a library emits nothing unless the host application
asks for it.
"""

import logging
import sys

_ROOT_LOGGER_NAME = "galaxi"


def configure_cli_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a stderr handler to the `galaxi` logger and return it.

    Idempotent: calling it more than once (e.g. one entry point delegating to
    another) will not stack duplicate handlers.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)

    # Keep records out of the root logger, so a handler installed there by a
    # dependency cannot suppress or duplicate these lines.
    logger.propagate = False

    if not any(getattr(h, "_galaxi_cli_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        handler._galaxi_cli_handler = True
        logger.addHandler(handler)

    return logger
