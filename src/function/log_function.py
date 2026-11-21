# ------------------------------ #
        # Python's packages

import logging
import os
import re
import sys
import warnings

# ------------------------------ #

"""
Logging for NICER-Pointing.

All modules write their messages with the shared logger::

    from function.log_function import log
    log.info("shown by default")
    log.debug("shown only with --verbose")
    log.log(RESULT, "final result, shown also with --quiet")
    log.warning("always shown, also with --quiet")

The console shows the level the user chose (--quiet, default or --verbose).
Every message, at all levels, is also written to ``run.log`` in the output
folder. Messages sent before that folder is known are kept in memory and
written to the file when it is opened.
"""

log = logging.getLogger("nicer_pointing")

# Level for the final results: between INFO and WARNING, so --quiet still shows them.
RESULT = 25
logging.addLevelName(RESULT, "RESULT")

CONSOLE_LEVELS = {"quiet": RESULT, "normal": logging.INFO, "verbose": logging.DEBUG}

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class _StripColorFormatter(logging.Formatter):
    """File formatter: removes the termcolor color codes from the message."""

    def format(self, record):
        return _ANSI_ESCAPE.sub("", super().format(record))


class _BufferHandler(logging.Handler):
    """Keeps every record in memory until run.log can be opened."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


_buffer_handler = None
_file_handler = None


def setup_logging(verbosity: str = "normal") -> None:
    """
    Sets up the console output at the chosen level and starts recording all
    messages for run.log.

    Args:
        verbosity (str): 'quiet' (warnings, errors and final results only),
            'normal' (default) or 'verbose' (all details).
    """
    global _buffer_handler
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(CONSOLE_LEVELS[verbosity])
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    _buffer_handler = _BufferHandler()
    log.addHandler(_buffer_handler)

    # Python warnings (numpy, matplotlib, ...) go to run.log only, unless
    # the user asked for --verbose.
    logging.captureWarnings(True)
    py_warnings = logging.getLogger("py.warnings")
    py_warnings.handlers.clear()
    py_warnings.propagate = False
    py_warnings.addHandler(_buffer_handler)
    if verbosity == "verbose":
        py_warnings.addHandler(console)
    warnings.simplefilter("default")


def start_log_file(output_dir: str) -> str:
    """
    Opens run.log in the output folder and writes into it every message
    recorded so far. From then on all messages go straight to the file.

    Args:
        output_dir (str): Output folder of the run.

    Returns:
        str: Path of run.log.
    """
    global _buffer_handler, _file_handler
    log_path = os.path.join(output_dir, "run.log").replace("\\", "/")
    _file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_StripColorFormatter("%(asctime)s %(levelname)-7s %(message)s",
                                                    datefmt="%H:%M:%S"))
    if _buffer_handler is not None:
        for record in _buffer_handler.records:
            _file_handler.handle(record)
        log.removeHandler(_buffer_handler)
        logging.getLogger("py.warnings").removeHandler(_buffer_handler)
    log.addHandler(_file_handler)
    logging.getLogger("py.warnings").addHandler(_file_handler)
    _buffer_handler = None
    return log_path
