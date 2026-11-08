"""Immediate, configurable user feedback without modifying scientific results."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging as std_logging
import os
import sys
import threading
import time
from numbers import Integral
from pathlib import Path

_LOGGER = std_logging.getLogger("cellscope")
_LOGGER.propagate = False
_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "quiet": 100}
_CURRENT = contextvars.ContextVar("cellscope_operation", default=None)
_WRAPPED = {}
_BANNER_SHOWN = False
_UNSET = object()


class Settings:
    """Console settings shared by all cellscope namespaces."""

    def __init__(self):
        level = os.getenv("CELLSCOPE_VERBOSITY", "info").lower()
        if os.getenv("CELLSCOPE_QUIET", "0").lower() in {"1", "true", "yes"}:
            level = "quiet"
        self.verbosity = level if level in _LEVELS else "info"
        self.banner = os.getenv("CELLSCOPE_BANNER", "1").lower() not in {"0", "false", "no"}
        try:
            self.heartbeat_interval = float(os.getenv("CELLSCOPE_HEARTBEAT_SECONDS", "30"))
        except ValueError:
            self.heartbeat_interval = 30
        self.device = os.getenv("CELLSCOPE_DEVICE", "auto").lower()
        self.stream = None  # resolve sys.stderr at emit time, including notebooks/capture

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
        self._device = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if isinstance(value, int):
            value = {0: "error", 1: "warning", 2: "info", 3: "debug"}.get(value, "invalid")
        value = str(value).lower()
        if value not in _LEVELS:
            raise ValueError("verbosity must be debug/info/warning/error/quiet or 0..3")
        self._verbosity = value
        _LOGGER.setLevel(_LEVELS[value])

    @property
    def heartbeat_interval(self):
        return self._heartbeat_interval

    @heartbeat_interval.setter
    def heartbeat_interval(self, value):
        value = float(value)
        if not 0 <= value < float("inf"):
            raise ValueError("heartbeat_interval must be finite and nonnegative; 0 disables it")
        self._heartbeat_interval = value


settings = Settings()


class _VerbosityFilter(std_logging.Filter):
    def filter(self, record):
        # warning is a warnings-only view; exceptions still reach the caller.
        return settings.verbosity != "warning" or record.levelno == std_logging.WARNING


if not any(isinstance(f, _VerbosityFilter) for f in _LOGGER.filters):
    _LOGGER.addFilter(_VerbosityFilter())


class _ConsoleHandler(std_logging.Handler):
    _cellscope_console = True

    def emit(self, record):
        stream = settings.stream if settings.stream is not None else sys.stderr
        try:
            stream.write(self.format(record) + "\n")
            stream.flush()
        except (OSError, TypeError, ValueError, UnicodeError):
            # A closed output pipe must never abort an analysis.
            pass


_FORMATTER = std_logging.Formatter(
    "[%(asctime)s] [cellscope] %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
_CONSOLE = next((h for h in _LOGGER.handlers if getattr(h, "_cellscope_console", False)), None)
if _CONSOLE is None:
    _CONSOLE = _ConsoleHandler()
    _CONSOLE.setFormatter(_FORMATTER)
    _LOGGER.addHandler(_CONSOLE)


def configure(*, verbosity=None, banner=None, heartbeat_interval=None, stream=_UNSET,
              log_file=_UNSET):
    """Configure feedback; optional UTF-8 log_file appends without duplicate handlers."""
    if verbosity is not None:
        settings.verbosity = verbosity
    if banner is not None:
        settings.banner = bool(banner)
    if heartbeat_interval is not None:
        settings.heartbeat_interval = heartbeat_interval
    if stream is not _UNSET:
        settings.stream = stream
    if log_file is not _UNSET:
        new_handler = None
        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            new_handler = std_logging.FileHandler(path, encoding="utf-8")
            new_handler._cellscope_file = True
            new_handler.setFormatter(_FORMATTER)
        for handler in list(_LOGGER.handlers):
            if getattr(handler, "_cellscope_file", False):
                _LOGGER.removeHandler(handler)
                handler.close()
        if new_handler is not None:
            _LOGGER.addHandler(new_handler)
    return settings


def info(message):
    _LOGGER.info(message)


def debug(message):
    _LOGGER.debug(message)


def warning(message):
    _LOGGER.warning(message)


def error(message):
    _LOGGER.error(message)


def verbose_enabled():
    return _LOGGER.isEnabledFor(std_logging.DEBUG)


def show_banner(version, *, force=False):
    """Show a plain ASCII logo once per process; no ANSI codes or dependencies."""
    global _BANNER_SHOWN
    if not settings.banner or not _LOGGER.isEnabledFor(std_logging.INFO) or (_BANNER_SHOWN and not force):
        return
    logo = r"""
   ____     _ _ ____
  / ___|___| | / ___|  ___ ___  _ __   ___
 | |   / _ \ | \___ \ / __/ _ \| '_ \ / _ \
 | |__|  __/ | |___) | (_| (_) | |_) |  __/
  \____\___|_|_|____/ \___\___/| .__/ \___|
                              |_|
"""
    stream = settings.stream if settings.stream is not None else sys.stderr
    with _CONSOLE.lock:
        try:
            stream.write(logo + f"  cellscope {version} | Single-cell transcriptomics\n\n")
            stream.flush()
        except (OSError, TypeError, ValueError, UnicodeError):
            return
    _BANNER_SHOWN = True


def progress(message):
    """Report a detailed DEBUG phase and update the long-operation heartbeat."""
    task = _CURRENT.get()
    phase = f"{task.label} | {message}" if task is not None else message
    while task is not None:
        task.phase = phase
        task = task.parent
    debug(message)


class _Task:
    def __init__(self, label):
        self.label = label
        self.parent = _CURRENT.get()
        self.phase = "working"
        self.started = time.monotonic()
        self.complete_detail = ""
        self.stop = threading.Event()
        self.thread = None

    def heartbeat(self, interval):
        while not self.stop.wait(interval):
            debug(f"RUNNING {self.label} | {self.phase} | elapsed {time.monotonic() - self.started:.1f}s")


@contextlib.contextmanager
def operation(label, detail="", *, sample=False):
    """Emit start/success/failure and a bounded-lifetime heartbeat for opaque work."""
    task = _Task(label)
    if task.parent is not None:
        task.parent.phase = label
    token = _CURRENT.set(task)
    emit = info if task.parent is None or sample else debug
    start_detail = detail if verbose_enabled() else detail.split(" | ", 1)[0]
    emit(f"START {label}" + (f" | {start_detail}" if start_detail else ""))
    if task.parent is None and settings.heartbeat_interval > 0 and verbose_enabled():
        task.thread = threading.Thread(
            target=task.heartbeat, args=(settings.heartbeat_interval,), daemon=True,
            name="cellscope-feedback",
        )
        task.thread.start()
    try:
        yield task
    except BaseException as exc:
        task.stop.set()
        if task.thread is not None:
            task.thread.join(timeout=0.2)
        status = "CANCELLED" if isinstance(exc, KeyboardInterrupt) else "FAIL"
        report_error = error if task.parent is None or sample else debug
        report_error(f"{status} {label} | {type(exc).__name__}: {str(exc)[:500]}"
              f" | elapsed {time.monotonic() - task.started:.1f}s")
        raise
    else:
        task.stop.set()
        if task.thread is not None:
            task.thread.join(timeout=0.2)
        suffix = f" | {task.complete_detail}" if task.complete_detail else ""
        emit(f"DONE {label}{suffix} | elapsed {time.monotonic() - task.started:.1f}s")
    finally:
        task.stop.set()
        if task.thread is not None:
            task.thread.join(timeout=0.2)
        _CURRENT.reset(token)


@contextlib.contextmanager
def batch(label, name, index, total, *, cells=None):
    """Known batch position, not an estimated completion percentage."""
    detail = f"{cells:,} cells" if cells is not None else ""
    with operation(f"{label} [{index}/{total}] {name}", detail, sample=True) as task:
        task.batch_name = str(name)
        yield task


def _shape(obj, mod="gex"):
    try:
        if hasattr(obj, "mod") and mod in obj.mod:
            obj = obj.mod[mod]
        if isinstance(getattr(obj, "n_obs", None), Integral) and isinstance(getattr(obj, "n_vars", None), Integral):
            return int(obj.n_obs), int(obj.n_vars)
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def instrument_function(function, label):
    """Preserve signature/return/exception semantics while adding feedback."""
    if getattr(function, "_cellscope_logged", False):
        return function
    if function in _WRAPPED:
        return _WRAPPED[function]
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        # Diagnostics must not replace the function's own argument errors.
        try:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            extras = arguments.get("kwargs", {})
            if isinstance(extras, dict):
                arguments = {**extras, **arguments}
        except TypeError:
            arguments = kwargs
        data = next((v for v in arguments.values() if _shape(v) is not None), None)
        mod = arguments.get("mod", "gex")
        before = _shape(data, mod)
        row_unit, column_unit = ("barcodes", "features") if label.startswith("io.") else ("cells", "genes")
        details = []
        if before is not None:
            details.append(f"{before[0]:,} {row_unit} x {before[1]:,} {column_unit}")
        for key in ("batch_key", "library_id", "layer", "method", "min_genes", "min_cells",
                    "max_pct_mito", "nmads", "n_top_genes", "n_comps", "n_pcs", "n_neighbors", "resolution", "epochs"):
            if key in arguments and (arguments[key] is not None or key == "layer"):
                value = "X" if key == "layer" and arguments[key] is None else arguments[key]
                if key == "layer" and arguments[key] is None and function.__name__ == "normalize":
                    current = data.mod[mod] if hasattr(data, "mod") and mod in data.mod else data
                    count_layer = arguments.get("counts_layer", "counts")
                    if hasattr(current, "layers") and count_layer in current.layers:
                        value = count_layer
                details.append(f"{key}={str(value)[:80]}")
        for key in ("path", "input_file", "output_file"):
            value = arguments.get(key)
            if isinstance(value, (str, Path)):
                details.append(f"{key}={value}")
        samples = arguments.get("samples")
        if hasattr(samples, "keys"):
            names = list(map(str, samples))
            details.append(f"{len(samples)} libraries: {', '.join(names[:8])}" + (" ..." if len(names) > 8 else ""))
        with operation(label, " | ".join(details)) as task:
            result = function(*args, **kwargs)
            after = _shape(result, mod) or _shape(data, mod)
            if after is not None:
                if before is not None and before != after:
                    task.complete_detail = (f"{before[0]:,} x {before[1]:,} -> "
                                            f"{after[0]:,} {row_unit} x {after[1]:,} {column_unit}")
                else:
                    task.complete_detail = f"{after[0]:,} {row_unit} x {after[1]:,} {column_unit}"
            elif hasattr(result, "columns") and hasattr(result, "shape"):
                task.complete_detail = f"{result.shape[0]:,} result rows"
            elif isinstance(result, Path):
                task.complete_detail = f"saved {result}"
            return result

    wrapped._cellscope_logged = True
    _WRAPPED[function] = wrapped
    return wrapped


def instrument_namespace(namespace, prefix):
    """Instrument already-loaded public functions without importing optional tools."""
    for name in getattr(namespace, "__all__", ()):
        function = vars(namespace).get(name)
        if not inspect.isfunction(function) or not function.__module__.startswith("cellscope."):
            continue
        if name in {"set_cell_ids", "de_methods", "read_json", "get_rank_array"}:
            continue
        wrapped = instrument_function(function, f"{prefix}.{name}")
        setattr(namespace, name, wrapped)
        # Also cover direct within-module calls, e.g. workflow -> PCA -> UMAP.
        module = sys.modules.get(function.__module__)
        if module is not None and getattr(module, function.__name__, None) is function:
            setattr(module, function.__name__, wrapped)


@contextlib.contextmanager
def relay_backend(logger, label, *, epochs=None):
    """Relay backend details at DEBUG, warnings/errors at their native level.

    Native file handlers and log levels remain unchanged. epochs is retained
    for API compatibility; DEBUG now shows every training epoch.
    """
    relay_task = _CURRENT.get()
    parent = relay_task
    scope = label
    while parent is not None:
        if hasattr(parent, "batch_name"):
            scope = f"{label} [{parent.batch_name}]"
            break
        parent = parent.parent

    class Relay(std_logging.Handler):
        def emit(self, record):
            if record.name.startswith("cellscope"):
                return
            message = record.getMessage()
            if any(word in message.lower() for word in ("epoch", "training", "posterior", "chunk", "inference", "writing")):
                task = relay_task
                while task is not None:
                    task.phase = f"{scope} | {message}"
                    task = task.parent
            level = std_logging.DEBUG if record.levelno < std_logging.WARNING else record.levelno
            _LOGGER.log(level, f"{scope} | {message}")

    handler = Relay(std_logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
