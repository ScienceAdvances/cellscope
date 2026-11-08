"""Feedback contracts: timely output, cleanup, unchanged results and signatures."""

import importlib
import inspect
import io
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cellscope as cs


@pytest.fixture(autouse=True)
def feedback_defaults():
    old = (cs.settings.verbosity, cs.settings.banner, cs.settings.heartbeat_interval, cs.settings.stream)
    cs.logging.configure(verbosity="info", heartbeat_interval=0, stream=None, log_file=None)
    yield
    cs.logging.configure(verbosity=old[0], banner=old[1], heartbeat_interval=old[2], stream=old[3], log_file=None)


def test_public_feedback_preserves_return_counts_and_signature(capsys):
    from cellscope.logging import configure

    assert configure is cs.logging.configure
    data = cs.datasets.toy_rna(n_cells=30, n_genes=20)
    before = data.X.copy()
    signature = inspect.signature(cs.pp.qc)
    assert signature == inspect.signature(cs.pp.qc.__wrapped__)
    assert cs.pp.qc(data) is data
    assert (before != data.X).nnz == 0
    output = capsys.readouterr()
    assert output.out == ""
    assert "START pp.qc" in output.err
    assert "DONE pp.qc" in output.err
    assert "30 cells x 20 genes" in output.err
    assert "elapsed" in output.err


def test_batch_names_and_final_qc_counts_are_visible(capsys):
    data = cs.datasets.toy_rna(n_cells=40, n_genes=30)
    data.obs["library"] = ["P1Pre"] * 20 + ["P8Post"] * 20
    result = cs.pp.qc(data, batch_key="library", min_genes=1, min_cells=1, filter=True)
    assert result.n_obs > 0
    output = capsys.readouterr().err
    assert "QC [1/2] P1Pre" in output
    assert "QC [2/2] P8Post" in output
    assert "QC P1Pre | cells" in output
    assert output.count("START pp.qc") == 1


def test_normalization_reports_actual_counts_fallback(capsys):
    data = cs.datasets.toy_rna(n_cells=30, n_genes=20)
    assert "counts" in data.layers
    cs.pp.normalize(data)
    assert "layer=counts" not in capsys.readouterr().err
    cs.settings.verbosity = "debug"
    cs.pp.normalize(data)
    assert "layer=counts" in capsys.readouterr().err


def test_failure_is_not_success_and_original_exception_is_preserved(capsys):
    original = ValueError("invalid test input")

    def broken(value):
        raise original

    wrapped = cs.logging.instrument_function(broken, "test.broken")
    with pytest.raises(ValueError) as caught:
        wrapped(1)
    assert caught.value is original
    output = capsys.readouterr().err
    assert "FAIL test.broken" in output
    assert "invalid test input" in output
    assert "DONE test.broken" not in output


def test_heartbeat_is_realtime_and_worker_is_stopped():
    emitted = threading.Event()

    class LiveStream(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if "RUNNING test.slow" in text:
                emitted.set()
            return result

    stream = LiveStream()
    cs.logging.configure(verbosity="debug", stream=stream, heartbeat_interval=0.01)

    def slow():
        assert emitted.wait(2), "No heartbeat while the function was still running"
        return 42

    assert cs.logging.instrument_function(slow, "test.slow")() == 42
    text = stream.getvalue()
    assert text.index("RUNNING test.slow") < text.index("DONE test.slow")
    assert not any(t.name == "cellscope-feedback" for t in threading.enumerate())
    emitted.clear()
    stream.seek(0)
    stream.truncate()

    def slow_failure():
        assert emitted.wait(2)
        raise RuntimeError("failed while running")

    with pytest.raises(RuntimeError):
        cs.logging.instrument_function(slow_failure, "test.slow_failure")()
    text = stream.getvalue()
    assert text.rindex("RUNNING test.slow_failure") < text.index("FAIL test.slow_failure")
    assert "DONE test.slow_failure" not in text
    assert not any(t.name == "cellscope-feedback" for t in threading.enumerate())


def test_quiet_does_not_change_computation(capsys):
    data = cs.datasets.toy_rna(n_cells=30, n_genes=20)
    cs.settings.verbosity = "quiet"
    cs.pp.qc(data)
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert "qc_pass" in data.obs


@pytest.mark.parametrize("banner,quiet,expected", [("1", "0", True), ("0", "0", False), ("1", "1", False)])
def test_import_banner_is_once_on_stderr_and_can_be_disabled(banner, quiet, expected):
    env = os.environ.copy()
    env.update(CELLSCOPE_BANNER=banner, CELLSCOPE_QUIET=quiet, CELLSCOPE_VERBOSITY="info",
               PYTHONPATH=str(Path(__file__).parents[1]))
    result = subprocess.run(
        [sys.executable, "-c", "import cellscope; import importlib; importlib.reload(cellscope); print('payload')"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout == "payload\n"
    assert result.stderr.count("Single-cell transcriptomics") == int(expected)


def test_reconfigure_and_reload_do_not_duplicate_log_handlers(tmp_path, capsys):
    path = tmp_path / "analysis.log"
    root_level = logging.getLogger().level
    for _ in range(2):
        cs.logging.configure(log_file=path)
    importlib.reload(cs)
    cs.logging.info("unique-feedback-event")
    assert path.read_text().count("unique-feedback-event") == 1
    assert capsys.readouterr().err.count("unique-feedback-event") == 1
    assert logging.getLogger().level == root_level


def test_backend_relay_details_are_debug_only_and_native_logs_unchanged(capsys):
    native = logging.getLogger("test.native.cellbender")
    native.setLevel(logging.INFO)
    native.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    native.addHandler(handler)
    before = list(native.handlers)
    try:
        with cs.logging.relay_backend(native, "CellBender", epochs=12):
            for epoch in [1, 2, 10, 12]:
                native.info("[epoch %03d] training loss", epoch)
            native.warning("backend warning")
        assert native.handlers == before
        assert "epoch 002" in stream.getvalue()
        output = capsys.readouterr().err
        assert "epoch" not in output
        cs.settings.verbosity = "debug"
        with cs.logging.relay_backend(native, "CellBender", epochs=12):
            native.info("[epoch 002] training loss")
        assert "epoch 002" in capsys.readouterr().err
        assert "backend warning" in output
    finally:
        native.removeHandler(handler)


def test_feedback_does_not_misinterpret_dataframe_columns_as_anndata():
    frame = pd.DataFrame({"n_obs": [1, 2], "n_vars": [3, 4]})

    def identity(data):
        return data

    assert cs.logging.instrument_function(identity, "test.frame")(frame) is frame


def test_batch_failure_has_no_false_completion(capsys):
    with pytest.raises(RuntimeError), cs.logging.operation("outer"), cs.logging.batch("Model", "P8Post", 2, 4):
        raise RuntimeError("training failed")
    output = capsys.readouterr().err
    assert "FAIL Model [2/4] P8Post" in output
    assert "DONE Model [2/4] P8Post" not in output


def test_settings_validate_levels_and_intervals():
    cs.settings.verbosity = 2
    assert cs.settings.verbosity == "info"
    with pytest.raises(ValueError):
        cs.settings.verbosity = "invalid"
    with pytest.raises(ValueError):
        cs.settings.heartbeat_interval = np.nan


@pytest.mark.parametrize("level", ["info", "debug", "warning", "quiet"])
def test_verbosity_contract_for_nested_steps_and_batches(level, capsys):
    cs.settings.verbosity = level
    with cs.logging.operation("outer", "40 cells | layer=X | epochs=150"):
        with cs.logging.operation("nested", "internal parameters"):
            cs.logging.progress("checking SHA256")
        with cs.logging.batch("CellBender", "P1Pre", 1, 4, cells=40):
            cs.logging.info("singlets=38 | doublets=2")
        cs.logging.warning("test warning")
        cs.logging.error("test error")
    output = capsys.readouterr().err
    if level == "quiet":
        assert output == ""
    elif level == "warning":
        assert "test warning" in output
        assert "test error" not in output
        assert "START" not in output and "DONE" not in output
    else:
        assert "START outer" in output and "DONE outer" in output
        assert "[1/4] P1Pre" in output and "singlets=38" in output
        assert ("checking SHA256" in output) == (level == "debug")
        assert ("START nested" in output) == (level == "debug")
        assert ("layer=X" in output) == (level == "debug")


def test_info_does_not_start_periodic_output_thread():
    cs.settings.heartbeat_interval = 0.001
    with cs.logging.operation("quietly working"):
        assert not any(t.name == "cellscope-feedback" for t in threading.enumerate())


@pytest.mark.parametrize("level", ["warning", "quiet"])
def test_noninformational_modes_hide_banner(level, capsys):
    cs.settings.verbosity = level
    cs.logging.show_banner("test", force=True)
    assert capsys.readouterr().err == ""


def test_backend_verbose_defaults_follow_debug():
    for level in ["info", "warning", "quiet", "debug"]:
        cs.settings.verbosity = level
        assert cs.logging.verbose_enabled() == (level == "debug")
