"""Offline contract tests for the M4 encoder-depth research harness (no model downloads)."""

import ast
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "research" / "encoder-depth-m4"
sys.path.insert(0, str(HERE))
import common  # noqa: E402


@pytest.mark.parametrize("script", ["common.py", "parity.py", "extract.py", "probe.py", "benchmark.py"])
def test_research_scripts_compile_without_importing_model(script):
    ast.parse((HERE / script).read_text(encoding="utf-8"), filename=script)


@pytest.mark.parametrize("depth,projected", [(6, False), (9, False), (12, True)])
def test_early_exit_only_executes_requested_layers(monkeypatch, depth, projected):
    hidden = torch.zeros(1, 8, 1024)
    lens = torch.tensor([8])
    run = Mock(return_value=hidden)
    monkeypatch.setattr(common.E, "subsample", lambda tower, mel, mask: (hidden, lens))
    monkeypatch.setattr(common.E, "run_layers", run)
    output, actual_lens = common.early(object(), None, None, depth)
    assert output is hidden
    assert actual_lens is lens
    assert run.call_args.args[3:5] == (0, depth)
    assert run.call_args.kwargs["proj"] == projected


def test_detects_exact_train_dev_audio_collision(tmp_path):
    a = tmp_path / "train.wav"
    b = tmp_path / "dev.wav"
    a.write_bytes(b"identical audio bytes")
    b.write_bytes(b"identical audio bytes")
    with pytest.raises(RuntimeError, match="TRAIN_DEV_LEAKAGE"):
        common.no_overlap([{"audio": str(a)}], [{"audio": str(b)}])


def test_stable_sample_ids(tmp_path):
    row = {"audio": str(tmp_path / "x.wav"), "text": "test"}
    assert common.sample_key(row) == common.sample_key(row)
    assert common.sample_key(row) != common.sample_key({**row, "text": "different"})
