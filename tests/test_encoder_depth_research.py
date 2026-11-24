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


@pytest.mark.parametrize(
    "script", ["common.py", "parity.py", "extract.py", "probe.py", "benchmark.py", "prepare_data.py", "analyze.py", "collect.py"]
)
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


def test_executed_params_split_projection_from_layers():
    class Tower:
        subsample_conv_projection = torch.nn.Linear(2, 2)  # 6 params
        layers = [torch.nn.Linear(3, 3) for _ in range(12)]  # 12 params each
        output_proj = torch.nn.Linear(4, 4)  # 20 params

    assert common.executed_params(Tower, 6, proj=False) == 6 + 6 * 12
    assert common.executed_params(Tower, 12, proj=False) == 6 + 12 * 12
    assert common.executed_params(Tower, 12, proj=True) == 6 + 12 * 12 + 20


def test_ctc_probe_recipe_is_layer_agnostic():
    pytest.importorskip("embed2asr")
    import probe

    assert probe.dim_of("L6") == probe.dim_of("L12") == 1024 and probe.dim_of("L12-Proj") == 1536
    rows = [{"ids": [1, 1], "frames": 2}, {"ids": [1, 1], "frames": 3}, {"ids": [], "frames": 9}, {"ids": [2], "frames": 1}]
    kept, dropped = probe.eligible(rows)  # "1 1" needs 3 frames (blank between repeats); empty target is invalid
    assert [r["frames"] for r in kept] == [3, 1] and dropped == 2
    lrs = [probe.lr_at(s, 100, 1e-3) for s in range(100)]
    assert max(lrs) <= 1e-3 and lrs[0] < lrs[4] and lrs[-1] < 1e-5


def test_linear_ctc_loss_and_backward_on_padded_batch():
    pytest.importorskip("embed2asr")
    import probe

    head = probe.LinearCTC(8, 6)
    x, lens = torch.randn(2, 10, 8), torch.tensor([10, 7])
    loss = probe.ctc(head, x, lens, [[1, 2, 2], [3]])
    loss.backward()
    assert torch.isfinite(loss) and head.linear.weight.grad.abs().sum() > 0


def test_reference_text_pipeline_and_overlap_audit():
    pytest.importorskip("embed2asr")
    import prepare_data as pd

    assert pd.reference("Hello, World!", "en") == "hello world"
    assert pd.reference("在2023年", "zh") == "在二零二三年"
    assert pd.reference("a g2g link", "en") is None  # digits that numnorm cannot verbalise are dropped, as in the pipeline
    train = [{"text": "a b", "speaker": "s1", "sha256": "x"}]
    other = [{"text": "a b", "speaker": "s2", "sha256": "y"}, {"text": "c", "speaker": "s1", "sha256": "x"}]
    drop, speakers, same_audio = pd.overlap_report(train, other)
    assert drop == {0} and speakers == ["s1"] and same_audio == 1
