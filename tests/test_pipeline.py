import json
from pathlib import Path

import pandas as pd
import pytest

from esr.corruptions import SEEN, UNSEEN
from esr.pipeline import build_datasets, run_robustness, run_training, summarize


def test_build_datasets_respects_subset(tiny_cfg):
    tr, va = build_datasets(tiny_cfg(max_train_clips=5))
    assert len(tr) == 5 and len(va) == 4 and tr.train and not va.train


def test_run_training_writes_artifacts(tiny_cfg):
    cfg = tiny_cfg()
    res = run_training(cfg)
    exp = Path(res["exp_dir"])
    for name in ("config.json", "history.csv", "last.pt", "best.pt"):
        assert (exp / name).exists(), name
    assert json.loads((exp / "config.json").read_text())["experiment"] == "cnn_baseline"
    assert len(res["history"]) == 1
    assert 0.0 <= res["best_val_mAP"] <= 1.0  # val lacks class 1 -> must not be NaN


def test_run_training_resumes(tiny_cfg):
    first = run_training(tiny_cfg(epochs=1))
    second = run_training(tiny_cfg(epochs=2))
    assert [h["epoch"] for h in second["history"]] == [0, 1]
    assert second["history"][0] == first["history"][0]


def test_run_training_resume_from_other_dir(tiny_cfg, tmp_path):
    first = run_training(tiny_cfg(epochs=1))
    cfg = tiny_cfg(epochs=2, out_dir=str(tmp_path / "session2"),
                   resume_from=str(Path(first["exp_dir"]) / "last.pt"))
    res = run_training(cfg)
    assert [h["epoch"] for h in res["history"]] == [0, 1]


def test_resume_from_finished_run_keeps_best_checkpoint(tiny_cfg, tmp_path):
    first = run_training(tiny_cfg(epochs=1))
    cfg = tiny_cfg(epochs=1, out_dir=str(tmp_path / "session2"),
                   resume_from=str(Path(first["exp_dir"]) / "last.pt"))
    run_training(cfg)  # nothing left to train, so no new best.pt is written
    assert (Path(cfg.out_dir) / cfg.experiment / "best.pt").exists()
    run_robustness(cfg)  # must not raise FileNotFoundError


def test_robustness_csv_flags_subset_runs(tiny_cfg):
    full = tiny_cfg(severities=(1,))
    run_training(full)
    assert not run_robustness(full)["subset"].any()
    quick = tiny_cfg(severities=(1,), max_eval_clips=5)
    assert run_robustness(quick)["subset"].all()


def test_run_training_effnet_robust_smoke(tiny_cfg):
    res = run_training(tiny_cfg("effnet_robust", batch_size=4))
    assert len(res["history"]) == 1


def test_run_robustness_rows_and_groups(tiny_cfg):
    cfg = tiny_cfg(severities=(1,))
    run_training(cfg)
    df = run_robustness(cfg)
    assert list(df.columns) == ["experiment", "condition", "group", "severity", "mAP", "subset"]
    assert len(df) == 1 + len(SEEN) + len(UNSEEN)
    assert df.set_index("condition").loc["clean", "group"] == "clean"
    assert set(df[df.group == "seen"].condition) == set(SEEN)
    assert set(df[df.group == "unseen"].condition) == set(UNSEEN)
    assert df["mAP"].between(0, 1).all()
    assert (Path(cfg.out_dir) / cfg.experiment / "robustness.csv").exists()


def test_run_robustness_missing_checkpoint_raises(tiny_cfg):
    with pytest.raises(FileNotFoundError, match="run_training"):
        run_robustness(tiny_cfg(experiment="effnet_standard"))


def test_summarize():
    df = pd.DataFrame([
        ("a", "clean", "clean", 0, 0.5), ("a", "white_noise", "seen", 1, 0.4), ("a", "reverb", "seen", 1, 0.3),
        ("a", "telephone", "unseen", 1, 0.25),
        ("b", "clean", "clean", 0, 0.6), ("b", "white_noise", "seen", 1, 0.6), ("b", "telephone", "unseen", 1, 0.3),
    ], columns=["experiment", "condition", "group", "severity", "mAP"])
    s = summarize(df).set_index("experiment")
    assert s.loc["a", "seen"] == pytest.approx(0.35)
    assert s.loc["a", "unseen_rel"] == pytest.approx(0.5)
    assert s.loc["b", "seen_rel"] == pytest.approx(1.0)
