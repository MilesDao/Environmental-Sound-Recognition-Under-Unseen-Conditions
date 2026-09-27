from pathlib import Path

import pandas as pd
import pytest

from esr.cli import build_config, main, parse_args

TINY = ["--clip-seconds", "1", "--n-mels", "64", "--batch-size", "4", "--num-workers", "0", "--no-amp",
        "--no-pretrained", "--warmup-epochs", "0"]


def _args(fake_root, out, *extra):
    return ["--data-path", str(fake_root), "--out-dir", str(out), *TINY, *extra]


def test_build_config_maps_flags(fake_root, tmp_path):
    cfg = build_config(parse_args(["train", "--experiment", "effnet_robust", *_args(fake_root, tmp_path),
                                   "--epochs", "3", "--lr", "0.01"]))
    assert cfg.experiment == "effnet_robust" and cfg.model == "efficientnet_b0"
    assert cfg.epochs == 3 and cfg.lr == 0.01 and cfg.batch_size == 4
    assert cfg.pretrained is False and cfg.amp is False
    assert cfg.data_root == str(fake_root)


def test_defaults_keep_experiment_settings(fake_root, tmp_path):
    cfg = build_config(parse_args(["train", "--experiment", "cnn_baseline", "--data-path", str(fake_root)]))
    assert cfg.lr == 1e-3 and cfg.epochs == 20 and cfg.batch_size == 32 and cfg.amp is True


def test_quick_flag_sets_smoke_test_subset_and_separate_out_dir(fake_root, tmp_path):
    cfg = build_config(parse_args(["train", "--experiment", "cnn_baseline", "--data-path", str(fake_root),
                                   "--out-dir", str(tmp_path / "outputs"), "--quick"]))
    assert (cfg.max_train_clips, cfg.max_eval_clips, cfg.epochs) == (2000, 1000, 2)
    assert cfg.out_dir == str(tmp_path / "outputs_quick")


def test_data_path_is_auto_discovered(fake_root):
    cfg = build_config(parse_args(["train", "--experiment", "cnn_baseline",
                                   "--search-root", str(fake_root.parent.parent)]))
    assert cfg.data_root == str(fake_root)


def test_unknown_experiment_is_rejected(fake_root):
    with pytest.raises(SystemExit):
        parse_args(["train", "--experiment", "nope", "--data-path", str(fake_root)])


def test_train_evaluate_compare_end_to_end(fake_root, tmp_path):
    out = tmp_path / "outputs"
    main(["train", "--experiment", "cnn_baseline", *_args(fake_root, out), "--epochs", "1"])
    main(["evaluate", "--experiment", "cnn_baseline", *_args(fake_root, out), "--severities", "1"])
    csv = out / "cnn_baseline" / "robustness.csv"
    assert csv.exists() and len(pd.read_csv(csv)) == 6

    quick_out = tmp_path / "outputs_quick" / "cnn_baseline"
    quick_out.mkdir(parents=True)
    pd.read_csv(csv).assign(subset=True, mAP=0.0).to_csv(quick_out / "robustness.csv", index=False)

    table = tmp_path / "summary.csv"
    main(["compare", "--inputs", str(tmp_path / "**" / "robustness.csv"), "--out", str(table)])
    summary = pd.read_csv(table)
    assert summary["experiment"].tolist() == ["cnn_baseline"]
    assert summary["clean"].iloc[0] > 0  # subset (smoke-test) rows were excluded


def test_compare_without_results_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="robustness.csv"):
        main(["compare", "--inputs", str(tmp_path / "*.csv")])


@pytest.mark.parametrize("script", ["train", "evaluate", "compare"])
def test_scripts_run_from_any_directory(script, tmp_path):
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    res = subprocess.run([sys.executable, str(repo / "scripts" / f"{script}.py"), "--help"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert f"esr {script}" in res.stdout
