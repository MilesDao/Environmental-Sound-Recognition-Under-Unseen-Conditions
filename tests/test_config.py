import json

import pytest

from esr.config import EXPERIMENTS, Config, find_data_root, make_config


def test_default_clip_samples_is_ten_seconds_at_16k():
    assert Config().clip_samples == 160_000


def test_make_config_applies_experiment_then_overrides():
    cfg = make_config("effnet_robust", epochs=3)
    assert cfg.experiment == "effnet_robust"
    assert cfg.model == "efficientnet_b0"
    assert cfg.corruption_aug_p == 0.5 and cfg.freq_mixstyle_p == 0.7
    assert cfg.epochs == 3


def test_make_config_unknown_experiment_raises():
    with pytest.raises(KeyError, match="cnn_baseline"):
        make_config("does_not_exist")


def test_experiment_presets():
    assert set(EXPERIMENTS) == {"cnn_baseline", "effnet_standard", "effnet_robust", "effnet_robust_v2",
                                "effnet_mixstyle", "effnet_corrupt", "effnet_eq"}
    assert make_config("cnn_baseline").model == "simple_cnn"
    knobs = ("freq_mixstyle_p", "corruption_aug_p", "eq_aug_p")
    assert all(getattr(make_config("effnet_standard"), k) == 0.0 for k in knobs)
    v1 = make_config("effnet_robust")  # must stay exactly the Experiment 1 recipe
    assert (v1.freq_mixstyle_p, v1.corruption_aug_p, v1.eq_aug_p) == (0.7, 0.5, 0.0)
    assert v1.train_snr_db == (5.0, 30.0) and v1.train_rt60_s == (0.2, 0.8)
    v2 = make_config("effnet_robust_v2")
    assert (v2.freq_mixstyle_p, v2.corruption_aug_p, v2.eq_aug_p) == (0.7, 0.5, 0.5)
    assert v2.train_snr_db == (0.0, 30.0) and v2.train_rt60_s == (0.2, 1.0)
    for name, knob in [("effnet_mixstyle", "freq_mixstyle_p"), ("effnet_corrupt", "corruption_aug_p"),
                       ("effnet_eq", "eq_aug_p")]:
        cfg = make_config(name)
        assert {k for k in knobs if getattr(cfg, k) > 0} == {knob}
        assert cfg.model == "efficientnet_b0" and cfg.pretrained


def test_to_dict_is_json_serializable():
    json.dumps(make_config("cnn_baseline").to_dict())


def test_find_data_root_on_fake_kaggle_layout(fake_root):
    assert find_data_root(str(fake_root.parent.parent)) == str(fake_root)


def test_find_data_root_deep_nesting(tmp_path):
    deep = tmp_path / "datasets" / "yousirui1" / "fsd50k" / "fsd50k"
    (deep / "FSD50K.ground_truth").mkdir(parents=True)
    (deep / "FSD50K.ground_truth" / "vocabulary.csv").write_text("0,A,/m/a\n")
    assert find_data_root(str(tmp_path)) == str(deep)


def test_find_data_root_missing_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="yousirui1/fsd50k"):
        find_data_root(str(tmp_path))
