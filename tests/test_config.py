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


def test_experiments_are_exactly_the_three_in_the_spec():
    assert set(EXPERIMENTS) == {"cnn_baseline", "effnet_standard", "effnet_robust"}
    assert make_config("cnn_baseline").model == "simple_cnn"
    assert make_config("effnet_standard").corruption_aug_p == 0.0


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
