import numpy as np
import pytest
import soundfile as sf
import torch

from esr.data import (
    FSD50KDataset, balanced_sample_weights, encode_labels, fix_length, load_split, load_vocab,
    load_waveforms, make_balanced_sampler, read_audio,
)


def test_load_vocab_order_and_mapping(fake_root):
    labels, l2i = load_vocab(fake_root)
    assert labels == ["Bark", "Siren", "Guitar", "Rain", "Speech"]
    assert l2i["Rain"] == 3


def test_encode_labels_multi_hot():
    y = encode_labels("Bark,Rain", {"Bark": 0, "Siren": 1, "Rain": 2})
    assert y.dtype == np.float32
    assert y.tolist() == [1.0, 0.0, 1.0]


def test_encode_labels_unknown_label_raises():
    with pytest.raises(KeyError):
        encode_labels("Nope", {"Bark": 0})


def test_load_split_sizes_paths_and_targets(fake_root):
    tr, ytr = load_split(fake_root, "train")
    va, yva = load_split(fake_root, "val")
    ev, yev = load_split(fake_root, "eval")
    assert (len(tr), len(va), len(ev)) == (12, 4, 8)
    assert ytr.shape == (12, 5) and yev.shape == (8, 5)
    assert "FSD50K.dev_audio_16k" in tr.path[0] and tr.path[0].endswith("1000.wav")
    assert "FSD50K.eval_audio_16k" in ev.path[0]
    assert yva[:, 1].sum() == 0  # val deliberately lacks class 1
    assert (ytr.sum(1) >= 1).all()


def test_load_split_bad_name_raises(fake_root):
    with pytest.raises(ValueError):
        load_split(fake_root, "test")


def test_read_audio_mono_float(fake_root):
    tr, _ = load_split(fake_root, "train")
    wav = read_audio(tr.path[0], 16000)
    assert wav.ndim == 1 and wav.dtype == np.float32 and len(wav) > 0


def test_read_audio_downmixes_stereo(tmp_path):
    p = tmp_path / "st.wav"
    sf.write(p, np.stack([np.ones(100), -np.ones(100)], 1).astype(np.float32) * 0.5, 16000)
    wav = read_audio(str(p), 16000)
    assert wav.shape == (100,) and np.allclose(wav, 0.0, atol=1e-3)


def test_read_audio_rejects_wrong_sample_rate(tmp_path):
    p = tmp_path / "hi.wav"
    sf.write(p, np.zeros(441, np.float32), 44100)
    with pytest.raises(ValueError, match="16000"):
        read_audio(str(p), 16000)


def test_fix_length_pads_short_clip():
    out = fix_length(np.ones(10, np.float32), 16)
    assert out.shape == (16,) and out[:10].sum() == 10 and out[10:].sum() == 0


def test_fix_length_crops_long_clip_from_start_without_rng():
    out = fix_length(np.arange(100, dtype=np.float32), 16)
    assert out.tolist() == list(range(16))


def test_fix_length_random_crop_stays_in_bounds():
    rng = np.random.default_rng(0)
    for _ in range(50):
        out = fix_length(np.arange(100, dtype=np.float32), 16, rng)
        assert out.shape == (16,) and np.all(np.diff(out) == 1)


def test_fix_length_empty_clip_becomes_silence():
    out = fix_length(np.zeros(0, np.float32), 8)
    assert out.shape == (8,) and not out.any()


def test_dataset_items_have_fixed_length(fake_root):
    tr, ytr = load_split(fake_root, "train")
    ds = FSD50KDataset(tr.path.tolist(), ytr, clip_samples=16000, sample_rate=16000, train=True)
    for i in range(len(ds)):
        wav, y = ds[i]
        assert wav.shape == (16000,) and wav.dtype == torch.float32
        assert y.shape == (5,)
    assert ds.targets is ytr


def test_balanced_weights_favor_rare_classes():
    targets = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], np.float32)
    w = balanced_sample_weights(targets)
    assert w[3] > w[0]
    sampler = make_balanced_sampler(targets, seed=0)
    assert len(list(sampler)) == 4


def test_load_waveforms_int16_matrix(fake_root):
    ev, _ = load_split(fake_root, "eval")
    waves = load_waveforms(ev.path.tolist(), clip_samples=16000, sample_rate=16000, num_threads=2)
    assert waves.shape == (8, 16000) and waves.dtype == np.int16
    assert np.abs(waves).max() > 1000
