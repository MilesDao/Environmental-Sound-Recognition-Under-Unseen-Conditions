import numpy as np
import torch

from esr.augment import freq_mixstyle, make_spec_transform, mixup, random_eq, spec_augment
from esr.config import make_config


def test_mixup_disabled_returns_inputs():
    wav, y = torch.randn(4, 100), torch.eye(4)
    w2, y2 = mixup(wav, y, 0.0, np.random.default_rng(0))
    assert w2 is wav and y2 is y


def test_mixup_mixes_labels_convexly():
    torch.manual_seed(0)
    wav, y = torch.randn(4, 100), torch.eye(4)
    w2, y2 = mixup(wav, y, 0.5, np.random.default_rng(0))
    assert w2.shape == wav.shape and y2.shape == y.shape
    assert torch.allclose(y2.sum(1), torch.ones(4))
    assert (y2 >= 0).all() and (y2.max(1).values >= 0.5 - 1e-6).all()


def test_spec_augment_masks_without_touching_input():
    torch.manual_seed(0)
    x = torch.ones(8, 1, 64, 200)
    out = spec_augment(x)
    assert out.shape == x.shape
    assert (out == 0).any()
    assert (x == 1).all()


def test_spec_augment_tiny_input_is_safe():
    x = torch.ones(2, 1, 3, 3)
    assert spec_augment(x).shape == x.shape


def test_freq_mixstyle_p0_is_identity():
    x = torch.randn(4, 1, 16, 20)
    assert freq_mixstyle(x, p=0.0) is x


def test_freq_mixstyle_changes_per_frequency_stats():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 16, 50) * torch.arange(1, 5).view(4, 1, 1, 1).float()
    out = freq_mixstyle(x, p=1.0)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert not torch.allclose(out, x)


def test_freq_mixstyle_batch_of_one_is_near_identity():
    x = torch.randn(1, 1, 16, 50)
    assert torch.allclose(freq_mixstyle(x, p=1.0), x, atol=1e-4)


def test_make_spec_transform_per_experiment():
    assert make_spec_transform(make_config("cnn_baseline", spec_augment=False)) is None
    tf = make_spec_transform(make_config("effnet_robust"))
    assert tf(torch.randn(4, 1, 64, 101)).shape == (4, 1, 64, 101)


def test_random_eq_p0_is_identity():
    x = torch.randn(3, 16000)
    assert torch.equal(random_eq(x, 0.0, torch.Generator().manual_seed(0)), x)


def test_random_eq_changes_every_clip_keeps_rms_and_is_deterministic():
    x = torch.randn(4, 16000, generator=torch.Generator().manual_seed(1))
    a = random_eq(x, 1.0, torch.Generator().manual_seed(0))
    b = random_eq(x, 1.0, torch.Generator().manual_seed(0))
    assert torch.equal(a, b) and a.shape == x.shape
    assert all(not torch.allclose(a[i], x[i], atol=1e-3) for i in range(4))
    assert torch.allclose(a.pow(2).mean(-1), x.pow(2).mean(-1), rtol=1e-3)


def test_random_eq_gain_stays_within_max_db():
    x = torch.randn(2, 16000, generator=torch.Generator().manual_seed(2))
    y = random_eq(x, 1.0, torch.Generator().manual_seed(0), max_db=6.0)
    ratio_db = 10 * torch.log10(torch.fft.rfft(y).abs().pow(2) / torch.fft.rfft(x).abs().pow(2))
    assert (ratio_db.amax(-1) - ratio_db.amin(-1) <= 12.0 + 1e-2).all()


def test_random_eq_silent_clip_stays_finite():
    assert torch.isfinite(random_eq(torch.zeros(2, 16000), 1.0, torch.Generator().manual_seed(0))).all()
