import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from esr.engine import (
    load_checkpoint, make_scheduler, predict, predict_array, save_checkpoint, set_seed, train_one_epoch,
)
from esr.models import build_model


@pytest.fixture
def setup(tiny_cfg):
    set_seed(0)
    cfg = tiny_cfg(corruption_aug_p=1.0, freq_mixstyle_p=0.0)
    model = build_model(cfg, n_classes=5)
    wav = torch.randn(8, cfg.clip_samples) * 0.1
    y = (torch.rand(8, 5) < 0.4).float()
    y[:, 0] = 1.0
    loader = DataLoader(TensorDataset(wav, y), batch_size=4, drop_last=True)
    return cfg, model, loader, wav


def _opt(model, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return opt, make_scheduler(opt, steps, 1), torch.amp.GradScaler("cuda", enabled=False)


def test_scheduler_warmup_then_cosine():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = make_scheduler(opt, total_steps=10, warmup_steps=2)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx(0.5) and lrs[1] == pytest.approx(1.0)
    assert lrs[-1] < lrs[2] and lrs[-1] >= 0.01


def test_train_one_epoch_updates_weights(setup):
    cfg, model, loader, _ = setup
    opt, sch, scaler = _opt(model, 2)
    before = [p.detach().clone() for p in model.parameters()]
    loss = train_one_epoch(model, loader, opt, sch, scaler, torch.device("cpu"), cfg,
                           np.random.default_rng(0), torch.Generator().manual_seed(0))
    assert np.isfinite(loss) and loss > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_predict_returns_probabilities(setup):
    cfg, model, loader, _ = setup
    y, p = predict(model, loader, torch.device("cpu"), amp=False)
    assert y.shape == p.shape == (8, 5)
    assert (p >= 0).all() and (p <= 1).all()


def test_predict_array_clean_vs_corrupted_and_deterministic(setup):
    cfg, model, _, wav = setup
    waves = (wav.numpy() * 32767).astype(np.int16)
    dev = torch.device("cpu")
    clean = predict_array(model, waves, dev, batch_size=3, amp=False)
    a = predict_array(model, waves, dev, 3, False, corruption="white_noise", severity=3, seed=1)
    b = predict_array(model, waves, dev, 3, False, corruption="white_noise", severity=3, seed=1)
    assert clean.shape == (8, 5)
    assert np.array_equal(a, b)
    assert not np.allclose(a, clean)


def test_checkpoint_roundtrip(setup, tmp_path):
    cfg, model, loader, _ = setup
    opt, sch, scaler = _opt(model, 2)
    model.frontend.set_stats(-3.0, 2.0)
    save_checkpoint(tmp_path / "c.pt", model, opt, sch, scaler, epoch=4, best_map=0.3, history=[{"epoch": 0}])
    other = build_model(cfg, n_classes=5)
    opt2, sch2, scaler2 = _opt(other, 2)
    state = load_checkpoint(tmp_path / "c.pt", other, opt2, sch2, scaler2)
    assert state["epoch"] == 4 and state["history"] == [{"epoch": 0}]
    assert other.frontend.mean.item() == pytest.approx(-3.0)
    for a, b in zip(model.state_dict().values(), other.state_dict().values()):
        assert torch.equal(a, b)
    assert not (tmp_path / "c.pt.tmp").exists()
