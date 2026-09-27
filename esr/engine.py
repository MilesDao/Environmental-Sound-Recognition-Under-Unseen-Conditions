"""Training loop, inference and checkpointing."""
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .augment import mixup, random_eq
from .corruptions import apply_corruption, random_train_corruption


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device, amp):
    return torch.autocast(device_type=device.type, enabled=bool(amp) and device.type == "cuda")


def make_scheduler(optimizer, total_steps, warmup_steps):
    def factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, **extra):
    state = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    tmp = Path(f"{path}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a Kaggle kill mid-save never corrupts the last good checkpoint


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, rng, gen, spec_transform=None):
    model.train()
    total, count = 0.0, 0
    for wav, y in loader:
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.corruption_aug_p > 0:
            wav = random_train_corruption(wav, cfg.corruption_aug_p, gen, cfg.sample_rate, cfg.train_snr_db,
                                          cfg.train_rt60_s)
        if cfg.eq_aug_p > 0:
            wav = random_eq(wav, cfg.eq_aug_p, gen, cfg.sample_rate, cfg.eq_max_db)
        wav, y = mixup(wav, y, cfg.mixup_alpha, rng)
        with _autocast(device, cfg.amp):
            logits = model(wav, spec_transform)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * wav.size(0)
        count += wav.size(0)
    return total / max(count, 1)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ys, ps = [], []
    for wav, y in loader:
        with _autocast(device, amp):
            logits = model(wav.to(device, non_blocking=True))
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def predict_array(model, waves, device, batch_size, amp, corruption=None, severity=1, seed=0, sample_rate=16000):
    model.eval()
    out = []
    for start in range(0, len(waves), batch_size):
        wav = torch.from_numpy(waves[start:start + batch_size].astype(np.float32) / 32767.0).to(device)
        if corruption is not None:  # one generator per clip: identical noise for any batch_size
            wav = torch.cat([
                apply_corruption(wav[i:i + 1], corruption, severity,
                                 torch.Generator().manual_seed(seed * 100003 + start + i), sample_rate)
                for i in range(wav.size(0))
            ])
        with _autocast(device, amp):
            logits = model(wav)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)
