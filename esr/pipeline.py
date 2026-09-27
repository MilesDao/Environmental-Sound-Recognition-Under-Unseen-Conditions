"""High-level entry points used by the Kaggle notebook."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .augment import make_spec_transform
from .corruptions import SEEN, UNSEEN
from .data import FSD50KDataset, load_split, load_vocab, load_waveforms, make_balanced_sampler
from .engine import (
    get_device, load_checkpoint, make_scheduler, predict, predict_array, save_checkpoint, set_seed,
    train_one_epoch,
)
from .metrics import mean_average_precision
from .models import build_model


def build_datasets(cfg):
    tr_meta, tr_y = load_split(cfg.data_root, "train")
    va_meta, va_y = load_split(cfg.data_root, "val")
    if cfg.max_train_clips:
        idx = np.random.default_rng(cfg.seed).permutation(len(tr_meta))[: cfg.max_train_clips]
        tr_meta, tr_y = tr_meta.iloc[idx].reset_index(drop=True), tr_y[idx]
    train_ds = FSD50KDataset(tr_meta["path"].tolist(), tr_y, cfg.clip_samples, cfg.sample_rate, train=True)
    val_ds = FSD50KDataset(va_meta["path"].tolist(), va_y, cfg.clip_samples, cfg.sample_rate, train=False)
    return train_ds, val_ds


def _loaders(cfg, train_ds, val_ds, device):
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                  persistent_workers=cfg.num_workers > 0)
    sampler = make_balanced_sampler(train_ds.targets, seed=cfg.seed) if cfg.balanced_sampling else None
    train_loader = DataLoader(train_ds, sampler=sampler, shuffle=sampler is None, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


@torch.no_grad()
def estimate_norm_stats(frontend, dataset, n_clips=256):
    frontend.set_stats(0.0, 1.0)
    device = frontend.mean.device
    feats = [frontend(dataset[i][0][None].to(device)).flatten() for i in range(min(n_clips, len(dataset)))]
    x = torch.cat(feats)
    return float(x.mean()), float(x.std())


def run_training(cfg):
    set_seed(cfg.seed)
    device = get_device()
    exp_dir = Path(cfg.out_dir) / cfg.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    labels, _ = load_vocab(cfg.data_root)
    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = _loaders(cfg, train_ds, val_ds, device)
    model = build_model(cfg, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg.epochs * steps, cfg.warmup_epochs * steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    last = exp_dir / "last.pt"
    if cfg.resume_from and not last.exists():
        shutil.copy(cfg.resume_from, last)
        for sibling in ("best.pt", "history.csv"):  # best.pt may never be rewritten in this session
            src = Path(cfg.resume_from).with_name(sibling)
            if src.exists() and not (exp_dir / sibling).exists():
                shutil.copy(src, exp_dir / sibling)
    if last.exists():
        state = load_checkpoint(last, model, optimizer, scheduler, scaler)
        start_epoch, best, history = state["epoch"] + 1, state["best_map"], list(state["history"])
        print(f"Resuming from {last} at epoch {start_epoch} (best val mAP {best:.4f})")
    else:
        mean, std = estimate_norm_stats(model.frontend, train_ds)
        model.frontend.set_stats(mean, std)
        start_epoch, best, history = 0, -1.0, []

    rng = np.random.default_rng(cfg.seed + start_epoch)
    gen = torch.Generator().manual_seed(cfg.seed + start_epoch)
    spec_transform = make_spec_transform(cfg)
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, rng, gen,
                               spec_transform)
        y, p = predict(model, val_loader, device, cfg.amp)
        val_map = mean_average_precision(y, p)
        history.append({"epoch": epoch, "train_loss": loss, "val_mAP": val_map,
                        "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0})
        print(f"[{cfg.experiment}] epoch {epoch + 1}/{cfg.epochs} loss={loss:.4f} "
              f"val_mAP={val_map:.4f} ({history[-1]['seconds']:.0f}s)")
        if val_map > best:
            best = val_map
            save_checkpoint(exp_dir / "best.pt", model, epoch=epoch, best_map=best, history=history)
        save_checkpoint(last, model, optimizer, scheduler, scaler, epoch=epoch, best_map=best, history=history)
        pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)
    return {"best_val_mAP": best, "history": history, "exp_dir": str(exp_dir)}


def run_robustness(cfg, checkpoint="best.pt", waves=None, targets=None):
    """Evaluate on the (uploader-disjoint) eval set: clean + every corruption x severity."""
    exp_dir = Path(cfg.out_dir) / cfg.experiment
    ckpt = exp_dir / checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found - call run_training(cfg) first (or set out_dir).")
    device = get_device()
    labels, _ = load_vocab(cfg.data_root)
    model = build_model(cfg, len(labels), pretrained=False).to(device)
    load_checkpoint(ckpt, model)

    if waves is None:
        meta, targets = load_split(cfg.data_root, "eval")
        if cfg.max_eval_clips:  # eval.csv is grouped by class, so take a seeded random subset, never the head
            idx = np.random.default_rng(cfg.seed).permutation(len(meta))[: cfg.max_eval_clips]
            meta, targets = meta.iloc[idx].reset_index(drop=True), targets[idx]
        print(f"Loading {len(meta)} eval clips into memory ...")
        waves = load_waveforms(meta["path"].tolist(), cfg.clip_samples, cfg.sample_rate)

    conditions = [("clean", "clean", 0)]
    conditions += [(name, "seen", s) for name in SEEN for s in cfg.severities]
    conditions += [(name, "unseen", s) for name in UNSEEN for s in cfg.severities]
    subset = bool(cfg.max_train_clips or cfg.max_eval_clips)  # smoke-test results must not mix with full runs
    rows = []
    for name, group, severity in conditions:
        scores = predict_array(model, waves, device, cfg.batch_size, cfg.amp,
                               corruption=None if group == "clean" else name,
                               severity=severity, seed=cfg.seed, sample_rate=cfg.sample_rate)
        m = mean_average_precision(targets, scores)
        rows.append({"experiment": cfg.experiment, "condition": name, "group": group,
                     "severity": severity, "mAP": m, "subset": subset})
        print(f"[{cfg.experiment}] {name:12s} sev={severity} ({group:6s}) mAP={m:.4f}")
    df = pd.DataFrame(rows, columns=["experiment", "condition", "group", "severity", "mAP", "subset"])
    df.to_csv(exp_dir / "robustness.csv", index=False)
    return df


def summarize(df):
    clean = df[df["group"] == "clean"].groupby("experiment")["mAP"].mean().rename("clean")
    groups = df[df["group"] != "clean"].groupby(["experiment", "group"])["mAP"].mean().unstack()
    out = pd.concat([clean, groups], axis=1)
    for g in ("seen", "unseen"):
        if g not in out:
            out[g] = np.nan
        out[f"{g}_rel"] = out[g] / out["clean"]
    return out[["clean", "seen", "unseen", "seen_rel", "unseen_rel"]].reset_index(names="experiment")
