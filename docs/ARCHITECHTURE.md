# Architecture — Environmental Sound Recognition Under Unseen Conditions

> This document describes **all** of the code that was built: the research goal, directory layout, data flow,
> the design of each module, the CNN architecture, the training loop, the robustness evaluation suite, the
> checkpoint/resume mechanism, how to run on Kaggle, the testing strategy, and the **points to watch** when reading
> or modifying the code.
>
> Related documents:
> - Design spec: [`docs/superpowers/specs/2026-09-27-esr-unseen-conditions-design.md`](superpowers/specs/2026-09-27-esr-unseen-conditions-design.md)
> - Implementation plan: [`docs/superpowers/plans/2026-09-27-esr-unseen-conditions.md`](superpowers/plans/2026-09-27-esr-unseen-conditions.md)
> - Quick start: [`README.md`](../README.md)

---

## Table of contents

1. [Goal and research question](#1-goal-and-research-question)
2. [Architecture overview](#2-architecture-overview)
3. [Directory layout](#3-directory-layout)
4. [The FSD50K dataset on Kaggle](#4-the-fsd50k-dataset-on-kaggle)
5. [End-to-end data flow](#5-end-to-end-data-flow)
6. [Module-by-module design](#6-module-by-module-design)
7. [CNN architecture](#7-cnn-architecture)
8. [Training loop](#8-training-loop)
9. [Robustness evaluation: seen / unseen conditions](#9-robustness-evaluation-seen--unseen-conditions)
10. [Checkpoints and resume](#10-checkpoints-and-resume)
11. [Output files](#11-output-files)
12. [Running on Kaggle](#12-running-on-kaggle)
13. [Testing strategy](#13-testing-strategy)
14. [Points to watch in the code (important)](#14-points-to-watch-in-the-code)
15. [Compute budget](#15-compute-budget)
16. [Known limitations and deferred work](#16-known-limitations-and-deferred-work)
17. [Extension ideas](#17-extension-ideas)
18. [Quick reference](#18-quick-reference)

---

## 1. Goal and research question

**Question:** How much does a CNN environmental-sound classifier trained on FSD50K lose when the test audio is recorded
under **acoustic conditions it never saw during training** — and which training strategies close that gap?

"Unseen conditions" are measured in two ways:

| Kind of domain shift | How it is produced | What it tells us |
|---|---|---|
| **Natural** | FSD50K's `eval` split was recorded by **different uploaders** than the `dev` split (different microphones, rooms, devices) | Clean eval mAP already measures generalization |
| **Controlled** | 7 corruption types applied to the eval set, 3 severities each | Pinpoints which kind of condition the model is weak on |

The seven corruptions are split into two groups:

- **SEEN** (used as training augmentation by the `effnet_robust` model): `white_noise`, `reverb`.
- **UNSEEN** (test-time only, **never** used in training): `brown_noise`, `telephone`, `clipping`, `speed`, `quantize`.

The key question of the experiment: *does robustness training on the SEEN group transfer to the UNSEEN group?*

**Seven experiments:**

| Name | Model | Pretrained | Augmentation | Comparison it answers |
|---|---|---|---|---|
| `cnn_baseline` | SimpleCNN (VGG-style) | No | SpecAugment + mixup | Reference point |
| `effnet_standard` | EfficientNet-B0 | ImageNet | SpecAugment + mixup | Effect of architecture + pretraining |
| `effnet_robust` | EfficientNet-B0 | ImageNet | + Freq-MixStyle (p=0.7) + SEEN corruptions (p=0.5) | Effect of robustness training |
| `effnet_robust_v2` | EfficientNet-B0 | ImageNet | + Freq-MixStyle (p=0.7) + SEEN corruptions (p=0.5, widened to SNR 0–30 dB / RT60 0.2–1.0 s) + random EQ (p=0.5) | Does a generic spectral-envelope augmentation close the `telephone` gap? |
| `effnet_mixstyle` | EfficientNet-B0 | ImageNet | + Freq-MixStyle (p=0.7) only | Ablation: MixStyle alone |
| `effnet_corrupt` | EfficientNet-B0 | ImageNet | + SEEN corruptions (p=0.5) only | Ablation: corruption augmentation alone |
| `effnet_eq` | EfficientNet-B0 | ImageNet | + random EQ (p=0.5) only | Ablation: random EQ alone |

---

## 2. Architecture overview

All logic lives in **one small Python package, `esr/`** (Environmental Sound Recognition), which is tested locally on CPU
against a synthetic FSD50K. On Kaggle, the notebook **only clones the repo and calls scripts** — it contains no logic.

```
┌──────────────────────────── Kaggle notebook (notebooks/kaggle_run.ipynb) ────────────────────────────┐
│  git clone repo  →  pip install timm  →  !python scripts/train.py / evaluate.py / compare.py          │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                │  sys.argv
                                   scripts/*.py (3-line wrappers)
                                                │
                                         esr/cli.py  ── parse_args → build_config → Config
                                                │
                  ┌─────────────────────────────┼──────────────────────────────┐
                  ▼                             ▼                              ▼
       pipeline.run_training          pipeline.run_robustness           cli.compare → pipeline.summarize
                  │                             │
     ┌────────────┼──────────────┐     ┌────────┼────────────┐
     ▼            ▼              ▼     ▼        ▼            ▼
  data.py     models.py      engine.py  data.load_waveforms  engine.predict_array
 (Dataset,   (LogMel +       (train loop,                     └─ corruptions.apply_corruption
  sampler)    CNN + attn)     checkpoint)
                  │              │
             features.py    augment.py  corruptions.random_train_corruption
                                         metrics.mean_average_precision
```

**Design principles:**

1. **One responsibility per file** — data, features, augmentation, corruptions, models, metrics, engine, pipeline and CLI are separate.
2. **Runs on CPU and CUDA** — no code path requires a GPU; AMP is only enabled when CUDA is available.
3. **All configuration in one `Config` dataclass** — the CLI is only a flag → field mapping layer.
4. **Reproducible** — evaluation noise comes from generators with fixed seeds, identical for every model.
5. **Survives Kaggle killing the session** — checkpoints are written atomically and resume is automatic.

---

## 3. Directory layout

```
.
├── README.md                     Quick start (local + Kaggle)
├── pyproject.toml                pytest config (pythonpath=".")
├── requirements-dev.txt          Development dependencies (CPU torch)
├── .gitignore
├── docs/
│   ├── ARCHITECHTURE.md          (this document)
│   └── superpowers/specs|plans/  Design spec and implementation plan
├── esr/                          MAIN PACKAGE
│   ├── __init__.py
│   ├── config.py                 Config, EXPERIMENTS, make_config, find_data_root
│   ├── data.py                   Metadata/audio loading, Dataset, balanced sampler, in-RAM eval loading
│   ├── features.py               LogMel front end (mean/std normalization buffers)
│   ├── augment.py                mixup, SpecAugment, Frequency-MixStyle
│   ├── corruptions.py            SEEN/UNSEEN corruption suite + training-time corruption augmentation
│   ├── models.py                 AttentionPool, SimpleCNN, TimmCNN, AudioModel, build_model
│   ├── metrics.py                mean_average_precision (macro mAP)
│   ├── engine.py                 seed, scheduler, checkpoints, train_one_epoch, predict, predict_array
│   ├── pipeline.py               run_training, run_robustness, summarize
│   └── cli.py                    Command-line interface: train / evaluate / compare
├── scripts/
│   ├── train.py                  → esr.cli.main(["train", ...])
│   ├── evaluate.py               → esr.cli.main(["evaluate", ...])
│   └── compare.py                → esr.cli.main(["compare", ...])
├── notebooks/
│   └── kaggle_run.ipynb          Kaggle notebook: clone repo + run scripts
└── tests/                        130 tests, run against synthetic data
    ├── conftest.py               Fixtures: fake FSD50K (fake_root) + tiny_cfg
    ├── test_config.py  test_data.py  test_models.py  test_augment.py
    ├── test_corruptions.py  test_metrics.py  test_engine.py  test_pipeline.py
    ├── test_cli.py               CLI + running the scripts via subprocess
    └── test_kaggle_runner.py     Every `!python scripts/...` line in the notebook must parse
```

**Import rules:** every module in `esr/` uses **relative imports** (`from .config import Config`), so the package works
wherever the repo is cloned. The scripts in `scripts/` add the repo root to `sys.path` themselves, so they run from
**any working directory** without `pip install -e .`.

---

## 4. The FSD50K dataset on Kaggle

Kaggle mirror: `yousirui1/fsd50k` (61,441 files, 21.4 GB). The layout was verified through the Kaggle API:

```
<DATA_ROOT>/                                   e.g. /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
├── FSD50K.dev_audio_16k/    40,966 wav  9.30 GB   16 kHz, mono, 16-bit PCM  (train + val)
├── FSD50K.eval_audio_16k/   10,231 wav  3.21 GB   16 kHz, mono, 16-bit PCM  (test)
├── FSD50K.eval_audio/       10,231 wav  8.84 GB   original 44.1 kHz         (NOT used)
├── FSD50K.ground_truth/
│   ├── vocabulary.csv       NO header: index,label,mid              (200 classes)
│   ├── dev.csv              fname,labels,mids,split   split ∈ {train: 36,796, val: 4,170}
│   └── eval.csv             fname,labels,mids                          (10,231 rows)
├── FSD50K.metadata/         json + collection/*.csv  (not used)
└── FSD50K.doc/              README.md, LICENSE
```

Important properties:

- **Multi-label:** 2.79 labels per clip on average, following the AudioSet ontology (parent labels are included, e.g.
  `"Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music"`). → The loss is **BCE** and the metric is **macro mAP**.
- **Variable clip length:** 0.3 s – 30 s. → Cropped/padded to 10 s.
- **Strong class imbalance.** → Balanced sampling.
- **`eval.csv` is ordered by class** (the first rows are all Electric_guitar). → Subsets must be **seeded random samples**,
  never `head(N)` (see [§14](#14-points-to-watch-in-the-code)).
- This mirror has **no 44.1 kHz dev folder**, so the pipeline **uses only the `*_16k` folders**; any file that is not 16 kHz
  is rejected with a clear error.

---

## 5. End-to-end data flow

### 5.1 Training

```
dev.csv ──load_split("train")──► (meta: fname,path) + targets [N,200] float32 multi-hot
                                        │
                     FSD50KDataset.__getitem__(i)
                        read_audio (soundfile, float32, mono, 16 kHz check)
                        fix_length → 160,000 samples (random crop if long, zero-pad if short)
                                        │   (wav [L], y [200])
         DataLoader (class-balanced WeightedRandomSampler, batch 32, 4 workers, drop_last)
                                        │   wav [B,160000] ─► GPU
               random_train_corruption (white noise / reverb, p=corruption_aug_p; ranges configurable via
                                         cfg.train_snr_db / cfg.train_rt60_s, default 5-30 dB / 0.2-0.8 s)
                                        │
                     random_eq (FilterAugment-style ±eq_max_db tilt, p=eq_aug_p; runs only if eq_aug_p > 0)
                                        │
                     waveform mixup (λ ~ Beta(0.5,0.5), λ ≥ 0.5)
                                        │
          AudioModel.forward(wav, spec_transform)
             LogMel: MelSpectrogram → log(x+1e-6) → (x-mean)/std   [B,1,128,1001]  (fp32, autocast off)
             spec_transform: Freq-MixStyle (robust) → SpecAugment
             CNN backbone → mean over frequency → [B,C,T'] → AttentionPool → logits [B,200]
                                        │
          BCEWithLogits(logits.float(), y_mixed) → GradScaler → AdamW → LambdaLR (warmup + cosine)
                                        │
       End of each epoch: predict(clean val) → macro mAP → save best.pt (if better) + last.pt + history.csv
```

### 5.2 Robustness evaluation

```
eval.csv ──load_split("eval")──► paths + targets
         load_waveforms (8 threads) → int16 matrix [10,231, 160,000] in RAM (~3.3 GB)
                                        │  (loaded ONCE for all 22 conditions)
   for (condition, group, severity) in [clean] + SEEN×{1,2,3} + UNSEEN×{1,2,3}:   → 22 passes
        predict_array: batch → float32/32767 → GPU
                       generator seed = seed*100003 + clip_index  → apply_corruption
                       → model → sigmoid → scores
        mean_average_precision(targets, scores)
                                        │
                     robustness.csv (experiment, condition, group, severity, mAP, subset, protocol)
```

---

## 6. Module-by-module design

### 6.1 `esr/config.py` — Configuration

**`Config` (dataclass)** — the single source of truth for every hyperparameter:

| Group | Field | Default | Notes |
|---|---|---|---|
| Paths | `data_root` | `""` | Folder that contains `FSD50K.ground_truth/` |
| | `out_dir` | `/kaggle/working/outputs` | Output goes to `<out_dir>/<experiment>/` |
| Experiment | `experiment`, `model`, `pretrained` | `effnet_robust`, `efficientnet_b0`, `True` | `model="simple_cnn"` or any `timm` model name |
| Audio | `sample_rate`, `clip_seconds` | 16000, 10.0 | `clip_samples` (property) = 160,000 |
| Features | `n_mels`, `n_fft`, `win_length`, `hop_length`, `f_min`, `f_max` | 128, 1024, 400 (25 ms), 160 (10 ms), 50, 8000 | |
| Optimization | `batch_size`, `epochs`, `lr`, `weight_decay`, `warmup_epochs` | 32, 20, 5e-4, 1e-2, 1 | |
| System | `num_workers`, `amp`, `seed` | 4, True, 42 | |
| Augmentation | `balanced_sampling`, `spec_augment`, `mixup_alpha`, `freq_mixstyle_p`, `corruption_aug_p` | True, True, 0.5, 0.0, 0.0 | `mixup_alpha=0` disables mixup |
| | `train_snr_db`, `train_rt60_s` | (5.0, 30.0), (0.2, 0.8) | Ranges used by `random_train_corruption` (the seen-family training augmentation) |
| | `eq_aug_p`, `eq_max_db` | 0.0, 12.0 | Probability and max ±dB tilt of `random_eq` (runs after `random_train_corruption`, before mixup) |
| Evaluation | `severities` | (1,2,3) | |
| Subsets | `max_train_clips`, `max_eval_clips` | 0 (= all) | Used by the `--quick` smoke test |
| Resume | `resume_from` | `""` | Path to a previous session's `last.pt` |

`to_dict()` turns the `severities` tuple into a list so it can be written as JSON (`config.json`).

**`EXPERIMENTS`** — seven presets that only hold what differs from the defaults:

```python
_EFFNET = dict(model="efficientnet_b0", pretrained=True, lr=5e-4)

EXPERIMENTS = {
    "cnn_baseline": dict(model="simple_cnn", pretrained=False, lr=1e-3),
    "effnet_standard": dict(_EFFNET),
    "effnet_robust": dict(_EFFNET, freq_mixstyle_p=0.7, corruption_aug_p=0.5),  # Experiment 1
    # v2: + random EQ, and seen-corruption ranges widened to cover severity 3 (0 dB SNR, RT60 1.0 s)
    "effnet_robust_v2": dict(_EFFNET, freq_mixstyle_p=0.7, corruption_aug_p=0.5, eq_aug_p=0.5,
                             train_snr_db=(0.0, 30.0), train_rt60_s=(0.2, 1.0)),
    # ablations: exactly one robustness component each (v1 ranges)
    "effnet_mixstyle": dict(_EFFNET, freq_mixstyle_p=0.7),
    "effnet_corrupt": dict(_EFFNET, corruption_aug_p=0.5),
    "effnet_eq": dict(_EFFNET, eq_aug_p=0.5),
}
```

**`make_config(experiment, **overrides)`** — precedence: *Config defaults* < *experiment preset* < *overrides*.
An unknown experiment name raises `KeyError` listing the valid names.

**`find_data_root(search_root="/kaggle/input", max_depth=5)`** — searches for `FSD50K.ground_truth/vocabulary.csv` at
**increasing depth** (0, 1, …, 5) and returns the parent of `FSD50K.ground_truth`. Reason: Kaggle mounts datasets at
different paths over time (`/kaggle/input/fsd50k/fsd50k` is depth 2; `/kaggle/input/datasets/yousirui1/fsd50k/fsd50k` is
depth 4). Searching shallow first avoids **listing folders with 40,000 audio files**. If nothing is found it raises
`FileNotFoundError` with the hint "Add Input → yousirui1/fsd50k".

### 6.2 `esr/data.py` — Data

| Function/Class | Role |
|---|---|
| `AUDIO_DIRS` | `train`/`val` → `FSD50K.dev_audio_16k`, `eval` → `FSD50K.eval_audio_16k` |
| `load_vocab(root)` | Reads `vocabulary.csv` (**no header**), sorted by `index` → `(labels, label2idx)` |
| `encode_labels(str, label2idx)` | `"A,B"` → float32 multi-hot vector [200]. Unknown label → `KeyError` (vocab mismatches surface early) |
| `load_split(root, split)` | Returns `(DataFrame[fname, path], targets [N,200])`. `split` ∉ {train,val,eval} → `ValueError` |
| `read_audio(path, sr)` | `soundfile.read(..., always_2d=True)` → average over channels (stereo downmix) → 1-D float32. **Wrong sample rate → `ValueError`** |
| `fix_length(wav, n, rng=None)` | Longer than `n`: crop (random if `rng` is given, from the start otherwise). Shorter: zero-pad at the end. Empty clip → all zeros |
| `FSD50KDataset` | Each item: `(torch.float32 [clip_samples], torch.float32 [200])`. Train = random crop; val/eval = first 10 s |
| `balanced_sample_weights(targets)` | A clip's weight = Σ (1000 / class frequency) over its labels (PSLA-style) |
| `make_balanced_sampler(targets, seed)` | `WeightedRandomSampler` with replacement and a fixed-seed generator |
| `load_waveforms(paths, ...)` | Loads many clips in parallel (8-thread pool) into an **int16** matrix `[N, clip_samples]` |

Details worth noting:
- In `__getitem__`, `np.random.default_rng()` is created **fresh for every item** (seeded from OS entropy), so DataLoader
  workers **do not produce the same crop positions** (the classic bug when a global `np.random` state is shared by workers).
- The eval set is stored as int16 instead of float32 to **halve RAM usage** (3.3 GB instead of 6.5 GB); it is converted back
  to float with `/32767` at prediction time.

### 6.3 `esr/features.py` — Log-mel front end

`LogMel(nn.Module)`:

```
wav [B, L]  ──MelSpectrogram(n_fft=1024, win=400, hop=160, 128 mels, 50–8000 Hz, power=2)──►  [B,128,T]
            ──log(x + 1e-6)──►  ──(x − mean) / std──►  unsqueeze(1)  ──►  [B, 1, 128, T]    T = L/160 + 1
```

- `mean` and `std` are **buffers** (`register_buffer`), so they are saved in the `state_dict`/checkpoint together with the
  model weights. Loading a checkpoint for evaluation therefore **does not need to recompute** the normalization statistics.
- `set_stats(mean, std)` sets them (std is floored at 1e-5 to avoid division by zero).
- The whole `forward` runs inside `torch.autocast(enabled=False)` and casts `wav.float()`, so **the log-mel is always computed
  in fp32**, even when AMP is on (the log of very small values in fp16 easily becomes `-inf`/NaN).
- `from_config(cfg)` builds the front end from a `Config`.

### 6.4 `esr/augment.py` — Augmentation

| Function | Domain | Description |
|---|---|---|
| `mixup(wav, y, alpha, rng)` | waveform | λ ~ Beta(α,α), forced to λ ≥ 0.5 (the original clip always dominates); labels are mixed too. `alpha ≤ 0` → inputs returned unchanged |
| `spec_augment(x, freq_width=24, time_width=100, n_masks=2)` | spectrogram | 2 frequency masks + 2 time masks, **independent per sample** in the batch. Implemented in-house, not via torchaudio |
| `_mask_along(x, max_width, axis)` | | Mask width capped at `size // 4`; sizes that are too small are skipped. Uses `masked_fill` (does not modify the input in place) |
| `freq_mixstyle(x, p, alpha=0.3)` | spectrogram | **Frequency-wise MixStyle** (Schmid et al., 2022): normalizes by the statistics of **each frequency band** (mean/std over channel+time), then mixes those statistics between samples in the batch. Simulates the frequency response of different microphones/channels |
| `make_spec_transform(cfg)` | | Chains Freq-MixStyle (if `freq_mixstyle_p>0`) **then** SpecAugment (if enabled). Nothing enabled → `None` |

Why MixStyle comes **before** SpecAugment: masking first would put zero regions into the spectrogram and skew the
per-frequency mean/std.

Why SpecAugment is implemented in-house: torchaudio ≥ 2.9 is in maintenance mode and has dropped many APIs; writing it
ourselves avoids a version dependency. In the whole package, torchaudio is used **only** for `MelSpectrogram`.

`random_eq(wav, p, gen, sample_rate, max_db=12.0, n_points=(3, 6))` — FilterAugment-style (Nam et al., 2022) random EQ on
the **waveform**, per clip with probability `p`: a gain curve that is piecewise-linear in dB over log-frequency through
3–6 random anchors in `[-max_db, +max_db]` (±12 dB by default), applied via one `rfft`/`irfft` pass, with the clip's RMS
restored afterwards so the augmentation changes tone, not loudness. In `train_one_epoch` it runs **after** the seen-family
`random_train_corruption` and **before** mixup, and only when `cfg.eq_aug_p > 0`.

### 6.5 `esr/corruptions.py` — Acoustic corruption suite

Every corruption function has **the same signature**:

```python
fn(wav: Tensor[B, L], severity: int ∈ {1,2,3}, gen: torch.Generator (CPU), sample_rate=16000) -> Tensor[B, L]
```

| Name | Group | Severity 1 / 2 / 3 | Implementation |
|---|---|---|---|
| `white_noise` | SEEN | SNR 20 / 10 / 0 dB | `colored_noise(exponent=0)` + `add_noise_at_snr` |
| `reverb` | SEEN | RT60 0.3 / 0.6 / 1.0 s | Synthetic RIR: white noise × `exp(-6.9078·t/RT60)` (exactly −60 dB at t=RT60), sample 0 = 1 (direct path), energy-normalized; FFT convolution; **keeps the signal's RMS** |
| `brown_noise` | UNSEEN | SNR 20 / 10 / 0 dB | Power spectrum ~ 1/f² (shaped in the FFT domain) |
| `telephone` | UNSEEN | pass band 100–5000 / 300–3400 / 500–2000 Hz | Mask in the `rfft` domain, then `irfft` |
| `clipping` | UNSEEN | threshold 50 % / 20 % / 5 % of peak | Clamp at ±threshold, then re-amplify to **keep the original peak amplitude** |
| `speed` | UNSEEN | ×1.05 / ×1.15 / ×1.3 | linear resample, zero-pad the tail (no anti-aliasing; content above ~8 kHz / factor folds back) |
| `quantize` | UNSEEN | 8 / 6 / 4 bits | uniform steps relative to the clip peak |

Helpers:

- `colored_noise(shape, exponent, gen, device)` — unit-RMS noise with spectrum ~ 1/f^exponent (0 white, 1 pink, 2 brown), DC removed.
- `add_noise_at_snr(wav, noise, snr_db)` — **SNR is measured on the audible part** (`_active_rms`: RMS over non-zero samples),
  so the zero padding of short clips does not "dilute" the signal level. A completely silent clip uses a −60 dBFS floor (1e-3)
  instead of producing NaN or adding nothing.
- `apply_corruption(wav, name, severity, gen, sr)` — looks up `CORRUPTIONS`; an unknown name raises `KeyError`.
- `random_train_corruption(wav, p, gen, sr, snr_db=(5.0, 30.0), rt60_s=(0.2, 0.8))` — training augmentation: for each clip,
  with probability `p`, applies **one** SEEN corruption with **continuous parameters** drawn from `snr_db`/`rt60_s` (the
  defaults shown; `run_training` passes `cfg.train_snr_db`/`cfg.train_rt60_s`, so the range is configurable per experiment
  preset). It only calls SEEN-group functions, so **UNSEEN corruptions cannot leak into training**.

**Important — determinism:** every random number is drawn **on the CPU** from the `torch.Generator` that is passed in, and only
then moved with `.to(device)`. The same seed therefore gives **exactly the same noise signal** on CPU and GPU, and every model is
evaluated on **the same corrupted audio** → a fair comparison.

**Protocol v2:** noise SNR is measured on the 50–8000 Hz band (`SNR_BAND_HZ`); v1 used broadband RMS, which made brown
noise ~29 dB milder in-band. `PROTOCOL` is written to every row, and `compare` keeps only the latest protocol.

### 6.6 `esr/models.py` — Models

- `AttentionPool(in_ch, n_classes)` — two 1×1 `Conv1d`s: one branch for per-frame logits (`cla`), one for attention weights
  (`att`, clamped to [−10, 10], then softmax over time). Output = Σ_t logit_t × att_t → **logits** [B, K].
- `SimpleCNN(n_classes, channels=(64,128,256,512))` — 4 blocks (Conv3×3-BN-ReLU ×2 + 2×2 AvgPool), mean over the frequency
  axis, dropout 0.3, `AttentionPool`. Trained from scratch.
- `TimmCNN(name, n_classes, pretrained)` — `timm.create_model(name, in_chans=1, num_classes=0, global_pool="")`; uses
  `forward_features` → mean over frequency → dropout → `AttentionPool`. With `in_chans=1`, timm **sums the RGB channels of the
  first conv** in the ImageNet weights to adapt them to 1-channel input.
- `AudioModel(frontend, net)` — wraps front end + network; `forward(wav, spec_transform=None)`: spectrogram augmentation is
  inserted **between** the front end and the CNN, so the model takes **raw waveforms** as input.
- `build_model(cfg, n_classes, pretrained=None)` — `None` means use `cfg.pretrained`. For evaluation, `run_robustness` calls it
  with `pretrained=False` so that **ImageNet weights are not downloaded again** (they would be overwritten by the checkpoint anyway).

Tensor shapes are detailed in [§7](#7-cnn-architecture).

### 6.7 `esr/metrics.py` — Metrics

`mean_average_precision(y_true, y_score)` — **macro mAP** (the FSD50K standard) via `sklearn.metrics.average_precision_score`,
**only over classes with ≥ 1 positive example**. Classes without positives are skipped (their AP would be undefined → NaN).
If no class has a positive example → `ValueError`. This situation is very common with subsets (`--quick`).

### 6.8 `esr/engine.py` — Engine

| Function | Description |
|---|---|
| `set_seed(seed)` | Seeds `random`, `numpy`, `torch` |
| `get_device()` | CUDA if available, otherwise CPU |
| `_autocast(device, amp)` | Enables autocast only when `amp=True` **and** the device is CUDA |
| `make_scheduler(opt, total_steps, warmup_steps)` | Per-**step** `LambdaLR`: linear warmup → cosine down to **0.01 × lr** (never 0) |
| `save_checkpoint(path, model, optimizer, scheduler, scaler, **extra)` | Writes `path.tmp`, then `os.replace` → **atomic write** |
| `load_checkpoint(path, model, optimizer, scheduler, scaler)` | `torch.load(map_location="cpu", weights_only=False)`; restores whatever objects are passed; returns the full state (including `epoch`, `best_map`, `history`) |
| `train_one_epoch(...)` | One epoch of training (details in [§8](#8-training-loop)); returns the mean loss |
| `predict(model, loader, device, amp)` | Iterates a DataLoader → `(y_true, sigmoid(logits))` as numpy |
| `predict_array(model, waves_int16, ..., corruption, severity, seed)` | Predicts on the in-RAM int16 matrix; every clip is corrupted with its own generator, seeded `seed·100003 + clip_index`, so eval noise no longer depends on `batch_size` |

### 6.9 `esr/pipeline.py` — High-level entry points

- `build_datasets(cfg)` — builds the train/val Datasets; `max_train_clips > 0` → a **seeded random subset**.
- `_loaders(cfg, ...)` — train DataLoader (balanced sampler or shuffle, `drop_last=True` so a batch of 1 cannot break BatchNorm)
  and val DataLoader; `pin_memory` with CUDA, `persistent_workers` when `num_workers > 0`.
- `estimate_norm_stats(frontend, dataset, n_clips=256)` — computes the log-mel mean/std over up to 256 training clips (first
  resetting the stats to (0,1) so it measures raw values).
- `run_training(cfg)` — see [§8](#8-training-loop) and [§10](#10-checkpoints-and-resume).
- `run_robustness(cfg, checkpoint="best.pt", waves=None, targets=None)` — see [§9](#9-robustness-evaluation-seen--unseen-conditions).
  Already-loaded `waves`/`targets` can be passed in to reuse the eval data when evaluating several models in one session.
- `summarize(df)` — collapses to one row per experiment:
  `clean`, `seen` (mean over every SEEN condition × severity), `unseen`, `seen_rel = seen/clean`, `unseen_rel = unseen/clean`.

### 6.10 `esr/cli.py` and `scripts/` — Command-line interface

Three subcommands:

| Command | Calls | What it does |
|---|---|---|
| `train` | `run_training(cfg)` | Trains one experiment (resumes automatically) |
| `evaluate` | `run_robustness(cfg, checkpoint)` | 22 conditions on the eval set → `robustness.csv` |
| `compare` | `compare(inputs, include_subset, out)` | Collects every `robustness.csv` → summary table + `summary.csv` + `summary_by_condition.csv` |

How flags map to `Config` (`build_config`):
1. A flag that is **not passed** **does not override** anything → the experiment preset's default stays (e.g. `cnn_baseline` keeps `lr=1e-3`).
2. `--quick` sets `max_train_clips=2000, max_eval_clips=1000, epochs=2` and changes `out_dir` to `..._quick`
   (`/kaggle/working/outputs_quick` or `<--out-dir>_quick`) → **smoke-test results never mix with real runs**.
3. `--no-amp`, `--no-pretrained`, `--severities 1 2 3` map directly.
4. If `--data-path` is omitted → `find_data_root(--search-root)`.
5. `--checkpoint` (evaluate only) defaults to `best.pt` in `<out_dir>/<experiment>/`, but also accepts an **absolute
   path** (e.g. a `best.pt` attached from `/kaggle/input/...`) — `run_robustness` always writes its results to
   `<out_dir>/<experiment>/`, even when the checkpoint itself comes from elsewhere.

`compare`:
- By default searches `/kaggle/working/outputs/*/robustness.csv` and `/kaggle/input/**/robustness.csv`.
- **Drops rows with `subset=True`** (`--quick` results) unless `--include-subset` is passed; if only subset results remain it fails
  with a clear message.
- **Keeps only the latest evaluation protocol** (the `protocol` column): rows from an older protocol are dropped, with a
  printed warning naming the affected experiments. CSVs without a `protocol` column (written before protocols existed)
  count as protocol 1.
- Prints the **source file** of each experiment so the user can check where every row of the table came from.
- Older CSVs without a `subset` column are treated as `False` (backward compatible).
- Writes the summary table to `--out` (default `summary.csv`) and a per-condition-and-severity relative-mAP table
  (`per_condition`) to `<out stem>_by_condition.csv` (default `summary_by_condition.csv`).

`scripts/{train,evaluate,compare}.py` are 3-line wrappers: they add the repo root to `sys.path`, then call
`main(["<command>", *sys.argv[1:]])`.

### 6.11 `notebooks/kaggle_run.ipynb` — Kaggle runner notebook

Contains no logic; only: clone the repo (`REPO_URL`, `BRANCH`) → `pip install timm` → set `EXPERIMENT`, `DATA_PATH` →
`--quick` smoke test → real run → `compare.py` → plot `summary.csv`. The test `test_kaggle_runner.py` checks that every
`!python scripts/...` line in the notebook **parses with the real CLI**, so the notebook cannot drift from the code.

---

## 7. CNN architecture

### 7.1 Why EfficientNet-B0 + attention pooling

| Candidate | Pros | Cons | Role |
|---|---|---|---|
| VGG-style CNN trained from scratch | Simple and transparent; comparable to the FSD50K paper baselines (VGG-like ≈ 0.43 mAP reported) | Needs many epochs, overfits rare classes | `cnn_baseline` |
| **EfficientNet-B0, ImageNet-pretrained, + attention pooling (PSLA recipe, Gong et al. 2021)** | Best accuracy-per-FLOP CNN on FSD50K (PSLA reports ≈ 0.55+ mAP with B2); ~5 M parameters, fits a T4 with AMP; attention ignores noisy frames | Needs Internet to download weights | `effnet_standard`, `effnet_robust` |
| PANNs CNN14 (AudioSet) | Strong audio transfer | 80 M parameters, slow on a T4 | Extension |
| AST / HTS-AT (transformers) | Top of the leaderboard | Not a CNN; too heavy for Kaggle quotas | Out of scope |

> The mAP figures above come from the published literature — verify them against the papers before citing.

**Why attention pooling helps robustness:** the model learns a weight for each time frame, so it can concentrate on frames
that contain the sound event and ignore frames that hold only noise or zero padding, instead of averaging everything equally
as global average pooling does.

### 7.2 Tensor shapes (10 s clip, 128 mels)

**EfficientNet-B0** (`effnet_*`):

```
wav                        [B, 160000]
LogMel                     [B, 1, 128, 1001]
spec_transform (train)     [B, 1, 128, 1001]
backbone.forward_features  [B, 1280, 4, 32]      (stride 32 on both axes)
mean(dim=2) over frequency [B, 1280, 32]
Dropout(0.3)
AttentionPool              [B, 200]  logits
```

**SimpleCNN** (`cnn_baseline`):

```
LogMel                     [B, 1, 128, 1001]
block(1→64)   + AvgPool2   [B, 64, 64, 500]
block(64→128) + AvgPool2   [B, 128, 32, 250]
block(128→256)+ AvgPool2   [B, 256, 16, 125]
block(256→512)+ AvgPool2   [B, 512, 8, 62]
mean over frequency        [B, 512, 62]
Dropout + AttentionPool    [B, 200]
```

Approximate parameter counts (hand-calculated): EfficientNet-B0 backbone ≈ 4.0 M + head ≈ 0.5 M; SimpleCNN ≈ 4.7 M + head ≈ 0.2 M.

---

## 8. Training loop

`run_training(cfg)` does the following:

1. `set_seed`, picks the device, creates `<out_dir>/<experiment>/`, writes `config.json`.
2. Builds datasets/loaders, the model, `AdamW(lr, weight_decay=0.01)`,
   a `LambdaLR` scheduler with `total_steps = epochs × steps_per_epoch` and `warmup = warmup_epochs × steps_per_epoch`,
   and `GradScaler("cuda", enabled=amp and CUDA available)`.
3. **Resume or initialize** (see [§10](#10-checkpoints-and-resume)). On a fresh start it estimates and sets the log-mel
   normalization statistics.
4. For each epoch from `start_epoch` to `epochs − 1`:
   - `train_one_epoch`: for each batch → move to device → (robust) `random_train_corruption` → `mixup` →
     `autocast` forward → `BCEWithLogits(logits.float(), y)` → `scaler.scale(loss).backward()` → `scaler.step` →
     `scaler.update` → `scheduler.step()` (**per step**, not per epoch).
   - `predict` on the **clean val set** → macro mAP.
   - Appends to `history` (epoch, train_loss, val_mAP, lr, seconds) and prints one progress line.
   - If val mAP improved → `best.pt` (weights + metadata only); always writes `last.pt` (full optimizer/scheduler/scaler
     state); rewrites `history.csv`.
5. Returns `{"best_val_mAP", "history", "exp_dir"}`.

**Model selection uses clean val only.** Corrupted data is never used to pick a checkpoint — if it were, the "unseen"
conditions would no longer be unseen.

---

## 9. Robustness evaluation: seen / unseen conditions

`run_robustness(cfg)`:

1. Checks that `<exp_dir>/best.pt` exists (if not → `FileNotFoundError` suggesting to run `train` first). `--checkpoint` may instead be an absolute path (for example, a checkpoint attached from `/kaggle/input`); either way, results always go to `<out_dir>/<experiment>/`.
2. Builds the model (`pretrained=False`) and loads the checkpoint.
3. Loads the eval set into RAM once (`load_waveforms`). If `max_eval_clips > 0`: picks a **seeded random subset**.
4. Runs 22 conditions: `clean` (severity 0) + 2 SEEN × 3 + 5 UNSEEN × 3.
5. Each condition → macro mAP → one row of `robustness.csv`.
6. The column `subset = bool(max_train_clips or max_eval_clips)` marks smoke-test results.

**How to read the results** (`summarize`):

| Column | Meaning |
|---|---|
| `clean` | mAP on the clean eval set — generalization to other uploaders' recordings |
| `seen`, `unseen` | Mean mAP over each corruption group |
| `seen_rel`, `unseen_rel` | Fraction of performance retained (1.0 = no degradation) |

The key comparisons: `effnet_standard` vs `cnn_baseline` (architecture + pretraining), and **`effnet_robust` vs
`effnet_standard` on the `unseen` column** (does robustness training transfer to conditions it never saw?).

Note: the "seen" label is defined **relative to which presets train on the SEEN corruptions** (`white_noise`, `reverb`
via `corruption_aug_p`): `effnet_robust`, `effnet_robust_v2` and `effnet_corrupt` do; `effnet_mixstyle` and `effnet_eq`
do not (their robustness component is MixStyle or EQ, not corruption augmentation). For `cnn_baseline`,
`effnet_standard`, `effnet_mixstyle` and `effnet_eq`, all seven corruptions are unseen; their `seen` column exists only
so all models are compared on the same set of conditions.

---

## 10. Checkpoints and resume

| File | Contents | When written |
|---|---|---|
| `last.pt` | model (including mean/std buffers) + optimizer + scheduler + scaler + `epoch`, `best_map`, `history` | End of **every** epoch |
| `best.pt` | model + `epoch`, `best_map`, `history` | When val mAP improves |

**Resume flow in `run_training`:**

```
if cfg.resume_from is set AND <exp_dir>/last.pt does not exist yet:
    copy resume_from → <exp_dir>/last.pt
    copy best.pt and history.csv that sit next to resume_from (if present and not already there)
if <exp_dir>/last.pt exists:
    load everything → start_epoch = epoch + 1, best, history
else:
    estimate normalization stats, start from epoch 0
```

- Re-running the same command in the **same session** → continues from `last.pt` automatically.
- In a **new Kaggle session** (after the 12-hour limit): add the previous version's output as an Input and pass
  `--resume-from /kaggle/input/<output>/outputs/<experiment>/last.pt`.
- `best.pt` is copied along because, if the remaining epochs never beat the old best, the new session would **never write
  `best.pt`**, and `evaluate` would fail after hours of training (this bug was found by the reviewer and fixed, with a test).
- Raising `--epochs` and re-running is also a way to "train more": the scheduler is rebuilt with the new `total_steps` and its
  step counter is restored.
- Atomic writes (`.tmp` + `os.replace`) → a session killed mid-save cannot corrupt the last good checkpoint.

---

## 11. Output files

```
/kaggle/working/outputs/<experiment>/          (smoke test: /kaggle/working/outputs_quick/<experiment>/)
├── config.json        The full Config of the run
├── history.csv        epoch, train_loss, val_mAP, lr, seconds
├── last.pt            Full checkpoint for resuming
├── best.pt            Best checkpoint by clean val mAP (used by evaluate)
└── robustness.csv     experiment, condition, group, severity, mAP, subset, protocol
/kaggle/working/summary.csv                    experiment, clean, seen, unseen, seen_rel, unseen_rel
/kaggle/working/summary_by_condition.csv       experiment, condition, group, severity, mAP, rel
```

---

## 12. Running on Kaggle

Setup: *Add Input* → `yousirui1/fsd50k`; *Settings* → Accelerator **GPU T4**, Internet **On** (to download ImageNet weights).

```python
!git clone -b main https://github.com/<your-user>/<your-repo>.git /kaggle/working/esr
%cd /kaggle/working/esr
!git pull origin main
!pip install -q "timm>=1.0"
```

```python
EXPERIMENT = "effnet_robust"   # cnn_baseline | effnet_standard | effnet_robust | effnet_robust_v2 | effnet_mixstyle | effnet_corrupt | effnet_eq
DATA_PATH = "/kaggle/input/datasets/yousirui1/fsd50k/fsd50k"
# 1) Smoke test (~15 min)
!python scripts/train.py    --experiment $EXPERIMENT --data-path $DATA_PATH --quick
!python scripts/evaluate.py --experiment $EXPERIMENT --data-path $DATA_PATH --quick --severities 2
# 2) Real run — use Save Version → Save & Run All so it runs in the background
!python scripts/train.py    --experiment $EXPERIMENT --data-path $DATA_PATH
!python scripts/evaluate.py --experiment $EXPERIMENT --data-path $DATA_PATH
# 3) After the experiments you ran (add their outputs as Inputs)
!python scripts/compare.py --out /kaggle/working/summary.csv
```

Recommended workflow: one version (commit) per experiment → add the versions' outputs as Inputs → run `compare.py`.
All flags: `python scripts/train.py --help`.

---

## 13. Testing strategy

- **130 tests**, running on CPU in a few tens of seconds, **downloading nothing** (`pretrained=False`).
- The `fake_root` fixture (`tests/conftest.py`) creates a **fake FSD50K** with the real folder names, the real CSV schemas
  (header-less vocabulary, comma-separated labels) and the same folder nesting as Kaggle; the audio is a sine tone per class,
  0.3–2.5 s long.
  - The val split **deliberately lacks class 1** → checks that mAP skips classes without positives.
- `tiny_cfg(...)`: 1 s clips, 64 mels, batch 4, 1 epoch, no AMP → the whole pipeline (train → resume → evaluate → compare) really runs.
- Per-module tests: config/discovery; data (stereo, wrong sample rate, empty/short/long clips); models (output shapes);
  augmentation; corruptions (measured SNR hits the target, telephone removes out-of-band tones, clipping keeps the peak,
  determinism by seed, silent clips); metrics; engine (scheduler, weights update, checkpoint round-trip); pipeline (resume,
  resume from another folder, subset flag); CLI (flag mapping, `--quick`, running the scripts via subprocess); notebook (every
  command parses).

**Running the tests on this machine:** Windows Application Control blocks the DLLs of freshly pip-installed packages (torch,
pandas), so the tests run in **WSL Ubuntu**:

```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/e/User/documentaries/USTH_codes/Environmental-Sound-Recognition-Under-Unseen-Conditions && ~/.venvs/esr/bin/python -m pytest -q"
```

On other machines (Linux/macOS/unblocked Windows): `pip install -r requirements-dev.txt && python -m pytest -q`.

Beyond the unit tests, the scripts were run end to end in WSL on fake data with **real 10 s clips, 128 mels and a pretrained
EfficientNet-B0**: auto-discovery at depth 4, training, resume, the 16 evaluation conditions and `compare` all worked.

---

## 14. Points to watch in the code

The points below are **the reason many "seemingly redundant" lines exist**. Keep them when modifying the code.

1. **Only the `*_16k` folders are used.** `read_audio` rejects anything that is not 16 kHz. Do not point it at `FSD50K.eval_audio` (44.1 kHz).
2. **`vocabulary.csv` has no header** — it is read with `header=None`. Reading it wrong drops the first class and shifts every label index.
3. **`eval.csv` is ordered by class** — subsets must be drawn with a seeded `permutation` (`pipeline.py`), never `iloc[:N]`;
   otherwise the subset contains only a few classes and its mAP is meaningless.
4. **mAP skips classes without positives** (`metrics.py`). So mAP on a subset is not directly comparable to mAP on the full set.
5. **The log-mel is always fp32** (`features.py` turns autocast off). Computing it in fp16 easily produces `-inf`/NaN.
6. **The loss uses `logits.float()`** — BCE in fp16 is less stable.
7. **Normalization statistics live in the checkpoint** (`mean`/`std` buffers). They are only estimated on a fresh start; resume does not recompute them.
8. **The scheduler steps per batch**, not per epoch. Changing `batch_size` when resuming skews the LR schedule.
9. **`drop_last=True` for training** — a final batch of 1 sample breaks BatchNorm.
10. **The random crop uses a fresh `np.random.default_rng()` per item** — avoids DataLoader workers producing the same crop.
11. **Noise is drawn on the CPU from the passed-in generator** and only then moved to the GPU. Do not switch to
    `torch.randn(..., device="cuda")`: results would differ between CPU/GPU and between runs.
12. **Evaluation noise is seeded per clip (protocol v2), so it no longer depends on `batch_size`.**
13. **SNR is computed on the audible part** (`_active_rms`), not on the zero padding — otherwise short clips would get noise
    5–15 dB weaker than their SNR label says.
14. **Silent clips** get noise at a −60 dBFS floor — no NaN, and never "nothing changes".
15. **Reverb keeps the RMS; clipping keeps the peak amplitude** — so the model cannot tell a corruption apart by loudness alone.
16. **No UNSEEN leakage:** `random_train_corruption` only calls `add_noise_at_snr` (white) and `apply_reverb`. Do not add UNSEEN
    corruptions there, or the experiment loses its meaning. `random_eq` (training only) is a smooth ±12 dB tilt. It overlaps
    *in kind* with `telephone` (both change the spectral envelope), so always report `telephone` results for
    `effnet_robust_v2` with that caveat.
17. **Checkpoints are selected on clean val only.** Do not evaluate corruptions during training.
18. **Checkpoints are written atomically** (`.tmp` + `os.replace`). Never write straight into `last.pt`.
19. **`resume_from` is only used when `last.pt` does not exist yet** — if the folder already has a `last.pt` (same session), that one wins.
20. **`--quick` writes to `outputs_quick`** and sets `subset=True`; `compare` drops those rows. Do not remove the `subset` column.
21. **`run_robustness` builds the model with `pretrained=False`** — the weights come from the checkpoint, so evaluation needs no Internet.
22. **`torch.load(weights_only=False)`** — checkpoints contain `history` (Python dicts/lists). Only load checkpoints you created yourself.
23. **Relative imports in `esr/`** and `sys.path` in `scripts/` — keep them so the repo runs right after `git clone`.
24. **`GradScaler("cuda", ...)`** requires torch ≥ 2.3 (Kaggle meets this).
25. **Only 1 GPU is used.** On a T4 ×2 machine, the second GPU sits idle.
26. **FFT convolution always pads to a power of two** (`_fft_size` in `corruptions.py`). Training reverb uses a random RT60, so
    without this every clip would need a differently sized FFT; on Kaggle that filled the cuFFT plan cache and crashed training
    with `CUFFT_INTERNAL_ERROR` in epoch 2. Do not remove the padding.

---

## 15. Compute budget

(Estimates — the notebook prints `seconds` per epoch; check after the first epoch and adjust `--epochs`.)

| Item | Estimate on T4 (AMP) |
|---|---|
| GPU memory, EfficientNet-B0, batch 32 × 128 × 1001 | ~8 GB (fits in 16 GB) |
| GPU memory, SimpleCNN | ~5–6 GB |
| RAM for the in-memory eval set (int16) | ~3.3 GB |
| Time per epoch | ~8–12 min |
| 20 training epochs | ~3–4 h (`effnet_robust` takes longer because of the corruption augmentation) |
| Evaluation, 22 conditions × 10,231 clips | ~40–60 min |
| `--quick` smoke test | ~15 min |

Kaggle limits: 12-hour sessions, ~30 GPU-hours per week → one version per experiment.

**Trial runs on CPU:** EfficientNet-B0 in fp32 with batch 32 and 10 s clips needs more than 7 GB of RAM. When simulating on a
personal machine/WSL, use `--batch-size 8`.

---

## 16. Known limitations and deferred work

Minor issues raised by the reviewer that are **not fixed yet** (they do not affect the correctness of the comparison when the
default settings are used):

| ID | Issue | Suggested fix |
|---|---|---|
| M-1 | ~~Evaluation noise depends on `batch_size`~~ — **resolved in v2**: `predict_array` seeds per clip (`seed·100003 + clip_index`) | Use a fixed evaluation batch size or seed per clip |
| M-2 | Resume replays the first session's sampler/augmentation order | Seed with `seed + start_epoch` or save the generator state |
| M-3 | Train + val loaders each keep 4 persistent workers on 4 CPUs | `num_workers=2`, no persistent workers for val |
| M-4 | `find_data_root` does not check that the `*_16k` folders exist | Require `FSD50K.dev_audio_16k` next to the hit |
| M-5 | Normalization stats come from the first 256 clips and include padded frames | Sample randomly, exclude padded frames |
| M-6 | (obsolete) text in the old notebook recommended T4×2/P100 | The new notebook only mentions T4 |
| M-7 | `--quick` still reads the full val set; `cudnn.benchmark` is not enabled | Subset the val set; set `torch.backends.cudnn.benchmark = True` |

Deliberate design choices (from the spec):
- Only the **first 10 s** of long clips are evaluated; labels are clip-level, so an event outside that window can be missed.
- `brown_noise` differs from `white_noise` only in spectral shape — it is "unseen" spectrally, but in the same "additive noise" family.
- SpecAugment's frequency masking loosely resembles `telephone`, but it is applied equally to all 3 experiments, so it is not leakage.
- With `f_min=50` and `n_fft=1024`, a few low mel bands may be empty (torchaudio warns) — harmless.

---

## 17. Extension ideas

- **Two-GPU training** (DDP via `torchrun`, as in the reference repo) — nearly doubles throughput on T4 ×2.
- **EfficientNet-B2** (set `model="efficientnet_b2"` in the `EXPERIMENTS` preset or add a `--model` flag) — PSLA uses B2.
- **Weight averaging** over the last epochs and **ensembling** — PSLA techniques.
- **PCEN** instead of log-mel — a front end known to be robust to gain changes and noise.
- **More UNSEEN corruptions**: codecs (MP3/Opus), babble noise, real RIRs, device changes.
- **Per-class analysis**: which classes degrade most under unseen conditions.
- **PANNs CNN14 / AST** as an upper bound.

---

## 18. Quick reference

| To change… | Look in |
|---|---|
| Default hyperparameters | `esr/config.py` → `Config` |
| Add/change an experiment | `esr/config.py` → `EXPERIMENTS` (the CLI `choices` update automatically) |
| Add a CLI flag | `esr/cli.py` → `_add_run_args` + `OVERRIDES` |
| Add a corruption | `esr/corruptions.py` → a function with the same signature + add it to `CORRUPTIONS` and `SEEN`/`UNSEEN` |
| Add an architecture | `esr/models.py` → `build_model` (any `timm` name already works via `model=`) |
| Change the input features | `esr/features.py` → `LogMel` |
| Change spectrogram augmentation | `esr/augment.py` → `make_spec_transform` |
| Change the metric | `esr/metrics.py` |
| Change resume/checkpointing | `esr/engine.py` + `esr/pipeline.py:run_training` |
| Run the tests | `python -m pytest -q` (on this machine: via WSL, see [§13](#13-testing-strategy)) |

**Common commands:**

```bash
# --experiment choices: cnn_baseline | effnet_corrupt | effnet_eq | effnet_mixstyle | effnet_robust | effnet_robust_v2 | effnet_standard
python scripts/train.py    --experiment effnet_robust [--quick] [--epochs N] [--batch-size B] [--resume-from PATH]
python scripts/evaluate.py --experiment effnet_robust [--quick] [--severities 1 2 3] [--checkpoint best.pt]
python scripts/compare.py  [--inputs GLOB ...] [--include-subset] [--out summary.csv]
```
