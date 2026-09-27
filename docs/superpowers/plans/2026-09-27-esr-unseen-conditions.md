# Environmental Sound Recognition Under Unseen Conditions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally tested Python package plus a generated Kaggle notebook. The notebook trains CNN sound classifiers on FSD50K and measures how their mAP degrades under seen and unseen acoustic conditions.

**Architecture:** A small package `esr/` (config, data, features, augment, corruptions, models, metrics, engine, pipeline) is unit-tested on CPU against a synthetic FSD50K-shaped fixture. `tools/build_notebook.py` inlines every module into `%%writefile` cells and appends the experiment cells (EDA, training, robustness eval, comparison), producing `notebooks/esr_fsd50k_kaggle.ipynb`. The user uploads that notebook to Kaggle and runs it with the `yousirui1/fsd50k` dataset attached.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.3, torchaudio (MelSpectrogram only), timm ≥ 1.0 (EfficientNet), soundfile, scikit-learn (average precision), pandas, numpy, matplotlib, nbformat, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-esr-unseen-conditions-design.md`

## Global Constraints

- Audio: only `FSD50K.dev_audio_16k` and `FSD50K.eval_audio_16k`. Sample rate exactly 16000 Hz. A file with any other rate raises `ValueError`.
- Labels: 200 classes from `FSD50K.ground_truth/vocabulary.csv` (no header: `index,label,mid`). `dev.csv` columns `fname,labels,mids,split` with `split ∈ {train,val}`. `eval.csv` columns `fname,labels,mids`. `labels` is a comma-separated string.
- Metric: macro mAP over classes with ≥ 1 positive (`sklearn.metrics.average_precision_score`).
- Model selection on **clean val mAP only**. Corrupted audio is never used for selection.
- Seen corruptions: `("white_noise", "reverb")`. Unseen corruptions: `("brown_noise", "telephone", "clipping")`. Severities `1, 2, 3`.
- Features: log-mel with n_fft 1024, win 400, hop 160, 128 mels, 50–8000 Hz, 10 s clips (160,000 samples).
- Experiments: exactly `cnn_baseline`, `effnet_standard`, `effnet_robust` (see spec §4).
- Every module must run on CPU (tests) and on CUDA with AMP (Kaggle). No code path may require a GPU.
- Tests must not download anything. Use `pretrained=False` in tests.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Kaggle mount path differs** (`/kaggle/input/fsd50k/fsd50k`, `/kaggle/input/fsd50k`, or the newer `/kaggle/input/datasets/yousirui1/fsd50k/fsd50k`). `find_data_root` must find it at any depth up to 5 → `test_find_data_root_deep_nesting` (Task 1).
2. **Very short, very long or empty clips** (FSD50K has 0.3 s clips and 30 s clips). They must pad or crop to exactly `clip_samples` without error → `test_fix_length_*` (Task 2).
3. **Silent (all-zero) or zero-padded clips under SNR-based noise.** They must give finite, non-zero noisy audio, never NaN → `test_noise_on_silent_clip_is_finite` (Task 5).
4. **A class with no positives in the val/eval subset** (common with `QUICK_RUN` subsets). mAP must skip that class, not return NaN → `test_map_skips_classes_without_positives` (Task 6) plus the fake val split that lacks class 1 (Task 8).
5. **Kaggle session killed or time limit reached.** Training must resume from `last.pt` in the same folder or from a previous version's `last.pt` (`resume_from`) with history intact → `test_run_training_resumes`, `test_run_training_resume_from_other_dir` (Task 8).

---

## File Structure

```
.
├── pyproject.toml                    pytest config (pythonpath=".")
├── requirements-dev.txt              local CPU dev dependencies
├── .gitignore
├── README.md                         how to build + run on Kaggle (Task 9)
├── esr/
│   ├── __init__.py
│   ├── config.py        Config dataclass, EXPERIMENTS, make_config, find_data_root
│   ├── data.py          vocab/split loading, read_audio, fix_length, FSD50KDataset, sampler, load_waveforms
│   ├── features.py      LogMel front end (normalization stats stored as buffers)
│   ├── augment.py       mixup, spec_augment, freq_mixstyle, make_spec_transform
│   ├── corruptions.py   seen/unseen corruption suite + random_train_corruption
│   ├── models.py        AttentionPool, SimpleCNN, TimmCNN, AudioModel, build_model
│   ├── metrics.py       mean_average_precision
│   ├── engine.py        seed/device, scheduler, checkpoints, train_one_epoch, predict, predict_array
│   └── pipeline.py      build_datasets, estimate_norm_stats, run_training, run_robustness, summarize
├── tools/build_notebook.py           package → notebooks/esr_fsd50k_kaggle.ipynb
├── notebooks/esr_fsd50k_kaggle.ipynb generated artifact that goes to Kaggle
└── tests/
    ├── conftest.py      synthetic FSD50K layout + tiny_cfg factory
    ├── test_config.py  test_data.py  test_models.py  test_augment.py
    ├── test_corruptions.py  test_metrics.py  test_engine.py  test_pipeline.py
    └── test_notebook.py
```

All `esr` modules use relative imports (`from .config import Config`) so that they work both locally and when `%%writefile` writes them into `/kaggle/working/esr/`.

---

### Task 1: Project scaffold, config and dataset discovery

**Files:**
- Create: `pyproject.toml`, `requirements-dev.txt`, `.gitignore`, `esr/__init__.py`, `esr/config.py`
- Test: `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Config` dataclass. Fields and defaults are listed in Step 5. It has property `clip_samples -> int` and method `to_dict() -> dict` (JSON-serializable).
  - `EXPERIMENTS: dict[str, dict]` with keys `cnn_baseline`, `effnet_standard`, `effnet_robust`.
  - `make_config(experiment: str, **overrides) -> Config`. Raises `KeyError` for an unknown experiment.
  - `find_data_root(search_root: str = "/kaggle/input", max_depth: int = 5) -> str`. Returns the folder that contains `FSD50K.ground_truth/`. Raises `FileNotFoundError` if there is none.
  - Test fixtures `fake_root` (a `Path` to the synthetic dataset root) and `tiny_cfg(experiment="cnn_baseline", **kw) -> Config`.

- [ ] **Step 1: Initialize the repo and dev environment**

```bash
git init
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
```

Create `requirements-dev.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.3
torchaudio>=2.3
timm>=1.0
soundfile>=0.12
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
nbformat>=5.9
pytest>=8
```

Create `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Create `.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
outputs/
*.pt
```

Run: `pip install -r requirements-dev.txt`
Expected: installs without error. `python -c "import torch, torchaudio, timm, soundfile, sklearn"` prints nothing.

- [ ] **Step 2: Write the shared test fixtures**

Create `tests/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from esr.config import make_config

CLASSES = ["Bark", "Siren", "Guitar", "Rain", "Speech"]
SR = 16000


def _tone(freq, seconds):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    """Synthetic FSD50K with the real folder names and CSV schemas.

    Mirrors Kaggle's nesting: <input>/fsd50k/fsd50k/FSD50K.*
    dev: 12 train + 4 val clips; val has no clip of class 1 ("Siren") on purpose.
    eval: 8 clips covering all 5 classes. Durations include 0.3 s and 2.5 s.
    """
    root = tmp_path_factory.mktemp("kaggle_input") / "fsd50k" / "fsd50k"
    gt = root / "FSD50K.ground_truth"
    dev_dir = root / "FSD50K.dev_audio_16k"
    eval_dir = root / "FSD50K.eval_audio_16k"
    for d in (gt, dev_dir, eval_dir):
        d.mkdir(parents=True)

    pd.DataFrame({"i": range(5), "label": CLASSES, "mid": [f"/m/{i}" for i in range(5)]}).to_csv(
        gt / "vocabulary.csv", header=False, index=False
    )

    durations = [0.3, 0.8, 1.5, 2.5]
    dev_rows = []
    for k in range(16):
        c1, c2 = k % 5, (k + 2) % 5
        split = "val" if k >= 12 else "train"  # val clips k=12..15 have c1 in {2,3,4,0}
        multi = k % 2 == 0 and not (split == "val" and c2 == 1)
        wav = _tone(200 + 150 * c1, durations[k % 4])
        labels = CLASSES[c1]
        if multi:
            wav = wav + _tone(200 + 150 * c2, durations[k % 4])
            labels = f"{CLASSES[c1]},{CLASSES[c2]}"
        sf.write(dev_dir / f"{1000 + k}.wav", wav, SR, subtype="PCM_16")
        dev_rows.append((1000 + k, labels, "/m/x", split))
    pd.DataFrame(dev_rows, columns=["fname", "labels", "mids", "split"]).to_csv(gt / "dev.csv", index=False)

    eval_rows = []
    for k in range(8):
        c = k % 5
        sf.write(eval_dir / f"{2000 + k}.wav", _tone(200 + 150 * c, durations[k % 4]), SR, subtype="PCM_16")
        eval_rows.append((2000 + k, CLASSES[c], "/m/x"))
    pd.DataFrame(eval_rows, columns=["fname", "labels", "mids"]).to_csv(gt / "eval.csv", index=False)
    return root


@pytest.fixture
def tiny_cfg(fake_root, tmp_path):
    def _make(experiment="cnn_baseline", **kw):
        base = dict(
            data_root=str(fake_root), out_dir=str(tmp_path / "out"), clip_seconds=1.0, n_mels=64,
            batch_size=4, epochs=1, num_workers=0, amp=False, pretrained=False, warmup_epochs=0,
        )
        base.update(kw)
        return make_config(experiment, **base)
    return _make
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'esr'`

- [ ] **Step 5: Implement**

Create `esr/__init__.py`:

```python
"""Environmental sound recognition under unseen conditions (FSD50K)."""
```

Create `esr/config.py`:

```python
"""Experiment configuration and Kaggle dataset discovery."""
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    data_root: str = ""
    out_dir: str = "/kaggle/working/outputs"
    experiment: str = "effnet_robust"
    model: str = "efficientnet_b0"  # "simple_cnn" or any timm model name
    pretrained: bool = True
    # audio / features
    sample_rate: int = 16000
    clip_seconds: float = 10.0
    n_mels: int = 128
    n_fft: int = 1024
    win_length: int = 400
    hop_length: int = 160
    f_min: float = 50.0
    f_max: float = 8000.0
    # optimisation
    batch_size: int = 32
    epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 1
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    # augmentation / robustness
    balanced_sampling: bool = True
    spec_augment: bool = True
    mixup_alpha: float = 0.5  # 0 disables mixup
    freq_mixstyle_p: float = 0.0
    corruption_aug_p: float = 0.0
    # evaluation / bookkeeping
    severities: tuple = (1, 2, 3)
    max_train_clips: int = 0  # 0 = use all
    max_eval_clips: int = 0  # 0 = use all
    resume_from: str = ""

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severities"] = list(self.severities)
        return d


EXPERIMENTS = {
    "cnn_baseline": dict(model="simple_cnn", pretrained=False, lr=1e-3),
    "effnet_standard": dict(model="efficientnet_b0", pretrained=True, lr=5e-4),
    "effnet_robust": dict(
        model="efficientnet_b0", pretrained=True, lr=5e-4, freq_mixstyle_p=0.7, corruption_aug_p=0.5
    ),
}


def make_config(experiment: str, **overrides) -> Config:
    if experiment not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {experiment!r}; choose one of {sorted(EXPERIMENTS)}")
    kw = dict(EXPERIMENTS[experiment])
    kw.update(overrides)
    return Config(experiment=experiment, **kw)


def find_data_root(search_root: str = "/kaggle/input", max_depth: int = 5) -> str:
    """Return the directory containing FSD50K.ground_truth/, searching shallow paths first."""
    base = Path(search_root)
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth + ["FSD50K.ground_truth", "vocabulary.csv"])
        hits = sorted(base.glob(pattern))
        if hits:
            return str(hits[0].parent.parent)
    raise FileNotFoundError(
        f"No FSD50K.ground_truth/vocabulary.csv under {search_root}. "
        "In Kaggle: Add Input -> search 'yousirui1/fsd50k' -> Add."
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements-dev.txt .gitignore esr/__init__.py esr/config.py tests/conftest.py tests/test_config.py docs/
git commit -m "feat: project scaffold, experiment config and FSD50K discovery" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Data loading

**Files:**
- Create: `esr/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `fake_root` fixture (Task 1)
- Produces:
  - `AUDIO_DIRS: dict[str, str]` (split → folder name)
  - `load_vocab(root) -> tuple[list[str], dict[str, int]]`
  - `encode_labels(label_str: str, label2idx: dict) -> np.ndarray float32 [C]`
  - `load_split(root, split: "train"|"val"|"eval") -> tuple[pd.DataFrame(columns fname,path), np.ndarray float32 [N, C]]`
  - `read_audio(path, sample_rate) -> np.ndarray float32 [L]` (mono). Raises `ValueError` on a sample-rate mismatch.
  - `fix_length(wav, n, rng: np.random.Generator | None = None) -> np.ndarray [n]`. `rng` gives a random crop. `None` gives a crop from the start.
  - `FSD50KDataset(paths, targets, clip_samples, sample_rate, train: bool)`. Items are `(torch.float32 [clip_samples], torch.float32 [C])`. It exposes `.targets`.
  - `balanced_sample_weights(targets) -> np.ndarray [N]`, `make_balanced_sampler(targets, num_samples=None, seed=0) -> WeightedRandomSampler`
  - `load_waveforms(paths, clip_samples, sample_rate, num_threads=8) -> np.ndarray int16 [N, clip_samples]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'esr.data'`

- [ ] **Step 3: Implement**

Create `esr/data.py`:

```python
"""FSD50K metadata + audio loading."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

AUDIO_DIRS = {
    "train": "FSD50K.dev_audio_16k",
    "val": "FSD50K.dev_audio_16k",
    "eval": "FSD50K.eval_audio_16k",
}


def load_vocab(root):
    vocab = pd.read_csv(Path(root) / "FSD50K.ground_truth" / "vocabulary.csv", header=None,
                        names=["index", "label", "mid"])
    labels = vocab.sort_values("index")["label"].tolist()
    return labels, {label: i for i, label in enumerate(labels)}


def encode_labels(label_str, label2idx):
    y = np.zeros(len(label2idx), np.float32)
    for label in str(label_str).split(","):
        y[label2idx[label]] = 1.0
    return y


def load_split(root, split):
    root = Path(root)
    gt = root / "FSD50K.ground_truth"
    if split in ("train", "val"):
        df = pd.read_csv(gt / "dev.csv")
        df = df[df["split"] == split]
    elif split == "eval":
        df = pd.read_csv(gt / "eval.csv")
    else:
        raise ValueError(f"split must be train/val/eval, got {split!r}")
    df = df.reset_index(drop=True)
    _, label2idx = load_vocab(root)
    paths = [str(root / AUDIO_DIRS[split] / f"{fname}.wav") for fname in df["fname"]]
    targets = np.stack([encode_labels(s, label2idx) for s in df["labels"]]).astype(np.float32)
    return pd.DataFrame({"fname": df["fname"], "path": paths}), targets


def read_audio(path, sample_rate):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, got {sr} Hz. Use the *_16k audio folders.")
    return wav.mean(axis=1)


def fix_length(wav, n, rng=None):
    if len(wav) >= n:
        start = 0 if rng is None else int(rng.integers(0, len(wav) - n + 1))
        return wav[start:start + n]
    return np.pad(wav, (0, n - len(wav)))


class FSD50KDataset(Dataset):
    def __init__(self, paths, targets, clip_samples, sample_rate, train):
        self.paths = list(paths)
        self.targets = targets
        self.clip_samples = clip_samples
        self.sample_rate = sample_rate
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wav = read_audio(self.paths[i], self.sample_rate)
        rng = np.random.default_rng() if self.train else None
        wav = fix_length(wav, self.clip_samples, rng).astype(np.float32)
        return torch.from_numpy(wav), torch.from_numpy(self.targets[i])


def balanced_sample_weights(targets):
    """PSLA-style: a clip's weight is the sum of 1/frequency over its labels."""
    class_freq = targets.sum(axis=0)
    class_w = 1000.0 / np.maximum(class_freq, 1.0)
    return (targets * class_w).sum(axis=1)


def make_balanced_sampler(targets, num_samples=None, seed=0):
    weights = torch.as_tensor(balanced_sample_weights(targets), dtype=torch.double)
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples or len(weights), replacement=True, generator=gen)


def load_waveforms(paths, clip_samples, sample_rate, num_threads=8):
    """Load clips into one int16 matrix (10 k eval clips x 10 s = 3.3 GB) for fast repeated evaluation."""
    out = np.zeros((len(paths), clip_samples), np.int16)

    def work(i):
        wav = fix_length(read_audio(paths[i], sample_rate), clip_samples)
        out[i] = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)

    with ThreadPoolExecutor(num_threads) as ex:
        list(ex.map(work, range(len(paths))))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add esr/data.py tests/test_data.py
git commit -m "feat: FSD50K metadata, audio loading, balanced sampler" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Log-mel front end and CNN models

**Files:**
- Create: `esr/features.py`, `esr/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `Config` (Task 1)
- Produces:
  - `LogMel(sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max)`. Also `LogMel.from_config(cfg)` and `.set_stats(mean, std)`. `forward(wav [B, L]) -> [B, 1, n_mels, T]` with `T = L // hop + 1`. Normalization buffers `mean` and `std` are saved in the state_dict.
  - `AttentionPool(in_ch, n_classes)`: `[B, C, T] -> logits [B, K]`
  - `SimpleCNN(n_classes, channels=(64,128,256,512))`, `TimmCNN(name, n_classes, pretrained)`: both map `[B, 1, F, T] -> logits [B, K]`
  - `AudioModel(frontend, net)` with attributes `.frontend` and `.net`, and `forward(wav [B, L], spec_transform=None) -> logits [B, K]`
  - `build_model(cfg, n_classes, pretrained=None) -> AudioModel` (`pretrained=None` means use `cfg.pretrained`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
import torch

from esr.config import make_config
from esr.features import LogMel
from esr.models import AttentionPool, AudioModel, SimpleCNN, TimmCNN, build_model


def _cfg(**kw):
    return make_config("cnn_baseline", clip_seconds=1.0, n_mels=64, **kw)


def test_logmel_shape():
    front = LogMel.from_config(_cfg())
    x = front(torch.randn(2, 16000) * 0.1)
    assert x.shape == (2, 1, 64, 101)
    assert torch.isfinite(x).all()


def test_logmel_silence_is_finite():
    assert torch.isfinite(LogMel.from_config(_cfg())(torch.zeros(1, 16000))).all()


def test_logmel_set_stats_normalizes_and_is_saved():
    front = LogMel.from_config(_cfg())
    raw = front(torch.randn(4, 16000) * 0.1)
    front.set_stats(raw.mean(), raw.std())
    x = front(torch.randn(4, 16000) * 0.1)
    assert abs(x.mean().item()) < 0.5
    sd = front.state_dict()
    assert "mean" in sd and "std" in sd and sd["std"].item() > 0


def test_attention_pool_shape():
    assert AttentionPool(32, 5)(torch.randn(3, 32, 7)).shape == (3, 5)


def test_simple_cnn_logits_shape():
    assert SimpleCNN(5)(torch.randn(2, 1, 64, 101)).shape == (2, 5)


def test_timm_efficientnet_logits_shape_without_download():
    net = TimmCNN("efficientnet_b0", 5, pretrained=False)
    assert net(torch.randn(2, 1, 64, 101)).shape == (2, 5)


def test_build_model_end_to_end_from_waveform():
    for exp in ("cnn_baseline", "effnet_standard"):
        cfg = make_config(exp, clip_seconds=1.0, n_mels=64)
        model = build_model(cfg, n_classes=5, pretrained=False)
        assert isinstance(model, AudioModel)
        logits = model(torch.randn(2, 16000) * 0.1)
        assert logits.shape == (2, 5)


def test_audio_model_applies_spec_transform():
    model = build_model(_cfg(), 5)
    seen = {}

    def spy(x):
        seen["shape"] = tuple(x.shape)
        return x

    model(torch.randn(2, 16000), spec_transform=spy)
    assert seen["shape"] == (2, 1, 64, 101)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_models.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'esr.features'`

- [ ] **Step 3: Implement the front end**

Create `esr/features.py`:

```python
"""Log-mel spectrogram front end."""
import torch
import torchaudio
from torch import nn


class LogMel(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=128,
                 f_min=50.0, f_max=8000.0):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=f_min, f_max=f_max, n_mels=n_mels, power=2.0,
        )
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.n_mels, cfg.f_min, cfg.f_max)

    @torch.no_grad()
    def set_stats(self, mean, std):
        self.mean.fill_(float(mean))
        self.std.fill_(max(float(std), 1e-5))

    def forward(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            x = torch.log(self.mel(wav.float()) + 1e-6)
            x = (x - self.mean) / self.std
        return x.unsqueeze(1)
```

- [ ] **Step 4: Implement the models**

Create `esr/models.py`:

```python
"""CNN classifiers with attention pooling over time."""
import torch
from torch import nn

from .features import LogMel


class AttentionPool(nn.Module):
    """PSLA/PANNs-style pooling: per-frame class logits weighted by a softmax attention over time."""

    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.cla = nn.Conv1d(in_ch, n_classes, kernel_size=1)
        self.att = nn.Conv1d(in_ch, n_classes, kernel_size=1)

    def forward(self, x):  # [B, C, T]
        att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        return (self.cla(x) * att).sum(dim=-1)


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """VGG-style baseline (PANNs CNN10-like), trained from scratch."""

    def __init__(self, n_classes, channels=(64, 128, 256, 512)):
        super().__init__()
        layers, cin = [], 1
        for c in channels:
            layers += [_conv_block(cin, c), nn.AvgPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.3)
        self.head = AttentionPool(cin, n_classes)

    def forward(self, x):  # [B, 1, F, T]
        x = self.features(x).mean(dim=2)  # average over frequency -> [B, C, T']
        return self.head(self.dropout(x))


class TimmCNN(nn.Module):
    """Any timm CNN (default EfficientNet-B0) on 1-channel spectrograms."""

    def __init__(self, name, n_classes, pretrained):
        super().__init__()
        import timm

        self.backbone = timm.create_model(name, pretrained=pretrained, in_chans=1, num_classes=0, global_pool="")
        self.dropout = nn.Dropout(0.3)
        self.head = AttentionPool(self.backbone.num_features, n_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x).mean(dim=2)
        return self.head(self.dropout(x))


class AudioModel(nn.Module):
    def __init__(self, frontend, net):
        super().__init__()
        self.frontend = frontend
        self.net = net

    def forward(self, wav, spec_transform=None):
        x = self.frontend(wav)
        if spec_transform is not None:
            x = spec_transform(x)
        return self.net(x)


def build_model(cfg, n_classes, pretrained=None):
    pretrained = cfg.pretrained if pretrained is None else pretrained
    if cfg.model == "simple_cnn":
        net = SimpleCNN(n_classes)
    else:
        net = TimmCNN(cfg.model, n_classes, pretrained)
    return AudioModel(LogMel.from_config(cfg), net)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_models.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add esr/features.py esr/models.py tests/test_models.py
git commit -m "feat: log-mel front end, SimpleCNN and EfficientNet with attention pooling" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Training augmentations (mixup, SpecAugment, Frequency-MixStyle)

**Files:**
- Create: `esr/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Consumes: `Config` fields `spec_augment` and `freq_mixstyle_p`
- Produces:
  - `mixup(wav [B,L], y [B,K], alpha: float, rng: np.random.Generator) -> (wav, y)`. `alpha <= 0` returns the inputs unchanged.
  - `spec_augment(x [B,1,F,T], freq_width=24, time_width=100, n_masks=2) -> x`. Does not modify the input in place.
  - `freq_mixstyle(x [B,1,F,T], p: float, alpha=0.3, eps=1e-6) -> x`
  - `make_spec_transform(cfg) -> Callable | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_augment.py`:

```python
import numpy as np
import torch

from esr.augment import freq_mixstyle, make_spec_transform, mixup, spec_augment
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_augment.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'esr.augment'`

- [ ] **Step 3: Implement**

Create `esr/augment.py`:

```python
"""Training-time augmentations on waveforms and spectrograms."""
import torch


def mixup(wav, y, alpha, rng):
    if alpha <= 0:
        return wav, y
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(wav.size(0), device=wav.device)
    return lam * wav + (1 - lam) * wav[perm], lam * y + (1 - lam) * y[perm]


def _mask_along(x, max_width, axis):
    b, size = x.size(0), x.size(axis)
    max_width = min(max_width, size // 4)
    if max_width < 1:
        return x
    width = torch.randint(0, max_width + 1, (b,), device=x.device)
    start = (torch.rand(b, device=x.device) * (size - width + 1)).long()
    idx = torch.arange(size, device=x.device)
    mask = (idx[None] >= start[:, None]) & (idx[None] < (start + width)[:, None])
    shape = [b, 1, 1, 1]
    shape[axis] = size
    return x.masked_fill(mask.view(shape), 0.0)


def spec_augment(x, freq_width=24, time_width=100, n_masks=2):
    for _ in range(n_masks):
        x = _mask_along(x, freq_width, axis=2)
        x = _mask_along(x, time_width, axis=3)
    return x


def freq_mixstyle(x, p, alpha=0.3, eps=1e-6):
    """Frequency-wise MixStyle (Schmid et al., 2022): mix per-frequency mean/std across the batch."""
    if p <= 0 or torch.rand(1).item() > p:
        return x
    b = x.size(0)
    mu = x.mean(dim=(1, 3), keepdim=True)
    sig = (x.var(dim=(1, 3), keepdim=True) + eps).sqrt()
    x_norm = (x - mu) / sig
    lam = torch.distributions.Beta(alpha, alpha).sample((b, 1, 1, 1)).to(x.device, x.dtype)
    perm = torch.randperm(b, device=x.device)
    mu_mix = lam * mu + (1 - lam) * mu[perm]
    sig_mix = lam * sig + (1 - lam) * sig[perm]
    return x_norm * sig_mix + mu_mix


def make_spec_transform(cfg):
    fns = []
    if cfg.freq_mixstyle_p > 0:
        fns.append(lambda x: freq_mixstyle(x, cfg.freq_mixstyle_p))
    if cfg.spec_augment:
        fns.append(spec_augment)
    if not fns:
        return None

    def transform(x):
        for fn in fns:
            x = fn(x)
        return x

    return transform
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add esr/augment.py tests/test_augment.py
git commit -m "feat: mixup, SpecAugment and frequency-wise MixStyle" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Corruption suite (seen vs unseen conditions)

**Files:**
- Create: `esr/corruptions.py`
- Test: `tests/test_corruptions.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - Constants `SNR_DB`, `RT60_S`, `BAND_HZ`, `CLIP_FRAC` (severity → parameter), `SEEN = ("white_noise", "reverb")`, `UNSEEN = ("brown_noise", "telephone", "clipping")`, and `CORRUPTIONS: dict[str, fn]`
  - Every corruption has the signature `fn(wav [B,L] float tensor, severity: int, gen: torch.Generator (CPU), sample_rate=16000) -> [B,L]` and returns a tensor on the same device.
  - `apply_corruption(wav, name, severity, gen, sample_rate=16000)`
  - `colored_noise(shape, exponent, gen, device="cpu")`, `add_noise_at_snr(wav, noise, snr_db)`, `apply_reverb(wav, rt60, sample_rate, gen)`
  - `random_train_corruption(wav, p, gen, sample_rate=16000)`: seen family only, with continuous parameters (SNR 5–30 dB, RT60 0.2–0.8 s)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_corruptions.py`:

```python
import math

import pytest
import torch

from esr.corruptions import (
    CORRUPTIONS, SEEN, UNSEEN, add_noise_at_snr, apply_corruption, colored_noise, random_train_corruption,
)

SR = 16000


def _tone(freq, n=SR, batch=2):
    t = torch.arange(n) / SR
    return (0.5 * torch.sin(2 * math.pi * freq * t)).repeat(batch, 1)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _db(x):
    return 10 * torch.log10(x.pow(2).mean(-1))


def test_registry_matches_spec():
    assert SEEN == ("white_noise", "reverb")
    assert UNSEEN == ("brown_noise", "telephone", "clipping")
    assert set(CORRUPTIONS) == set(SEEN + UNSEEN)


@pytest.mark.parametrize("name", ["white_noise", "reverb", "brown_noise", "telephone", "clipping"])
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_every_corruption_keeps_shape_and_is_finite(name, severity):
    x = _tone(440) + _tone(60)  # 60 Hz lies outside every telephone band, so every corruption must change x
    out = apply_corruption(x, name, severity, _gen())
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert not torch.allclose(out, x)


def test_unknown_corruption_raises():
    with pytest.raises(KeyError):
        apply_corruption(_tone(440), "nope", 1, _gen())


def test_white_noise_hits_target_snr():
    x = _tone(440)
    out = apply_corruption(x, "white_noise", 2, _gen())  # 10 dB
    snr = _db(x) - _db(out - x)
    assert torch.allclose(snr, torch.full_like(snr, 10.0), atol=0.3)


def test_brown_noise_is_low_frequency_heavy():
    n = colored_noise((1, SR), 2.0, _gen())
    spec = torch.fft.rfft(n).abs().pow(2)[0]
    assert spec[1:200].sum() > 10 * spec[4000:].sum()


def test_noise_on_silent_clip_is_finite():
    out = apply_corruption(torch.zeros(2, SR), "white_noise", 3, _gen())
    assert torch.isfinite(out).all() and out.abs().max() > 0
    out = add_noise_at_snr(torch.zeros(1, SR), torch.zeros(1, SR), 0.0)
    assert torch.isfinite(out).all()


def test_reverb_keeps_loudness_and_silence_stays_silent():
    x = _tone(440)
    out = apply_corruption(x, "reverb", 3, _gen())
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), x.pow(2).mean(-1).sqrt(), rtol=1e-3)
    assert torch.isfinite(apply_corruption(torch.zeros(1, SR), "reverb", 1, _gen())).all()


def test_telephone_removes_out_of_band_tone():
    low, mid = _tone(100), _tone(1000)
    assert _db(apply_corruption(low, "telephone", 2, _gen())).max() < _db(low).max() - 30
    assert torch.allclose(_db(apply_corruption(mid, "telephone", 2, _gen())), _db(mid), atol=0.5)


def test_clipping_saturates_and_keeps_peak():
    x = _tone(440)
    out = apply_corruption(x, "clipping", 3, _gen())
    assert torch.allclose(out.abs().amax(-1), x.abs().amax(-1), atol=1e-5)
    assert (out.abs() > 0.99 * x.abs().amax()).float().mean() > 0.5


def test_same_seed_same_output():
    x = _tone(440)
    for name in CORRUPTIONS:
        assert torch.equal(apply_corruption(x, name, 2, _gen(7)), apply_corruption(x, name, 2, _gen(7)))


def test_random_train_corruption_probability_extremes():
    x = _tone(440, batch=6)
    assert torch.equal(random_train_corruption(x, 0.0, _gen()), x)
    out = random_train_corruption(x, 1.0, _gen())
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert all(not torch.allclose(out[i], x[i]) for i in range(6))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_corruptions.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'esr.corruptions'`

- [ ] **Step 3: Implement**

Create `esr/corruptions.py`:

```python
"""Acoustic corruptions that simulate unseen recording conditions.

SEEN corruptions may be used as training augmentation; UNSEEN ones are test-only.
All functions take a CPU torch.Generator so evaluation noise is identical for every model.
"""
import torch

SNR_DB = {1: 20.0, 2: 10.0, 3: 0.0}
RT60_S = {1: 0.3, 2: 0.6, 3: 1.0}
BAND_HZ = {1: (100.0, 5000.0), 2: (300.0, 3400.0), 3: (500.0, 2000.0)}
CLIP_FRAC = {1: 0.5, 2: 0.2, 3: 0.05}


def _rms(x):
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt()


def colored_noise(shape, exponent, gen, device="cpu"):
    """Unit-RMS noise with power spectrum ~ 1/f**exponent (0 white, 1 pink, 2 brown)."""
    noise = torch.randn(tuple(shape), generator=gen)
    if exponent != 0:
        spec = torch.fft.rfft(noise, dim=-1)
        f = torch.arange(spec.size(-1), dtype=torch.float32)
        f[0] = 1.0
        noise = torch.fft.irfft(spec / f.pow(exponent / 2), n=shape[-1], dim=-1)
    noise = noise - noise.mean(dim=-1, keepdim=True)
    return (noise / _rms(noise).clamp_min(1e-8)).to(device)


def add_noise_at_snr(wav, noise, snr_db):
    # Silent clips get noise relative to a -60 dBFS floor instead of producing NaN/no-op.
    signal_rms = _rms(wav).clamp_min(1e-3)
    scale = signal_rms / (10 ** (snr_db / 20)) / _rms(noise).clamp_min(1e-8)
    return wav + noise * scale


def _fft_convolve(x, h):
    n = x.size(-1) + h.size(-1) - 1
    y = torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(h, n=n), n=n)
    return y[..., : x.size(-1)]


def _synthetic_rir(rt60, sample_rate, gen, n):
    length = max(int(rt60 * sample_rate), 16)
    t = torch.arange(length, dtype=torch.float32) / sample_rate
    rir = torch.randn((n, length), generator=gen) * torch.exp(-6.9078 * t / rt60)  # -60 dB at t = rt60
    rir[:, 0] = 1.0  # direct path
    return rir / rir.norm(dim=-1, keepdim=True)


def apply_reverb(wav, rt60, sample_rate, gen):
    rir = _synthetic_rir(rt60, sample_rate, gen, wav.size(0)).to(wav.device)
    wet = _fft_convolve(wav, rir)
    return wet * (_rms(wav) / _rms(wet).clamp_min(1e-8))


def white_noise(wav, severity, gen, sample_rate=16000):
    return add_noise_at_snr(wav, colored_noise(wav.shape, 0.0, gen, wav.device), SNR_DB[severity])


def brown_noise(wav, severity, gen, sample_rate=16000):
    return add_noise_at_snr(wav, colored_noise(wav.shape, 2.0, gen, wav.device), SNR_DB[severity])


def reverb(wav, severity, gen, sample_rate=16000):
    return apply_reverb(wav, RT60_S[severity], sample_rate, gen)


def telephone(wav, severity, gen, sample_rate=16000):
    lo, hi = BAND_HZ[severity]
    spec = torch.fft.rfft(wav, dim=-1)
    freqs = torch.fft.rfftfreq(wav.size(-1), d=1.0 / sample_rate).to(wav.device)
    spec = spec * ((freqs >= lo) & (freqs <= hi))
    return torch.fft.irfft(spec, n=wav.size(-1), dim=-1)


def clipping(wav, severity, gen, sample_rate=16000):
    frac = CLIP_FRAC[severity]
    t = wav.abs().amax(dim=-1, keepdim=True) * frac
    return torch.maximum(torch.minimum(wav, t), -t) / frac


CORRUPTIONS = {
    "white_noise": white_noise,
    "reverb": reverb,
    "brown_noise": brown_noise,
    "telephone": telephone,
    "clipping": clipping,
}
SEEN = ("white_noise", "reverb")
UNSEEN = ("brown_noise", "telephone", "clipping")


def apply_corruption(wav, name, severity, gen, sample_rate=16000):
    if name not in CORRUPTIONS:
        raise KeyError(f"Unknown corruption {name!r}; choose one of {sorted(CORRUPTIONS)}")
    return CORRUPTIONS[name](wav, severity, gen, sample_rate)


def random_train_corruption(wav, p, gen, sample_rate=16000):
    """Seen-family augmentation with continuous parameters, applied per clip with probability p."""
    if p <= 0:
        return wav
    out = wav.clone()
    for i in range(wav.size(0)):
        if torch.rand(1, generator=gen).item() >= p:
            continue
        x = wav[i:i + 1]
        if torch.rand(1, generator=gen).item() < 0.5:
            snr = 5.0 + 25.0 * torch.rand(1, generator=gen).item()
            out[i:i + 1] = add_noise_at_snr(x, colored_noise(x.shape, 0.0, gen, x.device), snr)
        else:
            rt60 = 0.2 + 0.6 * torch.rand(1, generator=gen).item()
            out[i:i + 1] = apply_reverb(x, rt60, sample_rate, gen)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_corruptions.py -v`
Expected: 25 passed (15 parametrized + 10)

- [ ] **Step 5: Commit**

```bash
git add esr/corruptions.py tests/test_corruptions.py
git commit -m "feat: seen/unseen acoustic corruption suite" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Metrics

**Files:**
- Create: `esr/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `mean_average_precision(y_true [N,K], y_score [N,K]) -> float`. It is macro AP over classes with at least one positive. It raises `ValueError` if no class has a positive.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import math

import numpy as np
import pytest

from esr.metrics import mean_average_precision


def test_perfect_scores_give_one():
    y = np.array([[1, 0], [0, 1], [1, 1]], np.float32)
    assert mean_average_precision(y, y.copy()) == pytest.approx(1.0)


def test_map_skips_classes_without_positives():
    y = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    s = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.5]], np.float32)
    m = mean_average_precision(y, s)
    assert not math.isnan(m) and m == pytest.approx(1.0)


def test_map_no_positives_at_all_raises():
    with pytest.raises(ValueError):
        mean_average_precision(np.zeros((3, 2)), np.random.rand(3, 2))


def test_random_scores_are_worse_than_perfect():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 10)) < 0.2).astype(np.float32)
    assert mean_average_precision(y, rng.random((200, 10))) < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'esr.metrics'`

- [ ] **Step 3: Implement**

Create `esr/metrics.py`:

```python
"""Evaluation metrics."""
import numpy as np
from sklearn.metrics import average_precision_score


def mean_average_precision(y_true, y_score):
    """Macro mAP (the FSD50K standard) over classes that have at least one positive."""
    y_true = np.asarray(y_true)
    keep = y_true.sum(axis=0) > 0
    if not keep.any():
        raise ValueError("No class has a positive example; mAP is undefined.")
    return float(average_precision_score(y_true[:, keep], np.asarray(y_score)[:, keep], average="macro"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add esr/metrics.py tests/test_metrics.py
git commit -m "feat: macro mAP metric" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Training / inference engine

**Files:**
- Create: `esr/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `build_model` (Task 3), `mixup` (Task 4), `apply_corruption`, `random_train_corruption` (Task 5)
- Produces:
  - `set_seed(seed)`, `get_device() -> torch.device`
  - `make_scheduler(optimizer, total_steps, warmup_steps) -> LambdaLR`. The LR factor rises linearly to 1 during warm-up, then follows a cosine down to 0.01.
  - `save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, **extra)`. The write is atomic.
  - `load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None) -> dict` (the full state, including extra keys)
  - `train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, rng, gen, spec_transform=None) -> float` (mean loss)
  - `predict(model, loader, device, amp) -> (y_true [N,K], y_score [N,K])`. Scores are sigmoid probabilities.
  - `predict_array(model, waves_int16 [N,L], device, batch_size, amp, corruption=None, severity=1, seed=0, sample_rate=16000) -> y_score [N,K]`. Batch `b` is corrupted with generator seed `seed * 100003 + b`, so the result is deterministic.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_engine.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'esr.engine'`

- [ ] **Step 3: Implement**

Create `esr/engine.py`:

```python
"""Training loop, inference and checkpointing."""
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .augment import mixup
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
            wav = random_train_corruption(wav, cfg.corruption_aug_p, gen, cfg.sample_rate)
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
    for b, start in enumerate(range(0, len(waves), batch_size)):
        wav = torch.from_numpy(waves[start:start + batch_size].astype(np.float32) / 32767.0).to(device)
        if corruption is not None:
            gen = torch.Generator().manual_seed(seed * 100003 + b)
            wav = apply_corruption(wav, corruption, severity, gen, sample_rate)
        with _autocast(device, amp):
            logits = model(wav)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_engine.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add esr/engine.py tests/test_engine.py
git commit -m "feat: training loop, deterministic corrupted inference, atomic checkpoints" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Pipeline (train with resume, robustness evaluation, summary)

**Files:**
- Create: `esr/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `build_datasets(cfg) -> (train_ds: FSD50KDataset, val_ds: FSD50KDataset)`. `max_train_clips` keeps a seeded random subset.
  - `estimate_norm_stats(frontend, dataset, n_clips=256) -> (mean: float, std: float)`
  - `run_training(cfg) -> {"best_val_mAP": float, "history": list[dict], "exp_dir": str}`. It writes `config.json`, `history.csv`, `last.pt` and `best.pt` into `<out_dir>/<experiment>/`. It resumes from `last.pt` if that file exists. Otherwise it copies `cfg.resume_from` to `last.pt` first.
  - `run_robustness(cfg, checkpoint="best.pt", waves=None, targets=None) -> pd.DataFrame` with columns `experiment, condition, group, severity, mAP`. Groups are `clean` (severity 0), `seen` and `unseen`. It also writes `robustness.csv`.
  - `summarize(df) -> pd.DataFrame` with columns `experiment, clean, seen, unseen, seen_rel, unseen_rel`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
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


def test_run_training_effnet_robust_smoke(tiny_cfg):
    res = run_training(tiny_cfg("effnet_robust", batch_size=4))
    assert len(res["history"]) == 1


def test_run_robustness_rows_and_groups(tiny_cfg):
    cfg = tiny_cfg(severities=(1,))
    run_training(cfg)
    df = run_robustness(cfg)
    assert list(df.columns) == ["experiment", "condition", "group", "severity", "mAP"]
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
```

Note: `tiny_cfg(experiment="effnet_standard")` in `test_run_robustness_missing_checkpoint_raises` passes `experiment` as the factory's first argument. That is valid because the factory's parameter is named `experiment`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'esr.pipeline'`

- [ ] **Step 3: Implement**

Create `esr/pipeline.py`:

```python
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
    rows = []
    for name, group, severity in conditions:
        scores = predict_array(model, waves, device, cfg.batch_size, cfg.amp,
                               corruption=None if group == "clean" else name,
                               severity=severity, seed=cfg.seed, sample_rate=cfg.sample_rate)
        m = mean_average_precision(targets, scores)
        rows.append({"experiment": cfg.experiment, "condition": name, "group": group,
                     "severity": severity, "mAP": m})
        print(f"[{cfg.experiment}] {name:12s} sev={severity} ({group:6s}) mAP={m:.4f}")
    df = pd.DataFrame(rows, columns=["experiment", "condition", "group", "severity", "mAP"])
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 8 passed (the EfficientNet smoke test takes about 10–30 s on CPU)

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests pass (≈ 81)

- [ ] **Step 6: Commit**

```bash
git add esr/pipeline.py tests/test_pipeline.py
git commit -m "feat: resumable training pipeline and robustness evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Kaggle notebook generator, notebook and README

**Files:**
- Create: `tools/build_notebook.py`, `notebooks/esr_fsd50k_kaggle.ipynb` (generated), `README.md`
- Test: `tests/test_notebook.py`

**Interfaces:**
- Consumes: all `esr/*.py` files. Public names used by the notebook cells: `make_config`, `find_data_root`, `load_vocab`, `load_split`, `read_audio`, `fix_length`, `LogMel`, `apply_corruption`, `SEEN`, `UNSEEN`, `run_training`, `run_robustness`, `summarize`.
- Produces: `MODULES: list[str]`, `build(out_path) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_notebook.py`:

```python
import ast
from pathlib import Path

import nbformat

from tools.build_notebook import MODULES, build

ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path):
    return nbformat.read(build(tmp_path / "nb.ipynb"), as_version=4)


def test_notebook_is_valid(tmp_path):
    nbformat.validate(_build(tmp_path))


def test_every_module_is_written_verbatim(tmp_path):
    cells = [c.source for c in _build(tmp_path).cells if c.cell_type == "code"]
    assert sorted(MODULES) == sorted(p.name for p in (ROOT / "esr").glob("*.py"))
    for name in MODULES:
        header = f"%%writefile esr/{name}\n"
        match = [c for c in cells if c.startswith(header)]
        assert len(match) == 1, name
        assert match[0][len(header):] == (ROOT / "esr" / name).read_text(encoding="utf-8")


def test_all_python_cells_parse(tmp_path):
    for cell in _build(tmp_path).cells:
        if cell.cell_type != "code" or cell.source.startswith("%%"):
            continue
        code = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("!", "%")))
        ast.parse(code)


def test_config_cell_exposes_experiment_switches(tmp_path):
    src = "\n".join(c.source for c in _build(tmp_path).cells if c.cell_type == "code")
    for token in ('EXPERIMENT = "effnet_robust"', "QUICK_RUN = True", 'RESUME_FROM = ""', "find_data_root("):
        assert token in src
```

Create `tools/__init__.py` as an empty file so that `from tools.build_notebook import ...` resolves (pytest `pythonpath=["."]`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_notebook.py -v`
Expected: collection ERROR `ModuleNotFoundError: No module named 'tools.build_notebook'`

- [ ] **Step 3: Implement the generator**

Create `tools/build_notebook.py`:

```python
"""Pack the tested esr/ package + experiment cells into one Kaggle notebook.

Usage:  python tools/build_notebook.py   ->  notebooks/esr_fsd50k_kaggle.ipynb
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__.py", "config.py", "data.py", "features.py", "augment.py", "corruptions.py",
           "models.py", "metrics.py", "engine.py", "pipeline.py"]

INTRO = """# Environmental Sound Recognition Under Unseen Conditions (FSD50K)

**Before running:**
1. *Add Input* → search **`yousirui1/fsd50k`** → Add.
2. *Settings* → **Accelerator: GPU T4 x2 (or P100)**, **Internet: On** (needed for ImageNet weights via `timm`).
3. Run with `QUICK_RUN = True` first (~10 min smoke test), then set it to `False` and use
   *Save Version → Save & Run All (Commit)* so the long run survives closing the browser.
4. Run once per experiment: `cnn_baseline`, `effnet_standard`, `effnet_robust`.

Seen conditions (used as training augmentation in `effnet_robust`): white noise, reverb.
Unseen conditions (never used in training): brown noise, telephone band-pass, clipping.
The eval split is also uploader-disjoint from dev, which is a natural recording-condition shift.
"""

SETUP = """!pip install -q "timm>=1.0"
import os
os.makedirs("esr", exist_ok=True)"""

TAIL = [
    ("md", "## 1. Configuration"),
    ("code", '''import sys
for _m in [m for m in list(sys.modules) if m == "esr" or m.startswith("esr.")]:
    del sys.modules[_m]  # pick up re-written modules when cells are re-run

import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from esr.config import make_config, find_data_root
from esr.data import load_vocab, load_split, read_audio, fix_length
from esr.features import LogMel
from esr.corruptions import apply_corruption, SEEN, UNSEEN
from esr.pipeline import run_training, run_robustness, summarize

EXPERIMENT = "effnet_robust"   # one of: cnn_baseline, effnet_standard, effnet_robust
QUICK_RUN = True               # True: 2,000 train clips, 1,000 eval clips, 2 epochs. False: the real run.
RESUME_FROM = ""               # e.g. "/kaggle/input/<your-output-dataset>/outputs/effnet_robust/last.pt"

DATA_ROOT = find_data_root("/kaggle/input")
overrides = dict(data_root=DATA_ROOT, out_dir="/kaggle/working/outputs", resume_from=RESUME_FROM)
if QUICK_RUN:
    overrides.update(max_train_clips=2000, max_eval_clips=1000, epochs=2, out_dir="/kaggle/working/outputs_quick")
cfg = make_config(EXPERIMENT, **overrides)
print("DATA_ROOT:", DATA_ROOT)
print(cfg)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - enable a GPU accelerator!")'''),
    ("md", "## 2. Explore the dataset"),
    ("code", '''labels, _ = load_vocab(DATA_ROOT)
tr_meta, tr_y = load_split(DATA_ROOT, "train")
va_meta, va_y = load_split(DATA_ROOT, "val")
ev_meta, ev_y = load_split(DATA_ROOT, "eval")
print(f"classes={len(labels)}  train={len(tr_meta)}  val={len(va_meta)}  eval={len(ev_meta)}")
print(f"labels per clip (train): {tr_y.sum(1).mean():.2f}")
freq = pd.Series(tr_y.sum(0), index=labels).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
freq.tail(20).plot.barh(ax=axes[0], title="20 most frequent classes (train)")
freq.head(20).plot.barh(ax=axes[1], title="20 rarest classes (train)")
plt.tight_layout(); plt.show()'''),
    ("code", '''import soundfile as sf
sample = tr_meta.sample(2000, random_state=0)
durs = np.array([sf.info(p).duration for p in sample["path"]])
plt.hist(durs, bins=60); plt.xlabel("seconds")
plt.title(f"Clip duration (2,000 train clips), median = {np.median(durs):.1f} s"); plt.show()
print(f"{(durs > cfg.clip_seconds).mean():.1%} of clips are longer than {cfg.clip_seconds:.0f} s (random-cropped in training)")'''),
    ("md", "## 3. What the test conditions look like (severity 2)"),
    ("code", '''wav = torch.from_numpy(fix_length(read_audio(ev_meta["path"][0], cfg.sample_rate), cfg.clip_samples))[None]
front = LogMel.from_config(cfg)
conds = [("clean", None)] + [(n, n) for n in SEEN + UNSEEN]
fig, axes = plt.subplots(len(conds), 1, figsize=(12, 2.2 * len(conds)))
gen = torch.Generator().manual_seed(0)
with torch.no_grad():
    for ax, (title, name) in zip(axes, conds):
        x = wav if name is None else apply_corruption(wav, name, 2, gen, cfg.sample_rate)
        ax.imshow(front(x)[0, 0].numpy(), origin="lower", aspect="auto", cmap="magma")
        tag = "" if name is None else (" - SEEN family" if name in SEEN else " - UNSEEN")
        ax.set_title(title + tag); ax.set_yticks([])
plt.tight_layout(); plt.show()
print("Labels of this clip:", [labels[i] for i in np.where(ev_y[0])[0]])'''),
    ("md", "## 4. Train (resumes automatically from `last.pt`)"),
    ("code", '''result = run_training(cfg)
hist = pd.DataFrame(result["history"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
hist.plot(x="epoch", y="train_loss", ax=ax[0], title="train loss")
hist.plot(x="epoch", y="val_mAP", ax=ax[1], title="clean val mAP")
plt.show()
print("best clean val mAP:", round(result["best_val_mAP"], 4))'''),
    ("md", "## 5. Robustness on the eval set (clean + seen + unseen conditions)"),
    ("code", '''rob = run_robustness(cfg)
display(rob.pivot_table(index=["group", "condition"], columns="severity", values="mAP").round(4))
display(summarize(rob).round(4))
order = ["clean"] + list(SEEN) + list(UNSEEN)
m = rob.groupby("condition")["mAP"].mean().reindex(order)
m.plot.bar(color=["gray"] + ["tab:blue"] * len(SEEN) + ["tab:red"] * len(UNSEEN),
           title=f"{cfg.experiment}: mAP averaged over severities (blue = seen family, red = unseen)")
plt.ylabel("mAP"); plt.show()'''),
    ("md", """## 6. Compare experiments
After each experiment finishes (committed version), open the version's *Output* and click
*New Dataset* (or add the notebook's output as an input). Add those outputs back here as inputs,
then re-run this cell. It collects every `robustness.csv` it can find."""),
    ("code", '''import glob
paths = sorted(set(glob.glob("/kaggle/input/**/robustness.csv", recursive=True)
                   + glob.glob(f"{cfg.out_dir}/*/robustness.csv")))
print("\\n".join(paths) or "no robustness.csv found yet")
if paths:
    allres = pd.concat([pd.read_csv(p) for p in paths]).drop_duplicates(["experiment", "condition", "severity"], keep="last")
    table = summarize(allres)
    display(table.round(4))
    table.set_index("experiment")[["clean", "seen", "unseen"]].plot.bar(title="mAP: clean vs seen vs unseen conditions")
    plt.ylabel("mAP"); plt.show()'''),
]


def build(out_path=ROOT / "notebooks" / "esr_fsd50k_kaggle.ipynb"):
    out_path = Path(out_path)
    cells = [new_markdown_cell(INTRO), new_code_cell(SETUP),
             new_markdown_cell("## 0. Write the `esr` package (tested locally, see repo `tests/`)")]
    for name in MODULES:
        src = (ROOT / "esr" / name).read_text(encoding="utf-8")
        cells.append(new_code_cell(f"%%writefile esr/{name}\n{src}"))
    for kind, text in TAIL:
        cells.append(new_markdown_cell(text) if kind == "md" else new_code_cell(text))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    return out_path


if __name__ == "__main__":
    print(build())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_notebook.py -v`
Expected: 4 passed

- [ ] **Step 5: Generate the notebook and write the README**

Run: `python tools/build_notebook.py`
Expected: prints `...notebooks/esr_fsd50k_kaggle.ipynb`

Create `README.md`:

````markdown
# Environmental Sound Recognition Under Unseen Conditions

CNN sound-event classifiers on FSD50K (Kaggle mirror `yousirui1/fsd50k`), evaluated on the
uploader-disjoint eval set under clean, **seen** (white noise, reverb) and **unseen**
(brown noise, telephone band, clipping) acoustic conditions.
Design: `docs/superpowers/specs/2026-09-27-esr-unseen-conditions-design.md`.

## Local development (CPU)
```bash
pip install -r requirements-dev.txt
python -m pytest -q                 # tests run on a synthetic FSD50K layout
python tools/build_notebook.py      # regenerate notebooks/esr_fsd50k_kaggle.ipynb after editing esr/
```

## Running on Kaggle
1. kaggle.com → Code → New Notebook → File → Import Notebook → `notebooks/esr_fsd50k_kaggle.ipynb`.
2. Add Input `yousirui1/fsd50k`. Accelerator GPU. Internet On.
3. Run all with `QUICK_RUN = True` (smoke test), then set `QUICK_RUN = False`, pick `EXPERIMENT`,
   and use Save Version → Save & Run All.
4. Repeat for `cnn_baseline`, `effnet_standard`, `effnet_robust`. Add the finished versions' outputs
   as inputs and run section 6 to compare.
5. If a run hits the 12 h limit, add its output as an input and set `RESUME_FROM` to its `last.pt`.

## Experiments
| name | model | robustness training |
|---|---|---|
| cnn_baseline | VGG-style CNN from scratch + attention pooling | – |
| effnet_standard | EfficientNet-B0 (ImageNet) + attention pooling | – |
| effnet_robust | EfficientNet-B0 (ImageNet) + attention pooling | FreqMixStyle + white-noise/reverb aug |
````

- [ ] **Step 6: Run the full suite once more**

Run: `python -m pytest -q`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add tools/ notebooks/esr_fsd50k_kaggle.ipynb README.md tests/test_notebook.py
git commit -m "feat: Kaggle notebook generator, generated notebook and README" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## After the plan: running on Kaggle (manual, by the user)

| Run | Setting | Expected time (T4, estimate) |
|---|---|---|
| Smoke test | `QUICK_RUN=True`, any experiment | ~10–15 min |
| `cnn_baseline` | `QUICK_RUN=False` | ~3–4 h train + ~40 min eval |
| `effnet_standard` | `QUICK_RUN=False` | ~3–4 h train + ~40 min eval |
| `effnet_robust` | `QUICK_RUN=False` | ~4–5 h train (corruption aug is extra work) + ~40 min eval |

Check the `seconds` column after epoch 1. If `epochs × seconds` is more than ~10 h, lower `epochs`
through `make_config(..., epochs=N)` or rely on `RESUME_FROM`.

What to report: the section 6 table (clean / seen / unseen mAP and the relative-robustness ratios),
the per-condition pivot tables, and the severity curves. The key comparisons are
`effnet_standard` vs `cnn_baseline` (architecture + pretraining) and `effnet_robust` vs
`effnet_standard` on **unseen** conditions (does robustness training transfer to corruptions it never saw?).
