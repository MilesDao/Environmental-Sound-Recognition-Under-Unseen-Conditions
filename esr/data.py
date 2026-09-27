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
