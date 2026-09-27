# Environmental Sound Recognition Under Unseen Conditions — Design

**Date:** 2026-09-27
**Dataset:** FSD50K, Kaggle mirror `yousirui1/fsd50k` (https://www.kaggle.com/datasets/yousirui1/fsd50k/data)
**Run target:** a single Kaggle notebook (`.ipynb`), GPU accelerator (T4 / P100), Internet ON.

---

## 1. Research question

> How much does a CNN sound-event classifier trained on FSD50K lose when test audio is recorded
> under acoustic conditions it never saw during training, and which training strategies close that gap?

"Unseen conditions" is made measurable in two ways:

1. **Natural domain shift.** FSD50K's `eval` split is uploader-disjoint from `dev` (different
   people, microphones, rooms). Clean eval mAP already measures generalization to unseen
   recording conditions.
2. **Controlled acoustic corruptions** applied to the eval set, at 3 severities each:

| Corruption    | Family | Severity 1 / 2 / 3                    | Used in training aug? |
|---------------|--------|---------------------------------------|-----------------------|
| `white_noise` | noise  | SNR 20 / 10 / 0 dB                    | **seen** (robust run) |
| `reverb`      | room   | RT60 0.3 / 0.6 / 1.0 s (synthetic RIR)| **seen** (robust run) |
| `brown_noise` | noise  | SNR 20 / 10 / 0 dB (1/f² spectrum)    | **unseen**            |
| `telephone`   | channel| band-pass 100–5000 / 300–3400 / 500–2000 Hz | **unseen**      |
| `clipping`    | device | clip at 50 % / 20 % / 5 % of peak     | **unseen**            |

The robust model is trained with white noise + reverb only (continuous random parameters),
then tested on the held-out corruption types. Evaluation noise is seeded per batch, so every
model sees the exact same corrupted audio.

## 2. Dataset structure (verified via Kaggle API, 61,441 files, 21.4 GB)

```
/kaggle/input/fsd50k/fsd50k/            <- DATA_ROOT (auto-discovered; mount path can vary)
├── FSD50K.dev_audio_16k/   40,966 wav   9.30 GB   16 kHz, mono, PCM16   (train + val)
├── FSD50K.eval_audio_16k/  10,231 wav   3.21 GB   16 kHz, mono, PCM16   (test)
├── FSD50K.eval_audio/      10,231 wav   8.84 GB   44.1 kHz original     (NOT used)
├── FSD50K.ground_truth/
│   ├── vocabulary.csv   no header: index,label,mid          (200 classes)
│   ├── dev.csv          fname,labels,mids,split   split ∈ {train: 36,796, val: 4,170}
│   └── eval.csv         fname,labels,mids                    (10,231 rows)
├── FSD50K.metadata/  class_info / dev_clips_info / eval_clips_info / pp_pnp_ratings (json)
│   └── collection/   collection_{dev,eval}.csv, vocabulary_collection_{dev,eval}.csv
└── FSD50K.doc/       README.md, LICENSE-DATASET
```

- `labels` is a comma-separated string, e.g. `"Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music"`.
  Multi-label: 2.79 labels per clip on average (AudioSet ontology, parents included).
- Audio file = `<audio dir>/<fname>.wav`. Clips are 0.3–30 s long, median ≈ 5–8 s.
- The dev set has no 44.1 kHz folder in this mirror, so the pipeline uses **only the `*_16k` folders**.
- The task is **multi-label classification**. The standard metric is **macro mAP** over the 200 classes.

## 3. Which CNN? (decision)

| Candidate | Pros | Cons | Role |
|---|---|---|---|
| VGG-style CNN from scratch (PANNs CNN10-like, 4 conv blocks) | Simple and transparent. Matches the FSD50K paper baselines (VGG-like ≈ 0.43 mAP reported) | Needs many epochs. Overfits rare classes | **Baseline** (`cnn_baseline`) |
| ResNet-18 / DenseNet from scratch | Well known | Reported weaker than VGG-like on FSD50K without pretraining | Not used |
| **EfficientNet-B0, ImageNet-pretrained, + attention pooling (PSLA recipe, Gong et al. 2021)** | Best accuracy-per-FLOP CNN on FSD50K (PSLA reports ≈ 0.55+ mAP with B2 + balanced sampling + augmentation). 5.3 M params, so it fits a T4 with AMP. Pretrained features transfer well. Attention pooling down-weights noisy frames | Needs Internet ON in Kaggle to download weights | **Main model** (`effnet_standard`, `effnet_robust`) |
| PANNs CNN14 (AudioSet-pretrained) | Strong audio transfer | 80 M params, slow on T4. Weights not in `timm` | Optional extension |
| AST / HTS-AT (transformers) | Top of the leaderboard | Not a CNN. Too heavy for Kaggle quotas | Out of scope |

**Decision:** EfficientNet-B0 (via `timm`, `in_chans=1`) on 128-bin log-mel spectrograms of
10 s crops, frequency-mean then attention pooling over time. For robustness, add
**Frequency-wise MixStyle** (Schmid et al. 2022, DCASE device generalization), which mixes
per-frequency statistics across the batch. This simulates unseen microphone/channel responses
without modeling them explicitly. Add waveform corruption augmentation from the seen family.
Upgrade path: set `model="efficientnet_b2"` when there is GPU time to spare. The numbers above are
as reported in the literature; verify them against the papers before citing.

## 4. Experiments

| Name | Model | Pretrained | Aug | Purpose |
|---|---|---|---|---|
| `cnn_baseline`    | SimpleCNN        | no       | SpecAugment + mixup | reference CNN |
| `effnet_standard` | EfficientNet-B0  | ImageNet | SpecAugment + mixup | effect of architecture + pretraining |
| `effnet_robust`   | EfficientNet-B0  | ImageNet | + FreqMixStyle (p=0.7) + seen corruptions (p=0.5) | effect of robustness training |

Common settings: 16 kHz, 10 s crops (random crop in training, first 10 s in eval), log-mel with
n_fft 1024, win 25 ms, hop 10 ms, 128 mels, 50–8000 Hz. Balanced sampling (PSLA). AdamW,
1 warm-up epoch plus cosine decay, 20 epochs, batch 32, AMP. Model selection uses **clean val mAP
only**. Corrupted data is never used for selection, because that would leak the unseen conditions.

**Reported metrics** (`robustness.csv`, `summarize()`): clean eval mAP. Mean mAP over the seen
family and over the unseen family. Relative robustness = corrupted mAP / clean mAP.

## 5. Compute budget (estimates; the notebook prints seconds/epoch so you can adjust)

- Training ≈ 8–12 min/epoch on T4 for EfficientNet-B0, so 20 epochs ≈ 3–4 h. SimpleCNN is similar.
- Robustness eval: 16 conditions × 10,231 clips held in RAM as int16 (3.3 GB) ≈ 30–45 min.
- One experiment per notebook run (Kaggle 12 h session limit, ~30 GPU-h/week). `QUICK_RUN=True`
  does a ~10 min smoke test on 2,000 clips first. `last.pt` allows resuming across sessions.

## 6. Code organization

The source of truth is a small, locally tested package `esr/`. `tools/build_notebook.py` packs every
module into `%%writefile esr/<module>.py` cells, followed by the experiment cells, and writes
`notebooks/esr_fsd50k_kaggle.ipynb`. Tests run locally on CPU against a tiny synthetic FSD50K
layout (sine tones), so the full pipeline is verified before anything is uploaded to Kaggle.
