# Environmental Sound Recognition Under Unseen Conditions

CNN sound-event classifiers on FSD50K (Kaggle mirror `yousirui1/fsd50k`), evaluated on the
uploader-disjoint eval set under clean, **seen** (white noise, reverb) and **unseen**
(brown noise, telephone band, clipping, speed-up, quantisation) acoustic conditions.
Design: `docs/superpowers/specs/2026-09-27-esr-unseen-conditions-design.md`.

## Local development (CPU)
```bash
pip install -r requirements-dev.txt
python -m pytest -q                 # tests run on a synthetic FSD50K layout
```

## Running on Kaggle (clone + run scripts)
Import `notebooks/kaggle_run.ipynb` (set `REPO_URL` to this repo), or paste these cells into a new notebook.
Add Input `yousirui1/fsd50k`, Accelerator **GPU T4**, Internet **On**.
```python
!git clone https://github.com/<your-user>/<your-repo>.git /kaggle/working/esr
%cd /kaggle/working/esr
!git pull origin main
!pip install -q "timm>=1.0"
```
```bash
# smoke test (~15 min) -> /kaggle/working/outputs_quick
!python scripts/train.py    --experiment effnet_robust --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k --quick
!python scripts/evaluate.py --experiment effnet_robust --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k --quick --severities 2
# full run (Save Version -> Save & Run All) -> /kaggle/working/outputs/effnet_robust/
!python scripts/train.py    --experiment effnet_robust --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
!python scripts/evaluate.py --experiment effnet_robust --data-path /kaggle/input/datasets/yousirui1/fsd50k/fsd50k
# after running the experiments you want to compare (their outputs added as inputs)
!python scripts/compare.py --out /kaggle/working/summary.csv
```
- `--data-path` can be omitted: the dataset is auto-discovered under `/kaggle/input`.
- Training resumes from `outputs/<experiment>/last.pt`. After a 12 h timeout, add that version's output as an
  input and pass `--resume-from /kaggle/input/<output>/outputs/<experiment>/last.pt`.
- Every flag: `python scripts/train.py --help` (`--epochs`, `--batch-size`, `--lr`, `--num-workers`, ...).
- Outputs per experiment: `config.json`, `history.csv`, `last.pt`, `best.pt`, `robustness.csv`.
- `compare` also writes `summary_by_condition.csv` (relative mAP per condition × severity) and skips results from
  an older evaluation protocol.

## Experiments
| name | model | robustness training |
|---|---|---|
| cnn_baseline | VGG-style CNN from scratch + attention pooling | – |
| effnet_standard | EfficientNet-B0 (ImageNet) + attention pooling | – |
| effnet_robust | EfficientNet-B0 (ImageNet) + attention pooling | FreqMixStyle + white-noise/reverb aug |
| effnet_robust_v2 | EfficientNet-B0 (ImageNet) + attention pooling | FreqMixStyle + white-noise (0–30 dB)/reverb (RT60 ≤ 1 s) aug + random EQ |
| effnet_mixstyle / effnet_corrupt / effnet_eq | EfficientNet-B0 | one robustness component each (ablation) |
