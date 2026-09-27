"""Command-line entry points: train, evaluate (robustness suite) and compare experiments.

    python scripts/train.py    --experiment effnet_robust [--quick] [--epochs N] ...
    python scripts/evaluate.py --experiment effnet_robust [--quick] ...
    python scripts/compare.py  --inputs "/kaggle/working/outputs/*/robustness.csv"
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from .config import EXPERIMENTS, find_data_root, make_config

QUICK = dict(max_train_clips=2000, max_eval_clips=1000, epochs=2)

# CLI flag -> Config field; flags left unset keep the experiment's defaults.
OVERRIDES = {
    "out_dir": "out_dir", "epochs": "epochs", "batch_size": "batch_size", "lr": "lr",
    "num_workers": "num_workers", "warmup_epochs": "warmup_epochs", "clip_seconds": "clip_seconds",
    "n_mels": "n_mels", "max_train_clips": "max_train_clips", "max_eval_clips": "max_eval_clips",
    "resume_from": "resume_from", "seed": "seed",
}


def _add_run_args(p):
    p.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--data-path", help="folder containing FSD50K.ground_truth/ (default: auto-discover)")
    p.add_argument("--search-root", default="/kaggle/input", help="where to auto-discover the dataset")
    p.add_argument("--out-dir", help="default /kaggle/working/outputs (outputs_quick with --quick)")
    p.add_argument("--quick", action="store_true", help="smoke test: 2,000 train / 1,000 eval clips, 2 epochs")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--warmup-epochs", type=int)
    p.add_argument("--clip-seconds", type=float)
    p.add_argument("--n-mels", type=int)
    p.add_argument("--max-train-clips", type=int)
    p.add_argument("--max-eval-clips", type=int)
    p.add_argument("--resume-from", help="last.pt from a previous Kaggle version")
    p.add_argument("--seed", type=int)
    p.add_argument("--severities", type=int, nargs="+", choices=[1, 2, 3])
    p.add_argument("--checkpoint", default="best.pt",
                   help="evaluate only: file name in <out-dir>/<experiment>/ or an absolute path "
                        "(e.g. a best.pt attached from /kaggle/input)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="esr", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_args(sub.add_parser("train", help="train one experiment (resumes from last.pt)"))
    _add_run_args(sub.add_parser("evaluate", help="clean + seen + unseen robustness on the eval set"))
    cmp = sub.add_parser("compare", help="summarize robustness.csv files from several experiments")
    cmp.add_argument("--inputs", nargs="+", default=["/kaggle/working/outputs/*/robustness.csv",
                                                     "/kaggle/input/**/robustness.csv"])
    cmp.add_argument("--include-subset", action="store_true", help="also use --quick (smoke-test) results")
    cmp.add_argument("--out", default="/kaggle/working/summary.csv")
    return parser.parse_args(argv)


def build_config(args):
    overrides = {}
    if args.quick:
        overrides.update(QUICK, out_dir="/kaggle/working/outputs_quick")
    for flag, field in OVERRIDES.items():
        value = getattr(args, flag)
        if value is not None:
            overrides[field] = value
    if args.quick and args.out_dir:
        overrides["out_dir"] = args.out_dir.rstrip("/\\") + "_quick"
    if args.severities:
        overrides["severities"] = tuple(args.severities)
    if args.no_amp:
        overrides["amp"] = False
    if args.no_pretrained:
        overrides["pretrained"] = False
    overrides["data_root"] = args.data_path or find_data_root(args.search_root)
    return make_config(args.experiment, **overrides)


def compare(inputs, include_subset=False, out=None):
    from .pipeline import per_condition, summarize

    paths = sorted({p for pattern in inputs for p in glob.glob(pattern, recursive=True)})
    if not paths:
        raise SystemExit(f"No robustness.csv found for {inputs}. Run evaluate first or add outputs as inputs.")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "subset" not in df:
            df["subset"] = False
        if "protocol" not in df:
            df["protocol"] = 1  # written before protocol versions existed
        df["source"] = p
        frames.append(df)
    allres = pd.concat(frames, ignore_index=True)
    if not include_subset:
        allres = allres[~allres["subset"].astype(bool)]
    if allres.empty:
        raise SystemExit("Only --quick (subset) results found; pass --include-subset to compare them anyway.")
    latest = allres["protocol"].max()
    stale = sorted(set(allres.loc[allres["protocol"] < latest, "experiment"]))
    if stale:
        print(f"Skipping protocol < {latest} results (re-run evaluate on them): {', '.join(stale)}")
    allres = allres[allres["protocol"] == latest]
    allres = allres.drop_duplicates(["experiment", "condition", "severity"], keep="last")
    print(f"Protocol {latest} results")
    for exp, src in allres.groupby("experiment")["source"].first().items():
        print(f"{exp}: {src}")
    table = summarize(allres)
    print(table.round(4).to_string(index=False))
    detail = per_condition(allres)
    print(detail.pivot_table(index=["group", "condition", "severity"], columns="experiment", values="rel")
          .round(3).to_string())
    if out:
        table.to_csv(out, index=False)
        detail.to_csv(Path(out).with_name(Path(out).stem + "_by_condition.csv"), index=False)
    return table


def main(argv=None):
    args = parse_args(argv)
    if args.command == "compare":
        return compare(args.inputs, args.include_subset, args.out)

    from .pipeline import run_robustness, run_training

    cfg = build_config(args)
    print(cfg)
    if args.command == "train":
        return run_training(cfg)
    return run_robustness(cfg, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main(sys.argv[1:])
