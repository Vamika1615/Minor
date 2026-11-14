"""Command-line entry point for the cross-distribution RLHF pipeline.

Examples (on a Kaggle GPU notebook):
    python run_pipeline.py --config A                 # all stages
    python run_pipeline.py --config B --stage rm      # only the reward model
    python run_pipeline.py --config A --stage eval --lmsys-csv /kaggle/input/.../train.csv
"""
import argparse
import os
import random
from dataclasses import replace

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

from rlhf.config import CONFIGS, SEED
from rlhf.data import build_dataset
from rlhf.evaluate import run_evaluation
from rlhf.train import train_dpo, train_reward_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=CONFIGS, default="A", help="A = balanced 2K+2K, B = HH-weighted 6K+2K, local = small run for an 8 GB Mac")
    ap.add_argument("--stage", choices=["data", "rm", "dpo", "eval", "all"], default="all")
    ap.add_argument("--lmsys-csv", default="", help="path to LMSYS train.csv (auto-detected on Kaggle)")
    ap.add_argument("--out-dir", default=None, help="output folder (default outputs/config_<A|B>)")
    args = ap.parse_args()

    cfg = replace(CONFIGS[args.config], lmsys_csv=args.lmsys_csv,
                  out_dir=args.out_dir or f"outputs/config_{args.config}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"Config {cfg.name} -> {cfg.out}")

    train, val, test = build_dataset(cfg)
    if args.stage in ("rm", "all"):
        train_reward_model(cfg, train, val)
    if args.stage in ("dpo", "all"):
        train_dpo(cfg, train, val)
    if args.stage in ("eval", "all"):
        run_evaluation(cfg, test)


if __name__ == "__main__":
    main()
