"""Stage 1: load HH-RLHF and LMSYS Arena, normalise to one schema, filter, balance, split."""
import ast
import json
import os
import random
from pathlib import Path

import pandas as pd

from .config import SEED, Config

LMSYS_CANDIDATES = [
    "train.csv",
    "data/train.csv",
    "/kaggle/input/lmsys-chatbot-arena/train.csv",
    "/kaggle/input/lmsys-chatbot-arena-human-preference-predictions/train.csv",
]


# ---------------------------------------------------------------- HH-RLHF
def parse_hh(example: dict) -> dict:
    """Split an HH-RLHF dialogue pair into (prompt, chosen, rejected) on the last Assistant turn."""
    chosen, rejected = example["chosen"], example["rejected"]
    tok = "\n\nAssistant:"
    if tok not in chosen or tok not in rejected:
        return {"prompt": "", "chosen": "", "rejected": "", "source": "hh_rlhf"}
    c_idx, r_idx = chosen.rfind(tok), rejected.rfind(tok)
    # Matches the notebooks exactly: the leading "Human:" is kept in the prompt.
    prompt = chosen[:c_idx].strip()
    return {
        "prompt": prompt,
        "chosen": chosen[c_idx + len(tok):].strip(),
        "rejected": rejected[r_idx + len(tok):].strip(),
        "source": "hh_rlhf",
    }


def load_hh() -> list:
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    recs = [parse_hh(r) for r in ds]
    recs = [r for r in recs if r["prompt"] and r["chosen"] and r["rejected"]]
    print(f"[data] HH-RLHF: {len(ds):,} rows -> {len(recs):,} valid")
    return recs


# ---------------------------------------------------------------- LMSYS
def safe_parse(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError, TypeError):
        try:
            return json.loads(x)
        except Exception:
            return [str(x)]


def last_turn(prompt_list, response_list):
    """Keep only the final user turn and final response (earlier turns of A and B
    come from different models, so concatenating them would contaminate context)."""
    prompts, responses = safe_parse(prompt_list), safe_parse(response_list)
    if not prompts:
        return "", ""
    return str(prompts[-1]).strip(), (str(responses[-1]).strip() if responses else "")


def find_lmsys_csv(cfg: Config) -> str:
    if cfg.lmsys_csv and os.path.exists(cfg.lmsys_csv):
        return cfg.lmsys_csv
    for p in LMSYS_CANDIDATES:
        if os.path.exists(p):
            return p
    hits = list(Path("/kaggle/input").glob("**/train.csv")) if Path("/kaggle/input").exists() else []
    for h in hits:
        cols = pd.read_csv(h, nrows=1).columns.str.strip()
        if {"winner_model_a", "winner_tie"} <= set(cols):
            return str(h)
    # Fallback: the same 57,477-row Arena file is published on the HuggingFace Hub.
    from huggingface_hub import hf_hub_download
    print("[data] LMSYS train.csv not found locally - downloading lmarena-ai/arena-human-preference-55k")
    return hf_hub_download("lmarena-ai/arena-human-preference-55k", "train.csv",
                           repo_type="dataset", local_dir="data")


def load_lmsys(cfg: Config) -> list:
    path = find_lmsys_csv(cfg)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    needed = {"winner_tie", "winner_model_a", "winner_model_b", "prompt", "response_a", "response_b"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"LMSYS CSV missing columns: {missing}")
    recs, ties = [], 0
    for row in df.itertuples(index=False):
        if row.winner_tie == 1:
            ties += 1
            continue
        prompt, resp_a = last_turn(row.prompt, row.response_a)
        _, resp_b = last_turn(row.prompt, row.response_b)
        if row.winner_model_a == 1:
            chosen, rejected = resp_a, resp_b
        elif row.winner_model_b == 1:
            chosen, rejected = resp_b, resp_a
        else:
            continue
        if prompt and chosen and rejected:
            recs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": "lmsys"})
    print(f"[data] LMSYS: {len(df):,} rows, {ties:,} ties dropped -> {len(recs):,} valid")
    return recs


# ---------------------------------------------------------------- cleaning / split
def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")


def truncate_and_filter(r: dict, max_chars: int, min_chars: int):
    p, c, x = (clean_text(r[k])[:max_chars] for k in ("prompt", "chosen", "rejected"))
    if min(len(p), len(c), len(x)) < min_chars:
        return None
    return {**r, "prompt": p, "chosen": c, "rejected": x}


def stratified_split(df: pd.DataFrame, fracs=(0.875, 0.0625, 0.0625)):
    """Split each source separately so every split contains both labelling regimes."""
    parts = ([], [], [])
    for _, sub in df.groupby("source"):
        sub = sub.reset_index(drop=True)
        n_tr = int(round(len(sub) * fracs[0]))
        n_va = int(round(len(sub) * fracs[1]))
        parts[0].append(sub.iloc[:n_tr])
        parts[1].append(sub.iloc[n_tr:n_tr + n_va])
        parts[2].append(sub.iloc[n_tr + n_va:])
    return tuple(pd.concat(p).sample(frac=1, random_state=SEED).reset_index(drop=True) for p in parts)


def build_dataset(cfg: Config):
    """Run the whole of Stage 1 and save train/val/test CSVs to cfg.out."""
    out = cfg.out
    if all((out / f"{s}.csv").exists() for s in ("train", "val", "test")):
        print(f"[data] Using cached splits in {out}")
        return tuple(pd.read_csv(out / f"{s}.csv").fillna("") for s in ("train", "val", "test"))

    random.seed(SEED)
    hh = [o for r in load_hh() for o in [truncate_and_filter(r, cfg.max_chars, cfg.min_chars)] if o]
    lm = [o for r in load_lmsys(cfg) for o in [truncate_and_filter(r, cfg.max_chars, cfg.min_chars)] if o]
    print(f"[data] after truncate-and-filter: HH {len(hh):,} | LMSYS {len(lm):,}")
    random.shuffle(hh)
    random.shuffle(lm)
    combined = hh[:cfg.hh_per_source] + lm[:cfg.lmsys_per_source]
    random.shuffle(combined)
    train, val, test = stratified_split(pd.DataFrame(combined), cfg.split)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(out / f"{name}.csv", index=False)
        print(f"[data] {name}: {len(d)} {d['source'].value_counts().to_dict()}")
    return train, val, test
