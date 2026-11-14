# ===================== Notebook cell 0 =====================
# RUN THIS FIRST - clears old TinyLlama RM weights (incompatible with DeBERTa)
# Only needed once after switching RM architecture. Safe to keep for future runs.
import shutil, os

for model_dir in ["/kaggle/working/reward_model", "/kaggle/working/finetuned_llm"]:
    if os.path.exists(model_dir):
        # Check if it's a DeBERTa model or old TinyLlama
        cfg_path = f"{model_dir}/adapter_config.json"
        if os.path.exists(cfg_path):
            import json as _json
            with open(cfg_path) as _f: cfg = _json.load(_f)  # BUG8: properly closed file handle
            base = cfg.get('base_model_name_or_path', '')
            if 'TinyLlama' in base and 'reward_model' in model_dir:
                shutil.rmtree(model_dir)
                print(f"  Deleted old TinyLlama RM at {model_dir} - will retrain with DeBERTa")
            else:
                print(f" {model_dir} looks correct ({base}) - keeping")
        else:
            shutil.rmtree(model_dir)
            print(f"  Deleted incomplete model dir: {model_dir}")
    else:
        print(f"  {model_dir} not found - will train fresh")

# ===================== Notebook cell 1 =====================
import os

path = "/kaggle/input/notebooks/sahil2604singh/mini-6/__results___files"

for root, dirs, files in os.walk(path):
    print("FOLDER:", root)
    print("FILES:", files)
    print("-"*50)

# ===================== Notebook cell 4 =====================
# === Set env vars FIRST - must happen before any bitsandbytes import ===
import os, sys
# FIXED: removed hardcoded BNB_CUDA_VERSION - let bitsandbytes auto-detect
# Setting this to a fixed value causes a warning when it mismatches PyTorch CUDA
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["WANDB_DISABLED"]  = "true"   # prevent TRL from importing broken wandb
os.environ["WANDB_MODE"]      = "disabled"
os.environ["WANDB_SILENT"]    = "true"

if os.path.exists("/kaggle/working"):
    BASE_DIR = "/kaggle/working"
elif os.path.exists("/content"):
    BASE_DIR = "/content"
else:
    BASE_DIR = os.path.abspath("./outputs")
    os.makedirs(BASE_DIR, exist_ok=True)

REWARD_MODEL_PATH  = f"{BASE_DIR}/reward_model"
FINETUNED_LLM_PATH = f"{BASE_DIR}/finetuned_llm"

# PERSISTENCE FIX: Kaggle clears /kaggle/working between sessions.
# After training, save a version (Save -> Quick Save) so outputs persist as
# /kaggle/input/<your-notebook-name>/reward_model on the next run.
# The block below auto-detects the saved model from either location.
import glob

def find_saved_model(model_dir_name):
    """Check working dir first, then any Kaggle input dataset."""
    working_path = f"{BASE_DIR}/{model_dir_name}"
    if os.path.exists(f"{working_path}/adapter_config.json"):
        return working_path
    # Search Kaggle input datasets (saved from previous notebook output)
    candidates = glob.glob(f"/kaggle/input/**/{model_dir_name}/adapter_config.json", recursive=True)
    if candidates:
        return os.path.dirname(candidates[0])
    return working_path   # return default path even if missing (will trigger training)

REWARD_MODEL_PATH  = find_saved_model("reward_model")
FINETUNED_LLM_PATH = find_saved_model("finetuned_llm")
PLOTS_DIR          = f"{BASE_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Base directory     : {BASE_DIR}")
print(f"Reward model path  : {REWARD_MODEL_PATH}  (exists: {os.path.exists(REWARD_MODEL_PATH)})")
print(f"Fine-tuned LLM path: {FINETUNED_LLM_PATH} (exists: {os.path.exists(FINETUNED_LLM_PATH)})")

# ===================== Notebook cell 5 =====================
# Install dependencies 
# torchao MUST be installed first and force-reinstalled:
# Kaggle ships torchao==0.10.0 but PEFT >=0.12 requires >=0.16.0.
# Without force-reinstall the old version stays cached in memory.
!pip install -q --upgrade --force-reinstall "torchao>=0.16.0" 2>&1 | tail -3

# Clear torchao from module cache so PEFT picks up the new version
import sys as _sys
for _k in list(_sys.modules.keys()):
    if "torchao" in _k:
        del _sys.modules[_k]

!pip install -q -U wandb 2>&1 | tail -2          # fix wandb proto mismatch before TRL imports it
!pip install -q -U bitsandbytes transformers trl peft accelerate \
    datasets tokenizers sentencepiece protobuf 2>&1 | tail -5

import bitsandbytes as bnb
import transformers, trl, peft
print("bnb        :", bnb.__version__)
print("transformers:", transformers.__version__)
print("trl        :", trl.__version__)
print("peft       :", peft.__version__)
import torchao; print("torchao    :", torchao.__version__)

# ===================== Notebook cell 6 =====================
import gc, json, ast, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
warnings.filterwarnings("always", category=RuntimeWarning)  # FIX: keep RuntimeWarnings (NaN score alerts) visible
sns.set_style("whitegrid")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("CUDA available :", torch.cuda.is_available())
print("GPU            :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if torch.cuda.is_available():
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("PyTorch        :", torch.__version__)

# GPU sanity check - confirm T4 x2 is active
!nvidia-smi | grep -E "GPU|MiB|Driver"

# ===================== Notebook cell 8 =====================
# === Step 1 - Load HH-RLHF helpful-base ===
# This downloads from HuggingFace. On Kaggle, make sure your kernel has internet enabled
# (Settings -> Internet: ON). On Colab, internet is on by default.
from datasets import load_dataset

HH_AVAILABLE = False
try:
    print(" Downloading Anthropic HH-RLHF helpful-base...")
    # FIXED: removed dead hh_raw_test download - test split was never used, wasted network time
    hh_train = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    HH_AVAILABLE = True
    print(f"    HH-RLHF loaded: {len(hh_train):,} train rows")
except Exception as e:
    print(f"     HH-RLHF download failed: {type(e).__name__}: {e}")
    print(f"   -> Will fall back to LMSYS-only (this still works fine for the project)")
    hh_train = None

# ===================== Notebook cell 9 =====================
# === Step 2 - Parse HH-RLHF into common schema ===
# HH-RLHF format: 'chosen' and 'rejected' are full dialogues like:
# "\n\nHuman: <prompt>\n\nAssistant: <response>"
# We need to split into (prompt, chosen_response, rejected_response).

def parse_hh(example):
    chosen = example["chosen"]; rejected = example["rejected"]
    split_token = "\n\nAssistant:"
    if split_token not in chosen or split_token not in rejected:
        return {"prompt": "", "chosen": "", "rejected": "", "source": "hh_rlhf"}
    c_idx = chosen.rfind(split_token)
    r_idx = rejected.rfind(split_token)
    prompt = chosen[:c_idx].strip()
    chosen_resp   = chosen[c_idx + len(split_token):].strip()
    rejected_resp = rejected[r_idx + len(split_token):].strip()
    if prompt.startswith("\n\nHuman:"):
        prompt = prompt[len("\n\nHuman:"):].strip()
    return {"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp,
            "source": "hh_rlhf"}

hh_records = []
if HH_AVAILABLE:
    parsed = hh_train.map(parse_hh, remove_columns=hh_train.column_names)
    hh_records = [r for r in parsed if r["prompt"] and r["chosen"] and r["rejected"]]
    print(f"HH-RLHF parsed: {len(hh_records):,} valid records")
else:
    print("HH-RLHF skipped (download failed)")

# ===================== Notebook cell 10 =====================
# === Step 3 - Load LMSYS train.csv ===
candidates = [
    "train.csv",
    "/kaggle/input/datasets/sahil2604singh/content/train.csv",
    "/kaggle/input/lmsys-chatbot-arena/train.csv",
    "/kaggle/input/lmsys-chatbot-arena-human-preference-predictions/train.csv",
]
DATA_PATH = next((p for p in candidates if os.path.exists(p)), None)
LMSYS_AVAILABLE = DATA_PATH is not None
if LMSYS_AVAILABLE:
    print(f" Found LMSYS at: {DATA_PATH}")
    lmsys_df = pd.read_csv(DATA_PATH)
    print(f"    LMSYS loaded: {len(lmsys_df):,} rows")
else:
    print("  LMSYS train.csv not found - will use HH-RLHF only")
    lmsys_df = None

if not (HH_AVAILABLE or LMSYS_AVAILABLE):
    raise RuntimeError("Neither HH-RLHF nor LMSYS could be loaded. Cannot proceed.")

# ===================== Notebook cell 11 =====================
# === Step 4 - Parse LMSYS into common schema ===
def safe_parse(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError, TypeError):  # TypeError: ast.literal_eval raises this for non-string inputs (e.g. NaN from pd.read_csv)
        try:
            return json.loads(x)
        except Exception:
            return [str(x)]

def flatten_conversation(prompt_list, response_list):
    prompts   = safe_parse(prompt_list)
    responses = safe_parse(response_list)
    if not prompts:                                          # BUG_C: guard empty prompts list
        return "", ""
    if len(prompts) == 1:
        resp = str(responses[0]).strip() if responses else ""  # BUG_C: guard empty responses list
        return str(prompts[0]).strip(), resp
    # BUG6: only last user turn as prompt - no cross-model context contamination
    final_prompt   = str(prompts[-1]).strip()
    final_response = str(responses[-1]).strip() if responses else ""
    return final_prompt, final_response

lmsys_records = []
if LMSYS_AVAILABLE:
    # FIX2: strip whitespace from column names - trailing spaces break itertuples attribute access
    lmsys_df.columns = lmsys_df.columns.str.strip()
    # BUG2: validate column names before itertuples (spaces/special chars break attribute access)
    expected = {"winner_tie", "winner_model_a", "winner_model_b", "prompt", "response_a", "response_b"}
    missing = expected - set(lmsys_df.columns)
    if missing:
        raise ValueError(f"LMSYS CSV is missing expected columns: {missing}")

    n_ties = 0
    for row in lmsys_df.itertuples(index=False):
        if row.winner_tie == 1:
            n_ties += 1
            continue
        prompt_a, resp_a = flatten_conversation(row.prompt, row.response_a)
        prompt_b, resp_b = flatten_conversation(row.prompt, row.response_b)
        if row.winner_model_a == 1:
            chosen, rejected = resp_a, resp_b
        elif row.winner_model_b == 1:
            chosen, rejected = resp_b, resp_a
        else:
            continue   # skip ambiguous / all-zero rows (NaN coercion, ties missed earlier)
        if not (chosen and rejected and prompt_a):
            continue
        lmsys_records.append({"prompt": prompt_a, "chosen": chosen,
                              "rejected": rejected, "source": "lmsys"})
    print(f"LMSYS parsed: {len(lmsys_records):,} valid records (dropped {n_ties:,} ties)")

# ===================== Notebook cell 12 =====================
# === Step 5 - Apply uniform length filter to BOTH ===
# BUG6 fixed: multi-turn LMSYS prompts were being rejected entirely when they
# exceeded MAX_CHARS (because flatten_conversation joins all turns with \n).
# Fix: truncate long fields instead of rejecting the whole record - this preserves
# LMSYS's prompt diversity (the stated goal) instead of silently keeping only
# single-turn conversations.
MAX_CHARS = 1500
MIN_CHARS = 20

def truncate_and_filter(r):
    p = r["prompt"]; c = r["chosen"]; x = r["rejected"]
    if not (p and c and x): return None
    # Truncate long fields rather than discarding the record
    p = p[:MAX_CHARS]; c = c[:MAX_CHARS]; x = x[:MAX_CHARS]
    # Still reject if too short after stripping
    if len(p) < MIN_CHARS or len(c) < MIN_CHARS or len(x) < MIN_CHARS: return None
    return {**r, "prompt": p, "chosen": c, "rejected": x}

# FIX1: walrus operator requires Python 3.8+; use nested list comprehension for 3.7 compat
hh_clean    = [out for r in hh_records    for out in [truncate_and_filter(r)] if out is not None]
lmsys_clean = [out for r in lmsys_records for out in [truncate_and_filter(r)] if out is not None]
print(f"After truncate-and-filter:")
print(f"  HH-RLHF : {len(hh_clean):,}  (dropped {len(hh_records)-len(hh_clean):,})")
print(f"  LMSYS   : {len(lmsys_clean):,}  (dropped {len(lmsys_records)-len(lmsys_clean):,})")

# ===================== Notebook cell 13 =====================
# === Step 6 - Sample-balanced combination ===
random.seed(SEED)
random.shuffle(hh_clean); random.shuffle(lmsys_clean)

PER_SOURCE = 2000   # 2000/source = 4000 total - good accuracy, ~2-3hr runtime   # reduced from 4000 to cut training time in half

if hh_clean and lmsys_clean:
    n_hh    = min(PER_SOURCE, len(hh_clean))
    n_lmsys = min(PER_SOURCE, len(lmsys_clean))
    combined = hh_clean[:n_hh] + lmsys_clean[:n_lmsys]
    print(f"Balanced mix: {n_hh:,} HH + {n_lmsys:,} LMSYS = {len(combined):,} total")
elif hh_clean:
    combined = hh_clean[:PER_SOURCE*2]
    print(f"  LMSYS unavailable - using HH only: {len(combined):,}")
elif lmsys_clean:
    combined = lmsys_clean[:PER_SOURCE*2]
    print(f"  HH unavailable - using LMSYS only: {len(combined):,}")
else:
    raise RuntimeError("Both hh_clean and lmsys_clean are empty after filtering. Cannot proceed.")

random.shuffle(combined)

# === Step 7 - Stratified train/val/test split ===
combined_df = pd.DataFrame(combined)

def stratified_split(df, train_n, val_n, test_n):
    splits_train, splits_val, splits_test = [], [], []
    for src in df["source"].unique():
        sub = df[df["source"] == src].reset_index(drop=True)
        frac = len(sub) / len(df)
        t_n = int(round(train_n * frac))
        v_n = int(round(val_n * frac))
        e_n = int(round(test_n * frac))
        splits_train.append(sub.iloc[:t_n])
        splits_val.append(sub.iloc[t_n:t_n+v_n])
        splits_test.append(sub.iloc[t_n+v_n:t_n+v_n+e_n])
    return (pd.concat(splits_train).sample(frac=1, random_state=SEED).reset_index(drop=True),
            pd.concat(splits_val).sample(frac=1, random_state=SEED).reset_index(drop=True),
            pd.concat(splits_test).sample(frac=1, random_state=SEED).reset_index(drop=True))

TOTAL = len(combined)
train_df, val_df, test_df = stratified_split(
    combined_df,
    int(TOTAL * 0.875),
    int(TOTAL * 0.0625),
    int(TOTAL * 0.0625),
)

# BUG2 fixed: original carved same 100 rows for both val and test -> leakage
# Now carves 200 unique rows in a single pass
if len(val_df) < 20 or len(test_df) < 20:
    print("  BUG5 WARNING: stratified split produced tiny val/test - falling back to tail-slice. Source balance NOT guaranteed.")
    val_df   = train_df.iloc[-200:-100].reset_index(drop=True)
    test_df  = train_df.iloc[-100:].reset_index(drop=True)
    train_df = train_df.iloc[:-200].reset_index(drop=True)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)}  ({train_df['source'].value_counts().to_dict()})")
print(f"  Val  : {len(val_df)}  ({val_df['source'].value_counts().to_dict()})")
print(f"  Test : {len(test_df)}  ({test_df['source'].value_counts().to_dict()})")

# ===================== Notebook cell 14 =====================
# === Step 8 - Visualize the combined dataset (jury slide) ===
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# (a) Source distribution
src_counts = combined_df["source"].value_counts()
axes[0].pie(src_counts.values, labels=src_counts.index,
            colors=["#10b981", "#f59e0b"], autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[0].set_title("Combined Dataset - Source Mix", fontweight="bold")

# (b) Prompt length distribution per source
for src, color in [("hh_rlhf", "#10b981"), ("lmsys", "#f59e0b")]:
    sub = combined_df[combined_df["source"] == src]
    if len(sub) > 0:
        axes[1].hist(sub["prompt"].str.len(), bins=30, alpha=0.6, label=src,
                     color=color, edgecolor="white")
axes[1].set_xlabel("Prompt length (chars)"); axes[1].set_ylabel("Count")
axes[1].set_title("Prompt Length Distribution by Source", fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)

# (c) Response length distribution per source (this is where they differ most)
for src, color in [("hh_rlhf", "#10b981"), ("lmsys", "#f59e0b")]:
    sub = combined_df[combined_df["source"] == src]
    if len(sub) > 0:
        axes[2].hist(sub["chosen"].str.len(), bins=30, alpha=0.6, label=src,
                     color=color, edgecolor="white")
axes[2].set_xlabel("Chosen-response length (chars)"); axes[2].set_ylabel("Count")
axes[2].set_title("Response Length by Source\n(LMSYS responses are longer)", fontweight="bold")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/dataset_composition.png", dpi=120, bbox_inches="tight")
plt.show()

# Sanity check
print("\n--- Sample HH-RLHF triple ---")
hh_ex = train_df[train_df["source"] == "hh_rlhf"].iloc[0] if (train_df["source"] == "hh_rlhf").any() else None
if hh_ex is not None:
    print(f"PROMPT  : {hh_ex['prompt'][:200]}")
    print(f"CHOSEN  : {hh_ex['chosen'][:200]}")
    print(f"REJECTED: {hh_ex['rejected'][:200]}")

print("\n--- Sample LMSYS triple ---")
lm_ex = train_df[train_df["source"] == "lmsys"].iloc[0] if (train_df["source"] == "lmsys").any() else None
if lm_ex is not None:
    print(f"PROMPT  : {lm_ex['prompt'][:200]}")
    print(f"CHOSEN  : {lm_ex['chosen'][:200]}")
    print(f"REJECTED: {lm_ex['rejected'][:200]}")

# ===================== Notebook cell 16 =====================
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType

# DeBERTa-v3-base: bidirectional encoder, purpose-built for scoring (184M params)
RM_BASE_MODEL = "microsoft/deberta-v3-base"
# bnb_config removed - DeBERTa-v3-base (184M) needs NO quantization (only 184M params).
# IMPORTANT: DeBERTa-v3 must load in float32 - float16 causes NaN in disentangled attention.
# BitsAndBytesConfig is only needed for TinyLlama (DPO training, defined in Cell 28 as bnb_dpo).

# ===================== Notebook cell 17 =====================
# # REWARD MODEL: LOAD IF EXISTS, ELSE TRAIN                       
# rm_already_trained = (
    os.path.exists(REWARD_MODEL_PATH)
    and os.path.exists(f"{REWARD_MODEL_PATH}/adapter_config.json")
)

rm_tokenizer = AutoTokenizer.from_pretrained(
    REWARD_MODEL_PATH if rm_already_trained else RM_BASE_MODEL
)
if rm_tokenizer.pad_token is None:
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
rm_tokenizer.padding_side = "right"

if rm_already_trained:
    print(" Found saved reward model - loading from disk (skipping training)")
    print(f"   Path: {REWARD_MODEL_PATH}")
    # DeBERTa: no quantization needed - 184M params fit easily in fp16
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        RM_BASE_MODEL, num_labels=1,
        torch_dtype=torch.float32,       # FIX: DeBERTa-v3 disentangled attention produces NaN in fp16/bf16 - must be float32
        device_map="auto",
    )
    rm_base.config.pad_token_id = rm_tokenizer.pad_token_id
    rm_base.config.return_dict = True   # suppress use_return_dict deprecation warning
    # FIX8: from_pretrained reads modules_to_save from adapter_config.json and
    # restores the trained classifier + pooler heads automatically (not just LoRA weights)
    reward_model = PeftModel.from_pretrained(rm_base, REWARD_MODEL_PATH).eval()
    rm_history = None
else:
    print(" No saved reward model found - training from scratch")
    print(f"   Will save to: {REWARD_MODEL_PATH}")

# ===================== Notebook cell 18 =====================
if not rm_already_trained:
    # DeBERTa: no 4-bit, no kbit prep needed
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        RM_BASE_MODEL, num_labels=1,
        torch_dtype=torch.float32,       # FIX: DeBERTa-v3 disentangled attention produces NaN in fp16/bf16 - must be float32
        device_map="auto",
    )
    rm_base.config.pad_token_id = rm_tokenizer.pad_token_id
    rm_base.config.return_dict = True  # FIX: suppress "use_return_dict deprecated" warning

    # DeBERTa attention module names differ from LLaMA
    rm_lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["query_proj", "key_proj", "value_proj"],
        bias="none",
        # FIX7: PEFT 0.19.1 + TaskType.SEQ_CLS auto-unfreezes and saves "classifier"
        # (confirmed: 1,770,241 trainable = 1,769,472 LoRA + 769 classifier params).
        # BUT pooler.dense (590,592 params) is FROZEN at random init - NEVER trained.
        # At cell-34 reload, a fresh base model gives pooler weights P2  training P1,
        # so LoRA adapters trained with P1 face wrong projections -> unreliable scores.
        # modules_to_save=["pooler"] makes it trainable, saves it in adapter_model.safetensors,
        # and restores it exactly via from_pretrained() - no double-wrapping of classifier.
        modules_to_save=["pooler"],
        # autocast_adapter_dtype removed - not supported in this PEFT version
        # fp16=False in RewardConfig already prevents the GradScaler crash
    )
    try:
        reward_model = get_peft_model(rm_base, rm_lora)
    except ImportError as _e:
        if "torchao" in str(_e):
            raise RuntimeError(
                "torchao version conflict. Fix: restart the kernel after\n"
                "the install cell runs, then re-run all cells.\n"
                f"Details: {_e}"
            ) from None
        raise
    reward_model.enable_input_require_grads()  # Required: gradient_checkpointing=True needs this for non-quantized PEFT models
    reward_model.print_trainable_parameters()
    print()

# ===================== Notebook cell 19 =====================
# BUG4 fixed: clean_text must run regardless of whether RM is pre-trained
# (test_df is used in eval sections 4 & 9 even when loading from disk)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")

for col in ["prompt", "chosen", "rejected"]:
    train_df[col] = train_df[col].apply(clean_text)
    val_df[col]   = val_df[col].apply(clean_text)
    test_df[col]  = test_df[col].apply(clean_text)

print(" Surrogate characters cleaned")

if not rm_already_trained:
    from datasets import Dataset

    # New trl RewardTrainer tokenizes internally - pass raw text only
    rm_train_ds = Dataset.from_pandas(train_df[["prompt", "chosen", "rejected"]].reset_index(drop=True))
    rm_val_ds   = Dataset.from_pandas(val_df[["prompt",  "chosen", "rejected"]].reset_index(drop=True))

    print(f"Dataset columns : {rm_train_ds.column_names}")
    print(f"Sizes           : {len(rm_train_ds)} train / {len(rm_val_ds)} val")

# ===================== Notebook cell 20 =====================
if not rm_already_trained:
    from trl import RewardConfig, RewardTrainer

    rm_config = RewardConfig(
        output_dir=f"{BASE_DIR}/rm_checkpoints",
        num_train_epochs=2,        # reduced from 3
        per_device_train_batch_size=8,   # increased from 4 (faster throughput)
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,   # reduced from 4
        gradient_checkpointing=True,
        learning_rate=2e-5,    # DeBERTa sweet spot: 1e-5 to 3e-5
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,   # triggers ~2x per epoch at PER_SOURCE=2000; increase if PER_SOURCE < 500
        save_strategy="no",
        fp16=False,            # FP32 - DeBERTa 184M fits on T4; avoids GradScaler unscale FP16 error
        max_length=512,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",   # paged_adamw_8bit is for quantized models only
        dataloader_num_workers=2,      # parallel data loading
        dataloader_pin_memory=True,    # faster GPU transfer
    )

    rm_trainer = RewardTrainer(
        model=reward_model,
        args=rm_config,
        processing_class=rm_tokenizer,
        train_dataset=rm_train_ds,
        eval_dataset=rm_val_ds,
    )
    rm_trainer.train()
    rm_history = rm_trainer.state.log_history

    reward_model.save_pretrained(REWARD_MODEL_PATH)
    rm_tokenizer.save_pretrained(REWARD_MODEL_PATH)
    with open(f"{REWARD_MODEL_PATH}/training_log.json", "w") as f:
        json.dump(rm_history, f)
    print(f" Reward model saved to {REWARD_MODEL_PATH}")
    print()
    print("=" * 60)
    print("  IMPORTANT - PREVENT DATA LOSS ON KAGGLE:")
    print("   Kaggle clears /kaggle/working when the session ends.")
    print("   To keep your trained model:")
    print("   1. Click  Save  ->  Quick Save  (top right)")
    print("   2. Next session: add this notebook's output as a dataset input")
    print("   3. The notebook will auto-detect and skip retraining")
    print("=" * 60)
    reward_model.eval()

# ===================== Notebook cell 22 =====================
@torch.no_grad()
def reward_score(prompt: str, response: str, max_len: int = 512) -> float:
    # FIX: get the actual device the model is on (device_map="auto" may
    # place it on cuda:1, not cuda:0 / DEVICE)
    rm_device = next(reward_model.parameters()).device

    inputs = rm_tokenizer(
        prompt, response,
        truncation=True, max_length=max_len,
        return_tensors="pt"
    ).to(rm_device)   # was .to(DEVICE)

    score = reward_model(**inputs).logits[0, 0].float().item()
    if not np.isfinite(score):
        # FIX: print directly - warnings.warn can be suppressed by filterwarnings
        print("  reward_score: NaN/inf score detected - "
              "DeBERTa-v3 must be in float32, not float16. "
              "Check model loading cell.", flush=True)
        return 0.0
    return score

@torch.no_grad()
def reward_batch(prompts, responses, max_len: int = 512, batch_size: int = 8):
    rm_device = next(reward_model.parameters()).device  # same fix for batch
    scores = []
    for i in range(0, len(prompts), batch_size):
        pb = prompts[i:i+batch_size]
        rb = responses[i:i+batch_size]
        inputs = rm_tokenizer(
            pb, rb,
            truncation=True, max_length=max_len,
            padding=True, return_tensors="pt"
        ).to(rm_device)   # was .to(DEVICE)
        raw = reward_model(**inputs).logits[:, 0].float()
        nan_count = (~torch.isfinite(raw)).sum().item()
        if nan_count > 0:
            # FIX: print directly - not suppressible by filterwarnings
            print(f"  reward_batch: {nan_count}/{len(raw)} NaN/inf scores - "
                  "DeBERTa-v3 must be in float32. Check model loading cell.", flush=True)
        scores.extend(raw.tolist())
    return scores

# ===================== Notebook cell 23 =====================
# === Per-source validation ===
def evaluate_subset(sub_df, label):
    if len(sub_df) == 0:
        return None
    p = sub_df["prompt"].tolist()
    c = sub_df["chosen"].tolist()
    r = sub_df["rejected"].tolist()
    sc = np.array(reward_batch(p, c))
    sr = np.array(reward_batch(p, r))
    # filter NaN scores - DeBERTa freshly-init classifier can produce NaN on some inputs
    valid = np.isfinite(sc) & np.isfinite(sr)
    if valid.sum() == 0:
        print(f"  {label}: all scores are NaN - skipping")
        return None
    if valid.sum() < len(sc):
        print(f"  {label}: filtered {(~valid).sum()} NaN pairs ({valid.sum()} kept)")
    sc, sr = sc[valid], sr[valid]
    return {
        "label": label,
        "n": len(sub_df),
        "mean_chosen":   float(sc.mean()),
        "mean_rejected": float(sr.mean()),
        "gap":           float((sc - sr).mean()),
        "accuracy":      float((sc > sr).mean()),
        "chosen_scores":   sc,
        "rejected_scores": sr,
    }

# Evaluate overall + per-source
overall = evaluate_subset(test_df, "Combined (overall)")
hh_eval    = evaluate_subset(test_df[test_df["source"] == "hh_rlhf"], "HH-RLHF only")
lmsys_eval = evaluate_subset(test_df[test_df["source"] == "lmsys"],   "LMSYS only")

print("=" * 70)
print(" Reward Model Validation - Held-out Test Set")
print("=" * 70)
for ev in [overall, hh_eval, lmsys_eval]:
    if ev is None:
        continue
    print(f"\n  {ev['label']} (n={ev['n']}):")
    print(f"    Mean reward (chosen)   : {ev['mean_chosen']:+.4f}")
    print(f"    Mean reward (rejected) : {ev['mean_rejected']:+.4f}")
    print(f"    Mean gap               : {ev['gap']:+.4f}")
    print(f"    Pairwise accuracy      : {ev['accuracy']:.3f}  (random=0.500)")
print("\n" + "=" * 70)

# Hard assertion
if overall and overall["mean_chosen"] > overall["mean_rejected"]:  # BUG4: guard None return
    print(" PASS: chosen > rejected on average. RM learned across both datasets.")
    if hh_eval and lmsys_eval:
        balanced = (abs(hh_eval["accuracy"] - lmsys_eval["accuracy"]) < 0.10)
        if balanced:
            print(" BALANCED: per-source accuracies within 10% of each other -> genuine generalization.")
        else:
            print(f"  IMBALANCED: HH acc={hh_eval['accuracy']:.3f} vs LMSYS acc={lmsys_eval['accuracy']:.3f}")
            print("    The RM is biased toward one source. Consider rebalancing the training mix.")
else:
    print("  WARNING: chosen NOT > rejected. RM did not learn properly.")

# ===================== Notebook cell 24 =====================
# === Reward distribution histogram, split by source (jury slide) ===
from matplotlib import gridspec

fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

evals = [("Combined", overall), ("HH-RLHF", hh_eval), ("LMSYS", lmsys_eval)]
for i, (name, ev) in enumerate(evals):
    if ev is None:
        continue
    # Top row: chosen vs rejected histograms
    ax = fig.add_subplot(gs[0, i])
    cs = ev["chosen_scores"][np.isfinite(ev["chosen_scores"])]   # drop NaN
    rs = ev["rejected_scores"][np.isfinite(ev["rejected_scores"])]
    if len(cs) == 0 or len(rs) == 0:
        ax.text(0.5, 0.5, "No valid scores", ha="center", va="center", transform=ax.transAxes)
        continue
    ax.hist(cs, bins=25, alpha=0.7, color="#10b981",
            label=f"chosen mu={ev['mean_chosen']:+.2f}", edgecolor="white")
    ax.hist(rs, bins=25, alpha=0.7, color="#ef4444",
            label=f"rejected mu={ev['mean_rejected']:+.2f}", edgecolor="white")
    ax.set_title(f"{name}  (n={ev['n']}, acc={ev['accuracy']*100:.1f}%)",
                 fontweight="bold")
    ax.set_xlabel("RM score"); ax.set_ylabel("Count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Bottom row: gap histograms
    ax = fig.add_subplot(gs[1, i])
    gaps = ev["chosen_scores"] - ev["rejected_scores"]
    ax.hist(gaps, bins=25, color="#6366f1", alpha=0.85, edgecolor="white")  # removed unused `colors` list
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("reward gap (chosen - rejected)"); ax.set_ylabel("Count")
    ax.set_title(f"{name} - gap distribution", fontweight="bold")
    ax.grid(alpha=0.3)

plt.suptitle("Reward Model Validation - Combined + Per-Source Breakdown",
             fontsize=14, fontweight="bold", y=1.005)
plt.savefig(f"{PLOTS_DIR}/rm_validation_per_source.png", dpi=120, bbox_inches="tight")
plt.show()

# ===================== Notebook cell 25 =====================
# === RM training loss curve ===
log_to_plot = None
if rm_already_trained and os.path.exists(f"{REWARD_MODEL_PATH}/training_log.json"):
    with open(f"{REWARD_MODEL_PATH}/training_log.json") as f:
        log_to_plot = json.load(f)
elif not rm_already_trained:
    log_to_plot = rm_history if "rm_history" in globals() else None  # dir()->globals(): reliable in Jupyter

if log_to_plot:
    train_logs = [(l["step"], l["loss"]) for l in log_to_plot
                  if "loss" in l and "eval_loss" not in l]
    eval_logs  = [(l["step"], l["eval_loss"]) for l in log_to_plot if "eval_loss" in l]

    fig, ax = plt.subplots(figsize=(10, 4))
    if train_logs:
        s, v = zip(*train_logs); ax.plot(s, v, label="train loss", color="#3b82f6")
    if eval_logs:
        s, v = zip(*eval_logs); ax.plot(s, v, label="eval loss", color="#ef4444",
                                        marker="o", markersize=5)
    ax.set_xlabel("Step"); ax.set_ylabel("Bradley-Terry Loss")
    ax.set_title("Reward Model Training (combined HH + LMSYS)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/rm_loss_curve.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("(No training log available)")

# ===================== Notebook cell 27 =====================
gc.collect(); torch.cuda.empty_cache()
print(f"VRAM allocated before DPO setup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ===================== Notebook cell 28 =====================
POLICY_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_BASE_MODEL)
if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

llm_already_trained = (
    os.path.exists(FINETUNED_LLM_PATH)
    and os.path.exists(f"{FINETUNED_LLM_PATH}/adapter_config.json")
)

if llm_already_trained:
    print(" Found saved fine-tuned LLM - will load (skipping DPO training)")
    print(f"   Path: {FINETUNED_LLM_PATH}")
else:
    print(" No saved fine-tuned LLM - will train with DPO")
    print(f"   Will save to: {FINETUNED_LLM_PATH}")

# ===================== Notebook cell 29 =====================
if not llm_already_trained:
    bnb_dpo = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        # FIX4: float16 not bfloat16 - T4 has no bf16 arithmetic; must match fp16 DPOConfig
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    policy_model = AutoModelForCausalLM.from_pretrained(
        POLICY_BASE_MODEL, quantization_config=bnb_dpo,
        torch_dtype=torch.float16,       # FIX: forces non-quantized layers (embedding, lm_head) to fp16, not bf16;
        # without this, TinyLlama's default bf16 layout causes GradScaler to crash
        # with: NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16
        device_map="auto",
    )
    policy_model.config.pad_token_id = policy_tokenizer.pad_token_id
    # FIX10: suppress "use_return_dict is deprecated, use return_dict" from transformers 5.x
    # Shows twice (policy + implicit ref model). Warning only, not a crash.
    policy_model.config.return_dict = True
    policy_model = prepare_model_for_kbit_training(policy_model)
    print(f"Policy loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ===================== Notebook cell 30 =====================
if not llm_already_trained:
    def format_for_dpo(ex):
        prompt_text = policy_tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        return {
            "prompt":   prompt_text,
            "chosen":   ex["chosen"],
            "rejected": ex["rejected"],
        }

    from datasets import Dataset
    # Sample 3000 from train (still source-mixed because train_df is already shuffled)
    dpo_train_df = train_df.sample(n=min(1500, len(train_df)), random_state=SEED).reset_index(drop=True)  # reduced from 3000
    dpo_val_df   = val_df.sample(n=min(200, len(val_df)),  random_state=SEED).reset_index(drop=True)

    dpo_train_ds = Dataset.from_pandas(
        dpo_train_df[["prompt", "chosen", "rejected"]]
    ).map(format_for_dpo, remove_columns=["prompt", "chosen", "rejected"])
    dpo_val_ds = Dataset.from_pandas(
        dpo_val_df[["prompt", "chosen", "rejected"]]
    ).map(format_for_dpo, remove_columns=["prompt", "chosen", "rejected"])

    print(f"DPO train: {len(dpo_train_ds)} | val: {len(dpo_val_ds)}")
    print(f"  Source mix in DPO train: {dpo_train_df['source'].value_counts().to_dict()}")

# ===================== Notebook cell 31 =====================
if not llm_already_trained:
    from trl import DPOConfig, DPOTrainer

    dpo_lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=f"{BASE_DIR}/dpo_checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=19,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=150,
        save_strategy="no",
        bf16=False,
        fp16=False,        # FIX: disable GradScaler entirely - 4-bit quant handles memory,
                           # fp32 accumulation avoids all fp16/bf16 GradScaler conflicts on T4
        beta=0.1,
        max_length=512,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    dpo_trainer = DPOTrainer(
        model=policy_model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dpo_train_ds,
        eval_dataset=dpo_val_ds,
        processing_class=policy_tokenizer,
        peft_config=dpo_lora,
    )

    dpo_trainer.train()
    dpo_history = dpo_trainer.state.log_history
    dpo_trainer.save_model(FINETUNED_LLM_PATH)
    policy_tokenizer.save_pretrained(FINETUNED_LLM_PATH)
    with open(f"{FINETUNED_LLM_PATH}/training_log.json", "w") as f:
        json.dump(dpo_history, f)
    print(f" Fine-tuned LLM saved to {FINETUNED_LLM_PATH}")
    del policy_model, dpo_trainer
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

# ===================== Notebook cell 32 =====================
dpo_log = None
if llm_already_trained and os.path.exists(f"{FINETUNED_LLM_PATH}/training_log.json"):
    with open(f"{FINETUNED_LLM_PATH}/training_log.json") as f:
        dpo_log = json.load(f)
elif not llm_already_trained:
    dpo_log = dpo_history if "dpo_history" in globals() else None
else:
    # FIX3: warn when no training log is available (avoids silent skip)
    print("  No DPO training log found on disk and training was not run this session - loss curves will be skipped.")

if dpo_log:
    train_loss = [(l["step"], l["loss"]) for l in dpo_log
                  if "loss" in l and "eval_loss" not in l]
    rc = [(l["step"], l["rewards/chosen"])  for l in dpo_log if "rewards/chosen" in l]
    rr = [(l["step"], l["rewards/rejected"]) for l in dpo_log if "rewards/rejected" in l]
    accs = [(l["step"], l["rewards/accuracies"]) for l in dpo_log if "rewards/accuracies" in l]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    if train_loss:
        s, v = zip(*train_loss); axes[0].plot(s, v, color="#3b82f6")
    axes[0].set_title("DPO Training Loss"); axes[0].set_xlabel("Step")
    axes[0].grid(alpha=0.3)

    if rc:
        s, v = zip(*rc); axes[1].plot(s, v, label="chosen", color="#10b981")
    if rr:
        s, v = zip(*rr); axes[1].plot(s, v, label="rejected", color="#ef4444")
    axes[1].set_title("DPO Implicit Rewards"); axes[1].set_xlabel("Step")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    if accs:
        s, v = zip(*accs); axes[2].plot(s, v, color="#8b5cf6")
        axes[2].axhline(0.5, color="gray", linestyle="--", label="random")
        axes[2].set_ylim(0, 1); axes[2].legend()
    axes[2].set_title("DPO Pairwise Accuracy"); axes[2].set_xlabel("Step")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/dpo_curves.png", dpi=120, bbox_inches="tight")
    plt.show()

# ===================== Notebook cell 34 =====================
# BUG3 fixed: exec() runs in its own scope and does not delete notebook globals
for v in ["policy_model", "dpo_trainer", "rm_base", "reward_model"]:
    if v in globals():
        del globals()[v]
gc.collect(); torch.cuda.empty_cache()
print(f"VRAM cleared. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ===================== Notebook cell 35 =====================
# BUG3: guard against running Section 6 without earlier cells
try:
    rm_tokenizer, RM_BASE_MODEL, REWARD_MODEL_PATH
    policy_tokenizer, POLICY_BASE_MODEL, FINETUNED_LLM_PATH
except NameError as e:
    raise RuntimeError(f"Missing variable: {e}. Run all cells from Section 1 first.") from None

# DeBERTa eval reload - no quantization needed
rm_base_for_eval = AutoModelForSequenceClassification.from_pretrained(
    RM_BASE_MODEL, num_labels=1,
    torch_dtype=torch.float32,       # FIX: DeBERTa-v3 disentangled attention produces NaN in fp16/bf16 - must be float32
    device_map="auto",
)
rm_base_for_eval.config.pad_token_id = rm_tokenizer.pad_token_id
rm_base_for_eval.config.return_dict = True  # FIX: suppress "use_return_dict deprecated" warning
# FIX9: from_pretrained restores LoRA adapters AND the trained classifier + pooler
# (because modules_to_save=["classifier","pooler"] is in adapter_config.json)
reward_model = PeftModel.from_pretrained(rm_base_for_eval, REWARD_MODEL_PATH).eval()
print(" Reward model reloaded.")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ===================== Notebook cell 36 =====================
# FIX6: float16 not bfloat16 - T4 does not support bf16 arithmetic; consistent with fp16 training
bnb_inf = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    POLICY_BASE_MODEL, quantization_config=bnb_inf,
    device_map="auto",  # FIX: dtype omitted with quantization_config - bnb_4bit_compute_dtype handles precision
).eval()
base_model.config.pad_token_id = policy_tokenizer.pad_token_id
print(f"Base model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ===================== Notebook cell 37 =====================
if not os.path.exists(FINETUNED_LLM_PATH):
    raise FileNotFoundError(
        f"Fine-tuned model not found at {FINETUNED_LLM_PATH}.\n"
        "Run Section 5 (DPO training) first, or check BASE_DIR is correct."
    )

ft_base = AutoModelForCausalLM.from_pretrained(
    POLICY_BASE_MODEL, quantization_config=bnb_inf,
    device_map="auto",  # FIX: dtype omitted with quantization_config - bnb_4bit_compute_dtype handles precision
)
ft_base.config.pad_token_id = policy_tokenizer.pad_token_id
finetuned_model = PeftModel.from_pretrained(ft_base, FINETUNED_LLM_PATH).eval()
print(f"Fine-tuned model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ===================== Notebook cell 39 =====================
@torch.no_grad()
def generate_response(model, prompt: str, max_new_tokens: int = 200) -> str:
    chat_text = policy_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    model_device = next(model.parameters()).device  # FIX: get real device (device_map="auto" may use cuda:1)
    inputs = policy_tokenizer(chat_text, return_tensors="pt", truncation=True,
                              max_length=512).to(model_device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        max_length=None,  # FIX: override model generation_config to suppress warning
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=policy_tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return policy_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compare_responses(user_prompt: str, verbose: bool = True) -> dict:
    base_response      = generate_response(base_model,      user_prompt)
    finetuned_response = generate_response(finetuned_model, user_prompt)
    base_score      = reward_score(user_prompt, base_response)
    finetuned_score = reward_score(user_prompt, finetuned_response)

    if verbose:
        print("=" * 78)
        print(f" PROMPT: {user_prompt}")
        print("=" * 78)
        print(f"\n BEFORE TRAINING (Base TinyLlama):")
        print(f"   Reward Score: {base_score:+.4f}")
        print(f"   Response: {base_response}")
        print(f"\n AFTER TRAINING (DPO-Aligned TinyLlama):")
        print(f"   Reward Score: {finetuned_score:+.4f}")
        print(f"   Response: {finetuned_response}")
        delta = finetuned_score - base_score
        if finetuned_score > base_score:
            denom = abs(base_score) if abs(base_score) > 1e-6 else 1.0
            pct = (delta / denom) * 100
            print(f"\n WINNER: Fine-tuned model (Delta = {delta:+.4f}, ~{pct:+.1f}%)")
        else:
            print(f"\n  Base model scored higher this turn (Delta = {delta:+.4f})")
            print("    Note: due to sampling, individual prompts may favor either model;")
            print("    the AVERAGE over many prompts is what shows DPO's effect.")
        print("=" * 78)

    return {
        "prompt": user_prompt,
        "base_response": base_response,
        "finetuned_response": finetuned_response,
        "base_score": base_score,
        "finetuned_score": finetuned_score,
        "delta": finetuned_score - base_score,
    }

print(" compare_responses() ready.")

# ===================== Notebook cell 41 =====================
DEMO_PROMPTS = [
    "What is neural network?",
    "Explain neural networks simply.",
    "How do I stay motivated while studying?",
    "I'm feeling anxious about an upcoming exam. Any advice?",
    "What's a healthy way to handle disagreements with friends?",
]

print(" Running 5-prompt demo reel...\n")
demo_results = []
for i, p in enumerate(DEMO_PROMPTS, 1):
    print(f"\n{'v'*3} Demo {i}/{len(DEMO_PROMPTS)} {'v'*3}")
    demo_results.append(compare_responses(p, verbose=True))

# ===================== Notebook cell 42 =====================
demo_df = pd.DataFrame(demo_results)
n_wins = int((demo_df["delta"] > 0).sum())
mean_base  = demo_df["base_score"].mean()
mean_ft    = demo_df["finetuned_score"].mean()
mean_delta = demo_df["delta"].mean()

print("\n" + "=" * 78)
print(f" AGGREGATE OVER {len(demo_df)} DEMO PROMPTS")
print("=" * 78)
print(f"  Mean reward - base       : {mean_base:+.4f}")
print(f"  Mean reward - fine-tuned : {mean_ft:+.4f}")
print(f"  Mean improvement (Delta)     : {mean_delta:+.4f}")
print(f"  DPO wins                 : {n_wins}/{len(demo_df)}")
print("=" * 78)

fig, ax = plt.subplots(figsize=(11, 4.5))
xs = np.arange(len(demo_df))
ax.bar(xs - 0.2, demo_df["base_score"],     width=0.4, color="#94a3b8", label="Base")
ax.bar(xs + 0.2, demo_df["finetuned_score"], width=0.4, color="#10b981", label="DPO")
ax.set_xticks(xs)
ax.set_xticklabels([p[:25] + "..." if len(p) > 25 else p for p in demo_df["prompt"]],
                   rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Reward Model score")
ax.set_title(f"Demo Prompts - Base vs. DPO  (mean Delta = {mean_delta:+.3f}, wins {n_wins}/{len(demo_df)})")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/demo_comparison.png", dpi=120, bbox_inches="tight")
plt.show()

demo_df.to_csv(f"{BASE_DIR}/demo_results.csv", index=False)
print(f"\n Saved demo results to {BASE_DIR}/demo_results.csv")

# ===================== Notebook cell 44 =====================
# Sample 25 prompts from each source if available
EVAL_PER_SOURCE = 25
# BUG8 fixed: use random sample instead of head() for proper evaluation
hh_eval_subset = test_df[test_df["source"] == "hh_rlhf"].sample(n=min(EVAL_PER_SOURCE, len(test_df[test_df["source"] == "hh_rlhf"])), random_state=SEED)
lm_eval_subset = test_df[test_df["source"] == "lmsys"].sample(n=min(EVAL_PER_SOURCE, len(test_df[test_df["source"] == "lmsys"])), random_state=SEED)
eval_subset = pd.concat([hh_eval_subset, lm_eval_subset]).reset_index(drop=True)
print(f"Eval set: {len(hh_eval_subset)} HH + {len(lm_eval_subset)} LMSYS = {len(eval_subset)} prompts")

from tqdm.auto import tqdm
batch_results = []
# itertuples() is faster than iterrows() even for 50 rows
for row in tqdm(eval_subset.itertuples(index=False), total=len(eval_subset), desc="Generating both models"):
    res = compare_responses(row.prompt, verbose=False)
    res["source"] = row.source
    batch_results.append(res)

batch_df = pd.DataFrame(batch_results) if batch_results else pd.DataFrame(
    columns=["prompt","base_response","finetuned_response",
             "base_score","finetuned_score","delta","source"])

# Overall + per-source win rates
def report_winrate(df, label):
    if len(df) == 0:
        return
    wr = float((df["delta"] > 0).mean())
    print(f"\n  {label}  (n={len(df)})")
    print(f"    Mean reward - base : {df['base_score'].mean():+.4f}")
    print(f"    Mean reward - DPO  : {df['finetuned_score'].mean():+.4f}")
    print(f"    Mean Delta             : {df['delta'].mean():+.4f}")
    print(f"    Win rate           : {wr*100:.1f}%  (random=50%)")

print("\n" + "=" * 70)
print(f" HELD-OUT EVALUATION - PER-SOURCE BREAKDOWN")
print("=" * 70)
report_winrate(batch_df, "Combined (overall)")
report_winrate(batch_df[batch_df["source"] == "hh_rlhf"], "HH-RLHF only")
report_winrate(batch_df[batch_df["source"] == "lmsys"],   "LMSYS only")
print("\n" + "=" * 70)

batch_df.to_csv(f"{BASE_DIR}/batch_eval.csv", index=False)

# ===================== Notebook cell 45 =====================
# === Headline figure: per-source win rates and reward distributions ===
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# (a) Per-source win rates as bars
sources = []
win_rates = []
for src, label in [("hh_rlhf", "HH-RLHF"), ("lmsys", "LMSYS")]:
    sub = batch_df[batch_df["source"] == src]
    if len(sub) == 0: continue
    sources.append(label)
    win_rates.append(float((sub["delta"] > 0).mean()) * 100)
overall_wr = float((batch_df["delta"] > 0).mean()) * 100
sources.append("Combined"); win_rates.append(overall_wr)

colors = ["#10b981", "#f59e0b", "#6366f1"][:len(sources)]
bars = axes[0].bar(sources, win_rates, color=colors, edgecolor="white", linewidth=2)
for bar, wr in zip(bars, win_rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{wr:.1f}%", ha="center", fontweight="bold", fontsize=11)
axes[0].axhline(50, color="gray", linestyle="--", label="random")
axes[0].set_ylim(0, 100); axes[0].set_ylabel("DPO win rate (%)")
axes[0].set_title("DPO Win Rate per Source", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")

# (b) Reward distributions (overall)
axes[1].hist(batch_df["base_score"],      bins=20, alpha=0.7, color="#94a3b8",
             label=f"Base (mu={batch_df['base_score'].mean():+.2f})", edgecolor="white")
axes[1].hist(batch_df["finetuned_score"], bins=20, alpha=0.7, color="#10b981",
             label=f"DPO (mu={batch_df['finetuned_score'].mean():+.2f})", edgecolor="white")
axes[1].axvline(batch_df["base_score"].mean(),      color="#475569", linestyle="--")
axes[1].axvline(batch_df["finetuned_score"].mean(), color="#047857", linestyle="--")
axes[1].set_xlabel("RM score"); axes[1].set_ylabel("Count")
axes[1].set_title(f"Reward Distributions (n={len(batch_df)})", fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)

# (c) Per-prompt deltas, color-coded by source
deltas = batch_df["delta"].values
order = np.argsort(deltas)[::-1]
sorted_deltas = deltas[order]
sorted_sources = batch_df["source"].values[order]
src_color = {"hh_rlhf": "#10b981", "lmsys": "#f59e0b"}
bar_colors = [src_color[s] if d > 0 else "#ef4444" for s, d in zip(sorted_sources, sorted_deltas)]
axes[2].bar(range(len(sorted_deltas)), sorted_deltas, color=bar_colors)
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_xlabel("Prompt (sorted by Delta)"); axes[2].set_ylabel("Deltareward (DPO - Base)")
axes[2].set_title(f"Per-Prompt Improvement\nGreen=HH win, Orange=LMSYS win, Red=loss",
                  fontweight="bold")
axes[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/before_after_headline.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"\n Headline figure saved to {PLOTS_DIR}/before_after_headline.png")

# ===================== Notebook cell 47 =====================
print(" Interactive comparison mode")
print("    Type any prompt and see base vs. DPO-aligned TinyLlama responses, side-by-side.")
print("    Type 'quit' (or 'exit', 'q') to stop.\n")

session_log = []

if os.environ.get("CI"):
    # Only skip in actual CI pipelines (GitHub Actions, etc.)
    print("(CI environment detected - skipping live demo.)")
else:
    while True:
        try:
            user_input = input("\n Enter your prompt (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOFError fires when stdin is truly non-interactive (automated run-all)
            print("\n(Non-interactive run detected - skipping live demo. Run this cell alone to interact.)")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print(f"\n Exited. {len(session_log)} prompts compared this session.")
            break
        result = compare_responses(user_input, verbose=True)
        session_log.append(result)

if session_log:
    session_df = pd.DataFrame(session_log)
    session_path = f"{BASE_DIR}/interactive_session.csv"
    session_df.to_csv(session_path, index=False)
    print(f"\n Session saved to {session_path}")
    print(f"   Mean Delta this session: {session_df['delta'].mean():+.4f}")
    print(f"   DPO wins: {(session_df['delta'] > 0).sum()}/{len(session_df)}")
