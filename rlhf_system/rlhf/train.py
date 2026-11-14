"""Stages 2 and 3: reward-model training (Bradley-Terry) and DPO alignment.

Both stages save a PEFT adapter plus training_log.json and are skipped when
the adapter already exists (delete the folder to retrain).
"""
import gc
import json

import pandas as pd
import torch

from .config import RM_BASE_MODEL, SEED, Config
from .models import has_adapter, load_policy_base, pick_device


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- Stage 2
def train_reward_model(cfg: Config, train: pd.DataFrame, val: pd.DataFrame):
    out = cfg.reward_model_dir
    if has_adapter(out):
        print(f"[rm] Found trained reward model at {out} - skipping training")
        return out

    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    tok = AutoTokenizer.from_pretrained(RM_BASE_MODEL)
    tok.padding_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(
        RM_BASE_MODEL, num_labels=1, torch_dtype=torch.float32,  # FP16 -> NaN in DeBERTa-v3
        device_map="auto" if pick_device() == "cuda" else None)
    base.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg.rm_lora_r, lora_alpha=cfg.rm_lora_alpha,
                      lora_dropout=0.05, target_modules=["query_proj", "key_proj", "value_proj"],
                      bias="none", modules_to_save=["pooler"])  # pooler must be trained AND saved
    model = get_peft_model(base, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    cols = ["prompt", "chosen", "rejected"]
    args = RewardConfig(
        output_dir=str(cfg.out / "rm_checkpoints"),
        num_train_epochs=cfg.rm_epochs,
        per_device_train_batch_size=cfg.rm_batch_size, per_device_eval_batch_size=cfg.rm_batch_size,
        gradient_accumulation_steps=cfg.rm_grad_accum, gradient_checkpointing=True,
        learning_rate=cfg.rm_lr, lr_scheduler_type="cosine", warmup_steps=cfg.rm_warmup_steps,
        logging_steps=25, eval_strategy="steps", eval_steps=100, save_strategy="no",
        fp16=False, max_length=cfg.max_len, remove_unused_columns=False,
        report_to="none", optim="adamw_torch", seed=SEED)
    trainer = RewardTrainer(model=model, args=args, processing_class=tok,
                            train_dataset=Dataset.from_pandas(train[cols].reset_index(drop=True)),
                            eval_dataset=Dataset.from_pandas(val[cols].reset_index(drop=True)))
    trainer.train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "training_log.json").write_text(json.dumps(trainer.state.log_history))
    print(f"[rm] saved to {out}")
    del model, trainer, base
    _free()
    return out


# ---------------------------------------------------------------- Stage 3
def _sample_dpo(cfg: Config, train: pd.DataFrame) -> pd.DataFrame:
    if cfg.dpo_hh_samples or cfg.dpo_lmsys_samples:
        hh = train[train.source == "hh_rlhf"]
        lm = train[train.source == "lmsys"]
        parts = [hh.sample(n=min(cfg.dpo_hh_samples, len(hh)), random_state=SEED),
                 lm.sample(n=min(cfg.dpo_lmsys_samples, len(lm)), random_state=SEED)]
        return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return train.sample(n=min(cfg.dpo_total, len(train)), random_state=SEED).reset_index(drop=True)


def train_dpo(cfg: Config, train: pd.DataFrame, val: pd.DataFrame):
    out = cfg.policy_dir
    if has_adapter(out):
        print(f"[dpo] Found DPO adapter at {out} - skipping training")
        return out
    if pick_device() != "cuda":
        print("[dpo] WARNING: no CUDA GPU - training without 4-bit quantisation will be very slow")

    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    policy, tok = load_policy_base(for_training=True)

    def fmt(ex):
        prompt = tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                         tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    cols = ["prompt", "chosen", "rejected"]
    tr = _sample_dpo(cfg, train)
    va = val.sample(n=min(cfg.dpo_val, len(val)), random_state=SEED)
    print(f"[dpo] train {len(tr)} {tr.source.value_counts().to_dict()} | val {len(va)}")
    tr_ds = Dataset.from_pandas(tr[cols].reset_index(drop=True)).map(fmt, remove_columns=cols)
    va_ds = Dataset.from_pandas(va[cols].reset_index(drop=True)).map(fmt, remove_columns=cols)

    lora = LoraConfig(r=cfg.dpo_lora_r, lora_alpha=cfg.dpo_lora_alpha, lora_dropout=0.05,
                      target_modules=cfg.dpo_target_modules, bias="none", task_type="CAUSAL_LM")
    args = DPOConfig(
        output_dir=str(cfg.out / "dpo_checkpoints"),
        num_train_epochs=cfg.dpo_epochs,
        per_device_train_batch_size=cfg.dpo_batch_size, per_device_eval_batch_size=cfg.dpo_batch_size,
        gradient_accumulation_steps=cfg.dpo_grad_accum, gradient_checkpointing=True,
        learning_rate=cfg.dpo_lr, lr_scheduler_type="cosine", warmup_steps=cfg.dpo_warmup_steps,
        logging_steps=20, eval_strategy="steps", eval_steps=150,
        save_strategy="steps" if cfg.dpo_load_best else "no", save_steps=150,
        load_best_model_at_end=cfg.dpo_load_best,
        bf16=False, fp16=False,  # T4 has no BF16; avoid GradScaler conflicts
        beta=cfg.dpo_beta, max_length=cfg.dpo_max_len or cfg.max_len, remove_unused_columns=False,
        report_to="none", optim="paged_adamw_8bit" if pick_device() == "cuda" else "adamw_torch",
        seed=SEED)
    trainer = DPOTrainer(model=policy, ref_model=None, args=args, train_dataset=tr_ds,
                         eval_dataset=va_ds, processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(out)
    (out / "training_log.json").write_text(json.dumps(trainer.state.log_history))
    print(f"[dpo] saved to {out}")
    del policy, trainer
    _free()
    return out
