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

# ===================== Notebook cell 13 =====================
# === Step 6 - Sample-balanced combination ===
random.seed(SEED)
random.shuffle(hh_clean); random.shuffle(lmsys_clean)

PER_SOURCE = 4000   # 2000/source = 4000 total - good accuracy, ~2-3hr runtime   # reduced from 4000 to cut training time in half

if hh_clean and lmsys_clean:
    n_hh    = min(int(PER_SOURCE * 1.5), len(hh_clean))
    n_lmsys = min(int(PER_SOURCE * 0.5), len(lmsys_clean))
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

# ===================== Notebook cell 18 =====================
if not rm_already_trained:
    # DeBERTa: no 4-bit, no kbit prep needed
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        RM_BASE_MODEL, num_labels=1,
        dtype=torch.float32,             # FIX: DeBERTa-v3 disentangled attention produces NaN in fp16/bf16 - must be float32
        device_map="auto",
    )
    rm_base.config.pad_token_id = rm_tokenizer.pad_token_id
    rm_base.config.return_dict = True

    # DeBERTa attention module names differ from LLaMA
    rm_lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=64, lora_alpha=128, lora_dropout=0.05,
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

# ===================== Notebook cell 20 =====================
if not rm_already_trained:
    from trl import RewardConfig, RewardTrainer

    rm_config = RewardConfig(
        output_dir=f"{BASE_DIR}/rm_checkpoints",
        num_train_epochs=3,        # increased from 2 - needs more epochs to converge
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=1e-5,    # reduced from 2e-5 - slower, more careful convergence
        lr_scheduler_type="cosine",
        warmup_steps=200,      # recalculated for 4 epochs (~1750 total steps, 10% warmup)
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
    # rebalanced: HH-heavy (HH gives cleaner reward signal) + larger total
    _hh_pool    = train_df[train_df["source"] == "hh_rlhf"]
    _lmsys_pool = train_df[train_df["source"] == "lmsys"]
    _hh_samp    = _hh_pool.sample(n=min(2000, len(_hh_pool)),   random_state=SEED)
    _lmsys_samp = _lmsys_pool.sample(n=min(1000, len(_lmsys_pool)), random_state=SEED)
    dpo_train_df = pd.concat([_hh_samp, _lmsys_samp]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    dpo_val_df   = val_df.sample(n=min(200, len(val_df)), random_state=SEED).reset_index(drop=True)

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
        r=32, lora_alpha=64, lora_dropout=0.05,           # r: 64->32 + MLP layers covers more of the model
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # MLP layers added
        bias="none", task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=f"{BASE_DIR}/dpo_checkpoints",
        num_train_epochs=2,          # 3->2: best checkpoint was step ~150; epoch 3 caused collapse
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=1e-5,          # 5e-5->1e-5: primary cause of fast memorisation
        lr_scheduler_type="cosine",
        warmup_steps=38,             # recalculated: 10% of ~375 steps (3000 samples, 2 epochs)
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=150,
        save_strategy="steps",
        save_steps=150,
        load_best_model_at_end=True,    # recovers best val-loss checkpoint, not collapsed final weights
        bf16=False,
        fp16=False,
        beta=0.15,           # 0.05->0.15: stronger KL regularisation - primary anti-overfit lever in DPO
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

    # KL monitor - verify training signal quality
    kl_entries = [e for e in dpo_history
                  if any(k in e for k in ['rewards/kl', 'logps/chosen', 'logps/rejected'])]
    if kl_entries:
        print("\n DPO training signal (last 5 log entries):")
        print(pd.DataFrame(kl_entries).tail(5).to_string(index=False))
    else:
        print("\n DPO log (last 5 entries):")
        print(pd.DataFrame(dpo_history).tail(5).to_string(index=False))

    dpo_trainer.save_model(FINETUNED_LLM_PATH)
    policy_tokenizer.save_pretrained(FINETUNED_LLM_PATH)
    with open(f"{FINETUNED_LLM_PATH}/training_log.json", "w") as f:
        json.dump(dpo_history, f)
    print(f" Fine-tuned LLM saved to {FINETUNED_LLM_PATH}")
    del policy_model, dpo_trainer
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
