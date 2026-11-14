"""Experiment configurations.

Config "A" reproduces notebook mini-6 (balanced 2K HH + 2K LMSYS).
Config "B" reproduces notebook mini-7 (HH-weighted 6K HH + 2K LMSYS).
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

RM_BASE_MODEL = "microsoft/deberta-v3-base"
POLICY_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SEED = 42


@dataclass
class Config:
    name: str = "A"
    # ---- data ----
    hh_per_source: int = 2000
    lmsys_per_source: int = 2000
    max_chars: int = 1500
    min_chars: int = 20
    split: tuple = (0.875, 0.0625, 0.0625)
    lmsys_csv: str = ""  # path to LMSYS train.csv; auto-detected if empty
    # ---- reward model ----
    rm_lora_r: int = 32
    rm_lora_alpha: int = 64
    rm_epochs: int = 2
    rm_lr: float = 2e-5
    rm_warmup_steps: int = 100
    rm_batch_size: int = 8
    rm_grad_accum: int = 2
    # ---- DPO ----
    dpo_hh_samples: int = 0       # 0 = sample dpo_total at random from the train split
    dpo_lmsys_samples: int = 0
    dpo_total: int = 1500
    dpo_val: int = 200
    dpo_lora_r: int = 16
    dpo_lora_alpha: int = 32
    dpo_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    dpo_epochs: int = 2
    dpo_lr: float = 5e-5
    dpo_beta: float = 0.10
    dpo_warmup_steps: int = 19
    dpo_load_best: bool = False
    dpo_batch_size: int = 4
    dpo_grad_accum: int = 4
    dpo_max_len: int = 0          # 0 = same as max_len
    # ---- evaluation ----
    eval_per_source: int = 25
    max_len: int = 512
    # ---- output ----
    out_dir: str = "outputs"

    @property
    def out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reward_model_dir(self) -> Path:
        return self.out / "reward_model"

    @property
    def policy_dir(self) -> Path:
        return self.out / "finetuned_llm"


CONFIG_A = Config(name="A")

CONFIG_B = replace(
    CONFIG_A,
    name="B",
    hh_per_source=6000, lmsys_per_source=2000,
    rm_lora_r=64, rm_lora_alpha=128, rm_epochs=3, rm_lr=1e-5, rm_warmup_steps=200,
    dpo_hh_samples=2000, dpo_lmsys_samples=1000, dpo_total=3000,
    dpo_lora_r=32, dpo_lora_alpha=64,
    dpo_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    dpo_lr=1e-5, dpo_beta=0.15, dpo_warmup_steps=38, dpo_load_best=True,
)

# Small, balanced run that trains on an 8 GB Apple-Silicon Mac (M1) in about an hour.
# Same method as config A, fewer pairs and shorter sequences.
CONFIG_LOCAL = replace(
    CONFIG_A,
    name="local",
    hh_per_source=1000, lmsys_per_source=1000,
    rm_epochs=1, rm_warmup_steps=10,
    dpo_total=400, dpo_val=60, dpo_epochs=1, dpo_warmup_steps=5,
    dpo_batch_size=1, dpo_grad_accum=8, dpo_max_len=256,
    eval_per_source=15, max_len=384,
)

CONFIGS = {"A": CONFIG_A, "B": CONFIG_B, "local": CONFIG_LOCAL}
