"""Model loading, scoring and generation shared by the pipeline and the web app.

Works on three kinds of hardware:
  * CUDA (Kaggle T4): policy in 4-bit NF4 with FP16 compute (needs bitsandbytes)
  * Apple Silicon (MPS): policy in FP16, no quantisation
  * CPU: policy in FP32 (slow, but works)
The DeBERTa-v3 reward model always runs in FP32 (it produces NaN in FP16).
"""
import os
from pathlib import Path

import torch

from .config import POLICY_BASE_MODEL, RM_BASE_MODEL


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def has_adapter(path) -> bool:
    return path is not None and (Path(path) / "adapter_config.json").exists()


# ---------------------------------------------------------------- reward model
class RewardModel:
    """DeBERTa-v3-base + LoRA adapter (+ trained pooler) returning a scalar score."""

    def __init__(self, adapter_dir=None, device=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or pick_device()
        self.trained = has_adapter(adapter_dir)
        tok_src = adapter_dir if self.trained else RM_BASE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        base = AutoModelForSequenceClassification.from_pretrained(
            RM_BASE_MODEL, num_labels=1, torch_dtype=torch.float32)
        base.config.pad_token_id = self.tokenizer.pad_token_id
        if self.trained:
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model = base.to(self.device).eval()

    @torch.no_grad()
    def score_batch(self, prompts, responses, max_len=512, batch_size=8):
        scores = []
        for i in range(0, len(prompts), batch_size):
            enc = self.tokenizer(list(prompts[i:i + batch_size]), list(responses[i:i + batch_size]),
                                 truncation=True, max_length=max_len, padding=True,
                                 return_tensors="pt").to(self.device)
            out = self.model(**enc).logits[:, 0].float()
            if (~torch.isfinite(out)).any():
                print("WARNING: NaN/inf reward scores - the reward model must run in FP32")
            scores.extend(out.cpu().tolist())
        return scores

    def score(self, prompt, response, max_len=512):
        return self.score_batch([prompt], [response], max_len=max_len)[0]


# ---------------------------------------------------------------- policy
def load_policy_base(device=None, for_training=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(POLICY_BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            POLICY_BASE_MODEL, quantization_config=bnb, torch_dtype=torch.float16, device_map="auto")
        if for_training:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(POLICY_BASE_MODEL, torch_dtype=dtype).to(device)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


class PolicyPair:
    """One copy of TinyLlama serving both the base and the DPO model.

    The DPO model is the base plus a LoRA adapter; the base answer is produced by
    temporarily disabling the adapter, so only one set of weights is in memory.
    """

    def __init__(self, adapter_dir=None, device=None):
        self.device = device or pick_device()
        model, self.tokenizer = load_policy_base(self.device)
        self.has_dpo = has_adapter(adapter_dir)
        if self.has_dpo:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        self.model = model.eval()

    def _input_device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _generate(self, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self._input_device())
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=temperature, top_p=top_p, repetition_penalty=1.15,
                                  pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate_base(self, prompt, **kw):
        if self.has_dpo:
            with self.model.disable_adapter():
                return self._generate(prompt, **kw)
        return self._generate(prompt, **kw)

    def generate_dpo(self, prompt, **kw):
        if not self.has_dpo:
            raise RuntimeError("No DPO adapter loaded")
        return self._generate(prompt, **kw)


def resolve_dir(*candidates):
    """Return the first candidate directory that contains a saved adapter."""
    for c in candidates:
        if c and has_adapter(c):
            return Path(c)
    return None


MODEL_SEARCH = ["models", "outputs/config_A", "outputs/config_B", "outputs/config_local", "outputs"]


def default_model_dirs(root="."):
    """Kaggle models in models/ win; otherwise use the newest pipeline output."""
    root = Path(root)
    rm = resolve_dir(os.environ.get("RLHF_RM_DIR"), *[root / d / "reward_model" for d in MODEL_SEARCH])
    pol = resolve_dir(os.environ.get("RLHF_DPO_DIR"), *[root / d / "finetuned_llm" for d in MODEL_SEARCH])
    return rm, pol
