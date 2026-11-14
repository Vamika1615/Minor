"""Stage 4: per-source evaluation of the reward model and of base vs. DPO policy."""
import json

import numpy as np
import pandas as pd

from .config import SEED, Config
from .models import PolicyPair, RewardModel

DEMO_PROMPTS = [
    "What is neural network?",
    "Explain neural networks simply.",
    "How do I stay motivated while studying?",
    "I'm feeling anxious about an upcoming exam. Any advice?",
    "What's a healthy way to handle disagreements with friends?",
]


def evaluate_reward_model(rm: RewardModel, test: pd.DataFrame, max_len=512) -> dict:
    results = {}
    subsets = {"combined": test,
               "hh_rlhf": test[test.source == "hh_rlhf"],
               "lmsys": test[test.source == "lmsys"]}
    for name, sub in subsets.items():
        if len(sub) == 0:
            continue
        c = np.array(rm.score_batch(sub.prompt.tolist(), sub.chosen.tolist(), max_len))
        r = np.array(rm.score_batch(sub.prompt.tolist(), sub.rejected.tolist(), max_len))
        ok = np.isfinite(c) & np.isfinite(r)
        c, r = c[ok], r[ok]
        results[name] = {"n": int(ok.sum()), "mean_chosen": float(c.mean()), "mean_rejected": float(r.mean()),
                         "gap": float((c - r).mean()), "accuracy": float((c > r).mean())}
    if "hh_rlhf" in results and "lmsys" in results:
        diff = abs(results["hh_rlhf"]["accuracy"] - results["lmsys"]["accuracy"])
        results["balanced"] = bool(diff < 0.10)
    return results


def compare(policy: PolicyPair, rm: RewardModel, prompt: str, seed=None) -> dict:
    base = policy.generate_base(prompt, seed=seed)
    dpo = policy.generate_dpo(prompt, seed=seed)
    sb, sd = rm.score(prompt, base), rm.score(prompt, dpo)
    return {"prompt": prompt, "base_response": base, "dpo_response": dpo,
            "base_score": sb, "dpo_score": sd, "delta": sd - sb}


def evaluate_policy(policy: PolicyPair, rm: RewardModel, test: pd.DataFrame, per_source=25) -> pd.DataFrame:
    rows = []
    for src in ("hh_rlhf", "lmsys"):
        sub = test[test.source == src]
        sub = sub.sample(n=min(per_source, len(sub)), random_state=SEED)
        for i, p in enumerate(sub.prompt):
            res = compare(policy, rm, p)
            res["source"] = src
            rows.append(res)
            print(f"[eval] {src} {i + 1}/{len(sub)} delta={res['delta']:+.3f}")
    return pd.DataFrame(rows)


def summarise_winrate(df: pd.DataFrame) -> dict:
    out = {}
    for name, sub in {"combined": df, "hh_rlhf": df[df.source == "hh_rlhf"],
                      "lmsys": df[df.source == "lmsys"]}.items():
        if len(sub):
            out[name] = {"n": len(sub), "mean_base": float(sub.base_score.mean()),
                         "mean_dpo": float(sub.dpo_score.mean()), "mean_delta": float(sub.delta.mean()),
                         "win_rate": float((sub.delta > 0).mean())}
    return out


def run_evaluation(cfg: Config, test: pd.DataFrame):
    rm = RewardModel(cfg.reward_model_dir)
    if not rm.trained:
        raise RuntimeError(f"No trained reward model in {cfg.reward_model_dir}")
    rm_res = evaluate_reward_model(rm, test, cfg.max_len)
    print("[eval] reward model:", json.dumps(rm_res, indent=2))

    policy = PolicyPair(cfg.policy_dir)
    if not policy.has_dpo:
        raise RuntimeError(f"No DPO adapter in {cfg.policy_dir}")
    demo = pd.DataFrame([compare(policy, rm, p) for p in DEMO_PROMPTS])
    held = evaluate_policy(policy, rm, test, cfg.eval_per_source)
    wr = summarise_winrate(held)
    print("[eval] held-out win rates:", json.dumps(wr, indent=2))

    demo.to_csv(cfg.out / "demo_results.csv", index=False)
    held.to_csv(cfg.out / "batch_eval.csv", index=False)
    (cfg.out / "metrics.json").write_text(json.dumps({"reward_model": rm_res, "dpo": wr}, indent=2))
    print(f"[eval] results written to {cfg.out}")
    return rm_res, wr
