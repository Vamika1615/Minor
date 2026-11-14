# RLHF System: Reward Modeling + DPO on Combined Human Preference Data

IGDTUW minor project (ECE, Semester VII). This folder turns the Kaggle notebooks
(`mini-6_working`, `mini-7-fixed`) into a reusable system with two parts:

| Part | What it does | Where it runs |
|---|---|---|
| `run_pipeline.py` | Data -> reward model -> DPO -> per-source evaluation | Kaggle / Colab NVIDIA GPU |
| `app.py` | Web app: base vs DPO TinyLlama side by side, scored by the reward model | This Mac (Apple M1), or any machine |

```
rlhf_system/
  app.py                 web app (Gradio)
  run_pipeline.py        training / evaluation CLI
  rlhf/
    config.py            configurations A (2K+2K) and B (6K+2K), as in the notebooks
    data.py              HH-RLHF + LMSYS loading, normalising, filtering, stratified split
    models.py            reward model + policy loading, scoring, generation (CUDA / MPS / CPU)
    train.py             reward-model training (Bradley-Terry) and DPO training
    evaluate.py          per-source RM accuracy, demo reel, held-out win rates
  models/                put the trained Kaggle adapters here
  outputs/               pipeline outputs and exported sessions
```

## 1. Run the web app on this Mac

```bash
cd "/Users/vamikabhardwaj/Desktop/minor project/rlhf_system"
.venv/bin/python app.py
```
Then open http://127.0.0.1:7860. The first start downloads TinyLlama (~2.2 GB) and DeBERTa-v3 (~0.7 GB).

### Plug in your trained models
From your saved Kaggle notebook version, open **Output** and download the two folders
`reward_model/` and `finetuned_llm/`. Put them here:

```
rlhf_system/models/reward_model/adapter_config.json   (+ adapter_model.safetensors, tokenizer files)
rlhf_system/models/finetuned_llm/adapter_config.json  (+ adapter_model.safetensors, tokenizer files)
```
Restart the app. The status table on the Compare tab should show both as **Loaded**. Until
then the app runs in base-only mode (base TinyLlama answers, no reward scores).

On the Mac the policy runs in FP16 on the M1 GPU (no 4-bit quantisation, which needs CUDA).
A 200-token answer takes roughly 10-30 s per model.

## 2. Train a small version on this Mac (already done)

```bash
cd "/Users/vamikabhardwaj/Desktop/minor project/rlhf_system"
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 .venv/bin/python run_pipeline.py --config local
```
Config `local` is config A scaled down for an 8 GB M1: 1,000 HH + 1,000 LMSYS pairs, 1 epoch,
DPO on 400 pairs with batch 1 and 256-token sequences. LMSYS data is downloaded from the
HuggingFace Hub (`lmarena-ai/arena-human-preference-55k`, the same file as the Kaggle
competition). It takes about 1.5 hours (reward model ~58 min, DPO ~22 min, evaluation ~12 min) and
writes everything to `outputs/config_local/`. The app loads these automatically when `models/` is empty.

Measured results of that run (126 test pairs, 30 held-out prompts):

| Metric | Combined | HH-RLHF | LMSYS |
|---|---|---|---|
| Reward-model pairwise accuracy | 57.1% | 52.4% | 61.9% |
| DPO win rate vs base | 57% | 60% | 53% |

These are small-sample numbers from a reduced run; the full Kaggle runs are in the report.

## 3. Re-train on Kaggle (GPU)

1. Upload this folder as a Kaggle dataset (or `git clone` it in the notebook).
2. Notebook settings: **GPU T4 x2**, **Internet on**, add the LMSYS `train.csv` dataset.
3. In a cell:
```bash
!pip install -q -r requirements-train.txt
!python run_pipeline.py --config A            # balanced 2K HH + 2K LMSYS (notebook mini-6)
!python run_pipeline.py --config B            # HH-weighted 6K + 2K (notebook mini-7)
```
Single stages: `--stage data | rm | dpo | eval`. Outputs go to `outputs/config_A/`:
`train/val/test.csv`, `reward_model/`, `finetuned_llm/`, `demo_results.csv`, `batch_eval.csv`,
`metrics.json`. Stages whose output already exists are skipped, so a crashed run can resume.
Copy `outputs/config_A/reward_model` and `outputs/config_A/finetuned_llm` into `models/` to use them in the app.
