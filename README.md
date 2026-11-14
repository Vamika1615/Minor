# Reward Modeling and Direct Preference Optimization on Combined Human Preference Data
### A Cross-Distribution RLHF Pipeline

**Minor Project, B.Tech Electronics and Communication Engineering (Semester VII), 2026–27**
**Indira Gandhi Delhi Technical University for Women (IGDTUW), Delhi**

| Team member | Enrolment no. |
|---|---|
| Saumya Shukla | 05901022023 |
| Kejal Garg | 07901022023 |
| Pranjal Jain | 08101022023 |
| Vamika Bhardwaj | 00411022024 |

**Project guide:** Prof. (Dr.) Ashwni Kumar, Pro Vice-Chancellor, IGDTUW

---

## Contents

1. [What this project is about](#1-what-this-project-is-about)
2. [How it works](#2-how-it-works)
3. [Results](#3-results)
4. [Repository layout](#4-repository-layout)
5. [Running the web app](#5-running-the-web-app)
6. [Training the models](#6-training-the-models)
7. [Building the report](#7-building-the-report)
8. [Limitations](#8-limitations)
9. [Future scope](#9-future-scope)
10. [Tech stack](#10-tech-stack)

---

## 1. What this project is about

Chat assistants such as ChatGPT and Claude are trained to answer the way people prefer, not
just to continue text. This step is called alignment. It is usually done with
**Reinforcement Learning from Human Feedback (RLHF)**: people compare two answers to the same
question and pick the better one. A **reward model** learns to predict those choices, and the
chat model is then trained to produce answers the reward model scores highly.

Most published work trains on a **single** preference dataset. Real preference data comes from
very different sources, though:

- **HH-RLHF (Anthropic):** paid annotators following written guidelines; mostly short,
  multi-turn helper dialogues.
- **LMSYS Chatbot Arena:** votes from the general public comparing two anonymous chatbots on
  whatever they chose to ask; longer, open-ended questions.

**Our question:** can one alignment pipeline learn from both kinds of preference data at once,
and does it work equally well on each? To answer that we report **every result per source**,
not just a single overall number. That decision turned out to matter: in one configuration an
overall win rate of exactly 50% hid a 24% vs 76% split between the two sources.

We also deliberately limited ourselves to **free hardware** (Kaggle's free T4 GPUs, and later an
8 GB Apple M1 laptop), using parameter-efficient methods (LoRA, 4-bit QLoRA) to fit
billion-parameter models.

### Objectives

| # | Objective | Status |
|---|---|---|
| O1 | One combined, source-tagged, stratified dataset | Met (40.5K HH + 36.5K LMSYS clean pairs) |
| O2 | DeBERTa-v3 reward model with LoRA | Met (3 trained reward models) |
| O3 | Per-source reward-model accuracy within a 10-point balance rule | Met |
| O4 | DPO-align TinyLlama-1.1B on a single T4 | Met (1.14 GB policy footprint) |
| O5 | Compare base vs DPO per source | Met (no clear overall gain; see results) |
| O6 | Reusable system with a web application | Met (`rlhf_system/`) |
| O7 | Honest critical analysis | Met (incl. a reward-hacking example) |

---

## 2. How it works

```
 HH-RLHF ──┐                                         ┌── per-source RM accuracy
           ├─> parse → common format → filter →      │
 LMSYS ────┘   balance → stratified split ──┬──> Reward model (DeBERTa-v3 + LoRA,
                                            │     Bradley–Terry loss)
                                            │              │ judges
                                            └──> DPO on TinyLlama-1.1B-Chat  ──> base vs DPO
                                                 (4-bit, LoRA, β-regularised)     win rate per source
```

1. **Data.** Both datasets are converted to `(prompt, chosen, rejected, source)`. Ties,
   empty or near-identical answers and overly long fields are removed. The two sources are
   balanced and split into train / validation / test so that every split contains both.
2. **Reward model.** `microsoft/deberta-v3-base` with LoRA on the attention projections and
   a trainable pooler, trained with the Bradley–Terry pairwise loss
   `−log σ(r(chosen) − r(rejected))`.
3. **Policy alignment with DPO.** `TinyLlama-1.1B-Chat` loaded in 4-bit NF4, LoRA adapters,
   paged 8-bit AdamW and gradient checkpointing, trained with **Direct Preference
   Optimization**, which learns from preference pairs directly without a separate RL loop
   (no PPO).
4. **Evaluation.** The reward model's pairwise accuracy on held-out pairs, and the win rate of
   DPO answers over base-model answers on held-out prompts — always split by source.

### Configurations

| | Config A (`mini-6_working`) | Config B (`mini-7-fixed`) | Config local (M1 laptop) |
|---|---|---|---|
| Training pairs | 2K HH + 2K LMSYS (balanced) | 6K HH + 2K LMSYS (HH-weighted) | 1K HH + 1K LMSYS |
| Hardware | Kaggle 2× T4 | Kaggle 2× T4 | Apple M1, 8 GB |
| DPO | 1.5K pairs, LoRA r=16 (attention), β=0.10, 2 epochs | 3K pairs (2:1 HH), LoRA r=32 (attention + MLP), β=0.15 | 400 pairs, 1 epoch, batch 1, 256 tokens |
| Time | ~2.5–3 h | ~3 h | ~1.5 h |

---

## 3. Results

All numbers below were **measured** by our code. The report also includes higher numbers from
an earlier paper draft; those could not be reproduced and are clearly marked as such in
Chapter 5 of the report.

### Reward model — pairwise accuracy on held-out pairs

| Config | Combined | HH-RLHF | LMSYS | Source gap |
|---|---|---|---|---|
| A (2K+2K) | **59.2%** (n=250) | 59.2% | 59.2% | 0.0 pts |
| B (6K+2K) | **59.6%** (n=500) | 57.9% | 64.8% | 6.9 pts |
| local (1K+1K) | **57.1%** (n=126) | 52.4% | 61.9% | 9.5 pts |

All three beat chance and stay within the 10-point balance rule, but a 184M-parameter encoder
plateaus around 57–60% regardless of data size.

### DPO — win rate of DPO answers over base answers (judged by the reward model)

| Config | Combined | HH-RLHF | LMSYS |
|---|---|---|---|
| A (25+25 prompts) | 50% | 56% | 44% |
| B (25+25 prompts) | 50% | **24%** | **76%** (p≈0.015 each) |
| local (15+15 prompts) | 57% | 60% | 53% |

DPO training itself looked successful (implicit-reward accuracy ≈80% in A, ≈95% in B), yet
held-out win rates were at chance overall.

### Key findings

1. **Expert and crowd preferences can be trained on together** — neither source swamps the other.
2. **The reward model is the weak link** at this scale (~59% accuracy), both as a training signal and as a judge.
3. **DPO training metrics do not prove alignment** — fitting stored pairs ≠ writing better answers.
4. **Always split by source** — config B's flat 50% hid opposite effects of 24% and 76%.
5. **A small judge can be fooled** — in a live demo the largest score gain went to a *wrong*
   answer ("2 + 3" in Java), a clear case of reward hacking.
6. **Reproducible on modest hardware** — the whole pipeline retrains on an 8 GB laptop in ~1.5 h.

---

## 4. Repository layout

```
.
├── README.md                         ← this file
├── RLHF_Minor_Project_Report.pdf     ← final report (138 pages)
├── RLHF_Minor_Report/                ← LaTeX source of the report
│   ├── main.tex                      preamble, cover, certificate, front matter
│   ├── chapters/                     ch1_introduction … ch6_conclusion, appendix
│   ├── figures/                      plots, diagrams, app screenshots, logo
│   ├── code/                         notebook code + demo outputs used in appendices
│   └── references.bib
├── rlhf_system/                      ← the packaged pipeline + web app
│   ├── app.py                        Gradio web app (base vs DPO, scored live)
│   ├── run_pipeline.py               CLI: --config A|B|local --stage data|rm|dpo|eval|all
│   ├── rlhf/                         config, data, models, train, evaluate modules
│   ├── outputs/config_local/         trained reward model + DPO adapters, metrics, CSVs, logs
│   ├── models/                       drop Kaggle-trained adapters here (take priority)
│   ├── requirements-app.txt
│   ├── requirements-train.txt
│   └── README.md                     detailed system instructions
├── mini-6_working (1).ipynb 2        ← original Kaggle notebook, config A
├── mini-7-fixed.ipynb 2              ← original Kaggle notebook, config B
├── RLHF_IEEE_Report (1) 3.docx       ← earlier paper draft
├── MinorProjectReport.pdf            ← university report template
└── igdtuw_logo.png
```

Not included in the repo (recreated automatically): the Python virtual environment, the
176 MB LMSYS `train.csv` (downloaded by the pipeline), LaTeX build files and training
checkpoints.

---

## 5. Running the web app

The app shows the **base** TinyLlama answer and the **DPO-aligned** answer side by side for any
prompt you type, scores both with the reward model, and declares a winner. It also has a
history table with CSV export, a Results tab drawn from `metrics.json`, and an About tab.
The trained adapters from the laptop run are already included, so it works straight after
cloning.

```bash
git clone https://github.com/Vamika1615/Minor.git
cd Minor/rlhf_system
python3 -m venv .venv
.venv/bin/pip install -r requirements-app.txt
.venv/bin/python app.py
```

Open http://127.0.0.1:7860. The first start downloads TinyLlama (~2.2 GB) and DeBERTa-v3
(~0.7 GB) from HuggingFace. On an M1 Mac a 200-token answer takes roughly 10–30 s per model.

> You do **not** need to vote for the app to work — the reward model picks the winner. Training
> happens separately via `run_pipeline.py`.

To use Kaggle-trained models instead, copy their `reward_model/` and `finetuned_llm/` folders
into `rlhf_system/models/` and restart.

---

## 6. Training the models

### On a Mac (8 GB Apple Silicon)

```bash
cd rlhf_system
.venv/bin/pip install -r requirements-train.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 .venv/bin/python run_pipeline.py --config local
```

About 1.5 hours (reward model ~58 min, DPO ~22 min, evaluation ~12 min). Output goes to
`outputs/config_local/`.

### On Kaggle (GPU T4 ×2, Internet on, LMSYS `train.csv` attached)

```bash
!pip install -q -r requirements-train.txt
!python run_pipeline.py --config A     # balanced 2K + 2K
!python run_pipeline.py --config B     # HH-weighted 6K + 2K
```

Run single stages with `--stage data|rm|dpo|eval`. Finished stages are skipped, so a crashed
run resumes where it stopped. Alternatively, run the original notebooks directly on Kaggle
(see Appendix D of the report).

---

## 7. Building the report

Requires a TeX distribution (MacTeX / TeX Live) with `latexmk`.

```bash
cd RLHF_Minor_Report
latexmk -pdf main.tex
```

The report follows the IGDTUW minor-project structure: cover, certificate, declaration,
acknowledgement, abstract, contents, lists of figures / tables / listings / abbreviations /
symbols, six chapters (Introduction, Literature Review, System Design and Methodology,
Implementation, Results and Discussion, Conclusion and Future Scope), references, and
appendices (full notebook code, config B changes, sample outputs, reproduction guide).

---

## 8. Limitations

- The aligned model is judged by a reward model of the same kind it was trained towards —
  weak (~59%), gameable and not independent. No human rating or external LLM judge was used.
- Small evaluation sets (15–25 prompts per source, one sample per model) can detect only
  large effects.
- Small models (184M reward model, 1.1B policy); results may not hold at 7B+.
- No supervised fine-tuning on our data before DPO.
- Helpfulness only — harmlessness and factual accuracy were not measured.
- Only two Kaggle data mixes; no single-source reward-model ablation.

---

## 9. Future scope

- **Better evaluation:** an independent strong LLM judge with position swapping, MT-Bench /
  AlpacaEval / RewardBench, 200+ prompts per source with significance tests.
- **Better reward models:** DeBERTa-v3-large or a 1–2B decoder reward model, per-source heads,
  a third source such as UltraFeedback or HelpSteer2.
- **Better policy training:** SFT before DPO, on-policy / iterative DPO, comparisons with
  IPO, KTO, ORPO and SimPO, and a 7–8B policy.
- **App:** show an independent judge's verdict next to the reward model's and let users vote,
  collecting new preference data.

---

## 10. Tech stack

Python · PyTorch (CUDA / Apple MPS) · HuggingFace Transformers, Datasets, PEFT, TRL
(`RewardTrainer`, `DPOTrainer`) · bitsandbytes (4-bit NF4) · Gradio · pandas · matplotlib ·
LaTeX (TikZ, listings, BibTeX with `ieeetr` style)

### Datasets and models

- [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) (helpful-base)
- [lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k)
- [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
- [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

### Key references

- Ouyang et al., *Training language models to follow instructions with human feedback* (2022)
- Bai et al., *Training a helpful and harmless assistant with RLHF* (2022)
- Rafailov et al., *Direct Preference Optimization* (2023)
- Hu et al., *LoRA* (2021); Dettmers et al., *QLoRA* (2023)
- Zheng et al., *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena* (2023)

The full bibliography is in the report.
