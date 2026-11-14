"""Local web app: compare base TinyLlama with the DPO-aligned model, scored by the reward model.

Run:  python app.py      then open http://127.0.0.1:7860

Models are picked up automatically (first match wins): $RLHF_RM_DIR / $RLHF_DPO_DIR, then
models/ (your Kaggle downloads), then outputs/config_A, config_B, config_local (pipeline runs).
Without them the app runs in base-only mode.
"""
import base64
import html
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
import pandas as pd

from rlhf.evaluate import DEMO_PROMPTS
from rlhf.models import PolicyPair, RewardModel, default_model_dirs, pick_device

ROOT = Path(__file__).parent
RM_DIR, DPO_DIR = default_model_dirs(ROOT)
DEVICE = pick_device()

print(f"[app] device={DEVICE}  reward_model={RM_DIR}  dpo_adapter={DPO_DIR}")
print("[app] loading TinyLlama (first run downloads ~2.2 GB)...")
POLICY = PolicyPair(DPO_DIR, DEVICE)
print("[app] loading reward model (first run downloads ~0.7 GB)...")
RM = RewardModel(RM_DIR, DEVICE)
print("[app] ready")

HISTORY: list = []
LOGO_B64 = base64.b64encode((ROOT / "assets" / "logo.png").read_bytes()).decode() \
    if (ROOT / "assets" / "logo.png").exists() else ""


def rel(path):
    p = Path(path)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


# ---------------------------------------------------------------- HTML pieces
def header_html():
    logo = f'<img src="data:image/png;base64,{LOGO_B64}" alt="IGDTUW">' if LOGO_B64 else ""
    return f"""
<div class="hdr">
  {logo}
  <div>
    <div class="hdr-title">Base vs DPO-aligned TinyLlama</div>
    <div class="hdr-sub">Reward Modeling and Direct Preference Optimization on Combined Human Preference Data</div>
    <div class="hdr-uni">Indira Gandhi Delhi Technical University for Women &middot; Department of ECE</div>
  </div>
</div>"""


def status_html():
    def chip(ok, label, detail):
        cls = "ok" if ok else "missing"
        mark = "&#9679;" if ok else "&#9675;"
        return f'<span class="chip {cls}">{mark} {label}<small>{html.escape(detail)}</small></span>'
    chips = [
        chip(True, "TinyLlama-1.1B-Chat", f"running on {DEVICE}"),
        chip(POLICY.has_dpo, "DPO adapter", rel(DPO_DIR) if POLICY.has_dpo else "not found"),
        chip(RM.trained, "Reward model", rel(RM_DIR) if RM.trained else "not found"),
    ]
    note = ""
    if not (POLICY.has_dpo and RM.trained):
        note = ('<div class="warn">Base-only mode: put <code>reward_model/</code> and <code>finetuned_llm/</code> '
                'in <code>rlhf_system/models/</code> and restart the app.</div>')
    return f'<div class="chips">{"".join(chips)}</div>{note}'


def card_head(title, score, kind):
    pill = f'<span class="pill {kind}">reward {score:+.3f}</span>' if score is not None else \
        '<span class="pill none">no score</span>'
    return f'<div class="card-head {kind}"><span>{title}</span>{pill}</div>'


def verdict_html(s_base, s_dpo, secs):
    timing = f'<span class="timing">{secs:.1f}s on {DEVICE}</span>'
    if s_base is None or s_dpo is None:
        msg = "Base-only mode: no DPO adapter loaded" if not POLICY.has_dpo else \
            "Reward model not loaded: answers shown without scores"
        return f'<div class="verdict neutral"><b>{msg}</b>{timing}</div>'
    d = s_dpo - s_base
    if d > 0:
        return (f'<div class="verdict dpo"><b>DPO-aligned model wins</b>'
                f'<span class="delta">&Delta; = {d:+.4f}</span>{timing}</div>')
    return (f'<div class="verdict base"><b>Base model wins</b>'
            f'<span class="delta">&Delta; = {d:+.4f}</span>{timing}</div>')


EMPTY_VERDICT = '<div class="verdict idle">Enter a prompt and press <b>Compare</b> to see both answers.</div>'


# ---------------------------------------------------------------- actions
def run(prompt, max_tokens, temperature, seed):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Type a prompt first.")
    seed = int(seed) if seed not in (None, "") else None
    kw = dict(max_new_tokens=int(max_tokens), temperature=float(temperature), seed=seed)

    t0 = time.time()
    base = POLICY.generate_base(prompt, **kw)
    dpo = POLICY.generate_dpo(prompt, **kw) if POLICY.has_dpo else ""
    s_base = RM.score(prompt, base) if RM.trained else None
    s_dpo = RM.score(prompt, dpo) if (RM.trained and POLICY.has_dpo) else None
    secs = time.time() - t0

    delta = None if s_base is None or s_dpo is None else s_dpo - s_base
    HISTORY.append({"Prompt": prompt[:80],
                    "Base": "" if s_base is None else round(s_base, 4),
                    "DPO": "" if s_dpo is None else round(s_dpo, 4),
                    "Delta": "" if delta is None else round(delta, 4),
                    "Winner": "" if delta is None else ("DPO" if delta > 0 else "Base")})
    return (verdict_html(s_base, s_dpo, secs),
            card_head("Base TinyLlama", s_base, "base"), base,
            card_head("DPO-aligned TinyLlama", s_dpo, "dpo"),
            dpo or "_No DPO adapter loaded._",
            pd.DataFrame(HISTORY[::-1]))


def export_history():
    if not HISTORY:
        raise gr.Error("Nothing to export yet.")
    path = ROOT / "outputs" / "interactive_session.csv"
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame(HISTORY).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------- results tab
def results_html():
    if RM_DIR is None or not (RM_DIR.parent / "metrics.json").exists():
        return ('<div class="warn">No evaluation results for the loaded models yet. Run '
                '<code>python run_pipeline.py --config local --stage eval</code>.</div>')
    m = json.loads((RM_DIR.parent / "metrics.json").read_text())
    rm, dp = m["reward_model"], m["dpo"]
    names = [("combined", "Combined"), ("hh_rlhf", "HH-RLHF"), ("lmsys", "LMSYS")]

    def tile(label, value, sub):
        return f'<div class="tile"><div class="tile-v">{value}</div><div class="tile-l">{label}</div><div class="tile-s">{sub}</div></div>'

    tiles = "".join([
        tile("Reward-model accuracy", f'{rm["combined"]["accuracy"]*100:.1f}%', f'{rm["combined"]["n"]} test pairs, chance = 50%'),
        tile("Per-source gap", f'{abs(rm["hh_rlhf"]["accuracy"]-rm["lmsys"]["accuracy"])*100:.1f} pp',
             "balanced" if rm.get("balanced") else "imbalanced (> 10 pp)"),
        tile("DPO win rate", f'{dp["combined"]["win_rate"]*100:.0f}%', f'{dp["combined"]["n"]} held-out prompts'),
        tile("Mean reward gain", f'{dp["combined"]["mean_delta"]:+.3f}', "DPO minus base"),
    ])

    def bar(v):
        return f'<div class="bar"><div style="width:{v*100:.0f}%"></div><span>{v*100:.1f}%</span></div>'

    rows_rm = "".join(
        f'<tr><td>{lab}</td><td>{rm[k]["n"]}</td><td>{rm[k]["mean_chosen"]:+.3f}</td><td>{rm[k]["mean_rejected"]:+.3f}</td>'
        f'<td>{rm[k]["gap"]:+.3f}</td><td>{bar(rm[k]["accuracy"])}</td></tr>' for k, lab in names if k in rm)
    rows_dp = "".join(
        f'<tr><td>{lab}</td><td>{dp[k]["n"]}</td><td>{dp[k]["mean_base"]:+.3f}</td><td>{dp[k]["mean_dpo"]:+.3f}</td>'
        f'<td>{dp[k]["mean_delta"]:+.3f}</td><td>{bar(dp[k]["win_rate"])}</td></tr>' for k, lab in names if k in dp)
    return f"""
<div class="res-title">Evaluation of the loaded models <code>{html.escape(RM_DIR.parent.name)}</code></div>
<div class="tiles">{tiles}</div>
<h3>Reward model on held-out test pairs</h3>
<table class="res"><tr><th>Split</th><th>n</th><th>Chosen mean</th><th>Rejected mean</th><th>Gap</th><th>Pairwise accuracy</th></tr>{rows_rm}</table>
<h3>DPO vs base on held-out prompts (judged by the reward model)</h3>
<table class="res"><tr><th>Split</th><th>n</th><th>Base mean</th><th>DPO mean</th><th>Mean &Delta;</th><th>DPO win rate</th></tr>{rows_dp}</table>
<p class="note">Small samples: with 15 prompts per source, win rates within about &plusmn;25 points of 50% are not
statistically different from chance.</p>"""


ABOUT_HTML = """
<div class="flow">
  <div class="step s1"><b>1. Data</b>Anthropic HH-RLHF (expert labels) + LMSYS Arena (crowd votes) &rarr; one source-tagged schema</div>
  <div class="arrow">&rarr;</div>
  <div class="step s2"><b>2. Reward model</b>DeBERTa-v3-base + LoRA, Bradley-Terry pairwise loss</div>
  <div class="arrow">&rarr;</div>
  <div class="step s3"><b>3. DPO</b>TinyLlama-1.1B-Chat + LoRA, trained on the same preference pairs</div>
  <div class="arrow">&rarr;</div>
  <div class="step s4"><b>4. Evaluation</b>Every metric reported separately for HH-RLHF and LMSYS</div>
</div>
<h3>Results of the full Kaggle runs (from the report)</h3>
<table class="res">
<tr><th>Metric</th><th>Config A (2K + 2K)</th><th>Config B (6K + 2K)</th></tr>
<tr><td>Reward-model accuracy, combined</td><td>59.2%</td><td>59.6%</td></tr>
<tr><td>Reward-model accuracy, HH-RLHF / LMSYS</td><td>59.2% / 59.2%</td><td>57.9% / 64.8%</td></tr>
<tr><td>DPO win rate, combined</td><td>50%</td><td>50%</td></tr>
<tr><td>DPO win rate, HH-RLHF / LMSYS</td><td>56% / 44%</td><td>24% / 76%</td></tr>
</table>
<p class="note">The reward model that scores answers in this app is the same kind used in the report: it is right only
about 6 times in 10 and can be fooled. Try the example <i>"what is 2 + 3 ?"</i> to see reward hacking.</p>"""


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
.hdr {display:flex; gap:18px; align-items:center; padding:18px 22px; border-radius:14px;
      background: linear-gradient(120deg, #2f4f24, #4a7a37); color:#fff; margin-bottom:6px;}
.hdr img {width:72px; height:72px; border-radius:50%; background:#fff; padding:3px;}
.hdr * {color:#fff !important;}
.hdr-title {font-size:26px; font-weight:700; line-height:1.15;}
.hdr-sub {font-size:14px; opacity:.92; margin-top:4px;}
.hdr-uni {font-size:12px; opacity:.75; margin-top:2px; letter-spacing:.3px;}
.chips {display:flex; flex-wrap:wrap; gap:8px; margin:4px 0;}
.chip {display:inline-flex; gap:6px; align-items:baseline; padding:6px 12px; border-radius:999px;
       font-size:13px; font-weight:600; border:1px solid;}
.chip small {font-weight:400; opacity:.8;}
.chip.ok {background:#ecf7ec; color:#22622b; border-color:#b9e0bd;}
.chip.missing {background:#fff4e5; color:#8a5300; border-color:#f5d39b;}
.warn {margin-top:8px; padding:10px 14px; border-radius:10px; background:#fff4e5; color:#7a4a00; font-size:14px;}
.verdict {display:flex; align-items:center; gap:14px; padding:14px 18px; border-radius:12px; font-size:17px;}
.verdict .delta {font-family:ui-monospace,monospace; font-size:15px;}
.verdict .timing {margin-left:auto; font-size:13px; opacity:.75;}
.verdict.dpo {background:#e6f4ea; color:#1d5e2c; border:1px solid #a8d8b5;}
.verdict.base {background:#eef1f5; color:#374151; border:1px solid #cdd3dc;}
.verdict.neutral {background:#fff4e5; color:#7a4a00; border:1px solid #f5d39b;}
.verdict.idle {display:block; background:transparent; color:#6b7280; border:1px dashed #cbd5e1; font-size:15px;}
.card-head {display:flex; justify-content:space-between; align-items:center; padding:10px 14px;
            border-radius:10px 10px 0 0; font-weight:700; font-size:15px;}
.card-head.base {background:#eef1f5; color:#374151;}
.card-head.dpo {background:#e6f4ea; color:#1d5e2c;}
.pill {font-size:12px; font-weight:600; padding:3px 10px; border-radius:999px; background:#fff; border:1px solid #d1d5db;
       font-family:ui-monospace,monospace;}
.card {border:1px solid #e5e7eb !important; border-radius:12px !important; gap:0 !important; overflow:hidden; background:#fff;}
.card .card-head {border-radius:0;}
.answer {min-height:260px; padding:6px 16px 12px; border:none !important;}
.answer .pending, .answer.pending {border:none !important;}
.tiles {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:10px 0 18px;}
.tile {padding:16px; border-radius:12px; border:1px solid #e5e7eb; background:#fafbf9;}
.tile-v {font-size:28px; font-weight:700; color:#2f4f24;}
.tile-l {font-size:14px; font-weight:600; margin-top:2px;}
.tile-s {font-size:12px; color:#6b7280;}
.res-title {font-size:18px; font-weight:700;}
table.res {width:100%; border-collapse:collapse; font-size:14px; margin-bottom:8px;}
table.res th {text-align:left; background:#f3f6f1; padding:8px 10px; border-bottom:2px solid #d5dfd0;}
table.res td {padding:8px 10px; border-bottom:1px solid #eceff1;}
.bar {position:relative; height:20px; background:#eef1f5; border-radius:6px; min-width:140px;}
.bar div {height:100%; background:#6aa25a; border-radius:6px;}
.bar span {position:absolute; left:8px; top:1px; font-size:12px; font-weight:600;}
.note {font-size:13px; color:#6b7280;}
.flow {display:flex; flex-wrap:wrap; align-items:stretch; gap:6px; margin:10px 0 18px;}
.step {flex:1 1 180px; padding:14px; border-radius:12px; font-size:13px; line-height:1.4;}
.step b {display:block; font-size:15px; margin-bottom:4px;}
.s1 {background:#e3eef8;} .s2 {background:#e6f4ea;} .s3 {background:#fcefdc;} .s4 {background:#efe7f6;}
.arrow {align-self:center; font-size:22px; color:#9ca3af;}
"""

THEME = gr.themes.Soft(primary_hue="green", secondary_hue="orange", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

with gr.Blocks(title="RLHF Comparison | IGDTUW") as demo:
    gr.HTML(header_html())
    with gr.Tab("Compare"):
        gr.HTML(status_html())
        with gr.Row(equal_height=True):
            prompt = gr.Textbox(label="Your prompt", lines=3, scale=5, autofocus=True,
                                placeholder="Ask anything, e.g. How do I stay motivated while studying?")
            go = gr.Button("Compare", variant="primary", scale=1, size="lg")
        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                max_tokens = gr.Slider(32, 300, value=200, step=8, label="Max new tokens")
                temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature")
                seed = gr.Number(value=42, label="Seed (blank = random)", precision=0)
        gr.Examples(DEMO_PROMPTS + ["what is 2 + 3 ?"], inputs=prompt, label="Try an example")
        verdict = gr.HTML(EMPTY_VERDICT)
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="card"):
                base_head = gr.HTML(card_head("Base TinyLlama", None, "base"))
                base_out = gr.Markdown(elem_classes="answer")
            with gr.Column(elem_classes="card"):
                dpo_head = gr.HTML(card_head("DPO-aligned TinyLlama", None, "dpo"))
                dpo_out = gr.Markdown(elem_classes="answer")
        with gr.Accordion("Session history", open=True):
            hist = gr.Dataframe(headers=["Prompt", "Base", "DPO", "Delta", "Winner"], interactive=False, wrap=True)
            with gr.Row():
                export = gr.Button("Export session to CSV", size="sm")
                export_file = gr.File(label="Exported file", height=60)
        outs = [verdict, base_head, base_out, dpo_head, dpo_out, hist]
        go.click(run, [prompt, max_tokens, temperature, seed], outs)
        prompt.submit(run, [prompt, max_tokens, temperature, seed], outs)
        export.click(export_history, None, export_file)
    with gr.Tab("Results"):
        gr.HTML(results_html())
    with gr.Tab("About the project"):
        gr.HTML(ABOUT_HTML)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)),
                                  theme=THEME, css=CSS)
