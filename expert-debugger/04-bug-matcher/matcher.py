"""Bug matcher: (root-cause analysis of a failed run, top-5 candidate bugs) -> {"bug": id | "NONE", "confidence", "why"}.

Qwen3-8B + QLoRA (Unsloth). The champion was reached in four data iterations, not by changing the model:
  v1  rejection sampling: keep only teacher trajectories whose final pick was right
  v2  cleaned data: STaR rationales (writer sees the gold, explains it from evidence; leaky or self-contradicting
      rationales filtered), one consistent policy for runs without logs, counterfactual augmentation
      (candidate order shuffled; truth removed -> target NONE), grouped validation split
  v3  2x data + out-of-fold ranker candidates (the matcher learns from the ranker's real mistakes) + a new
      platform slice; local 8B rationale writer instead of an API model (8B and 14B writers tied)
Serving: the matcher may override the ranker's top-1 only through a confidence gate tuned on validation.
In the full system it is the last model of the chain: it reads the DPO-tuned writer's root cause, the ranker's out-of-fold top-5, an
evidence probe (how many of each bug's known log lines appear in this run) and the deterministic graph facts of each candidate.
"""
from __future__ import annotations

import json
import os
import random
import re

PROMPT = """You match a finished root-cause analysis of ONE failed test run to the known bugs below.
Pick the bug that describes the SAME failure: same failing component, same origin and mechanism (what failed first and why).
A shared final symptom (timeout, hang, test FAIL, device lost, reset) is NOT enough: the origin must agree.
If no candidate shares the origin, answer NONE.

Root-cause analysis:
{rca}

Candidates (ranker order, ranker score, evidence probe, graph facts):
{cands}

Answer JSON only: {{"bug": "<id or NONE>", "confidence": 0-1, "why": "<one paragraph citing the deciding evidence>"}}"""


def render_candidates(cards: list[dict], probe: dict[str, str]) -> str:
    """One line per candidate; `facts` is the KG node's output, e.g. "same test: yes, same owner: no, filed before run: yes"."""
    return "\n".join(f"- {c['id']} [{c['score']:.2f}] {c['title']} | first fault: {c.get('first_fault', '?')} | "
                     f"signature lines found in this run: {probe.get(c['id'], 'unknown')} | graph facts: {c.get('facts', 'none')}"
                     for c in cards)


# ------------------------------------------------------------------ dataset recipe
def counterfactuals(row: dict, rng: random.Random) -> list[dict]:
    """Two cheap augmentations that taught the model the NONE answer and removed position bias."""
    out = []
    c = row["cands"][:]
    rng.shuffle(c)
    out.append({**row, "cands": c})
    if row["gold"] != "NONE":
        c2 = [x for x in row["cands"] if x["id"] != row["gold"]]
        out.append({**row, "cands": c2, "gold": "NONE",
                    "target": json.dumps({"bug": "NONE", "confidence": .3, "why": "No candidate shares the origin found in the analysis."})})
    return out


def star_rationale(row: dict, writer) -> str | None:
    """STaR-style rationale: the writer sees the gold answer and must justify it from the evidence only.
    Rejected when it leaks the answer without evidence, contradicts itself, or is boilerplate."""
    why = writer(f"Explain from the evidence why {row['gold']} is the right bug.\n{row['rca']}")
    if row["gold"] not in why or re.search(r"(?i)as (stated|given)|the answer is|cannot determine", why):
        return None
    return why


def to_chat(row: dict) -> dict:
    return {"messages": [{"role": "user", "content": PROMPT.format(rca=row["rca"], cands=render_candidates(row["cands"], row.get("probe", {})))},
                         {"role": "assistant", "content": row["target"]}]}


# ------------------------------------------------------------------ training (Unsloth QLoRA)
def train(rows: list[dict], out: str, base: str = "Qwen/Qwen3-8B", maxlen: int = 8192, epochs: int = 3, lr: float = 1e-4):
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(base, max_seq_length=maxlen, load_in_4bit=True, dtype=torch.float16)
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                             use_gradient_checkpointing="unsloth", random_state=0)
    def fmt(r):  # prompt/completion rows: the loss is on the answer tokens only
        m = to_chat(r)["messages"]
        return {"prompt": tok.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False),
                "completion": m[1]["content"] + tok.eos_token}
    tr = Dataset.from_list([fmt(r) for r in rows if r["split"] == "train"])
    va = Dataset.from_list([fmt(r) for r in rows if r["split"] == "val"])
    cfg = SFTConfig(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=epochs,
                    learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=.05, logging_steps=5,
                    eval_strategy="epoch", save_strategy="epoch", fp16=True, max_length=maxlen,
                    completion_only_loss=True, report_to=[])
    SFTTrainer(model=model, processing_class=tok, train_dataset=tr, eval_dataset=va, args=cfg).train()
    model.save_pretrained(f"{out}/adapter")


# ------------------------------------------------------------------ serving
def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def gated_pick(ranker_top: list[tuple[str, float]], answer: dict, tau: float = .7) -> str:
    """Override gate: the ranker's top-1 stands unless the matcher names another candidate with confidence >= tau.
    A NONE answer is surfaced as a note but never removes a confident ranker link."""
    pick, conf = answer.get("bug"), float(answer.get("confidence") or 0)
    ids = [b for b, _ in ranker_top]
    if pick in ids and conf >= tau:
        return pick
    return ids[0] if ids else "NONE"
