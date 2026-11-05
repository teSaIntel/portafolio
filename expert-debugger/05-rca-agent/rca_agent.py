"""RCA agent: a tool-using LLM that reads a failed run's evidence and writes a root-cause analysis, without
being told the answer. Evaluated end to end: its RCA goes through the bug matcher, so "right" means the RCA
led to the correct bug, or to a correct NONE (new failure, root cause stated in the RCA).

Pieces:
  tools        read-only, run-scoped, time-cut: run context, log digest, kernel errors / search, structured facts,
               signature cross-match against candidate bugs. Served over MCP (stdio) so an API model and the
               local model use the exact same tools.
  agent loop   chat with tool schemas; the model emits <tool_call>{json}</tool_call>; the loop executes,
               truncates results (4k chars) and feeds them back; max N turns; final answer is JSON.
  writer SFT   multi-turn behaviour cloning on the teacher's (API model) tool-calling transcripts, loss on
               assistant tokens only, 16k context with chunked attention to fit a 24 GB GPU.
  best-of-N    sample N RCAs at T=0.7, score with a critic (matcher agreement + evidence grounding), keep
               preference pairs -> DPO.
"""
from __future__ import annotations

import json
import math
import random
import re
import time

SYSTEM = """You are a hardware/software validation debugger. Find the root cause of ONE failed test run.
Use the tools to read the evidence. Identify the failing component and the FIRST fault in the chain, not the final symptom.
When done, answer JSON: {"root_cause": "...", "component": "...", "first_fault_line": "<verbatim line>",
"evidence": ["<verbatim lines>"], "known_bug_hint": "<id or NONE>", "confidence": 0-1}"""

TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": list(p)}}}
    for n, d, p in [
        ("run_context", "Metadata of the run: test, domain, host, timing, failure reason.", {}),
        ("log_digest", "Deduplicated, normalised error lines of the execution log.", {}),
        ("kernel_errors", "Kernel/driver error lines in the run window.", {}),
        ("kernel_search", "Search kernel log lines in the run window.", {"pattern": {"type": "string"}}),
        ("facts", "Structured facts extracted from the run (firmware, config, counters).", {}),
        ("cross_match", "Which signature lines of a candidate bug appear in this run.", {"bug": {"type": "string"}}),
    ]
]


# ------------------------------------------------------------------ agent loop
def parse_calls(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            out.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return out


def run_agent(generate, tools, run_id: str, max_turns: int = 8, cap: int = 4000, temp: float = 0.0) -> dict:
    """generate(messages, temp) -> assistant text. tools(name, args) -> str (already run-scoped and time-cut)."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Debug failed run {run_id}."}]
    t0, calls = time.time(), 0
    for _ in range(max_turns):
        text = generate(msgs, temp)
        msgs.append({"role": "assistant", "content": text})
        cs = parse_calls(text)
        if not cs:
            break
        for c in cs:
            calls += 1
            res = tools(c.get("name"), c.get("arguments") or {})
            msgs.append({"role": "tool", "name": c.get("name"), "content": res[:cap]})
    m = re.search(r"\{.*\}", msgs[-1]["content"], re.S)
    try:
        answer = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        answer = None
    return {"run": run_id, "answer": answer, "tool_calls": calls, "seconds": round(time.time() - t0, 1), "messages": msgs}


# ------------------------------------------------------------------ writer SFT (assistant-only loss, long context)
def encode(tok, messages, maxlen=16384):
    """Token ids + labels where only assistant turns are learned (tool outputs and prompts are context)."""
    ids, lab = [], []
    for i in range(1, len(messages) + 1):
        full = tok.apply_chat_template(messages[:i], tools=TOOLS, tokenize=False, enable_thinking=False)
        new = tok(full, add_special_tokens=False)["input_ids"][len(ids):]
        learn = messages[i - 1]["role"] == "assistant"
        ids += new
        lab += new if learn else [-100] * len(new)
    return (ids, lab) if len(ids) <= maxlen else None


def chunked_causal_attention(q, k, v, chunk=512):
    """Exact causal attention computed in query chunks: peak memory O(chunk * L) instead of O(L^2).
    Lets 16k-token trajectories train on a 24 GB card with QLoRA."""
    import torch
    out, L = [], q.shape[-2]
    scale = 1 / math.sqrt(q.shape[-1])
    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        att = (q[..., s:e, :] @ k[..., :e, :].transpose(-1, -2)) * scale
        mask = torch.ones(e - s, e, dtype=torch.bool, device=q.device).tril(diagonal=s)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        out.append(att @ v[..., :e, :])
    return torch.cat(out, -2)


def train_writer(model, tok, trajectories, epochs=2, lr=1e-4, ga=8, ckpt_every=10, out="writer", skip=()):
    """Custom loop: per-row forward, logits only on learned positions (full-vocab logits over 16k tokens OOM),
    cosine schedule, periodic checkpoints and a row skip-list (resume after a device fault on one row)."""
    import torch
    enc = [e for e in (encode(tok, t["messages"]) for t in trajectories) if e]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=.01)
    total = math.ceil(len(enc) * epochs / ga)
    warm = max(1, int(.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    base, step, acc, rng = model.get_base_model(), 0, 0, random.Random(0)
    model.train()
    while step < total:
        order = list(range(len(enc)))
        rng.shuffle(order)
        for i in order:
            if i in skip:
                continue
            ids, lab = enc[i]
            x = torch.tensor([ids], device=model.device)
            y = torch.tensor([lab], device=model.device)[:, 1:]
            h = base.model(input_ids=x).last_hidden_state[:, :-1]
            sel = y != -100
            loss = torch.nn.functional.cross_entropy(base.lm_head(h[sel]).float(), y[sel])
            (loss / ga).backward()
            acc += 1
            if acc % ga == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % ckpt_every == 0:
                    model.save_pretrained(f"{out}/step{step}")
                if step >= total:
                    break
    model.save_pretrained(f"{out}/adapter")


# ------------------------------------------------------------------ best-of-N + critic -> DPO pairs
def grounding(answer: dict, tool_text: str) -> float:
    """Share of cited evidence lines that really appear in the tool outputs (penalises invented log lines)."""
    ev = (answer or {}).get("evidence") or []
    return sum(1 for e in ev if e and e in tool_text) / len(ev) if ev else 0.0


def critic(sample: dict, matcher_right: bool) -> float:
    tool_text = "\n".join(m["content"] for m in sample["messages"] if m["role"] == "tool")
    return 1.0 * matcher_right + .5 * grounding(sample["answer"], tool_text) - .1 * (sample["answer"] is None)


def preference_pairs(samples_by_run: dict[str, list[tuple[dict, float]]], margin: float = .5) -> list[dict]:
    pairs = []
    for run, ss in samples_by_run.items():
        ss = sorted(ss, key=lambda x: -x[1])
        if len(ss) >= 2 and ss[0][1] - ss[-1][1] >= margin:
            pairs.append({"run": run, "chosen": ss[0][0]["messages"], "rejected": ss[-1][0]["messages"]})
    return pairs
