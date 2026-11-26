"""RCA writer: the local 8B model that reads a failed run's evidence and writes the root cause as an ordered error chain
(each link a verbatim log line). It is judged end to end: its root cause goes through the bug matcher, so "right" means it
led to the correct bug, or to a correct NONE (a new failure whose root cause is stated).

Pieces:
  tool layer     read-only, run-scoped, time-cut tools (run context, log digest, kernel errors / search, typed facts,
                 signature cross-match). Served over MCP so an API teacher and the local writer use the same tools.
                 They build the evidence digest the writer reads (`build_digest`), together with the expert preload.
  writer         one pass from the digest to a JSON answer. The tool loop (`run_agent`) is kept for the teacher's
                 transcripts and for tool-using variants; the champion answers from the digest.
  stage 1 BC     behaviour cloning (SFT) on expert transcripts, loss on assistant tokens only, 16k context.
  stage 2 DPO    rounds of: sample N answers per run -> critic (matcher agreement + evidence grounding) -> chosen/rejected
                 pairs -> DPO on top of the previous adapter. The reference model is the same model with the adapter off.
  repair         deterministic JSON salvage for the rare degenerate output (repetition loops, unclosed brackets).
"""
from __future__ import annotations

import json
import math
import random
import re
import time

SYSTEM = """You are a hardware/software validation debugger. Find the root cause of ONE failed test run.
Identify the failing component and the FIRST fault in the chain, not the final symptom.
Answer JSON: {"failure_mechanism": "...", "component": "...", "error_chain": ["<verbatim lines, causal order>"],
"evidence": ["<verbatim lines>"], "known_bug_hint": "<id or NONE>", "confidence": 0-1}"""

TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": list(p)}}}
    for n, d, p in [
        ("run_context", "Metadata of the run: test, domain, host, timing, failure reason.", {}),
        ("log_digest", "Deduplicated, normalised error lines of the execution log.", {}),
        ("kernel_errors", "Kernel/driver error lines of the whole run, grouped by template.", {}),
        ("kernel_search", "Search the kernel log of the whole run.", {"pattern": {"type": "string"}}),
        ("facts", "Typed facts extracted from the run without a model (firmware, config, counters).", {}),
        ("cross_match", "Which signature lines of a candidate bug appear in this run.", {"bug": {"type": "string"}}),
    ]
]


# ------------------------------------------------------------------ evidence digest and writer pass
def build_digest(tools, run_id: str, preload: dict | None = None, cap: int = 4000) -> str:
    """Runs the deterministic tools once and renders what the writer reads. `preload` is the expert preload
    (diff against a known-good run, test card)."""
    parts = [f"## {n}\n{tools(n, {})[:cap]}" for n in ("run_context", "log_digest", "kernel_errors", "facts")]
    if preload:
        parts.append("## new versus a known-good run\n" + "\n".join(preload.get("new_vs_good", [])[:40]))
    return f"Debug failed run {run_id}.\n\n" + "\n\n".join(parts)


def write(generate, digest: str, temp: float = 0.0) -> dict | None:
    """One pass: digest in, JSON root cause out."""
    text = generate([{"role": "system", "content": SYSTEM}, {"role": "user", "content": digest}], temp)
    return repair(text)


# ------------------------------------------------------------------ tool loop (teacher transcripts, tool-using variants)
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
    return {"run": run_id, "answer": repair(msgs[-1]["content"]), "tool_calls": calls,
            "seconds": round(time.time() - t0, 1), "messages": msgs}


# ------------------------------------------------------------------ deterministic repair (no model call)
def repair(text: str) -> dict | None:
    """Parse the writer's JSON; if it is degenerate, cut at the first unit repeated 3+ times back to back, close strings and
    brackets, and keep the result only if it parses and has a failure_mechanism. Evidence lists are de-duplicated."""
    m = re.search(r"\{.*", text or "", re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, _cut_repeats(s)):
        for tail in ("", "}", "]}", '"}', '"]}'):
            try:
                a = json.loads(cand + tail)
            except json.JSONDecodeError:
                continue
            if isinstance(a, dict) and a.get("failure_mechanism"):
                for k in ("evidence", "error_chain"):
                    if isinstance(a.get(k), list):
                        a[k] = list(dict.fromkeys(a[k]))
                return a
    return None


def _cut_repeats(s: str, unit: int = 12) -> str:
    m = re.search(r"(.{%d,}?)\1{2,}" % unit, s, re.S)
    return s[:m.end(1)] if m else s


# ------------------------------------------------------------------ stage 1: behaviour cloning (assistant-only loss, long context)
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
    Lets 16k-token trajectories train with QLoRA on a single data-centre GPU."""
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


def _cosine(opt, total, warm_frac):
    import torch
    warm = max(1, int(warm_frac * total))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def _logp(model, ids, lab):
    """(sum log p, mean NLL) of the learned tokens; logits only at learned positions (full-vocab logits over 16k tokens OOM)."""
    import torch
    base = model.get_base_model()
    x = torch.tensor([ids], device=model.device)
    y = torch.tensor([lab], device=model.device)[:, 1:]
    h = base.model(input_ids=x).last_hidden_state[:, :-1]
    sel = y != -100
    lp = torch.log_softmax(base.lm_head(h[sel]).float(), -1).gather(-1, y[sel][:, None]).squeeze(-1)
    return lp.sum(), -lp.mean()


def train_writer(model, tok, trajectories, epochs=2, lr=1e-4, ga=8, ckpt_every=10, out="writer", skip=()):
    """Stage 1 (behaviour cloning). Custom loop: one sequence per step, gradient accumulation, cosine schedule,
    periodic checkpoints and a row skip-list (resume after a device fault on one row)."""
    import torch
    enc = [e for e in (encode(tok, t["messages"]) for t in trajectories) if e]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    total = math.ceil(len(enc) * epochs / ga)
    sched, step, acc, rng = _cosine(opt, total, .05), 0, 0, random.Random(0)
    model.train()
    while step < total:
        order = list(range(len(enc)))
        rng.shuffle(order)
        for i in order:
            if i in skip:
                continue
            loss = _logp(model, *enc[i])[1]
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


# ------------------------------------------------------------------ critic and preference pairs
def grounding(answer: dict, tool_text: str) -> float:
    """Share of cited evidence lines that really appear in the evidence given (penalises invented log lines)."""
    ev = (answer or {}).get("evidence") or []
    return sum(1 for e in ev if e and e in tool_text) / len(ev) if ev else 0.0


def critic(sample: dict, matcher_right: bool) -> float:
    given = "\n".join(m["content"] for m in sample["messages"] if m["role"] in ("tool", "user"))
    return 1.0 * matcher_right + .5 * grounding(sample["answer"], given) - .1 * (sample["answer"] is None)


def preference_pairs(samples_by_run: dict[str, list[tuple[dict, float]]], margin: float = .5) -> list[dict]:
    """Best vs worst sample of the same run, kept only when the critic scores differ by at least `margin`."""
    pairs = []
    for run, ss in samples_by_run.items():
        ss = sorted(ss, key=lambda x: -x[1])
        if len(ss) >= 2 and ss[0][1] - ss[-1][1] >= margin:
            pairs.append({"run": run, "chosen": ss[0][0]["messages"], "rejected": ss[-1][0]["messages"]})
    return pairs


# ------------------------------------------------------------------ stage 2: DPO
def dpo_weight(margin: float, beta: float) -> float:
    """d/dmargin of -logsigmoid(beta * margin) is -beta * w with w = sigmoid(-beta * margin)."""
    return 1 / (1 + math.exp(beta * margin))


def train_dpo(model, tok, pairs, beta=.1, alpha=.5, epochs=2, lr=1e-4, ga=8, out="writer-dpo"):
    """Direct Preference Optimization of the writer on critic-ranked pairs.
        loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))) + alpha * NLL(chosen)
    pi_* are sequence log-probs of the learned (assistant) tokens under the current adapter; ref_* are the same with the adapter
    disabled (the starting model). The NLL term on the chosen answer keeps it from drifting away from valid JSON.
    Two 16k-token policy graphs do not fit in memory, so the weight w = sigmoid(-beta * margin) is computed first without
    gradients and each side is back-propagated alone with its exact coefficient (-beta*w for chosen, +beta*w for rejected)."""
    import torch
    enc = [(encode(tok, p["chosen"]), encode(tok, p["rejected"])) for p in pairs]
    enc = [(c, r) for c, r in enc if c and r]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    total = math.ceil(len(enc) * epochs / ga)
    sched, step, acc, rng = _cosine(opt, total, .1), 0, 0, random.Random(0)
    model.train()
    while step < total:
        order = list(range(len(enc)))
        rng.shuffle(order)
        for i in order:
            (ci, cl), (ri, rl) = enc[i]
            with torch.no_grad():
                with model.disable_adapter():
                    ref_c, ref_r = _logp(model, ci, cl)[0].item(), _logp(model, ri, rl)[0].item()
                pi_c, pi_r = _logp(model, ci, cl)[0].item(), _logp(model, ri, rl)[0].item()
            w = dpo_weight((pi_c - ref_c) - (pi_r - ref_r), beta)
            lp, nll = _logp(model, ci, cl)
            ((-beta * w * lp + alpha * nll) / ga).backward()          # chosen side
            lp, _ = _logp(model, ri, rl)
            ((beta * w * lp) / ga).backward()                          # rejected side
            acc += 1
            if acc % ga == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step >= total:
                    break
    model.save_pretrained(f"{out}/adapter")


def dpo_rounds(model, tok, runs, sample, judge, rounds=3, n=4, **kw):
    """Self-improvement loop: sample n answers per run from the current writer (T=0.7), score them with the critic
    (judge(sample, run) -> matcher agrees?), build pairs, run DPO, repeat. Each round starts from the previous adapter."""
    for k in range(rounds):
        by_run = {r["run"]: [(s, critic(s, judge(s, r))) for s in (sample(model, r) for _ in range(n))] for r in runs}
        train_dpo(model, tok, preference_pairs(by_run), out=f"writer-dpo-r{k + 1}", **kw)
