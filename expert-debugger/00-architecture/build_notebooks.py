"""Builds the Colab notebooks (self-contained: each inlines the modules it needs with %%writefile).
python build_notebooks.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
read = lambda p: open(os.path.join(ROOT, p)).read()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def inline(path):
    return code(f"%%writefile {os.path.basename(path)}\n" + read(path))


def nb(folder, name, cells, gpu=False):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
            "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    with open(os.path.join(ROOT, folder, name), "w") as fh:
        json.dump({"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}, fh, indent=1)
    print(folder, name, len(cells), "cells")


SYN = inline("00-architecture/synthetic.py")
DATA_NOTE = md("> All data in this notebook is **synthetic** (generated below). The real system runs on private validation data; "
               "the code paths are the same, the numbers in the README are from the real data.")
SETUP_CPU = code("!pip -q install scikit-learn matplotlib numpy scipy")

# ---------------------------------------------------------------- 01 classifier (production champion)
nb("01-failure-ranker", "failure_ranker_colab.ipynb", [
    md("# Bug classifier: the production \"suggested bugs\" champion (no LLM)\n"
       "Stage 1 multinomial LR over signature phrases + metadata + text → top-5 candidates → stage 2 pair re-ranker "
       "(pair features, base posterior, temporal burst, kNN votes, vendor hard rule) → conformal open-set router "
       "(LINK / SIBLINGS / NEW) → cold-start text rule for bugs with no training runs → explanation. "
       "It also produces the leak-free out-of-fold top-5 shortlist that the root-cause writer and matcher consume. "
       "Lifecycle: champion/challenger, forward ledger, guarded recalibration."),
    DATA_NOTE, SETUP_CPU, SYN, inline("01-failure-ranker/ranker.py"), inline("01-failure-ranker/classifier.py"),
    md("## 1. Train on the past, calibrate on the next window, test on bugs held out by group"),
    code("""
from synthetic import make_world, split
from classifier import *
bugs, runs = make_world(n_bugs=60, n_runs=2000, seed=2)
runs = split(runs, bugs, seed=2)
tr = sorted([r for r in runs if r["split"] == "train"], key=lambda r: r["start"]); test = [r for r in runs if r["split"] == "test"]
train, calib = tr[: int(.8 * len(tr))], tr[int(.8 * len(tr)):]
champ = Champion(K=5).fit(train, bugs)
trained = {r["label"] for r in train}
cb = [r for r in calib if r["label"] != "NONE"]
router = OpenSetRouter(alpha=.1).calibrate([champ.rank(r) for r in cb], [r["label"] for r in cb])
cold = TextColdStart().fit(bugs, trained).calibrate([r for r in calib if r["label"] in trained])
unseen = {r["label"] for r in test} - trained - {"NONE"}
metrics, routes = evaluate(champ, router, cold, test, unseen)
print(metrics); print(routes, "| cold-start theta", round(cold.theta, 3))
"""),
    md("## 2. What a person sees: top 3 (never hidden), the route and why"),
    code("""
r = next(r for r in test if r["label"] in trained - {"NONE"})
d = decide(champ, router, cold, r)
print("truth", r["label"]); print({k: d[k] for k in ("route", "top3", "set")}); print("why:", d["why"])
"""),
    md("## 3. Override rule on top of an agent verdict (applied only when the agent ran)"),
    code("""
ranked = champ.rank(r)
print(override_rule(ranked, {"bug": ranked[1][0], "confidence": .75}, probe={ranked[1][0]: .5, ranked[0][0]: .2}))
print(override_rule(ranked, {"bug": ranked[1][0], "confidence": .75}, probe={ranked[1][0]: 0}))
"""),
    md("## 4. Leak-free shortlist for the LLM stages\nThe writer and the matcher receive the top-5 from a ranker refit without the "
       "fold's own labels (out-of-fold). Recall@5 is the ceiling for everything downstream."),
    code("""
sl = shortlist_oof(train + calib, bugs, folds=5, K=5)
print("recall@1 / recall@5 on the train window (out-of-fold):", recall_at(sl, train + calib))
"""),
    md("## 5. Lifecycle: forward ledger → drift alarm → guarded recalibration"),
    code("""
import random
led = Ledger(); rng = random.Random(0)
fwd = [r for r in test if r["label"] in trained - {"NONE"}][:60]
for x in fwd:
    d = decide(champ, router, cold, x); led.record(x["run"], d["route"], d["bug"])
led.resolve({x["run"]: (x["label"] if rng.random() > .35 else "BUG-999") for x in fwd})   # simulate label drift
print(led.forward(), "recalibrate?", led.needs_recalibration())
new_router = guarded_recalibrate(router, [champ.rank(x) for x in cb], [x["label"] for x in cb],
                                 [champ.rank(x) for x in fwd], [led.rows[x["run"]]["truth"] for x in fwd])
print("alpha", router.alpha, "->", new_router.alpha)
reg = Registry(); print(reg.consider("champion-1", .81, {"C2": .1}), reg.consider("challenger", .80, {"C2": 1}))
"""),
])

# ---------------------------------------------------------------- 02 detector
nb("02-test-issue-detector", "issue_detector_colab.ipynb", [
    md("# Test-content / infrastructure issue detector\nFlags failures that are *not* product bugs, only at ≥90 % precision "
       "(a false positive would hide a real bug). Two arms: stored, precision-pruned rules (shown by name in the UI) and a "
       "TF-IDF + categorical logistic regression for what the rules miss."),
    DATA_NOTE, SETUP_CPU, SYN, inline("02-test-issue-detector/detector.py"),
    code("""
import random
from synthetic import make_world
from detector import IssueDetector, Rule, RuleTriage, Triage
bugs, runs = make_world(n_bugs=40, n_runs=2000, p_none=.25, seed=2)
rng = random.Random(0)
for r in runs:
    r["has_logs"] = rng.random() > .1
    if r["label"] != "NONE" and rng.random() < .08:      # some product bugs also log infrastructure-looking lines
        r["lines"].append("ERROR: node unreachable")
y = [int(r["label"] == "NONE") for r in runs]
groups = [r["label"] if r["label"] != "NONE" else r["family"] for r in runs]
det = IssueDetector().fit(runs, y, groups)
print(f"OOF average precision {det.oof_ap:.2f} | recall at P>=0.9: {det.oof_recall:.2f} | threshold {det.tau:.2f}")
"""),
    md("## Stored rules, pruned by measured precision\nA candidate rule survives only if its precision on labelled runs is at least 90%. "
       "Rules carry a not-contains guard against product signatures. The generic timeout rule below has no guard and also matches real bugs, so pruning drops it."),
    code("""
PRODUCT = ["engine reset", "page fault", "device lost", "link training", "ecc", "watchdog", "crc mismatch", "thermal", "banned", "fence", "first fault"]
candidates = [
    Rule("node unreachable", "infra", ["node unreachable"], PRODUCT),            # "and not contains a product signature"
    Rule("image not pullable", "infra", ["image pull failed"], PRODUCT),
    Rule("missing test binary", "infra", ["missing test binary"], PRODUCT),
    Rule("test script assertion", "test_content", ["assertion in test script"], PRODUCT),
    Rule("any timeout", "infra", ["timeout waiting for"]),                      # sounds right, also matches real bugs
]
rt = RuleTriage(candidates).prune(runs, y)
print("kept:", [ru.name for ru in rt.rules])
tri = Triage(rt, det)
flags = [(tri.flag(r), r["label"] == "NONE") for r in runs]
caught = [(why, pos) for (hit, why), pos in flags if hit]
print(f"flagged {len(caught)} | precision {sum(p for _, p in caught) / len(caught):.1%} | by rule: "
      f"{sum(w.startswith('rule') for w, _ in caught)}, by model: {sum(w.startswith('model') for w, _ in caught)}")
"""),
])

# ---------------------------------------------------------------- 03 knowledge (KB + KG)
nb("03-knowledge-layer", "knowledge_layer_colab.ipynb", [
    md("# Knowledge layer: knowledge base + knowledge graph\n"
       "**KB**: clean → quality gate → hash-idempotent ingest → chunk → incremental embed → BM25 / dense / RRF search, grouped by bug, "
       "time-cut, own run excluded. **KG**: extract → purify (counted rules) → load (hash diff) → openCypher projection; "
       "domain / bug / expert views, per-candidate facts for the matcher and leak-guarded personalised PageRank. Production runs both in one Postgres (pgvector + Apache AGE, see schema.sql)."),
    DATA_NOTE, code("!pip -q install numpy sentence-transformers"), SYN, inline("03-knowledge-layer/knowledge.py"),
    inline("03-knowledge-layer/kb_store.py"), inline("03-knowledge-layer/kg.py"),
    md("## 1. Synthetic bug reports and discussion threads"),
    code("""
import random
from datetime import timedelta
from synthetic import make_world
from kb_store import Doc, KnowledgeBase
bugs, runs = make_world(n_bugs=60, n_runs=1000, seed=3)
rng = random.Random(0)
for r in runs: r["host"] = f"node{rng.randrange(20):02d}"; r["test"] = f"{r['domain']}/{r['family']}"
docs = []
for b in bugs:
    docs.append(Doc("report", b["id"], None, b["created"], f"<p>{b['title']}</p> Symptom: {'. '.join(b['signature'])}. "
                    f"Root cause: {b['root_cause']} in {b['component']}_driver_{b['domain']}. See https://tracker/x/{b['id']}"))
    for r in [r for r in runs if r["label"] == b["id"]][:3]:
        docs.append(Doc("thread_comment", b["id"], r["run"], r["start"] + timedelta(days=1), f"Seen again on {r['run']}: " + "; ".join(r["lines"][:3])))
    docs.append(Doc("thread_comment", b["id"], None, b["created"] + timedelta(days=2), "+1"))
kb = KnowledgeBase()
print(kb.ingest(docs)); print("re-ingest:", kb.ingest(docs)); print(kb.stats())
"""),
    md("## 2. Search: time cut + own-run exclusion, then hybrid"),
    code("""
lab = [r for r in runs if r["label"] != "NONE"]
def recall(kb, k, mode="fuse"):
    return round(sum(r["label"] in [h["bug"] for h in kb.search(" ".join(r["lines"]), exclude_run=r["run"], k=k, mode=mode, cutoff=r["start"])]
                     for r in lab) / len(lab), 3)
print({k: recall(kb, k, "bm25") for k in (1, 5)}, "BM25")
"""),
    code("""
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
kb.embed = lambda xs: st.encode(xs, normalize_embeddings=True); print("embedded", kb.embed_new())
print({k: recall(kb, k, "fuse") for k in (1, 5)}, "BM25 + dense (RRF)")
"""),
    md("## 3. Knowledge graph: build, purify report, incremental reload, openCypher projection"),
    code("""
from kg import extract, purify, load, project_cypher, KG
nodes, edges = extract(runs, bugs, kb)
nodes, edges, report = purify(nodes, edges)
store = {}
print(report); print("first load", load(store, nodes, edges), "| second load", load(store, nodes, edges))
print(len(nodes), "nodes", len(edges), "edges |", len(project_cypher(nodes, edges)), "Cypher statements"); print(project_cypher(nodes, edges)[2][:200], "...")
"""),
    md("## 4. Read side: bug and domain views, graph-scoped expert search, leak-guarded PPR"),
    code("""
g = KG(nodes, edges, {f"Run:{r['run']}": r["start"] for r in runs})
r = max(lab, key=lambda x: x["start"])
print("truth", r["label"]); print("bug view", g.bug_view(r["label"], cutoff=r["start"], hide_run=f"Run:{r['run']}"))
print("expert", g.expert(r, kb, cutoff=r["start"]))
hit = sum(r["label"] in g.expert(r, kb, cutoff=r["start"]) for r in lab) / len(lab)
ppr = lambda r, **kw: list(g.ppr({f"Test:{r['test']}": 1, f"Host:{r['host']}": .5}, **kw))[:5]
leaky = sum(r["label"] in ppr(r) for r in lab[:200]) / 200
clean = sum(r["label"] in ppr(r, cutoff=r["start"], hide_run=f"Run:{r['run']}") for r in lab[:200]) / 200
print(f"expert top-5 {hit:.3f} | PPR recall@5 leaky {leaky:.3f} vs leak-free {clean:.3f}")
"""),
    md("## 5. Candidate facts for the matcher (KG node)\nDeterministic facts per candidate bug: shared test, host, owner with the bug's earlier "
       "linked runs. The right bug shares them far more often than a wrong candidate, so they are strong evidence for the matcher."),
    code("""
import random
from collections import Counter
rng2 = random.Random(1); stat = {"gold": Counter(), "wrong": Counter()}; n = {"gold": 0, "wrong": 0}
for r in rng2.sample(lab, 150):
    for b in {r["label"], rng2.choice([x["id"] for x in bugs if x["id"] != r["label"]])}:
        k = "gold" if b == r["label"] else "wrong"
        f = g.candidate_facts(r, b, cutoff=r["start"]); n[k] += 1
        for name in ("same_test", "same_host", "same_owner"): stat[k][name] += f[name]
print({k: {m: f"{v / n[k]:.0%}" for m, v in stat[k].items()} for k in stat})
"""),
])

# ---------------------------------------------------------------- 04 matcher (GPU)
nb("04-bug-matcher", "bug_matcher_colab.ipynb", [
    md("# Bug matcher: QLoRA fine-tune\nQwen3 + Unsloth. Production uses **Qwen3-8B** (one 24 GB GPU, 8k context); this notebook defaults "
       "to **Qwen3-0.6B** so it fits a free Colab T4. Input: root-cause analysis + ranker top-5 candidates with evidence probe and graph facts. Output: JSON pick or NONE."),
    DATA_NOTE, code("!pip -q install unsloth trl datasets scikit-learn"), SYN, inline("01-failure-ranker/ranker.py"),
    inline("04-bug-matcher/matcher.py"),
    md("## 1. Dataset: ranker candidates → prompts, with counterfactual augmentation"),
    code("""
import json, random
from synthetic import make_world, split
from ranker import Ranker
from matcher import counterfactuals, PROMPT, render_candidates, train, parse, gated_pick
bugs, runs = make_world(n_bugs=40, n_runs=900, seed=4); runs = split(runs, bugs)
B = {b["id"]: b for b in bugs}
rk = Ranker().fit([r for r in runs if r["split"] == "train"], bugs)
rng = random.Random(0); rows = []
for r in runs:
    top = rk.rank(r, 5)
    if not top: continue
    cands = [{**B[b], "score": s, "first_fault": B[b]["signature"][-1],
              "facts": f"same test family: {'yes' if B[b]['family'] == r['family'] else 'no'}, same owner: {'yes' if B[b]['owner'] == r['owner'] else 'no'}"}
             for b, s in top]
    gold = r["label"] if r["label"] in [c["id"] for c in cands] else "NONE"
    rca = "Evidence: " + " | ".join(l for l in r["lines"] if ":" in l)     # stand-in for the RCA writer's output
    row = {"split": r["split"], "rca": rca, "cands": cands, "gold": gold, "label": r["label"], "top": top,
           "target": json.dumps({"bug": gold, "confidence": .9 if gold != "NONE" else .3, "why": "origin matches" if gold != "NONE" else "no shared origin"})}
    rows += [row] + (counterfactuals(row, rng) if r["split"] == "train" else [])
for r in rows:
    if r["split"] == "test" and rng.random() < .3: r["split"] = "val"
print(len(rows), sum(r["split"] == "train" for r in rows), "train rows")
"""),
    md("## 2. Train (≈10 min on a T4 with 0.6B)"),
    code("""
BASE = "unsloth/Qwen3-0.6B"   # production: Qwen/Qwen3-8B on a 24 GB GPU
train(rows, "matcher-out", base=BASE, maxlen=2048, epochs=2, lr=2e-4)
"""),
    md("## 3. Evaluate: matcher alone vs matcher behind the confidence gate"),
    code("""
import torch
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("matcher-out/adapter", max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)
def ask(row):
    p = PROMPT.format(rca=row["rca"], cands=render_candidates(row["cands"], {}))   # same prompt as training, probe unknown here
    x = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to("cuda")
    y = model.generate(x, max_new_tokens=120, do_sample=False)
    return parse(tok.decode(y[0, x.shape[1]:], skip_special_tokens=True))
test = [r for r in rows if r["split"] == "test" and r["label"] != "NONE"]
ans = [ask(r) for r in test]
ranker = sum(r["top"][0][0] == r["label"] for r in test) / len(test)
alone = sum(a.get("bug") == r["label"] for a, r in zip(ans, test)) / len(test)
gate = sum(gated_pick(r["top"], a) == r["label"] for a, r in zip(ans, test)) / len(test)
print(f"ranker {ranker:.1%} | matcher {alone:.1%} | matcher + gate {gate:.1%}")
"""),
], gpu=True)

# ---------------------------------------------------------------- 05 rca writer (GPU)
nb("05-rca-writer", "rca_writer_colab.ipynb", [
    md("# RCA writer: behaviour cloning, then DPO\nThe local writer reads an evidence digest built by read-only tools (it never sees the answer), "
       "writes a JSON root cause, and is judged end to end through the matcher. Shows: the tool layer and digest, assistant-only-loss SFT "
       "(behaviour cloning), best-of-N sampling -> critic -> preference pairs, a **DPO** step with the reference model = adapter off, "
       "and the deterministic JSON repair. Defaults to Qwen3-0.6B for a free T4; production is Qwen3-8B at 16k context."),
    DATA_NOTE, code("!pip -q install unsloth trl datasets"), SYN, inline("05-rca-writer/rca_writer.py"),
    md("## 1. Read-only, run-scoped tools over a synthetic run"),
    code("""
import json, random
from synthetic import make_world
from rca_writer import write, build_digest, run_agent, TOOLS, SYSTEM, critic, preference_pairs, train_writer, train_dpo, dpo_weight, repair
bugs, runs = make_world(n_bugs=20, n_runs=200, seed=5)
B = {b["id"]: b for b in bugs}
def tools_for(run):
    def call(name, args):
        if name == "run_context": return json.dumps({k: str(run[k]) for k in ("run", "domain", "family", "start")})
        if name == "log_digest": return "\\n".join(run["lines"])
        if name == "kernel_errors": return "\\n".join(l for l in run["lines"] if ":" in l)
        if name == "kernel_search": return "\\n".join(l for l in run["lines"] if args.get("pattern", "").lower() in l.lower())
        if name == "facts": return json.dumps({"host_class": "gpu-node", "fw": "1.2.3"})
        if name == "cross_match":
            b = B.get(args.get("bug"), {"signature": []}); return json.dumps({s: s in run["lines"] for s in b["signature"]})
        return "unknown tool"
    return call
run = next(r for r in runs if r["label"] != "NONE")
print(build_digest(tools_for(run), run["run"])[:600])
"""),
    md("## 2. Zero-shot writer with a local model: digest in, JSON out"),
    code("""
import torch
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("unsloth/Qwen3-0.6B", max_seq_length=8192, load_in_4bit=True)
FastLanguageModel.for_inference(model)
def generate(msgs, temp):
    x = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to("cuda")
    y = model.generate(x, max_new_tokens=400, do_sample=temp > 0, temperature=temp or None)
    return tok.decode(y[0, x.shape[1]:], skip_special_tokens=True)
print(json.dumps(write(generate, build_digest(tools_for(run), run["run"])), indent=1))
"""),
    md("## 3. Stage 1: behaviour cloning on expert transcripts (assistant tokens only)\nHere the teacher is scripted; in production it is "
       "an expert model's real transcripts (~600 trajectories, up to 16k tokens)."),
    code("""
def teacher(run):
    b = B.get(run["label"]); ff = next((l for l in run["lines"] if ":" in l), "")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_digest(tools_for(run), run["run"])},
            {"role": "assistant", "content": json.dumps({"failure_mechanism": b["root_cause"] if b else "infrastructure",
                                                         "component": b["component"] if b else "infra", "error_chain": [ff],
                                                         "evidence": [ff], "known_bug_hint": run["label"], "confidence": .8})}]
traj = [{"messages": teacher(r)} for r in runs[:120]]
FastLanguageModel.for_training(model)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
train_writer(model, tok, traj, epochs=1, lr=2e-4, ga=4, out="writer-bc")
"""),
    md("## 4. Best-of-N -> critic -> preference pairs\nSample 4 answers per run at T=0.7. The critic rewards agreement with the matcher "
       "(here: the right bug hint) plus evidence that really appears in the digest. Best vs worst becomes a (chosen, rejected) pair."),
    code("""
FastLanguageModel.for_inference(model)
by_run = {}
for r in [x for x in runs if x["label"] != "NONE"][:10]:
    digest, ss = build_digest(tools_for(r), r["run"]), []
    for _ in range(4):
        a = write(generate, digest, temp=.7)
        s = {"answer": a, "messages": [{"role": "user", "content": digest}, {"role": "assistant", "content": json.dumps(a)}]}
        ss.append((s, critic(s, (a or {}).get("known_bug_hint") == r["label"])))     # production: the matcher's verdict
    by_run[r["run"]] = ss
pairs = preference_pairs(by_run)
print(len(pairs), "preference pairs")
"""),
    md("## 5. Stage 2: DPO\n`loss = -log sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))) + alpha * NLL(chosen)`. "
       "The reference is the same model with the adapter disabled. The weight `sigmoid(-beta * margin)` is computed without gradients so "
       "each 16k-token side can be back-propagated alone. Production repeats sample -> critic -> pairs -> DPO for three rounds."),
    code("""
print("weight when the policy has not moved:", dpo_weight(0, .1), "| when it already prefers the chosen by 30 nats:", round(dpo_weight(30, .1), 3))
FastLanguageModel.for_training(model)
train_dpo(model, tok, pairs, beta=.1, alpha=.5, epochs=1, lr=1e-4, ga=2, out="writer-dpo")
"""),
    md("## 6. Deterministic repair of a degenerate output"),
    code("""
print(repair('{"failure_mechanism": "link retrain loop", "error_chain": ["gt: engine reset", "gt: engine reset", "pcie: link down'))
"""),
], gpu=True)

# ---------------------------------------------------------------- 00 end to end
nb("00-architecture", "end_to_end_colab.ipynb", [
    md("# Expert debugger: end-to-end pipeline (curated)\nFailed run -> evidence -> triage (issue detector) -> ranker + open-set router -> "
       "knowledge layer -> investigation agent graph (XP preload -> writer -> verifier / revise loop -> Done) -> matcher with graph facts -> "
       "confidence gate -> human approval. The CPU stages and the graph control flow run here on synthetic data; the writer and the "
       "matcher are LLMs in the real system and are replaced by tiny stand-ins so the wiring can be run end to end."),
    DATA_NOTE, code("!pip -q install scikit-learn matplotlib numpy scipy"), SYN, inline("00-architecture/data_pipeline.py"),
    inline("00-architecture/agent_graph.py"),
    inline("01-failure-ranker/ranker.py"), inline("02-test-issue-detector/detector.py"), inline("03-knowledge-layer/knowledge.py"),
    code("""
import random
from collections import Counter
from synthetic import make_world, split
from ranker import Ranker, calibrate_tau
from detector import IssueDetector
from knowledge import KB
from agent_graph import xp_preload, run_graph, kg_facts, SEVERE
bugs, runs = make_world(n_bugs=50, n_runs=1800, p_none=.2, seed=7); runs = split(runs, bugs)
for r in runs: r["has_logs"] = True
train = [r for r in runs if r["split"] == "train"]; test = [r for r in runs if r["split"] == "test"]
det = IssueDetector().fit(train, [int(r["label"] == "NONE") for r in train], [r["label"] if r["label"] != "NONE" else r["family"] for r in train])
rk = Ranker().fit(train, bugs)
kb = KB([{**b, "symptom": b["signature"][0], "first_fault": b["signature"][-1], "match_phrases": b["signature"]} for b in bugs])
tau = calibrate_tau([(rk.rank(r, 1)[0][1], rk.rank(r, 1)[0][0] == r["label"]) for r in train if r["label"] != "NONE" and rk.rank(r, 1)], .85)
BUG = {b["id"]: b for b in bugs}
BASELINE = ["loading module", "heartbeat ok", "config applied"]   # lines of a known-good run
"""),
    md("## Stand-ins for the two LLM stages\nThe writer quotes evidence lines; the matcher scores candidates on the root cause plus graph facts. "
       "In the real system both are fine-tuned 8B models; only the control flow around them is shown here."),
    code("""
def toy_writer(ev):
    first = (ev["new_vs_good"] or ev["lines"])[0]
    return {"mechanism": first, "chain": [first]}

def toy_revise(ev, answer, issues):
    sev = [l for l in ev["lines"] if SEVERE.search(l)]
    if not sev or sev[0] in answer["chain"]: return None
    return {"mechanism": sev[0], "chain": [sev[0]] + answer["chain"]}

def toy_matcher(run, answer, pool):
    scored = []
    for b in pool:
        hit = sum(any(sig in q for q in answer["chain"]) for sig in BUG[b]["signature"])
        facts = kg_facts(run, BUG[b])
        scored.append((hit + .5 * facts["same_family"] + .25 * facts["same_owner"] - 5 * (not facts["filed_before_run"]), b))
    scored.sort(reverse=True)
    return scored
"""),
    code("""
def decide(r):
    flag, _ = det.flag(r)
    if flag: return "NOT-A-BUG", None, None
    top = rk.rank(r, 5)
    if top and top[0][1] >= tau: return "LINK", top[0][0], "ranker"
    pool = list(dict.fromkeys([b for b, _ in top] + [c["id"] for c in kb.search(" ".join(r["lines"]), 5, cutoff=r["start"])]))
    ev = xp_preload(r["lines"], BASELINE)
    trace = run_graph(ev, toy_writer, toy_revise)
    scored = toy_matcher(r, trace.answer, pool)
    if not scored or scored[0][0] < 1: return "NEW-BUG", None, trace.done
    margin = scored[0][0] - (scored[1][0] if len(scored) > 1 else 0)
    return ("LINK" if margin >= 1 else "REVIEW"), scored[0][1], trace.done
dec = [(r, *decide(r)) for r in test]
print(Counter(d for _, d, _, _ in dec))
print("graph exits:", Counter(x for *_, x in dec if x not in (None, "ranker")))
link = [(r, b) for r, d, b, _ in dec if d == "LINK"]
print(f"auto-LINK precision {sum(r['label'] == b for r, b in link) / max(1, len(link)):.1%} on {len(link)} runs")
nab = [r for r, d, *_ in dec if d == "NOT-A-BUG"]
print(f"NOT-A-BUG precision {sum(r['label'] == 'NONE' for r in nab) / max(1, len(nab)):.1%} on {len(nab)} runs")
"""),
])
