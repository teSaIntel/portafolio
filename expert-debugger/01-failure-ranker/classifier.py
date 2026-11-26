"""Bug classifier: the production champion behind "Suggested bugs" (no LLM).

    failed run ──► stage 1  base classifier      LR over [signature-phrase hits | metadata one-hot | text tf-idf]
               ──► top-K candidates (unthresholded, K=5)
               ──► stage 2  re-ranker            pair LR (ranker.py) + base posterior + temporal burst + kNN votes
               ──► hard rules                    a bug tagged for another platform vendor sinks to the bottom
               ──► open-set router               split-conformal on the absolute top logit: LINK / SIBLINGS / NEW
               ──► cold-start text rule          bug with no training runs whose report text matches strongly
               ──► explanation                   the features that pushed the score (coefficient x value)
    lifecycle: champion/challenger retraining, live refit on new human links, forward ledger + guarded recalibration
    downstream: the leak-free top-5 shortlist (shortlist_oof) is what the root-cause writer and the matcher receive
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack

from ranker import Ranker

META = ["domain", "owner", "family"]


# ================================================================ stage 1: base classifier
class BaseClassifier:
    """One multinomial LR over three channels. Status, CI result and comments are never features (label leak).
    Failing runs are ranked among bugs only (the NONE column is dropped at prediction time)."""

    def __init__(self, C: float = 100.0, phrase_weight: float = 2.0):
        self.C, self.w = C, phrase_weight

    def _phrases(self, runs):
        text = [" ".join(r["lines"]).lower() for r in runs]
        return csr_matrix([[sum(s.lower() in t for s in b["signature"]) / len(b["signature"]) for b in self.bugs]
                           for t in text]) * self.w

    def _X(self, runs, fit=False):
        meta = [[r.get(k, "") for k in META] for r in runs]
        text = [" ".join(r["lines"]) for r in runs]
        if fit:
            self.oh = OneHotEncoder(handle_unknown="ignore").fit(meta)
            self.tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(text)
        return hstack([self._phrases(runs), self.oh.transform(meta), self.tf.transform(text)]).tocsr()

    def fit(self, runs, bugs):
        self.bugs = bugs
        self.clf = LogisticRegression(C=self.C, max_iter=3000).fit(self._X(runs, fit=True), [r["label"] for r in runs])
        return self

    def rank(self, run, k=5):
        p = self.clf.predict_proba(self._X([run]))[0]
        out = [(c, float(v)) for c, v in zip(self.clf.classes_, p) if c != "NONE"]
        return sorted(out, key=lambda x: -x[1])[:k]          # unthresholded: low-prior truths stay in the pool

    def why(self, run, bug, top=5):
        """Features pushing `bug` up for this run: coefficient x value (what the UI shows next to a suggestion)."""
        if bug not in self.clf.classes_:
            return []
        x = self._X([run]).toarray()[0]
        contrib = self.clf.coef_[list(self.clf.classes_).index(bug)] * x
        names = [f"phrase:{b['id']}" for b in self.bugs] + list(self.oh.get_feature_names_out(META)) + \
            [f"text:{t}" for t in self.tf.get_feature_names_out()]
        return [(names[i], round(float(contrib[i]), 3)) for i in np.argsort(-contrib)[:top] if contrib[i] > 0]


# ================================================================ stage 2: re-ranker over the candidates
def _jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / (len(a | b) or 1)


class Champion:
    """Stage 1 + stage 2. Stage 2 only re-orders stage 1's top-K, so recall@K is its hard ceiling;
    K=5 (instead of 3, and without the 0.02 probability floor) lifted that ceiling by 3-8 points."""

    def __init__(self, K=5, C2=0.1, burst_days=7, knn=7):
        self.K, self.C2, self.burst, self.knn = K, C2, timedelta(days=burst_days), knn

    def _extra(self, r, bid, p_base):
        recent = sum(1 for h in self.hist if h["label"] == bid and r["start"] - self.burst <= h["start"] < r["start"])
        near = sorted((h for h in self.hist if h["start"] < r["start"]), key=lambda h: -_jacc(h["lines"], r["lines"]))[: self.knn]
        votes = sum(h["label"] == bid for h in near) / self.knn
        vendor_clash = int(bool(r.get("vendor")) and bool(self.pair.bugs[bid].get("vendor")) and
                           r["vendor"] != self.pair.bugs[bid]["vendor"])
        return [math.log(p_base + 1e-4), math.log1p(recent), votes, vendor_clash]

    def _rows(self, r, cands):
        return np.array([self.pair._feat(r, self.pair.bugs[b]) + self._extra(r, b, p) for b, p in cands])

    def fit(self, runs, bugs, oof_base: dict | None = None):
        self.base = BaseClassifier().fit(runs, bugs)
        self.pair = Ranker().fit(runs, bugs)
        self.hist = sorted((r for r in runs if r["label"] != "NONE"), key=lambda r: r["start"])
        X, y = [], []
        for r in self.hist:
            cands = (oof_base or {}).get(r["run"]) or self.base.rank(r, self.K)   # out-of-fold when available
            if r["label"] not in [b for b, _ in cands]:
                continue
            X.append(self._rows(r, cands)); y += [int(b == r["label"]) for b, _ in cands]
        self.rr = LogisticRegression(C=self.C2, max_iter=3000, class_weight="balanced").fit(np.vstack(X), y)
        return self

    def rank(self, r):
        cands = self.base.rank(r, self.K)
        logit = self.rr.decision_function(self._rows(r, cands))
        logit = np.where(self._rows(r, cands)[:, -1] > 0, -1e3, logit)   # hard rule: vendor clash goes last
        order = np.argsort(-logit)
        return [(cands[i][0], float(logit[i])) for i in order]


# ================================================================ open-set router (split conformal)
@dataclass
class OpenSetRouter:
    """Softmax-based conformal (1 - p) cannot say NEW: a closed set always has a winner (NEW recall 0%).
    The absolute top logit can (seen-vs-new AUROC .92). Nonconformity = -logit of the true candidate,
    calibrated only on runs whose truth was a candidate. Prediction set = candidates with logit >= -q.
        |set| == 1 -> LINK      |set| > 1 -> SIBLINGS (diff tools / agent)      |set| == 0 -> NEW"""
    alpha: float = .1
    q: float = 0.0

    def calibrate(self, ranked: list[list[tuple[str, float]]], truth: list[str]):
        s = sorted(-dict(rk)[t] for rk, t in zip(ranked, truth) if t in dict(rk))
        n = len(s)
        self.q = s[min(n - 1, math.ceil((n + 1) * (1 - self.alpha)) - 1)]
        return self

    def route(self, ranked):
        keep = [b for b, l in ranked if l >= -self.q]
        return ("LINK" if len(keep) == 1 else "SIBLINGS" if keep else "NEW"), keep


# ================================================================ cold-start text rule
class TextColdStart:
    """Label-free retriever over the bug reports (tf-idf of title + signature + root cause). It fires only for a
    bug that has NO training runs (the classifier cannot know it) and only above theta, the smallest score with
    <= 2% fires on the calibration window, where every such fire is wrong by construction."""

    def fit(self, bugs, trained_bugs: set[str]):
        self.ids = [b["id"] for b in bugs]
        self.cold = {b for b in self.ids if b not in trained_bugs}
        docs = [" ".join([b.get("title", ""), *b["signature"], b.get("root_cause", "")]) for b in bugs]
        self.tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(docs)
        self.M = self.tf.transform(docs)
        return self

    def top(self, r):
        s = cosine_similarity(self.tf.transform([" ".join(r["lines"])]), self.M)[0]
        i = int(np.argmax(s))
        return self.ids[i], float(s[i])

    def calibrate(self, calib_runs, max_fire=.02, floor=.3):
        """calib_runs: runs whose bug HAS training runs, so any cold fire on them is a false fire."""
        sc = sorted((s for b, s in map(self.top, calib_runs) if b in self.cold), reverse=True)
        k = int(max_fire * len(calib_runs))
        self.theta = max(floor, sc[k] + 1e-6 if len(sc) > k else 0.0)
        return self

    def fire(self, r):
        b, s = self.top(r)
        return b if b in self.cold and s >= self.theta else None


def decide(champ: Champion, router: OpenSetRouter, cold: TextColdStart, r) -> dict:
    """Final answer shown to people: always the top 3 (never hidden), plus the route."""
    ranked = champ.rank(r)
    b = cold.fire(r)
    if b:
        return {"route": "LINK", "bug": b, "top3": [b] + [x for x, _ in ranked[:2]], "why": "cold-start text rule"}
    route, keep = router.route(ranked)
    return {"route": route, "bug": keep[0] if route == "LINK" else None, "top3": [x for x, _ in ranked[:3]],
            "set": keep, "why": champ.base.why(r, ranked[0][0])}


def override_rule(ranked, agent: dict | None, probe: dict[str, float] | None = None, low=.34, min_conf=.7):
    """The classifier is usually more accurate than the agent, so it stays the default. The agent overrides only when
    confidence in the classifier's pick is low, measured by the probe: the share of that bug's known evidence lines
    found in this run's real logs. It must also name a bug whose own probe evidence is higher than the top's."""
    top = ranked[0][0]
    if not agent or not agent.get("bug"):
        return top
    b, probe = agent["bug"], probe or {}
    if b != top and agent.get("confidence", 0) >= min_conf and probe.get(top, 0) < low and probe.get(b, 0) > probe.get(top, 0):
        return b
    return top


# ================================================================ lifecycle
@dataclass
class Registry:
    """Champion/challenger: a challenger (hyper-parameter grid) is promoted only if it beats the champion on the
    same folds of the newest window. Triggers: bootstrap, new bugs, window rollover, incremental links."""
    versions: list = field(default_factory=list)

    def consider(self, name, score, params):
        best = max((v["score"] for v in self.versions if v["promoted"]), default=-1)
        v = {"name": name, "score": score, "params": params, "promoted": score > best}
        self.versions.append(v)
        return v["promoted"]


@dataclass
class Ledger:
    """Forward check (the drift alarm). Every day: record the live answer for failed runs nobody has linked yet;
    later, score those predictions against the links people actually made."""
    rows: dict = field(default_factory=dict)

    def record(self, run_id, route, bug):
        self.rows.setdefault(run_id, {"route": route, "bug": bug, "truth": None})

    def resolve(self, links: dict[str, str]):
        for rid, b in links.items():
            if rid in self.rows:
                self.rows[rid]["truth"] = b

    def forward(self):
        done = [v for v in self.rows.values() if v["truth"] and v["route"] == "LINK"]
        prec = sum(v["bug"] == v["truth"] for v in done) / len(done) if done else None
        return {"linked": len(done), "link_precision": prec}

    def needs_recalibration(self, floor=.85, min_n=10):
        f = self.forward()
        return f["linked"] >= min_n and f["link_precision"] < floor


def guarded_recalibrate(router: OpenSetRouter, base_ranked, base_truth, ledger_ranked, ledger_truth, grid=(.05, .1, .15, .2, .3)):
    """Refit the LINK route on base + ledger rows; keep it only if base precision drops < 2 pt and ledger
    precision does not drop. The server swaps the route in place (no champion rebuild)."""
    def prec(rt, ranked, truth):
        ok = [rk[0][0] == t for rk, t in zip(ranked, truth) if rt.route(rk)[0] == "LINK"]
        return sum(ok) / len(ok) if ok else 0.0
    b0, l0 = prec(router, base_ranked, base_truth), prec(router, ledger_ranked, ledger_truth)
    best = router
    for a in grid:
        cand = OpenSetRouter(alpha=a).calibrate(base_ranked + ledger_ranked, base_truth + ledger_truth)
        b1, l1 = prec(cand, base_ranked, base_truth), prec(cand, ledger_ranked, ledger_truth)
        if b1 >= b0 - .02 and l1 > l0:
            best, l0 = cand, l1
    return best


def evaluate(champ, router, cold, runs, unseen=frozenset()):
    m = defaultdict(list)
    for r in runs:
        d = decide(champ, router, cold, r)
        if r["label"] == "NONE":
            m["none_not_linked"].append(d["route"] != "LINK")
            continue
        m["top1"].append(d["top3"][0] == r["label"]); m["top3"].append(r["label"] in d["top3"])
        if d["route"] == "LINK":
            m["link_precision"].append(d["bug"] == r["label"])
        m["link_rate"].append(d["route"] == "LINK")
        if r["label"] in unseen:
            m["unseen_top1"].append(d["top3"][0] == r["label"])
    return {k: (round(100 * sum(v) / len(v), 1), len(v)) for k, v in m.items() if v}, Counter(
        decide(champ, router, cold, r)["route"] for r in runs)


def shortlist_oof(runs, bugs, folds=5, K=5, seed=0):
    """Leak-free top-K shortlist for the downstream LLM stages. Each fold is ranked by a champion refit WITHOUT that fold's
    labels and links, so the writer and matcher never see a ranking that already knew the answer."""
    import random
    idx = list(range(len(runs)))
    random.Random(seed).shuffle(idx)
    out = {}
    for f in range(folds):
        held = set(idx[f::folds])
        champ = Champion(K=K).fit([r for i, r in enumerate(runs) if i not in held], bugs)
        for i in held:
            out[runs[i]["run"]] = champ.rank(runs[i])
    return out


def recall_at(shortlist, runs, ks=(1, 5)):
    bug_runs = [r for r in runs if r["label"] != "NONE"]
    return {k: round(100 * sum(r["label"] in [b for b, _ in shortlist[r["run"]][:k]] for r in bug_runs) / len(bug_runs), 1)
            for k in ks}
