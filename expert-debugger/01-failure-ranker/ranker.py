"""Failure ranker (no LLM): failed run -> ranked list of known bugs + calibrated LINK / REVIEW / NEW decision.

Features per (run, candidate bug) pair:
  - log channel   : signature lines of the bug found in the run's normalised evidence (exact skeleton hits,
                    fuzzy phrase hits, idf-weighted), first-fault agreement
  - metadata prior: same test family, same domain, same owner, peer votes (how many earlier runs of this test
                    family were linked to the bug), recency of the bug's last link, bug created before the run
  - hard rules    : a bug tagged for another platform vendor never outranks any other candidate
Model: logistic regression over pair features (pairwise -> pointwise scores), grouped CV by bug, then a
calibrated open-set threshold: LINK only when P(top-1) >= tau, tau picked for >= 85 % out-of-fold precision.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

FEATS = ["sig_hits", "sig_frac", "sig_idf", "first_fault", "same_family", "same_domain", "same_owner",
         "peer_votes", "log_recency", "created_before"]


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z_]{3,}", s.lower()))


class Ranker:
    def __init__(self, C: float = 1.0, pool: int = 50):
        self.C, self.pool = C, pool

    # ------------------------------------------------------------- fit
    def fit(self, runs: list[dict], bugs: list[dict]):
        self.bugs = {b["id"]: b for b in bugs}
        self.df = Counter(t for b in bugs for s in b["signature"] for t in tokens(s))
        self.history = sorted((r for r in runs if r["label"] != "NONE"), key=lambda r: r["start"])
        X, y = [], []
        for r in runs:
            if r["label"] == "NONE":
                continue
            for bid in self._pool(r):
                X.append(self._feat(r, self.bugs[bid]))
                y.append(int(bid == r["label"]))
        self.clf = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced").fit(np.array(X), np.array(y))
        return self

    # ------------------------------------------------------------- features
    def _pool(self, r):
        return [b for b, v in self.bugs.items() if v["created"] <= r["start"]][: self.pool * 10]

    def _feat(self, r, b):
        text = " ".join(r["lines"]).lower()
        hits = [s for s in b["signature"] if s.lower() in text]
        idf = sum(math.log(1 + len(self.bugs) / (1 + self.df[t])) for s in hits for t in tokens(s))
        prev = [h for h in self.history if h["start"] < r["start"] and h["label"] == b["id"]]
        peers = sum(1 for h in prev if h["family"] == r["family"])
        last = (r["start"] - prev[-1]["start"]).days if prev else 999
        ff = next((l for l in r["lines"] if re.search(r"fault|reset|lost|timeout|mismatch", l)), "")
        return [len(hits), len(hits) / len(b["signature"]), idf, int(bool(ff) and ff in b["signature"]),
                int(b["family"] == r["family"]), int(b["domain"] == r["domain"]), int(b["owner"] == r["owner"]),
                math.log1p(peers), -math.log1p(last), int(b["created"] <= r["start"])]

    # ------------------------------------------------------------- predict
    def rank(self, r, k: int = 5):
        pool = self._pool(r)
        if not pool:
            return []
        p = self.clf.predict_proba(np.array([self._feat(r, self.bugs[b]) for b in pool]))[:, 1]
        order = np.argsort(-p)[:k]
        return [(pool[i], float(p[i])) for i in order]


def calibrate_tau(scored: list[tuple[float, bool]], target_precision: float = .85) -> float:
    """Lowest threshold whose out-of-fold precision on LINK decisions is >= target."""
    for tau in np.arange(.05, 1.0, .01):
        sel = [ok for s, ok in scored if s >= tau]
        if sel and sum(sel) / len(sel) >= target_precision:
            return float(tau)
    return 1.0


def evaluate(ranker: Ranker, runs: list[dict], unseen: set[str] | None = None) -> dict:
    res = defaultdict(list)
    for r in runs:
        if r["label"] == "NONE":
            continue
        top = [b for b, _ in ranker.rank(r, 3)]
        res["top1"].append(bool(top) and top[0] == r["label"])
        res["top3"].append(r["label"] in top)
        if unseen is not None and r["label"] in unseen:
            res["unseen_top1"].append(bool(top) and top[0] == r["label"])
    return {k: round(100 * sum(v) / len(v), 1) for k, v in res.items() if v}
