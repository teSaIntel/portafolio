"""Test-content / infrastructure issue detector: is this failure *not* a product bug?

A false positive hides a real bug, so the model only fires at an operating point with >= 90 % out-of-fold
precision. Features: TF-IDF over the failure reason, the failed stage and the error lines of the execution log
(numbers and paths masked), plus one-hot categorical flags (test family, owner, domain, evidence present or not).
Labels: runs humans tagged as test-content issues (positives) vs runs linked to a product bug (negatives).

Two arms behind one verdict (Triage): editable, stored rules (precision-pruned, each tagged infra or test_content, shown by name in the
UI) run first; the logistic regression above is the model arm for what the rules do not cover.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

CATS = ["family", "owner", "domain", "has_logs"]


class IssueDetector:
    def __init__(self, C: float = 3.0, min_precision: float = .9):
        self.C, self.min_precision = C, min_precision

    def _text(self, rows):
        return [" ".join(r["lines"]) for r in rows]

    def _cats(self, rows):
        return [[str(r.get(c, "")) for c in CATS] for r in rows]

    def _x(self, rows, fit=False):
        if fit:
            self.tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            self.oh = OneHotEncoder(handle_unknown="ignore")
            return hstack([self.tf.fit_transform(self._text(rows)), self.oh.fit_transform(self._cats(rows))]).tocsr()
        return hstack([self.tf.transform(self._text(rows)), self.oh.transform(self._cats(rows))]).tocsr()

    def fit(self, rows, y, groups):
        """Grouped CV gives out-of-fold scores; the threshold is the best-recall point with OOF precision >= target."""
        oof = np.zeros(len(rows))
        for tr, te in GroupKFold(5).split(rows, y, groups):
            m = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
            m.fit(self._x([rows[i] for i in tr], fit=True), np.array(y)[tr])
            oof[te] = m.predict_proba(self._x([rows[i] for i in te]))[:, 1]
        p, r, t = precision_recall_curve(y, oof)
        ok = [(r[i], t[i]) for i in range(len(t)) if p[i] >= self.min_precision]
        self.tau = max(ok)[1] if ok else 1.0
        self.oof_ap = average_precision_score(y, oof)
        self.oof_recall = max(ok)[0] if ok else 0.0
        self.model = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(self._x(rows, fit=True), y)
        return self

    def flag(self, row) -> tuple[bool, float]:
        s = float(self.model.predict_proba(self._x([row]))[:, 1][0])
        return s >= self.tau, s


@dataclass
class Rule:
    """One stored triage rule: fires when any `any_of` phrase is in the failure reason / log and no `none_of` phrase is."""
    name: str
    kind: str                                   # "infra" | "test_content"
    any_of: list[str]
    none_of: list[str] = field(default_factory=list)

    def fires(self, row) -> bool:
        t = " ".join(row["lines"]).lower()
        return any(p in t for p in self.any_of) and not any(p in t for p in self.none_of)


class RuleTriage:
    """Rules live in a shared store (users can add their own); base rules are pruned by measured precision."""

    def __init__(self, rules: list[Rule]):
        self.rules = rules

    def prune(self, rows, y, min_precision: float = .9):
        """Keep a rule only if its precision on labelled runs (non-bug = positive) is at least `min_precision`."""
        keep = []
        for ru in self.rules:
            hit = [yy for r, yy in zip(rows, y) if ru.fires(r)]
            if hit and sum(hit) / len(hit) >= min_precision:
                keep.append(ru)
        self.rules = keep
        return self

    def match(self, row):
        return next((ru for ru in self.rules if ru.fires(row)), None)


class Triage:
    """Rules first (explainable by rule name), then the model arm at its >= 90% precision threshold."""

    def __init__(self, rules: RuleTriage, model: IssueDetector):
        self.rules, self.model = rules, model

    def flag(self, row) -> tuple[bool, str]:
        ru = self.rules.match(row)
        if ru:
            return True, f"rule:{ru.name} ({ru.kind})"
        hit, score = self.model.flag(row)
        return hit, f"model:{score:.2f}"
