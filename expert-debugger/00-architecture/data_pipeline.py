"""Data-collection pipeline: failed CI runs -> normalised evidence -> leak-free labelled datasets.

Every model in the system trains and is evaluated on what this module produces. The sources are interfaces:
plug in your CI server, log store and bug tracker. Nothing here calls an LLM.

    runs  = ci.failed_runs(since)                       # failed test runs + metadata
    ev    = collect_evidence(run, logs)                 # windowed, normalised, deduplicated log lines
    rows  = build_dataset(runs, tracker, logs)          # one JSONL row per run, with its label and split
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Protocol


# ---------------------------------------------------------------- sources (interfaces)
@dataclass
class Run:
    id: str
    test: str              # test case / suite id
    domain: str            # feature area (power, memory, reset, ...)
    host: str
    start: datetime
    end: datetime
    reason: str            # one-line failure reason from the harness
    meta: dict = field(default_factory=dict)   # owner, config, build, ...


@dataclass
class Bug:
    id: str
    title: str
    description: str
    created: datetime
    state: str


class CI(Protocol):
    def failed_runs(self, since: datetime) -> Iterable[Run]: ...
    def artifact_text(self, run: Run, name: str) -> str | None: ...


class LogStore(Protocol):
    def search(self, index: str, host: str, start: datetime, end: datetime, size: int = 2000) -> list[str]: ...


class Tracker(Protocol):
    def links(self) -> dict[str, tuple[str, datetime]]: ...   # run id -> (bug id, link time)
    def bug(self, bug_id: str) -> Bug: ...


# ---------------------------------------------------------------- normalisation
_MASKS = [
    (re.compile(r"^\[?\s*\d+\.\d+\]\s*"), ""),                                          # kernel monotonic stamp
    (re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:?\d\d)?"), "<TS>"),
    (re.compile(r"\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b", re.I), "<PCI>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<UUID>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<HEX>"),
    (re.compile(r"\b[a-z]+-?\d{2,}[a-z0-9-]*\b", re.I), "<HOST>"),
    (re.compile(r"\b\d{3,}\b"), "<N>"),
]
_SIGNAL = re.compile(r"error|fail|fatal|timeout|hang|lost|reset|panic|oops|corrupt|denied|mismatch|assert", re.I)


def normalise(line: str) -> str:
    """Mask volatile tokens so the same failure gives the same string on every host and day."""
    s = line.strip()
    for rx, rep in _MASKS:
        s = rx.sub(rep, s)
    return re.sub(r"\s+", " ", s)


def skeleton(line: str) -> str:
    """Hash key of a normalised line (used for dedup, clustering and signature matching)."""
    return hashlib.sha1(normalise(line).lower().encode()).hexdigest()[:16]


# ---------------------------------------------------------------- evidence
INDEXES = ("kernel", "exec", "tracked", "alerts")     # kernel log, execution log, harness failure records, alerts
PAD = timedelta(minutes=15)


def collect_evidence(run: Run, logs: LogStore, ci: CI | None = None, cap: int = 40) -> dict[str, list[str]]:
    """Run-scoped evidence: every index in [start - 15 min, end + 15 min] on the run's host, normalised,
    deduplicated by skeleton, signal lines first, capped per source. Falls back to CI artifacts when an
    index has no data for the run (some suites never reach the log store)."""
    out = {}
    for ix in INDEXES:
        lines = logs.search(ix, run.host, run.start - PAD, run.end + PAD)
        if not lines and ci is not None:
            lines = (ci.artifact_text(run, f"{ix}.log") or "").splitlines()
        seen, keep = set(), []
        for ln in sorted(lines, key=lambda l: not _SIGNAL.search(l)):
            k = skeleton(ln)
            if k not in seen:
                seen.add(k)
                keep.append(normalise(ln))
            if len(keep) >= cap:
                break
        out[ix] = keep
    return out


def first_fault(ev: dict[str, list[str]]) -> str | None:
    """Earliest signal line across sources: a cheap, deterministic proxy of the root-cause line."""
    for ix in ("kernel", "exec", "tracked"):
        for ln in ev.get(ix, []):
            if _SIGNAL.search(ln):
                return ln
    return None


# ---------------------------------------------------------------- labels and leak-free splits
def build_dataset(runs: list[Run], tracker: Tracker, logs: LogStore, ci: CI | None = None, seed: int = 0,
                  big_group: int = 8, frac=(0.7, 0.1, 0.2)) -> list[dict]:
    """One row per failed run.
    label   : linked bug id, or NONE when the run was triaged as not-a-bug.
    split   : grouped by bug, so a bug is in exactly one of train / val / test (measures generalisation to
              bugs never seen in training). Very large groups are split by run instead, so frequent bugs are
              represented in every split.
    cutoff  : the run's start time. Any retrieval or tool built on top must hide bugs filed after it, and
              must hide the run's own link (prevents label leakage through the knowledge base)."""
    links = tracker.links()
    rows = []
    for r in runs:
        bug, linked = links.get(r.id, (None, None))
        rows.append({"run": r.id, "test": r.test, "domain": r.domain, "reason": normalise(r.reason), "meta": r.meta,
                     "evidence": collect_evidence(r, logs, ci), "label": bug or "NONE",
                     "cutoff": r.start.isoformat(), "linked": linked.isoformat() if linked else None})
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    keys = sorted(k for k in groups if k != "NONE")
    rng.shuffle(keys)
    n = len(keys)
    cut = [int(frac[0] * n), int((frac[0] + frac[1]) * n)]
    for i, k in enumerate(keys):
        g = groups[k]
        if len(g) >= big_group:
            for j, row in enumerate(sorted(g, key=lambda x: x["cutoff"])):
                row["split"] = "train" if j < .7 * len(g) else ("val" if j < .8 * len(g) else "test")
        else:
            for row in g:
                row["split"] = "train" if i < cut[0] else ("val" if i < cut[1] else "test")
    for row in groups.get("NONE", []):
        row["split"] = rng.choices(["train", "val", "test"], frac)[0]
    return rows


def unseen_mask(rows: list[dict]) -> list[bool]:
    """True for evaluation rows whose bug never appears in train: the honest 'new failure' slice."""
    train_bugs = {r["label"] for r in rows if r["split"] == "train"}
    return [r["split"] != "train" and r["label"] != "NONE" and r["label"] not in train_bugs for r in rows]


def out_of_fold(rows: list[dict], fit, predict, folds: int = 5, seed: int = 0) -> dict[str, list[str]]:
    """Out-of-fold candidate lists for training downstream models: the matcher must learn from the ranker's
    real mistakes, not from a ranker that has already seen the answer."""
    bugs = sorted({r["label"] for r in rows})
    random.Random(seed).shuffle(bugs)
    fold_of = {b: i % folds for i, b in enumerate(bugs)}
    out = {}
    for f in range(folds):
        model = fit([r for r in rows if fold_of[r["label"]] != f])
        for r in rows:
            if fold_of[r["label"]] == f:
                out[r["run"]] = predict(model, r)
    return out


def write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(path, len(rows), Counter(r["split"] for r in rows))
