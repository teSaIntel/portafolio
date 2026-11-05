"""Knowledge layer: structured bug cards + hybrid search + a small knowledge graph.

1. Bug cards (LLM, once per bug, cached): free-text bug reports are long and noisy (console pastes, HTML).
   An LLM turns each into a card: component, symptom, first fault, match phrases (verbatim log lines),
   distinguishing detail vs sibling bugs, workaround. Cards are validated: a match phrase that never occurs in
   any linked run's evidence is dropped.
2. Hybrid search: BM25 (exact log vocabulary) + dense embeddings (paraphrase) fused with reciprocal-rank fusion.
   Used by the agent as a tool and to widen the ranker's candidate pool.
3. Knowledge graph: nodes = bugs, components, tests, owners, root causes; edges = links, supersedes, shares
   component, co-occurs. Queried by the agent for neighbourhoods ("siblings of this bug").
Everything respects the run's cutoff: bugs filed after the run, and the run's own link, are invisible.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

CARD_PROMPT = """You turn one bug report into a structured card for a log-matching system.
Return JSON with keys: component, symptom, first_fault, match_phrases (3-8 verbatim log lines that identify THIS bug),
distinguishing (what separates it from similar bugs), workaround.
Only copy phrases that appear verbatim in the report. Do not invent log lines.

Report:
{report}"""


def make_card(report: str, llm) -> dict:
    card = json.loads(llm(CARD_PROMPT.format(report=report[:12000])))
    card["match_phrases"] = [p for p in card.get("match_phrases", []) if p and p in report]  # verbatim only
    return card


def validate_card(card: dict, linked_evidence: list[str]) -> dict:
    """Drop phrases that never hit a run linked to this bug (they would only add noise to retrieval)."""
    text = "\n".join(linked_evidence).lower()
    card["match_phrases"] = [p for p in card["match_phrases"] if p.lower() in text] or card["match_phrases"][:2]
    return card


# ------------------------------------------------------------------ hybrid search
def _tok(s):
    return re.findall(r"[a-z0-9_]{2,}", s.lower())


class BM25:
    def __init__(self, docs: list[str], k1=1.4, b=.75):
        self.d = [_tok(x) for x in docs]
        self.avg = sum(map(len, self.d)) / max(1, len(self.d))
        self.df = Counter(t for doc in self.d for t in set(doc))
        self.k1, self.b, self.N = k1, b, len(docs)

    def scores(self, q: str):
        qt = _tok(q)
        out = []
        for doc in self.d:
            tf = Counter(doc)
            s = 0.0
            for t in qt:
                if t in tf:
                    idf = math.log(1 + (self.N - self.df[t] + .5) / (self.df[t] + .5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / self.avg))
            out.append(s)
        return out


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion of several ranked id lists."""
    score = defaultdict(float)
    for r in rankings:
        for pos, i in enumerate(r):
            score[i] += 1 / (k + pos + 1)
    return sorted(score, key=lambda i: -score[i])


class KB:
    def __init__(self, cards: list[dict], embed=None):
        self.cards = cards
        self.docs = [" ".join([c["title"], c.get("symptom", ""), c.get("first_fault", ""), *c.get("match_phrases", [])]) for c in cards]
        self.bm25 = BM25(self.docs)
        self.embed = embed
        self.E = embed(self.docs) if embed else None

    def search(self, query: str, k: int = 10, cutoff=None, hide: set | None = None) -> list[dict]:
        s = self.bm25.scores(query)
        bm = sorted(range(len(self.docs)), key=lambda i: -s[i])
        lists = [bm]
        if self.embed is not None:
            q = self.embed([query])[0]
            sims = (self.E @ q) / ((self.E ** 2).sum(1) ** .5 * (q @ q) ** .5 + 1e-9)
            lists.append(list(sims.argsort()[::-1]))
        out = []
        for i in rrf(*lists):
            c = self.cards[i]
            if (cutoff and c["created"] > cutoff) or (hide and c["id"] in hide):
                continue                                   # time cut + label hiding: no leakage
            out.append(c)
            if len(out) == k:
                break
        return out


# ------------------------------------------------------------------ knowledge graph
class Graph:
    def __init__(self):
        self.adj = defaultdict(set)

    def add(self, a, rel, b):
        self.adj[a].add((rel, b))
        self.adj[b].add((f"~{rel}", a))

    def neighbourhood(self, node, depth=2):
        seen, frontier = {node}, [node]
        for _ in range(depth):
            nxt = []
            for n in frontier:
                for _, m in self.adj[n]:
                    if m not in seen:
                        seen.add(m)
                        nxt.append(m)
            frontier = nxt
        return seen - {node}

    def siblings(self, bug):
        comps = {b for r, b in self.adj[bug] if r == "component"}
        return {x for c in comps for r, x in self.adj[c] if r == "~component" and x != bug}
