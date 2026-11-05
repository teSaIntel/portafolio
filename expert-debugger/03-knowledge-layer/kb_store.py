"""Knowledge base: everything people wrote about bugs (reports, discussion threads, run comments), searchable.

    sources ─► clean (normalise, quality ok|short|noise, mask ids/URLs, extract entities)
            ─► ingest (content-hash idempotent upsert; noise is stored but never chunked)
            ─► chunk (<= 1100 chars, 120 overlap)
            ─► embed (incremental, only quality-ok chunks; halfvec(1024) + HNSW in Postgres/pgvector)
            ─► search: bm25 | dense | fuse (RRF k=60), grouped by bug, time-cut, own-run excluded

Production storage is Postgres + pgvector (schema.sql); this module keeps the same logic in memory so the
notebook runs anywhere. Daily refresh re-ingests only new/edited docs and embeds only new chunks (~20 s).
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from knowledge import BM25, rrf

# ---------------------------------------------------------------- clean
_URL = re.compile(r"https?://\S+")
_ID = re.compile(r"\b(BUG|RUN)-\d+\b")
ENTITY = {
    "hex": re.compile(r"\b0x[0-9a-f]{4,}\b", re.I),
    "path": re.compile(r"(?:/[\w.-]+){2,}"),
    "symbol": re.compile(r"\b[a-z]+_[a-z0-9_]{3,}\b"),
    "ticket": re.compile(r"\b[A-Z]{2,}-\d{2,}\b"),
    "logline": re.compile(r"^\s*\w+: .*(?:error|fail|fault|timeout|reset|lost).*$", re.I | re.M),
}


@dataclass
class Doc:
    kind: str                 # report | thread_comment | run_comment
    bug: str | None
    run: str | None
    ts: datetime
    text: str
    meta: dict = field(default_factory=dict)
    quality: str = "ok"
    entities: dict = field(default_factory=dict)

    @property
    def key(self):
        return hashlib.sha1(f"{self.kind}|{self.bug}|{self.run}|{self.ts}".encode()).hexdigest()

    @property
    def hash(self):
        return hashlib.sha1(self.text.encode()).hexdigest()


def clean(d: Doc) -> Doc:
    t = re.sub(r"<[^>]+>", " ", d.text)                       # html
    ents = {k: sorted(set(rx.findall(t)))[:50] for k, rx in ENTITY.items()}
    t = _URL.sub("<URL>", t)
    t = _ID.sub(lambda m: f"{m.group(1)}-<ID>", t)            # ids masked: queries can't match by number
    t = re.sub(r"\s+", " ", t).strip()
    words = len(t.split())
    noise = re.fullmatch(r"(?i)(\+1|thanks|ok|done|same here|any update\??|ping)\W*", t) is not None
    d.text, d.entities = t, ents
    d.quality = "noise" if noise else "short" if words < 8 else "ok"
    return d


def chunks(text: str, size: int = 1100, overlap: int = 120) -> list[str]:
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):                                      # cut on a sentence/space boundary when possible
            k = max(text.rfind(". ", i, j), text.rfind(" ", i, j))
            j = k + 1 if k > i + size // 2 else j
        out.append(text[i:j].strip())
        if j == len(text):
            break
        i = j - overlap
    return out


# ---------------------------------------------------------------- store
class KnowledgeBase:
    def __init__(self, embed=None, dim: int = 1024):
        self.docs: dict[str, Doc] = {}
        self.chunk_text, self.chunk_doc, self.vec = [], [], {}
        self.embed, self.dim = embed, dim
        self._bm25 = None

    def ingest(self, docs: list[Doc]) -> dict:
        rep = defaultdict(int)
        for d in map(clean, docs):
            old = self.docs.get(d.key)
            if old and old.hash == d.hash:
                rep["unchanged"] += 1
                continue
            rep["edited" if old else "new"] += 1
            if old:                                            # drop the old chunks of an edited doc
                keep = [i for i, k in enumerate(self.chunk_doc) if k != d.key]
                self.chunk_text = [self.chunk_text[i] for i in keep]
                self.chunk_doc = [self.chunk_doc[i] for i in keep]
                self.vec = {}
            self.docs[d.key] = d
            if d.quality == "noise":
                rep["noise"] += 1
                continue
            for c in chunks(d.text):
                self.chunk_text.append(c)
                self.chunk_doc.append(d.key)
        self._bm25 = BM25(self.chunk_text) if self.chunk_text else None
        return dict(rep)

    def embed_new(self, batch: int = 64) -> int:
        """Incremental: only quality-ok chunks without a vector. Returns how many were embedded."""
        if self.embed is None:
            return 0
        todo = [i for i, k in enumerate(self.chunk_doc) if i not in self.vec and self.docs[k].quality == "ok"]
        for s in range(0, len(todo), batch):
            ids = todo[s: s + batch]
            for i, v in zip(ids, self.embed([self.chunk_text[i] for i in ids])):
                self.vec[i] = np.asarray(v, dtype=np.float16)  # halfvec: half the space, same recall
        return len(todo)

    # ------------------------------------------------------------- search
    def _visible(self, i, cutoff, exclude_run):
        d = self.docs[self.chunk_doc[i]]
        if exclude_run is not None and d.run == exclude_run:  # never read the run's own comments (leak)
            return False
        return cutoff is None or d.ts <= cutoff

    def search(self, query: str, exclude_run: str | None, k: int = 10, mode: str = "fuse", cutoff=None):
        """exclude_run is REQUIRED (pass None explicitly only for non-evaluation use). Groups chunks by bug.
        Degrades to BM25 if the embedder is down."""
        q = _ID.sub(lambda m: f"{m.group(1)}-<ID>", query)
        vis = [i for i in range(len(self.chunk_text)) if self._visible(i, cutoff, exclude_run)]
        if not vis or self._bm25 is None:
            return []
        sp = self._bm25.scores(q)
        sparse = sorted(vis, key=lambda i: -sp[i])[:200]
        ranking = sparse
        if mode in ("dense", "fuse") and self.embed is not None and self.vec:
            try:
                qv = np.asarray(self.embed([q])[0], dtype=np.float32)
                dv = [i for i in vis if i in self.vec]
                dense = sorted(dv, key=lambda i: -float(self.vec[i].astype(np.float32) @ qv))[:200]
                ranking = dense if mode == "dense" else rrf(sparse, dense)
            except Exception:                                   # embed server down -> BM25
                ranking = sparse
        out, seen = [], set()
        for i in ranking:
            b = self.docs[self.chunk_doc[i]].bug
            if b and b not in seen:
                seen.add(b)
                out.append({"bug": b, "chunk": self.chunk_text[i][:300], "kind": self.docs[self.chunk_doc[i]].kind})
            if len(out) >= k:
                break
        return out

    def thread(self, bug: str, cutoff=None, cap: int = 80) -> list[str]:
        ds = sorted((d for d in self.docs.values() if d.bug == bug and d.quality != "noise" and
                     (cutoff is None or d.ts <= cutoff)), key=lambda d: d.ts)
        return [f"[{d.kind} {d.ts:%Y-%m-%d}] {d.text}" for d in ds[:cap]]

    def by_entity(self, kind: str, value: str) -> set[str]:
        return {d.bug for d in self.docs.values() if value in d.entities.get(kind, []) and d.bug}

    def stats(self):
        q = defaultdict(int)
        for d in self.docs.values():
            q[d.quality] += 1
        return {"docs": len(self.docs), "chunks": len(self.chunk_text), "vectors": len(self.vec), **q}
