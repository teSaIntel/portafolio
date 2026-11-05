"""Knowledge graph: who/what/where around every bug, built deterministically from labelled tracker data (no LLM).

Layers (one Postgres): documents (kb_store) | curated graph tables kg.node / kg.edge (source of truth, hash-diffed,
provenance + confidence per edge) | Apache AGE openCypher projection rebuilt from kg.* on every run.
GraphRAG was considered and rejected: its indexing needs an LLM per document, while here nodes and edges come
from structured, human-labelled data. Its useful idea (per-domain communities) is kept as `domain_view`.

    extract ─► purify (every rule counts into a report) ─► load (node hash diff) ─► project (Cypher, UNWIND batches)
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

LABELS = {"Run", "Bug", "Test", "Domain", "Owner", "Cycle", "Config", "Host", "Job", "Entity"}
EDGES = {  # type: (from, to, provenance, default confidence)
    "LINKED_TO": ("Run", "Bug", "tracker_link", 1.0),      # 0.7 when the run did not fail
    "OF_TEST": ("Run", "Test", "enrich", 1.0),
    "IN_DOMAIN": ("Test", "Domain", "test_registry", 1.0),
    "OWNED_BY": ("Test", "Owner", "test_registry", 1.0),
    "IN_CYCLE": ("Run", "Cycle", "enrich", 1.0),
    "ON_CONFIG": ("Run", "Config", "enrich", 1.0),
    "RAN_ON": ("Run", "Host", "enrich", 1.0),
    "BUILT_BY": ("Run", "Job", "enrich", 1.0),
    "BUG_IN_DOMAIN": ("Bug", "Domain", "inferred_majority", None),   # conf = share of the bug's runs in the domain
    "MENTIONS": ("Bug", "Entity", "text_mention", .6),
    "RELATED_TO": ("Bug", "Bug", "text_mention", .5),
}


@dataclass(frozen=True)
class Edge:
    src: str
    rel: str
    dst: str
    prov: str
    conf: float


def nid(label, key):
    return f"{label}:{key}"


# ---------------------------------------------------------------- extract (no judgement)
def extract(runs: list[dict], bugs: list[dict], kb=None) -> tuple[dict, list[Edge]]:
    nodes, edges = {}, []

    def node(label, key, **p):
        n = nid(label, key)
        nodes.setdefault(n, {"label": label, "key": key, **p})
        return n

    for b in bugs:
        bn = node("Bug", b["id"], title=b.get("title", ""), created=str(b["created"]))
        for other in b.get("mentions_bugs", []):
            edges.append(Edge(bn, "RELATED_TO", nid("Bug", other), "text_mention", .5))
        if kb is not None:
            ents = [e for d in kb.docs.values() if d.bug == b["id"] for k in ("symbol", "hex", "ticket") for e in d.entities.get(k, [])]
        else:
            ents = b.get("entities", [])
        for e in ents:
            edges.append(Edge(bn, "MENTIONS", node("Entity", e), "text_mention", .6))
    for r in runs:
        rn = node("Run", r["run"], start=str(r["start"]))
        tn = node("Test", r.get("test", r["family"]))
        edges += [Edge(rn, "OF_TEST", tn, "enrich", 1.0),
                  Edge(tn, "IN_DOMAIN", node("Domain", r["domain"]), "test_registry", 1.0),
                  Edge(tn, "OWNED_BY", node("Owner", r["owner"]), "test_registry", 1.0)]
        for k, rel, lab in (("host", "RAN_ON", "Host"), ("config", "ON_CONFIG", "Config"), ("cycle", "IN_CYCLE", "Cycle")):
            if r.get(k):
                edges.append(Edge(rn, rel, node(lab, r[k]), "enrich", 1.0))
        if r["label"] != "NONE":
            edges.append(Edge(rn, "LINKED_TO", nid("Bug", r["label"]), "tracker_link", 1.0 if r.get("failed", True) else .7))
    return nodes, edges


# ---------------------------------------------------------------- purify
def purify(nodes: dict, edges: list[Edge], hub_min: int = 8, hub_share: float = .05):
    rep = Counter()
    bugs = {n for n, v in nodes.items() if v["label"] == "Bug"}
    out = []
    for e in dict.fromkeys(edges):                                     # dedupe, keep order
        if e.src == e.dst:
            rep["self_loop"] += 1; continue
        if e.rel == "RELATED_TO" and e.dst not in bugs:               # numbers in text are often run/test ids
            rep["dangling_related"] += 1; continue
        if e.rel == "LINKED_TO" and e.dst not in bugs:                # a human link to an unknown bug: keep a stub
            nodes[e.dst] = {"label": "Bug", "key": e.dst.split(":", 1)[1], "stub": True}; bugs.add(e.dst); rep["stub"] += 1
        out.append(e)
    rep["duplicate"] = len(edges) - len(dict.fromkeys(edges))
    # hub filter: an entity in 1 bug connects nothing (stays in the KB); one in > max(8, 5% of bugs) is boilerplate
    ent_bugs = defaultdict(set)
    for e in out:
        if e.rel == "MENTIONS":
            ent_bugs[e.dst].add(e.src)
    cap = max(hub_min, hub_share * len(bugs))
    drop = {x for x, s in ent_bugs.items() if len(s) <= 1 or len(s) > cap}
    rep["singleton_entity"] = sum(len(ent_bugs[x]) <= 1 for x in drop)
    rep["hub_entity"] = len(drop) - rep["singleton_entity"]
    out = [e for e in out if e.dst not in drop]
    for x in drop:
        nodes.pop(x, None)
    # derived: bug domain by majority of its linked runs (confidence = share), test of each run
    run_test = {e.src: e.dst for e in out if e.rel == "OF_TEST"}
    test_dom = {e.src: e.dst for e in out if e.rel == "IN_DOMAIN"}
    votes = defaultdict(Counter)
    for e in out:
        if e.rel == "LINKED_TO" and e.src in run_test and run_test[e.src] in test_dom:
            votes[e.dst][test_dom[run_test[e.src]]] += 1
    for b, c in votes.items():
        d, n = c.most_common(1)[0]
        out.append(Edge(b, "BUG_IN_DOMAIN", d, "inferred_majority", round(n / sum(c.values()), 2)))
        rep["mixed_domain_bug"] += len(c) > 1
    return nodes, out, dict(rep)


# ---------------------------------------------------------------- load (hash diff) and project (openCypher)
def load(store: dict, nodes: dict, edges: list[Edge]) -> dict:
    h = {n: hashlib.sha1(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest() for n, v in nodes.items()}
    old = store.get("hash", {})
    diff = {"new": len(h.keys() - old.keys()), "removed": len(old.keys() - h.keys()),
            "changed": sum(1 for n in h.keys() & old.keys() if h[n] != old[n])}
    store.update(hash=h, nodes=nodes, edges=edges)                    # edges are replaced wholesale
    return diff


def project_cypher(nodes: dict, edges: list[Edge], graph: str = "hgraph", batch: int = 500) -> list[str]:
    """Statements for Apache AGE: drop+create the graph, one UNWIND per label / relation batch. AGE has no
    `SET n += map`, so each label carries a fixed property list."""
    q = [f"SELECT drop_graph('{graph}', true);", f"SELECT create_graph('{graph}');"]
    by_label = defaultdict(list)
    for n, v in nodes.items():
        by_label[v["label"]].append({"id": n, "key": v["key"]})
    for lab, rows in by_label.items():
        for i in range(0, len(rows), batch):
            lit = "[" + ",".join("{id:%s,key:%s}" % (json.dumps(r["id"]), json.dumps(str(r["key"]))) for r in rows[i:i + batch]) + "]"
            q.append(f"SELECT * FROM cypher('{graph}', $$ UNWIND {lit} AS r CREATE (:{lab} {{id:r.id, key:r.key}}) $$) AS (a agtype);")
    by_rel = defaultdict(list)
    for e in edges:
        by_rel[e.rel].append(e)
    for rel, es in by_rel.items():
        for i in range(0, len(es), batch):
            lit = "[" + ",".join("{s:%s,d:%s,p:%s,c:%s}" % (json.dumps(e.src), json.dumps(e.dst), json.dumps(e.prov), e.conf)
                                 for e in es[i:i + batch]) + "]"
            q.append(f"SELECT * FROM cypher('{graph}', $$ UNWIND {lit} AS r MATCH (a {{id:r.s}}), (b {{id:r.d}}) "
                     f"CREATE (a)-[:{rel} {{prov:r.p, conf:r.c}}]->(b) $$) AS (a agtype);")
    return q


# ---------------------------------------------------------------- read side (agent tools)
class KG:
    def __init__(self, nodes: dict, edges: list[Edge], run_time: dict | None = None):
        self.nodes, self.edges = nodes, edges
        self.out, self.inn = defaultdict(list), defaultdict(list)
        for e in edges:
            self.out[e.src].append(e); self.inn[e.dst].append(e)
        self.run_time = run_time or {}

    def _links(self, bug, cutoff=None, hide_run=None):
        """Runs linked to a bug, as visible at `cutoff`: the run being debugged and later links are hidden."""
        return [e.src for e in self.inn[bug] if e.rel == "LINKED_TO" and e.src != hide_run and
                (cutoff is None or self.run_time.get(e.src, cutoff) < cutoff)]

    def domain_view(self, domain: str, cutoff=None, hide_run=None) -> list[tuple[str, int]]:
        d = nid("Domain", domain)
        bugs = [e.src for e in self.inn[d] if e.rel == "BUG_IN_DOMAIN"]
        return sorted(((b, len(self._links(b, cutoff, hide_run))) for b in bugs), key=lambda x: -x[1])

    def bug_view(self, bug: str, cutoff=None, hide_run=None) -> dict:
        b = bug if ":" in bug else nid("Bug", bug)
        runs = self._links(b, cutoff, hide_run)
        tests = Counter(e.dst for r in runs for e in self.out[r] if e.rel == "OF_TEST")
        hosts = Counter(e.dst for r in runs for e in self.out[r] if e.rel == "RAN_ON")
        rel = [e.dst for e in self.out[b] if e.rel == "RELATED_TO"]
        return {"runs": len(runs), "tests": tests.most_common(5), "hosts": hosts.most_common(5), "related": rel,
                "domain": [(e.dst, e.conf) for e in self.out[b] if e.rel == "BUG_IN_DOMAIN"]}

    def expert(self, run: dict, kb, k: int = 5, cutoff=None) -> list[str]:
        """The graph picks the bugs of the run's domain, the KB ranks their text against the run's lines."""
        pool = {b.split(":", 1)[1] for b, _ in self.domain_view(run["domain"], cutoff, nid("Run", run["run"]))}
        hits = kb.search(" ".join(run["lines"]), exclude_run=run["run"], k=200, cutoff=cutoff)
        return [h["bug"] for h in hits if h["bug"] in pool][:k]

    def ppr(self, seeds: dict[str, float], damping: float = .5, iters: int = 30, cutoff=None, hide_run=None) -> dict[str, float]:
        """HippoRAG-style personalised PageRank from the run's entities (test, host, config, ...) over the
        undirected graph; mass that lands on Bug nodes ranks them. Redundant with the classifier's hand-built
        graph features (neutral as a feature), useful as an independent recall channel.
        Leak guard: the run being debugged and every link made at/after `cutoff` are invisible to the walk
        (without it, recall@5 is ~99%: the walk simply follows the run's own link)."""
        adj = defaultdict(list)
        for e in self.edges:
            if e.rel == "LINKED_TO" and (e.src == hide_run or (cutoff is not None and self.run_time.get(e.src, cutoff) >= cutoff)):
                continue
            adj[e.src].append((e.dst, e.conf)); adj[e.dst].append((e.src, e.conf))
        z = sum(seeds.values()) or 1
        reset = {n: v / z for n, v in seeds.items()}
        p = dict(reset)
        for _ in range(iters):
            nxt = defaultdict(float)
            for n, m in p.items():
                nb = adj.get(n, [])
                tot = sum(w for _, w in nb) or 1
                for x, w in nb:
                    nxt[x] += damping * m * w / tot
            for n, v in reset.items():
                nxt[n] += (1 - damping) * v
            p = nxt
        return {n.split(":", 1)[1]: v for n, v in sorted(p.items(), key=lambda x: -x[1]) if n.startswith("Bug:")}
