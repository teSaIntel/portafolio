"""Investigation stage as an explicit agent graph (curated, model calls are pluggable callables).

    XP preload -> Writer -> loop { Verifier -> Done? -> Revise } -> (matcher runs on the final answer)

The writer and reviser are LLM calls in the real system; here they are plain functions so the control flow, the deterministic
verifier and the Done exit can be read and tested on their own."""
import re
from dataclasses import dataclass, field

SEVERE = re.compile(r"ecc uncorrectable|watchdog|thermal trip|device lost|hang|fatal|link training failed", re.I)


def xp_preload(failing: list[str], passing: list[str], card: str = "") -> dict:
    """Expert preload: what changed against a known-good run, plus the test definition card (no model involved)."""
    good = set(passing)
    return {"lines": failing, "new_vs_good": [l for l in failing if l not in good], "card": card}


def verify(answer: dict, ev: dict, min_grounded: float = .5) -> list[str]:
    """Deterministic checks on the writer's answer; returns the issues that should trigger a revision."""
    issues = []
    text = " ".join([answer.get("mechanism", "")] + answer.get("chain", [])).lower()
    missed = [l for l in ev["lines"] if SEVERE.search(l) and not any(w in text for w in re.findall(r"[a-z]{4,}", l.lower())[:3])]
    if missed:
        issues.append(f"severe signature never mentioned: {missed[0]}")
    chain = answer.get("chain", [])
    if chain and sum(q in ev["lines"] for q in chain) / len(chain) < min_grounded:
        issues.append("evidence quotes not found in the given evidence")
    return issues


@dataclass
class Trace:
    answer: dict
    done: str                       # verified | stable | revise_no_json | max_steps
    rounds: list = field(default_factory=list)


def run_graph(ev: dict, write, revise, max_steps: int = 2) -> Trace:
    """write(ev) -> answer; revise(ev, answer, issues) -> answer or None. Exits through a Done reason, never loops forever."""
    answer = write(ev)
    rounds = []
    for _ in range(max_steps):
        issues = verify(answer, ev)
        rounds.append({"issues": issues})
        if not issues:
            return Trace(answer, "verified", rounds)
        new = revise(ev, answer, issues)
        if new is None:
            return Trace(answer, "revise_no_json", rounds)
        if new == answer:
            return Trace(answer, "stable", rounds)
        answer = new
    return Trace(answer, "max_steps", rounds)


def kg_facts(run: dict, bug: dict) -> dict:
    """Deterministic graph facts per candidate bug, fed to the matcher next to the root cause."""
    return {"same_family": run["family"] == bug["family"], "same_owner": run["owner"] == bug["owner"],
            "filed_before_run": bug["created"] <= run["start"]}
