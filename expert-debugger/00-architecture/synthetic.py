"""Synthetic failure generator for the demo notebooks (no real data is published).
Each bug has a signature (2-3 log lines), a domain, an owner and a test family; runs of that bug emit the signature
mixed with background noise, sometimes miss a line, and 'sibling' bugs share most of their signature, which is the
hard case in the real system. Not-a-bug runs are infrastructure or test-content problems."""
import random
from datetime import datetime, timedelta

DOMAINS = ["power", "memory", "reset", "virtualization", "media", "compute", "fabric", "firmware"]
COMPONENTS = ["gt", "dma", "pcie", "fw", "mem", "sched", "link", "irq", "pmu", "vf"]
SYMPTOMS = ["engine reset", "timeout waiting for fence", "page fault at <HEX>", "device lost", "link training failed",
            "ECC uncorrectable", "watchdog expired", "CRC mismatch", "thermal trip", "context banned"]
NOISE = ["test step <N> started", "allocating <N> buffers", "INFO: iteration <N> passed", "loading module", "heartbeat ok",
         "perf counter sample <N>", "config applied", "mounting share", "INFO: retrying connection"]
GENERIC = ["FAIL: test step <N> returned non-zero", "ERROR: timeout waiting for test harness", "FAIL: assertion in test script"]
INFRA = ["ERROR: node unreachable", "ERROR: container image pull failed", "FAIL: missing test binary", "ERROR: disk quota exceeded"]


def make_world(n_bugs=40, n_runs=1200, p_none=.18, seed=0):
    rng = random.Random(seed)
    t0 = datetime(2026, 1, 1)
    bugs = []
    for i in range(n_bugs):
        sib = bugs[rng.randrange(len(bugs))] if bugs and rng.random() < .35 else None
        comp = sib["component"] if sib else rng.choice(COMPONENTS)
        sig = list(sib["signature"][:2]) if sib else [f"{comp}: {rng.choice(SYMPTOMS)}", f"{comp}: {rng.choice(SYMPTOMS)} on <PCI>"]
        sig.append(f"{comp}: first fault code {rng.randrange(16, 255):#x}")
        bugs.append({"id": f"BUG-{i:03d}", "domain": sib["domain"] if sib else rng.choice(DOMAINS), "component": comp,
                     "owner": f"owner{rng.randrange(8)}", "family": f"T{rng.randrange(12)}", "signature": sig,
                     "created": t0 + timedelta(days=rng.randrange(0, 200)),
                     "title": f"[{comp}] {sig[0].split(': ', 1)[1]}", "root_cause": f"{comp} {sig[-1].split(': ', 1)[1]}"})
    w = [1 / (i + 1) ** .8 for i in range(n_bugs)]          # long tail: a few bugs own most failures
    runs = []
    for j in range(n_runs):
        start = t0 + timedelta(days=rng.uniform(0, 260))
        if rng.random() < p_none:
            lines, label, b = ([rng.choice(INFRA)] if rng.random() < .6 else [rng.choice(GENERIC)]), "NONE", None
        else:
            b = rng.choices(bugs, w)[0]
            if b["created"] > start + timedelta(days=30):
                start = b["created"] + timedelta(days=rng.uniform(0, 30))
            lines, label = [s for s in b["signature"] if rng.random() > .3] or b["signature"][:1], b["id"]
            if rng.random() < .25:
                lines.append(rng.choice(GENERIC + INFRA))      # product bugs also trip generic/infra-looking lines
        lines += rng.sample(NOISE, 4)
        rng.shuffle(lines)
        runs.append({"run": f"RUN-{j:05d}", "label": label, "start": start, "lines": lines,
                     "domain": b["domain"] if b else rng.choice(DOMAINS), "owner": b["owner"] if b else f"owner{rng.randrange(8)}",
                     "family": b["family"] if b and rng.random() < .7 else f"T{rng.randrange(12)}"})
    return bugs, runs


def split(runs, bugs, seed=0):
    """Grouped by bug: 30% of bugs are test-only, so the test split measures bugs never seen in training."""
    rng = random.Random(seed)
    ids = [b["id"] for b in bugs]
    rng.shuffle(ids)
    test_bugs = set(ids[: int(.3 * len(ids))])
    for r in runs:
        r["split"] = "test" if (r["label"] in test_bugs or (r["label"] == "NONE" and rng.random() < .3)) else "train"
        # frequent bugs also appear in test (seen-bug slice)
        if r["label"] not in test_bugs and r["label"] != "NONE" and rng.random() < .2:
            r["split"] = "test"
    return runs
