"""Regenerates every figure of the portfolio from the aggregate results (no raw data needed).
python make_figures.py  -> PNGs next to each model's README."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, ACC, ACC2, MUTED, GOOD, BAD = "#1f2937", "#2563eb", "#7c3aed", "#9ca3af", "#059669", "#dc2626"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "figure.dpi": 140})


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, folder, name), bbox_inches="tight")
    plt.close(fig)


def bars(ax, labels, vals, colors, fmt="{:.1f}%", ymax=100):
    b = ax.bar(labels, vals, color=colors, width=.62)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + ymax * .015, fmt.format(v), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, ymax)


# 1. Headline: held-out test set, 119 bug-linked failures (same failures for every system)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
lab = ["No-LLM\nranker", "Commercial LLM\nagent + gate", "Local 8B\nmatcher", "Local 8B\nmatcher + gate"]
bars(ax[0], lab, [83 / 119 * 100, 87 / 119 * 100, 88 / 119 * 100, 92 / 119 * 100], [MUTED, ACC2, ACC, GOOD])
ax[0].set_title("Top-1 correct bug, held-out test (n=119)")
ax[0].set_ylabel("% of failures")
bars(ax[1], ["No-LLM ranker", "Commercial LLM\nagent", "Local 8B matcher"], [46 / 72 * 100, 30 / 72 * 100, 48 / 72 * 100], [MUTED, ACC2, ACC])
ax[1].set_title("Bugs never seen in training (n=72)")
save(fig, "00-architecture", "fig-headline.png")

# 2. Matcher evolution
fig, ax = plt.subplots(figsize=(8.5, 4.4))
steps = ["Zero-shot\n8B", "SFT v1\n(rejection\nsampling)", "SFT v2\n(cleaned data,\nSTaR gold)", "SFT v3\n(2x data,\nleak-free ranks)"]
vals = [72, 74, 81, 88]
ax.plot(steps, [v / 119 * 100 for v in vals], marker="o", lw=2.5, color=ACC)
for i, v in enumerate(vals):
    ax.annotate(f"{v / 119 * 100:.1f}%", (i, v / 119 * 100), textcoords="offset points", xytext=(0, 9), ha="center")
ax.axhline(68 / 119 * 100, ls="--", color=ACC2)
ax.text(3.05, 68 / 119 * 100 - 3.2, "commercial LLM agent", color=ACC2, ha="right", fontsize=9)
ax.set_ylim(50, 85)
ax.set_ylabel("top-1 correct (%)")
ax.set_title("Bug matcher: four training iterations, same test set")
save(fig, "04-bug-matcher", "fig-matcher-evolution.png")

# 3. What was tried for the matcher and rejected (delta in correct picks vs the champion of the day)
fig, ax = plt.subplots(figsize=(8.5, 4))
tried = ["Self-consistency\nvote (N=5)", "Prompt optimisation\n(GEPA)", "FSM agent\nvs ReAct", "Bug-id\naliasing", "14B instead\nof 8B"]
delta = [-3, 2, 0, -1, 1]
ax.bar(tried, delta, color=[BAD if d < 0 else MUTED for d in delta], width=.6)
ax.axhline(0, color=INK, lw=.8)
ax.axhspan(-3, 3, color=MUTED, alpha=.15)
ax.text(4.4, 2.4, "run-to-run noise", ha="right", fontsize=9, color=INK)
ax.set_ylabel("Δ correct picks")
ax.set_title("Ideas benchmarked and rejected (all within noise or worse)")
save(fig, "04-bug-matcher", "fig-rejected.png")

# 4. Ranker model scoreboard (same folds)
fig, ax = plt.subplots(figsize=(9, 4.4))
models = ["Phrase-only\nLR", "XGBoost\n(tuned)", "LR + LLM\npre-processing", "SVM", "LR structured\n+ phrases"]
t1, t3 = [71, 79, 84, 87, 87], [81, 91, 92, 96, 95]
x = range(len(models))
ax.bar([i - .2 for i in x], t1, width=.4, color=ACC, label="top-1")
ax.bar([i + .2 for i in x], t3, width=.4, color=MUTED, label="top-3")
ax.set_xticks(list(x), models)
ax.set_ylim(50, 100)
ax.legend(frameon=False)
ax.set_ylabel("%")
ax.set_title("Ranker model selection (grouped CV)")
save(fig, "01-failure-ranker", "fig-ranker-models.png")

fig, ax = plt.subplots(figsize=(7, 4))
bars(ax, ["Dense\nembeddings", "Fine-tuned\ncross-encoder", "Production\nranker"], [0.2, 71.7, 76.1], [BAD, MUTED, GOOD])
ax.set_title("Retrieval alternatives (top-1, same TRs)")
save(fig, "01-failure-ranker", "fig-retrieval-alternatives.png")

# 5. Detector
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
bars(ax[0], ["Text only", "Categorical\nonly", "Text +\ncategorical"], [63, 75, 78], [MUTED, MUTED, GOOD], fmt="{:.0f}")
ax[0].set_title("Average precision (×100)")
bars(ax[1], ["Hand rules\n(P=0.34)", "Model\n(P≥0.90)"], [23, 33], [BAD, GOOD], fmt="{:.0f}%", ymax=50)
ax[1].set_title("Recall at the operating point")
save(fig, "02-test-issue-detector", "fig-detector.png")

# 6. Knowledge layer
fig, ax = plt.subplots(figsize=(8.5, 4.2))
bars(ax, ["Candidate recall\n(hybrid KB)", "+ query\nrewriting", "Graph expert\ntop-5", "Root-cause-first\nsearch, far misses"],
     [95.2, 95.9, 65, 75], [ACC, ACC, MUTED, GOOD])
ax.set_title("Knowledge layer: where it helps")
save(fig, "03-knowledge-layer", "fig-knowledge.png")

# 7. RCA agent: local zero-shot writer vs commercial model, same failures, same matcher
fig, ax = plt.subplots(figsize=(8.5, 4.2))
splits = ["Test", "Validation", "Virtualisation"]
loc, api = [73, 35, 12], [81, 32, 10]
x = range(3)
ax.bar([i - .2 for i in x], api, width=.4, color=ACC2, label="Commercial LLM RCA")
ax.bar([i + .2 for i in x], loc, width=.4, color=ACC, label="Local 8B RCA (0 API cost)")
for i in x:
    ax.text(i - .2, api[i] + 1, api[i], ha="center", fontsize=9)
    ax.text(i + .2, loc[i] + 1, loc[i], ha="center", fontsize=9)
ax.set_xticks(list(x), splits)
ax.set_ylabel("correct bug picks downstream")
ax.legend(frameon=False)
ax.set_title("Root-cause writer: local model vs commercial API")
save(fig, "05-rca-writer", "fig-writer.png")

# 8. Debugger second look (one domain, rescue = ranker wrong at top-1 but truth in top-3)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
bars(ax, ["Ranker only", "Deterministic\nre-ranker", "+ LLM judge", "+ evidence tiers\n+ situational flags"], [0, 56, 78, 89], [MUTED, ACC, ACC2, GOOD], fmt="{:.0f}%")
ax.set_title("Second-look debugger: rescue rate on ranker misses")
save(fig, "05-rca-writer", "fig-second-look.png")

# 9. Bug classifier champion (fleet snapshot, 1,591 runs / 292 bug-linked) and its open-set router
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
bars(ax[0], ["Top-1", "Top-3", "LINK\nprecision", "+ override\nrule (top-1)"], [78.8, 90.8, 85.4, 81.5], [ACC, ACC, GOOD, ACC2])
ax[0].set_title("Suggested-bugs champion, fleet-wide (no LLM)")
bars(ax[1], ["Singleton sets\n(auto LINK)", "Accuracy of\nsingletons", "Coverage\n(alpha .1)", "NEW recall\n(real new bugs)"], [82, 97.8, 90, 70], [ACC, GOOD, MUTED, ACC2], fmt="{:.0f}%", ymax=112)
ax[1].set_title("Conformal open-set router (AUROC seen vs new .92)")
save(fig, "01-failure-ranker", "fig-classifier.png")

# 10. Knowledge base retrieval and the leakage that was caught; knowledge graph build
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
bars(ax[0], ["BM25", "Dense", "Hybrid\n(RRF)"], [72, 78.5, 79.3], [MUTED, ACC, GOOD], fmt="{:.1f}%")
ax[0].set_title("KB: comment -> bug retrieval, top-1 (n=135)")
bars(ax[1], ["Looked like\n(leaky)", "Leak-free", "Pinned\nbaseline"], [57.8, 49.0, 52.4], [BAD, ACC, MUTED], fmt="{:.1f}%")
ax[1].set_title("KB as a ranking feature: the gain was leakage")
save(fig, "03-knowledge-layer", "fig-kb.png")
print("figures written")
