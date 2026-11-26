"""Chart for the latest champion system, from aggregate results (no raw data needed).
python make_fig_champion.py  -> fig-champion.png next to this file."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, ACC, ACC2, MUTED, GOOD = "#1f2937", "#2563eb", "#7c3aed", "#9ca3af", "#059669"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "figure.dpi": 140})


def bars(ax, labels, vals, colors):
    b = ax.bar(labels, vals, color=colors, width=.62)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, 100)


fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
bars(ax[0], ["Ranker\n(no root cause)", "First 8B\nwriter (SFT)", "Champion 8B\nwriter (DPO)", "Commercial\nLLM agent"],
     [137 / 197 * 100, 127 / 197 * 100, 129 / 197 * 100, 131 / 197 * 100], [MUTED, ACC2, ACC, MUTED])
ax[0].set_title("Writer alone, same 197 failures")
ax[0].set_ylabel("% correct bug")
bars(ax[1], ["Ranker only", "Writer + matcher\n+ expert preload"], [192 / 258 * 100, 200 / 258 * 100], [MUTED, GOOD])
ax[1].set_title("Full chain, 258 failures")
x = [0, 1]
w = .36
for i, (lab, vals, col) in enumerate([("Ranker only", [114 / 175 * 100, 51 / 103 * 100], MUTED),
                                      ("Writer + matcher", [120 / 175 * 100, 59 / 103 * 100], GOOD)]):
    pos = [p + (i - .5) * w for p in x]
    b = ax[2].bar(pos, vals, width=w, color=col, label=lab)
    for r, v in zip(b, vals):
        ax[2].text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10, color=INK)
ax[2].set_xticks(x, ["All newest cycles\n(n=175)", "Bugs never seen\n(n=103)"])
ax[2].set_ylim(0, 100)
ax[2].set_title("Held-out newest cycles")
ax[2].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig-champion.png"), bbox_inches="tight")

# Writer lineage: behaviour cloning only vs the DPO champion, same 197 failures
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
bars(ax[0], ["First writer\n(SFT only)", "Champion\n(SFT + DPO)", "Commercial\nLLM agent"], [127 / 197 * 100, 129 / 197 * 100, 131 / 197 * 100],
     [ACC2, ACC, MUTED])
ax[0].set_title("Correct bug, 197 failures")
bars(ax[1], ["First writer\n(SFT only)", "Champion\n(SFT + DPO)"], [80 / 119 * 100, 85 / 119 * 100], [ACC2, ACC])
ax[1].set_title("Main test split (n=119)")
for a, (t, v) in zip([ax[2]], [("Ranker's right answer wrongly abandoned", [24, 21])]):
    b = a.bar(["First writer\n(SFT only)", "Champion\n(SFT + DPO)"], v, color=[ACC2, ACC], width=.62)
    for r, x in zip(b, v):
        a.text(r.get_x() + r.get_width() / 2, x + .6, str(x), ha="center", fontsize=10, color=INK)
    a.set_title(t, fontsize=11); a.set_ylim(0, 30)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "05-rca-writer", "fig-writer-champion.png"), bbox_inches="tight")
