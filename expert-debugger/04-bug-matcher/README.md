# 4 · Bug matcher (fine-tuned local 8B)

![evolution](fig-matcher-evolution.png)

**Champion:** Qwen3-8B, LoRA SFT v3. **88/119 (73.9%) top-1** on the held-out test, **92/119 (77.3%)** with the confidence gate,
and **66.7% on unseen bugs** vs 41.7% for a commercial LLM agent. It runs on one local GPU at zero API cost.

## How it got there
| Version | Change | Correct /119 |
|---|---|---|
| Zero-shot 8B | prompt only | 72 |
| SFT v1 | human decisions as targets | 74 |
| SFT v2 | + counterfactual candidates (gold removed → NONE) | 81 |
| **SFT v3** | + STaR-style rationales, out-of-fold ranker scores | **88** |
| Commercial LLM agent | for reference | 68 |

## Training recipe
- Prompt: run evidence + the top-K candidate bug cards with ranker scores. Answer: a short rationale, then `BUG-ID` or `NONE`.
- Loss on completion tokens only, LoRA r=16, 3 epochs, LR 1e-4, 6k context, trained with Unsloth on XPU.
- Confidence gate: auto-LINK at p ≥ 0.7, otherwise human review.

## What didn't work
![rejected](fig-rejected.png)

Self-consistency voting (−3), GEPA prompt optimisation (+2, within noise), a finite-state agent (0), bug-ID aliasing (−1),
and a 14B model (+1, not worth twice the compute). Because the 8B is as good, it gets more iterations.

## Files
`matcher.py` (prompt, counterfactuals, rationales, Unsloth `train`, parse, `gated_pick`) · `bug_matcher_colab.ipynb`
(GPU, Qwen3-0.6B on a Colab T4, the same code as the 8B)

---
[← Back to the overview](../README.md) · The notebook runs on synthetic data; the numbers above come from the real (private) evaluation.
