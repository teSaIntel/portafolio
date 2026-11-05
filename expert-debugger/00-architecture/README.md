# End-to-end architecture and data pipeline

![headline](fig-headline.png)

**Results.** On held-out failures, the local stack (ranker → matcher → gate) gets the right bug top-1 **77.3%** of the time,
at **0 API cost**, vs **73.1%** for a commercial LLM agent. On bugs never seen in training it gets **66.7%** vs **41.7%**.

## How the pieces connect
1. **Evidence**: run metadata, log skeletons and telemetry within ±15 min of the failure.
2. **Test-issue detector** (02): takes out infra and test-script failures early, at precision ≥ 0.9.
3. **Failure ranker** (01): scores every open bug against the run and keeps the top-K shortlist.
4. **Knowledge layer** (03): bug cards, BM25 + rank fusion and a bug graph. It widens the shortlist when the ranker misses.
5. **RCA agent** (05): a tool-using local 8B model that investigates and writes the root cause.
6. **Bug matcher** (04): a fine-tuned 8B model that reads the evidence and the RCA and picks a bug or NONE.
7. **Confidence gate**: auto-LINK when confident, otherwise human review. NONE with a root cause means a new bug.
8. **Feedback**: human decisions become new labels for the next retrain.

## Files
| File | What |
|---|---|
| `data_pipeline.py` | Collectors (interfaces), normalisation/skeletons, evidence window, grouped time splits, unseen mask, out-of-fold ranks |
| `synthetic.py` | Synthetic world generator: bugs, sibling bugs, a long tail, infra noise |
| `end_to_end_colab.ipynb` | Detector → ranker → gate on synthetic data, with decision counts and precision |
| `make_figures.py` | Regenerates every chart from the aggregate results |
| `build_notebooks.py` | Builds all Colab notebooks from the `.py` modules |

---
[← Back to the overview](../README.md) · The notebook runs on synthetic data; the numbers above come from the real (private) evaluation.
