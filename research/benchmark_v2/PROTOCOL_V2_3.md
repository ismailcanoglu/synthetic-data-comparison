# Benchmark v2.3 — inference implementation review

**NOT YET CONFIRMATORY.** This branch adds paired t inference and Holm correction, but does not implement the full 30-seed runner or freeze CTGAN epoch settings.

## Primary analysis

- For each scenario × classifier × non-baseline method: 30 within-seed AP differences versus baseline.
- Two-sided paired t test (29 degrees of freedom), unadjusted paired t 95% CI.
- One Holm family across 40 comparisons (5 scenarios × 2 classifiers × 4 non-baseline methods).
- CI are **not multiplicity-adjusted**. Holm-adjusted p-values are reported separately.
- Requires all 30 seeds, all five methods, both models, both threshold policies, consistent test fingerprints, and no duplicate rows.
- AP uses only the default-policy row; validation-tuned AP would be a duplicate threshold-independent observation.
- Assumptions include independence across seeds and a suitable sampling distribution of paired differences. Report diagnostics and consider robustness/sensitivity analysis before publication.

## Tests

```powershell
python -m pytest research/benchmark_v2/test_paired_inference_v2_3.py -q
```

## Not ready for main run

The v2 benchmark currently trains one CTGAN per scenario/seed/epoch and evaluates both classifiers on the same synthetic data. LR=100 and RF=10 require separate CTGAN training and model-specific outputs, or a pre-specified common epoch rule. Neither is yet implemented.

Next: implement a runner with **new** evaluation seeds, fixed epochs selected using development data only, complete output validation, failure policy, and frozen config. Only then execute the main experiment.
