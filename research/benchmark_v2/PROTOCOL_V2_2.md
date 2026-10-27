# Benchmark v2.2 — Confirmatory protocol design (NOT YET FROZEN)

**Status: DESIGN / REVIEW.** Existing S1 results from seeds 0–4, including validation-selected epochs, are exploratory. Neither a new seed range nor this file alone establishes independence: all settings must be frozen before looking at the new test outcomes.

## Proposed design

- Scenarios: s1, s2a, s2b, s3a, s3b.
- Methods: baseline, class_weight, SMOTE, CTGAN, oracle.
- Classifiers: LR and RF; all method/classifier pairs use identical held-out test observations for each scenario/seed.
- Independent evaluation seeds: 10000–10029, **subject to audit that these have not been inspected**; development seeds: 0–4.
- Dataset sizes: fit 750, validation 250, test 2000 per scenario/seed; train/validation stratified; test generated separately.
- Primary endpoint: paired difference in test average precision (AP), method minus baseline, for each scenario/model. AP is threshold independent; report it only once, using the default policy row.
- Secondary endpoints: ROC-AUC, F1 at 0.5, F1 at validation-tuned threshold, balanced accuracy, structural residual gap and generation runtime. Report uncertainty and multiple comparisons transparently.
- CTGAN epochs: **NOT YET FROZEN**. Existing exploratory validation results suggest different values for LR/RF. Current benchmark v2 runs one shared epoch for both classifiers, so a per-classifier selection requires a separate implementation. Decide and implement one shared epoch rule or distinct per-model training before freeze.
- Augmentation budget: match majority class count in fit set, using fit data only.
- Threshold tuning: validation only, never test. Reusing validation for epoch and threshold selection can induce selection bias; ideally reserve separate selection/threshold splits.
- Primary statistical unit: seed, paired by scenario/model. There are 40 primary method-vs-baseline comparisons (4 methods × 2 models × 5 scenarios). Pre-specify Holm correction or a smaller primary family and a paired uncertainty interval procedure before reporting significance.
- Quality: class-conditional structural residual distributions, not only mean gap; MMD and correlation differences may be added after implementation and tests.
- Failure handling: predeclare all failures; do not silently drop seeds or replace missing values. Freeze an intention-to-evaluate policy before main runs.
- Reproducibility: record commit SHA, environment versions, generator settings, test fingerprints, epoch-selection decision and manifest.

## Preflight

`confirmatory_protocol.py` validates a *user-supplied* frozen JSON protocol and requires complete result coverage and consistent test hashes. Its analysis computes descriptive paired AP differences and **normal-approximation intervals only**, not corrected significance tests. It does **not** verify whether a freeze actually preceded the experiment and does **not** launch training. Do not claim the 30-seed study is confirmatory until the design choices above are resolved and the implementation is reviewed.

Run unit tests:

```powershell
python -m pytest research/benchmark_v2/test_confirmatory_protocol.py -q
```

**Next implementation:** separate development-only epoch selection from the confirmatory runner; support evaluation seed offsets and fixed CTGAN epoch rule; validate all output rows and failure policy; add multiplicity-corrected inference and quality-distribution summaries; then freeze the full configuration and run.
