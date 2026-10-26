# Research artifacts — synthetic / simulated / augmented data

This folder contains **exploratory** material created after the initial D1–D4 study. The original files at the repository root remain unchanged.

## Status
- Earlier D1–D4 experiments: existing repository files.
- Generator v2: five controlled scenarios; pilot-calibrated, **not preregistered**.
- Benchmark v1: exploratory comparisons (baseline, class weighting, SMOTE, oracle simulation); CTGAN was not executed in the most recent environment.
- Do **not** interpret small-seed pilot differences as confirmatory significance.

## Methodological distinctions
- CTGAN: learned conditional/tabular generative method.
- SMOTE: interpolation-based oversampling.
- Oracle simulation: samples from the known data-generating process and therefore uses privileged information.
- S1: overlapping imbalanced classes; S2a/easy and S2b/moderate nonlinear rings; S3a nuisance-feature test shift; S3b informative-feature test shift.
- S3b must not be described as strict covariate shift without checking P(Y|X).
- Report `average_precision_score` as **average precision (AP)**, not an ambiguously defined PR-AUC.

## Next work
Run pinned-environment CTGAN, independent repetitions, fidelity/structural-diversity analyses, multiplicity-aware paired statistical comparisons, and final protocol freeze before publication.
