# Controlled Artificial Data Benchmark — executable exploratory package v1

## Status and limits

**EXPLORATORY / NOT A FINAL PAPER RESULT.** The included run has **3 seeds (0,1,2)**, five scenarios, 80-tree Random Forest, 750 model-fit + 250 validation + 2,000 test observations. Results are not merged with previous D4 runs, which use different generators/settings. A 10-seed run was attempted but exceeded this session's execution budget; the complete 3-seed run is retained. CTGAN was **not installed or run**. This is not a full CTGAN–SMOTE–simulation comparison yet.

## Commands

```bash
python -m pip install numpy pandas scipy scikit-learn imbalanced-learn
python -m unittest -v test_benchmark.py
python benchmark.py --seeds 3 --rf-trees 80 --output outputs
python analyze.py
# Optional after separately installing/pinning ctgan:
python benchmark.py --seeds 30 --rf-trees 250 --methods baseline class_weight smote oracle ctgan --output confirmatory_candidate
```

The final command is only a **candidate**; do not call it confirmatory until all protocol choices and CTGAN versions are frozen. Run the script from its own directory. A full 30-seed CTGAN run can require substantial compute.

## Experimental design

- Five scenarios: S1 overlapping imbalanced classes; S2a easy rings; S2b moderate rings; S3a nuisance-feature test shift; S3b informative-feature test intervention. See `generators.py`.
- Model fit 750, untouched validation 250, untouched test 2,000. Validation is used only for decision threshold selection. No preprocessing is fit on validation/test data.
- Class weighting is an important non-generative baseline.
- SMOTE adds exactly majority-minus-minority observations from training fold only, `k_neighbors=5`.
- Oracle draws class-1 examples from **the known training DGP**, which is privileged information unavailable in real-data applications; it is not a fair deployable method without an explicit oracle advantage disclaimer. Under S3a/S3b it does not access the shifted test DGP.
- CTGAN fits joint continuous features plus discrete label to the model-fit fold only, then conditions on label 1. CTGAN may fail due to dependency/API differences or inadequate conditional sampling; all failures must be reported, not silently excluded. Its global RNG handling is not fully isolated; production runs should audit determinism.
- Primary proposed downstream metric: **Average Precision (AP)**. ROC-AUC, F1 (default 0.5 and validation-selected threshold), balanced accuracy secondary. AP is threshold-free, and AP is not identical to trapezoidal PR-AUC.
- Generated-vs-reference fidelity compares **minority class only**. The independent reference is drawn from the unshifted *training* DGP (500 observations), not the test distribution. This evaluates fidelity to the training mechanism, not test-shift adaptation.
- MMD²: biased RBF estimate, gamma=1/median squared pairwise distance of reference-only standardized samples. Reference sample size 500; generated sample size depends on training imbalance. Its finite-sample positive bias means it need not be zero even for identical distributions.
- Correlation MAE: upper-triangle Pearson correlation difference. Nearest-train distance is a **diversity/proximity diagnostic**, not a privacy guarantee. Unique fraction rounded to 4 decimals is a weak duplicate check, not a sufficient diversity metric.
- Structural residual uses observed-feature relationships `x3≈x0*x1`, `x4≈x0²`, `x5≈sin(x2)`; observation noise means even true DGP samples have nonzero residual. Compare residual **against oracle/reference**, not against zero.
- Generation wall time excludes model training; method runtime is therefore **not a complete computational cost comparison**.
- Paired effect intervals are descriptive, based on only 3 seeds; not suitable for significance claims. Future confirmatory analysis requires more repeats, Holm multiplicity correction and explicit distinction between seed-level vs dataset-level inference.

## Output inventory

- `outputs/results_per_seed.csv`: downstream predictions summarized by method/model/threshold.
- `outputs/results_summary.csv`: mean and standard deviation across seeds.
- `outputs/quality_per_seed.csv`, `outputs/quality_summary.csv`: generative fidelity diagnostics.
- `outputs/paired_effects_exploratory.csv`: seed-paired differences versus baseline with exploratory intervals.
- `outputs/failures.csv`: generation/model errors (empty means no errors in the completed 3-seed run).
- `outputs/manifest.json`: settings and software versions.

## Remaining prerequisites before publication

1. Install and pin CTGAN + PyTorch and verify conditional generation/reproducibility on a single seed before scaling.
2. Freeze DGP parameters and the primary endpoint **without inspecting new augmentation outcomes**. S2b noise 0.85 was chosen from baseline-only pilot, so document its calibration.
3. Define S3b formally as informative-feature intervention; do not claim strict `P(Y|X)` invariance without proof.
4. Choose a computationally feasible 30-seed replication design, sample-size/imbalance sensitivity, and evaluate random effects across *different DGPs* if making general claims.
5. Add fidelity null/reference calibration and stronger distributional tests, plus model/generator runtime and memory.
6. Verify CTGAN is stable under scarce minority samples and report failure rates.
7. Freeze multiplicity family and analysis script, then run a new, independent set of seeds for final reporting.
8. Journal extension: independently sourced real datasets and an **estimated** simulator (not an oracle), with suitable assumptions and leakage controls.
