# Benchmark v2.1 — exploratory review, NOT a frozen confirmatory protocol

This branch extends benchmark v2 without overwriting the v2 implementation or any local results.

## Validation-only CTGAN epoch exploration

```powershell
python research/benchmark_v2/select_ctgan_epochs.py --seeds 5 --scenarios s1 --epochs 10 50 100 --output research/benchmark_v2/epoch_selection
```

Outputs `validation_per_seed.csv`, `validation_summary.csv`, `selected_epochs.csv` and `selection_manifest.json`. The selector never generates or scores a test set. It ranks epochs by mean validation AP separately for LR and RF; ties prefer fewer epochs. Training is repeated per epoch and seed. Selection on this validation set makes its reported score optimistic.

## Existing multi-seed paired analysis

```powershell
python research/benchmark_v2/analyze_paired.py research/benchmark_v2/outputs_ctgan_10_5seeds research/benchmark_v2/outputs_ctgan_50_5seeds research/benchmark_v2/outputs_ctgan_100_5seeds --output research/benchmark_v2/paired_analysis
```

Pairs CTGAN and baseline by scenario, seed, model and epoch, checks identical test SHA-256, and computes within-seed AP/F1/ROC-AUC/balanced-accuracy differences. Uses only the `default` threshold policy to avoid counting duplicate AP measurements. **These are exploratory test-set analyses; do not use their results to choose the final epoch.**

## Before the main 30-seed study

1. Reserve an independent tuning/selection dataset or use nested development splits. Keep the final test sets untouched for model/epoch selection. Threshold tuning and epoch selection on the same validation set may induce selection bias.
2. Choose and freeze CTGAN training procedure, epochs, model settings, minority augmentation budget, data generator settings, metrics, failure handling and paired statistical plan *before* confirmatory evaluation.
3. Clarify whether epochs are selected per classifier or a single shared epoch; predefine the rule.
4. Evaluate all five scenarios and all five methods using identical per-seed held-out test sets. Store split fingerprints, package versions, generator configuration and code commit.
5. Report paired seed differences and uncertainty intervals; avoid treating two threshold-policy rows or repeated epoch measurements as independent samples.
6. Expand quality reporting to distributions and class-conditional fidelity, not only mean structural residual gap.

**No experiment results or tests have been executed by the GitHub connector.** Run the commands locally and inspect failures before considering merge.
