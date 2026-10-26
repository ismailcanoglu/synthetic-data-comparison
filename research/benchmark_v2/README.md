# Benchmark v2 — exploratory protocol

This is a **review branch**, not a frozen confirmatory experiment. It reuses `research/generator_v2/generators.py` without changing the existing pilot or benchmark v1.

## Smoke run

```powershell
python -m pytest research/benchmark_v2/test_benchmark_v2.py -v
python research/benchmark_v2/benchmark_v2.py --seeds 1 --scenarios s1 --ctgan-epochs 10 --output research/benchmark_v2/outputs_smoke
```

Outputs: `results_per_seed.csv`, `results_summary.csv`, `quality_per_seed.csv`, `splits.csv`, `failures.csv`, and `manifest.json`.

**Design:** 750 fit / 250 validation / 2000 test; validation-only F1 threshold selection; unchanged test data for all methods; 10 epochs CTGAN for plumbing only; seed-specific SHA-256 test fingerprints. Oracle draws from the *training* DGP only, not the shifted test DGP. CTGAN is fit on minority-class training observations to generate minority samples. Its class-conditional training differs from the mixed-class conditional CTGAN approach in benchmark v1, so compare results cautiously.

**Interpretation:** AP and ROC-AUC are threshold-independent and duplicated across the two threshold-policy rows: do not count those rows as independent replicates. Structural residual is computed on observed noisy features and compared with an independent reference sample; zero is not expected. The simple mean residual gap is descriptive, not a full fidelity test.

**Before confirmatory runs:** inspect the augmentation budget, CTGAN architecture and epoch selection, test shift assumptions, multi-seed reproducibility, quality metric choice, statistical analysis plan, and dataset scenario definitions. Commit a protocol freeze separately. Outputs should not be interpreted as confirmatory evidence.

**Note:** Creating this branch remotely does not alter an existing local `feature/benchmark-integration` branch.
