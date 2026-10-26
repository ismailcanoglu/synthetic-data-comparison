# Generator separability pilot — diagnostic, not confirmatory

Run `python pilot.py` in an environment with numpy, pandas and scikit-learn. The script imports original generator functions from `../audit_stage1/D4_extension/` and does not use SMOTE or CTGAN.

Settings: seeds 0–9, n_train=1000, n_test=2000, Random Forest 80 trees, LR standardization and max_iter=2000, fixed threshold 0.5. This is **not** an exact D4 reproduction (D4 used 250 trees and five different seeds).

Results are in `pilot_per_seed.csv` and `pilot_summary.csv`.

Observations: D2 RF nearly perfectly separates classes; adjust D2 noise/ring overlap before final study if aiming to measure meaningful augmentation differences. D3 shifts a nuisance variable and has small AP/ROC-AUC differences vs D1, so do not claim a general domain-shift robustness evaluation. For D1/D3, ranking metrics can be materially stronger than F1 at fixed threshold.

Next: preregister a severity grid for class overlap and informative vs nuisance covariate shifts; verify marginal distributions and label mechanisms; then freeze generators and execute larger seeded diagnostic study. Do not tune generator difficulty based on preferred SMOTE/CTGAN rankings.
