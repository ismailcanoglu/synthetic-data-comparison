# D4 — Decision thresholds, class weights and augmentation

## Research question / Araştırma sorusu
Does apparent minority-F1 improvement reflect new training information, or decision-boundary/threshold changes?
Azınlık F1 artışı yeni eğitim bilgisinden mi yoksa karar sınırı/eşik değişiminden mi kaynaklanıyor?

## Design
- D1/D2/D3 data-generating functions are reused unchanged.
- For each seed: 1,000 original training records split into 750 fitting / 250 validation records, stratified; 2,000 untouched test records.
- Fit-only oversampling (SMOTE), conditional CTGAN, and oracle minority simulation. Never augment validation/test.
- Baseline, balanced class weights, SMOTE, oracle simulation, CTGAN (optional).
- Logistic Regression and Random Forest; fixed 0.5 and validation-selected thresholds.
- Validation threshold grid 0.05..0.95, maximizing minority F1; the same fixed test set is evaluated once per model.
- Report F1, precision, recall, PR-AUC, AUROC, balanced accuracy; paired per-seed differences.
- Oracle simulation uses the known generative process, so is an informed reference, not a fair information-matched competitor.

## Install and run
```powershell
python -m pip install numpy pandas scikit-learn imbalanced-learn ctgan torch
python experiment_d4.py --skip-ctgan --seeds 11
python experiment_d4.py --epochs 100
```

Results are written under `results_d4/` and do not overwrite D1/D2/D3 results. D4 scores **must not be directly compared** with earlier experiments as the fit set is smaller. Compare methods **within D4**, using identical splits. For a stronger paper, expand to 20–30 independent seeds and consider a nested-validation design; tuning and comparing on the same validation set can favor more flexible methods. D3's shift is a nuisance-variable shift and may not be a strong concept shift.

## Interpretation
- If threshold tuning improves minority F1 without improving PR-AUC, it primarily changes operating point, not score ranking.
- Class weighting changes the loss and learned boundary without adding examples.
- Augmentation gains beyond tuned/weighted baselines are more informative than gains beyond default-only baselines.
