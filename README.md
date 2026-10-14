# Bilingual conference study / İki dilli bildiri çalışması

## Installation / Kurulum
Python 3.11 recommended. In PowerShell:

```powershell
python -m pip install -r requirements.txt
python experiment_d1.py
```

Quick check without CTGAN: `python experiment_d1.py --skip-ctgan`

Outputs: `results/d1_per_seed.csv`, `d1_summary.csv`, `paired_differences.csv`, and `failures.csv`.

CTGAN may take several minutes per seed on CPU. If CTGAN fails, the run is explicitly logged, not silently replaced. For the prior experiment, CTGAN was missing and therefore no CTGAN performance results were available. `prior_d1_per_seed.csv` is the original user-provided result file; it is not an output of the updated program. The abstract reports the earlier completed results, not hypothetical CTGAN results.

## Research caveats / Araştırma sınırlılıkları
- Oracle-informed simulation knows the true generating mechanism and has privileged information.
- The five seeds provide preliminary descriptive evidence, not strong inferential evidence.
- No test-set observations are used in fitting generators.
- SMOTE and CTGAN are trained on training data only; the synthetic-data category overlaps with augmentation conceptually.
- Run the new application before citing new PR-AUC or paired bootstrap outputs.
- Conference rules: English, 350–500 words, one page, no figures/tables/equations/references, 3–5 keywords.
