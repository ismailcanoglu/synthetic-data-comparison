# Research extension / Araştırma genişletmesi

## TR
- **D2:** Azınlık sınıfı halka, çoğunluk sınıfı merkezde; doğrusal olmayan karar sınırı.
- **D3:** Eğitim/test arasında `z2` nuisance özelliği kayıyor; etiket öncülü korunuyor. Bu bir covariate shift örneğidir; bütün dağılım kaymalarını temsil etmez.
- Baseline, SMOTE, CTGAN, oracle-informed simulation; her seed için eşit azınlık örneği ekleme bütçesi.
- Fidelity: bağımsız oracle azınlık referansına göre RBF MMD². **Düşük daha iyi.** Bu referans sadece kontrollü deneyde erişilebilir ve CTGAN/SMOTE eğitiminde kullanılmaz.
- Performans: minority F1, macro F1, balanced accuracy, AUROC, PR-AUC.
- Her senaryo 5 seed ile başlar. Bootstrap güven aralıkları keşifseldir; 5 seed ile güçlü anlamlılık iddiası yapılmamalı.
- D1 önceki deneydir; D2/D3 sonuçları mevcut değildir ve çalıştırılmalıdır.
- CTGAN sürümüne bağlı `enable_gpu` argümanı gerekirse `cuda=False` ile değiştirilebilir.

## EN
- **D2:** Ring-shaped minority class vs central majority class; nonlinear decision boundary.
- **D3:** A nuisance feature shifts between train and test while the class prior stays constant; a controlled covariate-shift scenario.
- Baseline, SMOTE, CTGAN and oracle-informed simulation; matched added minority-sample budgets.
- Fidelity: RBF MMD² against an independent oracle minority reference (lower is better), not used for training.
- Predictive metrics: minority F1, macro F1, balanced accuracy, AUROC and PR-AUC.
- Five paired seeds initially; confidence intervals are exploratory, not definitive significance evidence.
- Oracle simulation has privileged access to the DGP and is an upper-reference comparator, not a fair real-world deployment competitor.

## Run / Çalıştırma
```powershell
python -m pip install -r requirements.txt
python experiment_d2_d3.py --skip-ctgan --seeds 11
python experiment_d2_d3.py --epochs 100
```
Outputs under `results/`: `d2_d3_per_seed.csv`, `d2_d3_summary.csv`, `d2_d3_quality.csv`, `d2_d3_paired_differences.csv`, `d2_d3_failures.csv`.

## Caution / Dikkat
This is a new controlled simulation study, not an independent real-world validation. Fidelity and predictive quality are different targets; the proposed association must be calculated from outputs rather than assumed. The original D1 DGP has a label-conditional feature mechanism; the D3 generator likewise explicitly defines the shift. Preserve generated data, seeds, package versions and exact train/test splits when publishing.
