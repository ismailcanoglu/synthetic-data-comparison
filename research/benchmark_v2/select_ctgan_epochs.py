"""Select CTGAN epochs on validation AP only; never inspect test data.

Usage:
python research/benchmark_v2/select_ctgan_epochs.py --seeds 5 --scenarios s1 --epochs 10 50 100
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_v2 import Settings, augment, derived_seed, sample


def choose_epochs(seeds, scenarios, epochs, output, trees=80):
    if seeds < 1 or any(e < 1 for e in epochs):
        raise ValueError("seeds and epochs must be positive")
    records = []
    for scenario in scenarios:
        for seed in range(seeds):
            # Deliberately no test sample or test scores in this module.
            xa, ya = sample(1000, seed, scenario, "train", Settings())
            xf, xv, yf, yv = train_test_split(
                xa, ya, train_size=750, test_size=250,
                random_state=derived_seed(seed, 10), stratify=ya)
            for epoch in epochs:
                generated = augment("ctgan", xf, yf, scenario, seed, Settings(), epoch)
                xt = np.vstack((xf, generated))
                yt = np.r_[yf, np.ones(len(generated), dtype=int)]
                models = {
                    "LR": make_pipeline(StandardScaler(), LogisticRegression(
                        max_iter=2000, random_state=derived_seed(seed, 82))),
                    "RF": RandomForestClassifier(n_estimators=trees, n_jobs=-1,
                        random_state=derived_seed(seed, 83)),
                }
                for name, model in models.items():
                    model.fit(xt, yt)
                    score = average_precision_score(yv, model.predict_proba(xv)[:, 1])
                    records.append(dict(scenario=scenario, seed=seed, epoch=epoch,
                                        model=name, validation_ap=score))
    df = pd.DataFrame(records)
    summary = df.groupby(["scenario", "model", "epoch"], as_index=False).agg(
        mean_validation_ap=("validation_ap", "mean"),
        sd_validation_ap=("validation_ap", "std"),
        n=("seed", "nunique"))
    # Deterministic tie-break: prefer fewer epochs.
    selected = (summary.sort_values(["scenario", "model", "mean_validation_ap", "epoch"],
                                    ascending=[True, True, False, True])
                .drop_duplicates(["scenario", "model"]))
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "validation_per_seed.csv", index=False)
    summary.to_csv(output / "validation_summary.csv", index=False)
    selected.to_csv(output / "selected_epochs.csv", index=False)
    (output / "selection_manifest.json").write_text(json.dumps({
        "status": "EXPLORATORY_VALIDATION_SELECTION",
        "metric": "mean_validation_average_precision",
        "tie_break": "lowest_epoch",
        "seeds": list(range(seeds)), "scenarios": scenarios, "epochs": epochs,
        "warning": "Selection reuses a single validation set per seed; do not report it as unbiased performance."
    }, indent=2), encoding="utf-8")
    return selected


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--scenarios", nargs="+", default=["s1"])
    p.add_argument("--epochs", type=int, nargs="+", default=[10, 50, 100])
    p.add_argument("--rf-trees", type=int, default=80)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "epoch_selection")
    a = p.parse_args()
    print(choose_epochs(a.seeds, a.scenarios, a.epochs, a.output, a.rf_trees).to_string(index=False))
