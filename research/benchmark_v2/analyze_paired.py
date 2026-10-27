"""Paired exploratory analysis of per-seed results, never summary rows."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def analyze(paths, output):
    frames = []
    for path in paths:
        folder = Path(path)
        manifest = pd.read_json(folder / "manifest.json", typ="series")
        epoch = int(manifest["ctgan_epochs"])
        df = pd.read_csv(folder / "results_per_seed.csv")
        df = df[df["threshold_policy"] == "default"].copy()
        df["epoch"] = epoch
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)
    keys = ["scenario", "seed", "model"]
    baseline = all_data[all_data.method == "baseline"]
    ctgan = all_data[all_data.method == "ctgan"]
    merged = ctgan.merge(baseline, on=keys + ["epoch"], suffixes=("_ctgan", "_baseline"),
                         validate="one_to_one")
    if len(merged) != len(ctgan):
        raise ValueError("Missing baseline pairs")
    if not (merged.test_sha256_ctgan == merged.test_sha256_baseline).all():
        raise ValueError("Test fingerprints mismatch")
    for metric in ("ap", "roc_auc", "f1", "balanced_accuracy"):
        merged[f"{metric}_difference"] = merged[f"{metric}_ctgan"] - merged[f"{metric}_baseline"]
    # Repeated epochs use the same seeds; avoid treating them as independent replications.
    result = merged.groupby(["scenario", "model", "epoch"], as_index=False).agg(
        n_pairs=("seed", "nunique"),
        ap_diff_mean=("ap_difference", "mean"),
        ap_diff_sd=("ap_difference", "std"),
        f1_diff_mean=("f1_difference", "mean"),
        f1_diff_sd=("f1_difference", "std"))
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "paired_per_seed.csv", index=False)
    result.to_csv(output / "paired_summary.csv", index=False)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "paired_analysis")
    a = p.parse_args()
    print(analyze(a.folders, a.output).to_string(index=False))
