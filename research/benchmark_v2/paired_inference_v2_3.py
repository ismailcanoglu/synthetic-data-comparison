"""Pre-registered paired inference utilities for benchmark v2.3.

No training, test-set tuning, or automatic selection of epochs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from confirmatory_protocol import validate_protocol


def holm_adjust(pvalues):
    """Holm familywise-error correction, monotone and capped at one."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite numbers in [0, 1]")
    n = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        (n - np.arange(n)) * p[order]))
    return adjusted


def paired_inference(per_seed, protocol, output):
    """One pre-specified family of 40 AP comparisons; one row per pair."""
    validate_protocol(protocol)
    if protocol.get("multiplicity") != "holm_all_40_ap":
        raise ValueError("Freeze Holm correction across all 40 primary AP comparisons")
    if protocol.get("confidence_level") != 0.95:
        raise ValueError("Only frozen 95% confidence intervals supported")
    df = pd.read_csv(per_seed)
    keys = ["scenario", "seed", "method", "model", "threshold_policy"]
    required = set(keys + ["ap", "test_sha256"])
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df.duplicated(keys).any():
        raise ValueError("Duplicate observations")
    expected = {(s, seed, m, model, policy)
                for s in protocol["scenarios"] for seed in protocol["evaluation_seeds"]
                for m in protocol["methods"] for model in protocol["models"]
                for policy in ("default", "val_tuned")}
    if set(map(tuple, df[keys].itertuples(index=False, name=None))) != expected:
        raise ValueError("Missing or extra result rows")
    if (df.groupby(["scenario", "seed"])["test_sha256"].nunique() != 1).any():
        raise ValueError("Test fingerprint mismatch")
    df = df[df.threshold_policy == "default"]
    baseline = df[df.method == "baseline"]
    others = df[df.method != "baseline"]
    paired = others.merge(baseline, on=["scenario", "seed", "model"],
                          suffixes=("_method", "_baseline"), validate="many_to_one")
    if len(paired) != len(others):
        raise ValueError("Incomplete baseline pairs")
    paired["ap_difference"] = paired.ap_method - paired.ap_baseline
    rows = []
    for (scenario, model, method), group in paired.groupby(
            ["scenario", "model", "method_method"], sort=True):
        x = group.ap_difference.to_numpy(dtype=float)
        n = len(x)
        if n != 30 or not np.isfinite(x).all():
            raise ValueError("Expected 30 finite paired differences")
        mean = float(x.mean())
        sd = float(x.std(ddof=1))
        se = sd / np.sqrt(n)
        if se == 0:
            pvalue = 1. if mean == 0 else 0.
            lo = hi = mean
        else:
            stat = mean / se
            pvalue = float(2 * student_t.sf(abs(stat), n - 1))
            critical = float(student_t.ppf(.975, n - 1))
            lo, hi = mean - critical * se, mean + critical * se
        rows.append(dict(scenario=scenario, model=model, method=method,
                         n=n, mean_ap_difference=mean, sd_ap_difference=sd,
                         ci95_lower=lo, ci95_upper=hi, p_raw=pvalue))
    summary = pd.DataFrame(rows)
    if len(summary) != 40:
        raise ValueError("Expected exactly 40 primary comparisons")
    summary["p_holm"] = holm_adjust(summary.p_raw.to_numpy())
    summary["reject_at_0_05"] = summary.p_holm < 0.05
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_primary_per_seed.csv", index=False)
    summary.to_csv(output / "paired_primary_inference.csv", index=False)
    (output / "inference_manifest.json").write_text(json.dumps({
        "status": "ANALYSIS_ONLY",
        "method": "two-sided paired t-test on per-seed AP differences",
        "interval": "unadjusted 95% paired t interval",
        "multiple_testing": "Holm correction across all 40 primary comparisons",
        "assumptions": "independent evaluation seeds; approximately suitable paired differences",
        "warning": "Intervals are unadjusted for multiplicity; no automatic prospective-freeze verification."
    }, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "v2_3_inference")
    a = p.parse_args()
    cfg = json.loads(a.protocol.read_text(encoding="utf-8"))
    print(paired_inference(a.results, cfg, a.output).to_string(index=False))
