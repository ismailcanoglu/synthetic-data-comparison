"""Preflight and paired inference for a future, independently generated confirmatory run.

Does NOT run training or convert exploratory outputs into confirmatory evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t as student_t

SCENARIOS = ("s1", "s2a", "s2b", "s3a", "s3b")
METHODS = ("baseline", "class_weight", "smote", "ctgan", "oracle")
MODELS = ("LR", "RF")
POLICIES = ("default", "val_tuned")
METRICS = ("ap", "roc_auc", "f1", "balanced_accuracy")


def validate_protocol(protocol):
    if protocol.get("status") != "FROZEN_BEFORE_CONFIRMATORY_RUN":
        raise ValueError("Protocol must be explicitly frozen before the run")
    seeds = protocol["evaluation_seeds"]
    if len(seeds) != 30 or len(set(seeds)) != 30 or any(not isinstance(s, int) for s in seeds):
        raise ValueError("Exactly 30 distinct integer evaluation seeds required")
    if not protocol["development_seeds"] or len(set(protocol["development_seeds"])) != len(protocol["development_seeds"]):
        raise ValueError("Development seeds must be nonempty and distinct")
    if set(seeds) & set(protocol["development_seeds"]):
        raise ValueError("Evaluation and development seeds overlap")
    if set(protocol["scenarios"]) != set(SCENARIOS) or set(protocol["methods"]) != set(METHODS):
        raise ValueError("All five scenarios and methods are required")
    if set(protocol["models"]) != set(MODELS):
        raise ValueError("Both models required")
    if protocol["primary_metric"] != "ap" or protocol["primary_policy"] != "default":
        raise ValueError("Primary endpoint must be threshold-independent AP")
    epochs = protocol.get("ctgan_epochs_by_model")
    if not isinstance(epochs, dict) or set(epochs) != set(MODELS):
        raise ValueError("Freeze CTGAN epochs for both LR and RF")
    if any(type(value) is not int or value < 1 for value in epochs.values()):
        raise ValueError("CTGAN epoch counts must be positive integers")
    if protocol.get("selection_uses_evaluation_results", True):
        raise ValueError("Evaluation data cannot be used for hyperparameter selection")
    return True


def analyze_confirmatory(results, protocol, output):
    validate_protocol(protocol)
    df = pd.read_csv(results)
    required = {"scenario", "seed", "method", "model", "threshold_policy", "test_sha256", *METRICS}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    expected = {(s, seed, m, model, policy)
                for s in SCENARIOS for seed in protocol["evaluation_seeds"]
                for m in METHODS for model in MODELS for policy in POLICIES}
    actual = list(zip(df.scenario, df.seed, df.method, df.model, df.threshold_policy))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete, extra, or duplicate scenario/seed/method/model/policy rows")
    if (df.groupby(["scenario", "seed"])["test_sha256"].nunique() != 1).any():
        raise ValueError("Test fingerprints differ within scenario/seed")
    primary = df[df.threshold_policy == protocol["primary_policy"]]
    baseline = primary[primary.method == "baseline"]
    others = primary[primary.method != "baseline"]
    keys = ["scenario", "seed", "model"]
    paired = others.merge(baseline, on=keys, suffixes=("_method", "_baseline"),
                          validate="many_to_one")
    if len(paired) != len(others):
        raise ValueError("Unpaired observations")
    paired["ap_difference"] = paired["ap_method"] - paired["ap_baseline"]
    paired["comparison"] = paired["method_method"]
    rows = []
    # Seed is the independent unit; compare within identical scenario and model.
    for (scenario, model, method), group in paired.groupby(["scenario", "model", "comparison"]):
        x = group.ap_difference.to_numpy(dtype=float)
        n = len(x)
        sd = float(np.std(x, ddof=1))
        se = sd / np.sqrt(n)
        # Two-sided paired (one-sample differences) t test, 95% t interval.
        if n < 2:
            raise ValueError("At least two paired seeds required")
        if sd == 0:
            p_value = 1.0 if float(x.mean()) == 0 else 0.0
        else:
            p_value = float(ttest_1samp(x, popmean=0).pvalue)
        critical = float(student_t.ppf(0.975, df=n - 1))
        rows.append(dict(scenario=scenario, model=model, method=method, n=n,
                         mean_ap_difference=float(x.mean()), sd_ap_difference=sd,
                         ci95_t_lower=float(x.mean() - critical * se),
                         ci95_t_upper=float(x.mean() + critical * se),
                         p_unadjusted=p_value))
    # Holm step-down correction over the full family of 40 primary comparisons.
    if len(rows) != 40:
        raise ValueError(f"Expected 40 primary comparisons, got {len(rows)}")
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_unadjusted"])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = max(adjusted, min(1.0, (len(rows) - rank) * rows[index]["p_unadjusted"]))
        rows[index]["p_holm"] = adjusted
        rows[index]["significant_holm_0_05"] = adjusted < 0.05
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_primary.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "paired_primary_summary.csv", index=False)
    (output / "analysis_manifest.json").write_text(json.dumps({
        "status": "ANALYSIS_ONLY_NOT_PROOF_OF_PROSPECTIVE_FREEZE",
        "primary": "paired AP differences versus baseline",
        "ci": "two-sided 95% Student t interval on paired seed differences (unadjusted)",
        "comparisons": "4 methods x 2 models x 5 scenarios = 40",
        "test": "two-sided one-sample t-test on paired AP differences",
        "multiplicity": "Holm step-down across all 40 primary comparisons; alpha=0.05",
    }, indent=2), encoding="utf-8")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--results", type=Path)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "confirmatory_analysis")
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    if args.results:
        print(analyze_confirmatory(args.results, protocol, args.output).to_string(index=False))
    else:
        print("Protocol shape valid. This does not certify prospective registration.")


if __name__ == "__main__":
    main()
