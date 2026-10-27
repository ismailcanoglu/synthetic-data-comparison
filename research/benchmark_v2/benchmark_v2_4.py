"""Benchmark v2.4: checkpointable runner with per-classifier CTGAN epochs.

Exploratory by default. No confirmatory claims without independently frozen protocol.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_v2 import Settings, sample, augment, derived_seed, digest, residual, METHODS

MODELS = ("LR", "RF")
FIELDS = ["scenario", "seed", "method", "model", "threshold_policy",
          "threshold", "ap", "roc_auc", "f1", "balanced_accuracy", "test_sha256"]


def validate_config(cfg):
    required = {"status", "scenarios", "methods", "seeds", "ctgan_epochs_by_model",
                "fit_n", "validation_n", "test_n", "rf_trees"}
    if not required.issubset(cfg):
        raise ValueError(f"Missing config keys: {required - set(cfg)}")
    if cfg["status"] not in ("EXPLORATORY", "FROZEN_BEFORE_CONFIRMATORY_RUN"):
        raise ValueError("Unknown protocol status")
    if not cfg["seeds"] or len(cfg["seeds"]) != len(set(cfg["seeds"])):
        raise ValueError("Seeds must be nonempty and unique")
    if any(not isinstance(s, int) or s < 0 for s in cfg["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if not cfg["scenarios"] or not set(cfg["scenarios"]).issubset(
            {"s1", "s2a", "s2b", "s3a", "s3b"}):
        raise ValueError("Unknown scenario")
    if not cfg["methods"] or not set(cfg["methods"]).issubset(METHODS):
        raise ValueError("Unknown method")
    if set(cfg["ctgan_epochs_by_model"]) != set(MODELS):
        raise ValueError("Epochs required for LR and RF")
    if any(not isinstance(e, int) or e <= 0 for e in cfg["ctgan_epochs_by_model"].values()):
        raise ValueError("Epochs must be positive integers")
    if any(not isinstance(cfg[k], int) or cfg[k] < 1 for k in
           ("fit_n", "validation_n", "test_n", "rf_trees")):
        raise ValueError("Sample sizes and tree count must be positive")
    return cfg


def check_protocol_match(cfg, protocol):
    """Require exact agreement before any frozen run; no training or test access."""
    validate_config(cfg)
    validate_protocol(protocol)
    if cfg["status"] != "FROZEN_BEFORE_CONFIRMATORY_RUN":
        raise ValueError("Confirmatory config must be frozen")
    comparisons = {
        "seeds": "evaluation_seeds",
        "scenarios": "scenarios",
        "methods": "methods",
        "ctgan_epochs_by_model": "ctgan_epochs_by_model",
    }
    for config_key, protocol_key in comparisons.items():
        if cfg[config_key] != protocol[protocol_key]:
            raise ValueError(f"Protocol mismatch: {config_key}")
    if cfg["seeds"] != list(range(10000, 10030)):
        raise ValueError("Unexpected evaluation seeds")
    if protocol["development_seeds"] != list(range(9100, 9110)):
        raise ValueError("Unexpected development seeds")
    if cfg["ctgan_epochs_by_model"] != {"LR": 10, "RF": 10}:
        raise ValueError("Unexpected CTGAN epoch selection")
    return True


def model_for(name, seed, weighted, trees):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced" if weighted else None,
            random_state=derived_seed(seed, 82)))
    if name == "RF":
        return RandomForestClassifier(n_estimators=trees, n_jobs=-1,
            class_weight="balanced" if weighted else None,
            random_state=derived_seed(seed, 83))
    raise ValueError(name)


def run(cfg, output, protocol=None):
    validate_config(cfg)
    if cfg["status"] == "FROZEN_BEFORE_CONFIRMATORY_RUN":
        if protocol is None:
            raise ValueError("Frozen run requires --protocol")
        check_protocol_match(cfg, protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to silently mix outputs from different configurations.
    canonical = json.dumps(cfg, sort_keys=True).encode("utf-8")
    signature = hashlib.sha256(canonical).hexdigest()
    protocol_signature = (hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest()
                          if protocol is not None else None)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["config_sha256"] != signature or previous.get("protocol_sha256") != protocol_signature:
            raise ValueError("Output directory already belongs to another configuration")
    else:
        manifest_path.write_text(json.dumps({
            "config_sha256": signature, "protocol_sha256": protocol_signature,
            "config": cfg, "protocol": protocol, "python": platform.python_version(),
            "status": cfg["status"]
        }, indent=2), encoding="utf-8")
    results_path = output / "results_per_seed.csv"
    failures_path = output / "failures.csv"
    results = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    failures = pd.read_csv(failures_path).to_dict("records") if failures_path.exists() else []
    # A unit is complete only when both threshold-policy rows are present.
    expected_policies = {"default", "val_tuned"}
    grouped = {}
    for r in results:
        key = (r["scenario"], int(r["seed"]), r["method"], r["model"])
        grouped.setdefault(key, []).append(r["threshold_policy"])
    for key, policies in grouped.items():
        if len(policies) != 2 or set(policies) != expected_policies:
            raise RuntimeError(f"Partial or duplicate checkpoint for {key}; repair before resuming")
    completed = set(grouped)
    settings = Settings()
    for scenario in cfg["scenarios"]:
        for seed in cfg["seeds"]:
            x_all, y_all = sample(cfg["fit_n"] + cfg["validation_n"],
                                  seed, scenario, "train", settings)
            xf, xv, yf, yv = train_test_split(x_all, y_all,
                train_size=cfg["fit_n"], test_size=cfg["validation_n"],
                random_state=derived_seed(seed, 10), stratify=y_all)
            xt, yt = sample(cfg["test_n"], seed, scenario, "test", settings)
            test_hash = digest(xt, yt)
            for method in cfg["methods"]:
                for model_name in MODELS:
                    key = (scenario, seed, method, model_name)
                    if key in completed:
                        continue
                    try:
                        start = time.perf_counter()
                        if method in ("baseline", "class_weight"):
                            xtrain, ytrain = xf, yf
                        else:
                            epochs = cfg["ctgan_epochs_by_model"][model_name]
                            generated = augment(method, xf, yf, scenario, seed, settings, epochs)
                            xtrain = np.vstack((xf, generated))
                            ytrain = np.r_[yf, np.ones(len(generated), dtype=int)]
                        model = model_for(model_name, seed, method == "class_weight",
                                          cfg["rf_trees"])
                        model.fit(xtrain, ytrain)
                        pv = model.predict_proba(xv)[:, 1]
                        pt = model.predict_proba(xt)[:, 1]
                        candidates = np.unique(np.r_[0., pv, 1.])
                        f1_values = [f1_score(yv, pv >= t, zero_division=0) for t in candidates]
                        threshold = float(candidates[int(np.argmax(f1_values))])
                        for policy, t in (("default", .5), ("val_tuned", threshold)):
                            pred = pt >= t
                            results.append(dict(scenario=scenario, seed=seed, method=method,
                                model=model_name, threshold_policy=policy, threshold=t,
                                ap=average_precision_score(yt, pt),
                                roc_auc=roc_auc_score(yt, pt),
                                f1=f1_score(yt, pred, zero_division=0),
                                balanced_accuracy=balanced_accuracy_score(yt, pred),
                                test_sha256=test_hash))
                        completed.add(key)
                        print(f"OK {scenario} seed={seed} {method}/{model_name} "
                              f"{time.perf_counter()-start:.1f}s", flush=True)
                    except Exception as exc:
                        failures.append(dict(scenario=scenario, seed=seed, method=method,
                                             model=model_name, error=repr(exc)))
                        print(f"FAILED {key}: {exc}", flush=True)
                    # Persist after each unit so interruption loses at most one unit.
                    pd.DataFrame(results, columns=FIELDS).to_csv(results_path, index=False)
                    pd.DataFrame(failures, columns=["scenario","seed","method","model","error"]).to_csv(
                        failures_path, index=False)
    df = pd.DataFrame(results, columns=FIELDS)
    expected = len(cfg["scenarios"]) * len(cfg["seeds"]) * len(cfg["methods"]) * 2 * 2
    if len(df) != expected or df.duplicated(
            ["scenario","seed","method","model","threshold_policy"]).any():
        raise RuntimeError(f"Incomplete experiment: {len(df)}/{expected} result rows. "
                           f"See {failures_path}")
    if df[list(("ap", "roc_auc", "f1", "balanced_accuracy", "threshold"))].isna().any().any():
        raise RuntimeError("Missing numeric values in results")
    if not np.isfinite(df[["ap", "roc_auc", "f1", "balanced_accuracy", "threshold"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite numeric values in results")
    if (df.groupby(["scenario", "seed"]).test_sha256.nunique() != 1).any():
        raise RuntimeError("Test fingerprint mismatch")
    df.groupby(["scenario", "method", "model", "threshold_policy"], as_index=False).agg(
        n=("seed", "size"), ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std")).to_csv(
        output / "results_summary.csv", index=False)
    print(f"COMPLETE {len(df)}/{expected} rows; recorded failures={len(failures)}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
    protocol = json.loads(a.protocol.read_text(encoding="utf-8-sig")) if a.protocol else None
    if a.preflight_only:
        if protocol is None:
            raise ValueError("--preflight-only requires --protocol")
        check_protocol_match(cfg, protocol)
        print("PREFLIGHT PASS: config matches protocol; no training was run.")
    else:
        run(cfg, a.output, protocol)
