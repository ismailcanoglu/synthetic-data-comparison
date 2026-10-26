"""Exploratory benchmark v2. Run: python research/benchmark_v2/benchmark_v2.py --seeds 1 --scenarios s1 --ctgan-epochs 10"""
import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generator_v2"))
from generators import SCENARIOS, Settings, sample

METHODS = ("baseline", "class_weight", "smote", "oracle", "ctgan")


def derived_seed(seed, stream):
    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def digest(x, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def residual(x):
    return (np.abs(x[:, 3] - x[:, 0] * x[:, 1])
            + np.abs(x[:, 4] - x[:, 0] ** 2)
            + np.abs(x[:, 5] - np.sin(x[:, 2]))) / 3


def augment(method, x, y, scenario, seed, settings, epochs):
    minority = int(y.sum())
    n_add = int((y == 0).sum() - minority)
    if n_add <= 0:
        raise ValueError("No minority augmentation required")
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        if minority < 6:
            raise ValueError("SMOTE requires at least 6 minority examples")
        sampler = SMOTE(sampling_strategy={1: minority + n_add},
                        random_state=derived_seed(seed, 41))
        xx, _ = sampler.fit_resample(x, y)
        return xx[len(x):]
    if method == "oracle":
        chunks = []
        count = 0
        for attempt in range(100):
            xx, yy = sample(max(300, n_add * 12), derived_seed(seed, 1000 + attempt),
                            scenario, "train", settings)
            chunks.append(xx[yy == 1])
            count += int((yy == 1).sum())
            if count >= n_add:
                return np.vstack(chunks)[:n_add]
        raise RuntimeError("Oracle could not generate enough minority observations")
    if method == "ctgan":
        import inspect
        import random
        import torch
        from ctgan import CTGAN
        # Class-conditional training avoids unreliable post-hoc class filtering.
        df = pd.DataFrame(x[y == 1], columns=[f"x{i}" for i in range(x.shape[1])])
        s = derived_seed(seed, 51)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        kwargs = dict(epochs=epochs, batch_size=100, verbose=False)
        signature = inspect.signature(CTGAN)
        if "enable_gpu" in signature.parameters:
            kwargs["enable_gpu"] = False
        elif "cuda" in signature.parameters:
            kwargs["cuda"] = False
        model = CTGAN(**kwargs)
        model.fit(df)
        generated = model.sample(n_add).to_numpy(dtype=float)
        if generated.shape != (n_add, x.shape[1]) or not np.isfinite(generated).all():
            raise ValueError("Invalid CTGAN samples")
        return generated
    raise ValueError(method)


def run(seeds, scenarios, methods, output, epochs=10, trees=80,
        fit_n=750, val_n=250, test_n=2000):
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    records, quality, failures, splits = [], [], [], []
    for scenario in scenarios:
        for seed in range(seeds):
            x_all, y_all = sample(fit_n + val_n, seed, scenario, "train", settings)
            x_fit, x_val, y_fit, y_val = train_test_split(
                x_all, y_all, train_size=fit_n, test_size=val_n,
                random_state=derived_seed(seed, 10), stratify=y_all)
            x_test, y_test = sample(test_n, seed, scenario, "test", settings)
            test_hash = digest(x_test, y_test)
            splits.append(dict(scenario=scenario, seed=seed, test_sha256=test_hash,
                               n_fit=len(x_fit), n_val=len(x_val), n_test=len(x_test),
                               fit_minority=int(y_fit.sum()), val_minority=int(y_val.sum()),
                               test_minority=int(y_test.sum())))
            rx, ry = sample(10000, derived_seed(seed, 71), scenario, "train", settings)
            reference = rx[ry == 1][:500]
            scaler = StandardScaler().fit(x_fit)
            for method in methods:
                try:
                    t0 = time.perf_counter()
                    generated = None
                    if method not in ("baseline", "class_weight"):
                        generated = augment(method, x_fit, y_fit, scenario, seed, settings, epochs)
                        x_train = np.vstack([x_fit, generated])
                        y_train = np.r_[y_fit, np.ones(len(generated), dtype=int)]
                    else:
                        x_train, y_train = x_fit, y_fit
                    elapsed = time.perf_counter() - t0
                    if generated is not None:
                        # Structural error is compared with the reference baseline, not zero.
                        quality.append(dict(scenario=scenario, seed=seed, method=method,
                            n_added=len(generated), generation_seconds=elapsed,
                            residual_mean=float(residual(generated).mean()),
                            reference_residual_mean=float(residual(reference).mean()),
                            residual_mean_gap=float(abs(residual(generated).mean() - residual(reference).mean())),
                            unique_rounded_4dp=float(len(np.unique(generated.round(4), axis=0)) / len(generated))))
                    for model_name in ("LR", "RF"):
                        weighted = method == "class_weight"
                        if model_name == "LR":
                            model = make_pipeline(StandardScaler(), LogisticRegression(
                                max_iter=2000, class_weight="balanced" if weighted else None,
                                random_state=derived_seed(seed, 82)))
                        else:
                            model = RandomForestClassifier(n_estimators=trees, n_jobs=-1,
                                class_weight="balanced" if weighted else None,
                                random_state=derived_seed(seed, 83))
                        model.fit(x_train, y_train)
                        pv = model.predict_proba(x_val)[:, 1]
                        pt = model.predict_proba(x_test)[:, 1]
                        candidates = np.unique(np.r_[0., pv, 1.])
                        f1_values = [f1_score(y_val, pv >= t, zero_division=0) for t in candidates]
                        tuned = float(candidates[int(np.argmax(f1_values))])
                        for policy, threshold in (("default", .5), ("val_tuned", tuned)):
                            pred = pt >= threshold
                            records.append(dict(scenario=scenario, seed=seed, method=method,
                                model=model_name, threshold_policy=policy, threshold=threshold,
                                ap=average_precision_score(y_test, pt),
                                roc_auc=roc_auc_score(y_test, pt),
                                f1=f1_score(y_test, pred, zero_division=0),
                                balanced_accuracy=balanced_accuracy_score(y_test, pred),
                                test_sha256=test_hash))
                except Exception as exc:
                    failures.append(dict(scenario=scenario, seed=seed, method=method,
                                         error=repr(exc)))
    pd.DataFrame(records).to_csv(output / "results_per_seed.csv", index=False)
    pd.DataFrame(quality).to_csv(output / "quality_per_seed.csv", index=False)
    pd.DataFrame(failures, columns=["scenario", "seed", "method", "error"]).to_csv(
        output / "failures.csv", index=False)
    pd.DataFrame(splits).to_csv(output / "splits.csv", index=False)
    if records:
        df = pd.DataFrame(records)
        summary = df.groupby(["scenario", "method", "model", "threshold_policy"], as_index=False).agg(
            n=("seed", "size"), ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
            f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"))
        summary.to_csv(output / "results_summary.csv", index=False)
    manifest = dict(status="EXPLORATORY_NOT_CONFIRMATORY", generator="research/generator_v2/generators.py",
        settings=asdict(settings), seeds=list(range(seeds)), scenarios=scenarios, methods=methods,
        fit_n=fit_n, validation_n=val_n, test_n=test_n, rf_trees=trees,
        ctgan_epochs=epochs, python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return records, quality, failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["s1"])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--ctgan-epochs", type=int, default=10)
    p.add_argument("--rf-trees", type=int, default=80)
    p.add_argument("--output", type=Path, default=ROOT / "benchmark_v2" / "outputs_smoke")
    a = p.parse_args()
    _, _, failures = run(a.seeds, a.scenarios, a.methods, a.output, a.ctgan_epochs, a.rf_trees)
    print(f"Output: {a.output}; failures: {len(failures)}")
