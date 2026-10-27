import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirmatory_protocol import validate_protocol, analyze_confirmatory, SCENARIOS, METHODS


def protocol():
    return {
        "status": "FROZEN_BEFORE_CONFIRMATORY_RUN",
        "evaluation_seeds": list(range(10000, 10030)),
        "development_seeds": list(range(5)),
        "scenarios": list(SCENARIOS),
        "methods": list(METHODS),
        "models": ["LR", "RF"],
        "primary_metric": "ap",
        "primary_policy": "default",
        "ctgan_epochs_by_model": {"LR": 10, "RF": 10},
        "selection_uses_evaluation_results": False,
    }


def test_valid_protocol():
    assert validate_protocol(protocol())


def test_reject_overlap():
    p = protocol()
    p["evaluation_seeds"][0] = 0
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol(p)


def test_reject_unfrozen():
    p = protocol()
    p["status"] = "EXPLORATORY"
    with pytest.raises(ValueError, match="frozen"):
        validate_protocol(p)


def test_reject_incomplete_results(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame([dict(scenario="s1", seed=10000, method="baseline",
                       model="LR", threshold_policy="default", test_sha256="abc",
                       ap=0.5, roc_auc=0.6, f1=0.4, balanced_accuracy=0.5)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        analyze_confirmatory(path, protocol(), tmp_path / "out")


def test_reject_duplicate_seed():
    p = protocol()
    p["evaluation_seeds"][1] = p["evaluation_seeds"][0]
    with pytest.raises(ValueError, match="distinct"):
        validate_protocol(p)
